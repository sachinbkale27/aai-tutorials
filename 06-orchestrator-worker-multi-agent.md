# 06 · Orchestrator–Worker Multi-Agent

> Decompose one task into specialist sub-agents, fan them out in parallel, and fold their findings back into one answer — and know when that coordination actually earns its cost.

## 1. Mental model — the pattern, and when it's justified

The orchestrator–worker pattern has three roles:

1. **Orchestrator (planner):** reads the task and decides *which* specialists to dispatch and in what order. It does not do the work itself.
2. **Workers (specialists):** each owns a narrow job — one system prompt, one small toolset, its own isolated context window. It answers one sub-question and returns a finding.
3. **Synthesizer (reducer):** merges the workers' findings into a single coherent output.

```
                    ┌──> worker A ──┐
   task ─> orchestrator ─> worker B ──┼──> synthesizer ─> answer
                    └──> worker C ──┘
             (plan)     (fan-out, parallel)   (reduce)
```

**When it's justified — genuine decomposition.** The pattern pays off when the task splits into sub-questions that are:

- **Independent** — worker B doesn't need worker A's output to start. That independence is what lets you run them in parallel and what makes the wall-clock win real.
- **Heterogeneous** — each needs different tools, different context, or different expertise (query logs vs. read metrics vs. `git blame` vs. search a runbook KB). One agent juggling all of it thrashes: too many tools in one prompt, context bloated with irrelevant detail.
- **Individually verifiable** — you can look at one worker's finding and judge it on its own.

An incident is the textbook case: "why did checkout-api start throwing 5xx?" genuinely decomposes into *what do the logs say*, *what do the metrics say*, *what changed in the code*, *is there a runbook* — four independent lookups against four different systems.

**When it's over-engineering.** If the sub-tasks are really one chain of reasoning, or they all hit the same tool with the same context, you've faked the decomposition. Symptoms:

- Workers that just re-ask the same question with cosmetically different prompts.
- A "worker" whose output never changes the final answer (delete it).
- Sequential dependencies dressed up as fan-out (A→B→C is a pipeline, not a fan-out — use a single agent or a plain function chain).

The honest test: **would a single competent agent with all the tools do as well, cheaper?** If yes, use one agent (see [02-tool-and-function-calling.md]). Multi-agent buys you parallelism, context isolation, and specialization — you pay for it in coordination tokens and latency floor (the synthesizer waits for the slowest worker). Only spend it when decomposition is real.

## 2. Smallest working example — standalone runnable

No framework. An orchestrator picks workers, `asyncio.gather` fans them out in parallel, a synthesizer reduces. Everything an LLM would do is stubbed as a plain function so the *pattern* is naked. Runnable with `python3 example.py`.

```python
import asyncio, time

# ── 3 WORKERS: each a specialist with its own narrow job ──────────────────
# In production each of these is an LLM call with its own system prompt + tools.
async def log_worker(alert: str) -> str:
    await asyncio.sleep(1.0)                       # pretend: query a log backend
    return "logs: NullPointerException in CartService.total(), first seen 02:14 UTC"

async def metric_worker(alert: str) -> str:
    await asyncio.sleep(1.0)                       # pretend: pull dashboards
    return "metrics: 5xx 0.2%→12% at 02:14, exactly at deploy marker v482"

async def code_worker(alert: str) -> str:
    await asyncio.sleep(1.0)                       # pretend: search recent diffs
    return "code: v482 changed CartService.total(), removed a null guard (author: dana)"

WORKERS = {"logs": log_worker, "metrics": metric_worker, "code": code_worker}

# ── ORCHESTRATOR: decide WHICH workers to run ─────────────────────────────
def plan(alert: str) -> list[str]:
    # Deterministic here for clarity. Section 4 shows the LLM-driven version
    # that returns a SUBSET based on the alert.
    return ["logs", "metrics", "code"]

# ── SYNTHESIZER: reduce findings into one root cause + proposed fix ────────
def synthesize(alert: str, findings: dict[str, str]) -> str:
    joined = "\n".join(f"  - {k}: {v}" for k, v in findings.items())
    # A real synthesizer is an LLM call over `joined`. Stubbed correlation:
    return (f"ALERT: {alert}\nFINDINGS:\n{joined}\n"
            "ROOT CAUSE (confidence: high): deploy v482 removed a null guard in "
            "CartService.total(), causing NPEs → 5xx.\n"
            "PROPOSED REMEDIATION (needs approval): roll back v482.")

# ── THE LOOP: plan → fan-out (parallel) → reduce ──────────────────────────
async def orchestrate(alert: str) -> str:
    chosen = plan(alert)
    t0 = time.perf_counter()
    # fan-out: all chosen workers run concurrently, not one after another
    results = await asyncio.gather(*(WORKERS[name](alert) for name in chosen))
    findings = dict(zip(chosen, results))
    print(f"[{len(chosen)} workers finished in {time.perf_counter()-t0:.2f}s "
          f"(sequential would be ~{len(chosen)*1.0:.0f}s)]")
    return synthesize(alert, findings)

if __name__ == "__main__":
    alert = "5xx rate on checkout-api spiked to 12% at 02:14 UTC"
    print(asyncio.run(orchestrate(alert)))
```

Run it and you'll see three 1-second workers finish in ~1s wall-clock, not 3s — that parallel speedup is the whole reason to fan out independent work. Swap each stub for an LLM/tool call and you have the real thing.

The three moving parts to internalize: **`plan()`** (selection), **`asyncio.gather`** (fan-out), **`synthesize()`** (reduce). Everything else in a production system hangs off these.

## 3. How the On-Call Copilot uses it

The Copilot is exactly this pattern, config-driven. The roster lives in YAML, the loop in `incident.py`, the graph shape in `graph.py`.

### The roster — `config/agents.yaml`

Every role is data, not code (`config/agents.yaml:15-74`). Editing prompts/models/tools means editing YAML, then `POST /api/agents/reload` — no redeploy.

- **`orchestrator`** (`agents.yaml:15-28`) — gets the *stronger* model (`gpt-4o`, while workers default to `gpt-4o-mini`, `agents.yaml:10-13`). Its role: "decide WHICH specialist workers to dispatch and in what order… Prefer the cheapest sufficient set of workers; don't call a worker whose output won't change the diagnosis." It carries **few-shot exemplars** (`agents.yaml:24-28`) mapping an alert → a worker plan, e.g. a disk-usage alert plans only `[metric_fetcher, runbook_retriever]` (no logs, no code). This is the LLM-routing prompt, ready for M1.

- **Four workers** (`agents.yaml:30-57`), each a specialist with its own narrow toolset:
  - `log_analyzer` → tool `query_logs`: extract error signatures, stack traces, first-seen timestamp; cite exact log lines.
  - `metric_fetcher` → tools `fetch_metrics, list_deploys`: error rate, latency, saturation, deploy markers around the window.
  - `code_searcher` → tools `search_code, recent_diffs`: implicated code paths and suspicious recent commits with author.
  - `runbook_retriever` → tool `search_runbooks`, plus a `rag` block (`collection: runbooks, top_k: 4`, `agents.yaml:55-57`): **agentic RAG** — it *decides whether* a runbook is relevant and refuses to force a citation if nothing fits.

  Note the deliberate specialization: each worker sees only 1–2 tools and one job. That's context isolation — `code_searcher` is never distracted by log-query tools.

- **`synthesis`** (`agents.yaml:59-66`) — also `gpt-4o`. The reducer: combine findings into (1) one most-likely root cause with confidence, (2) the evidence, (3) a concrete proposed remediation from the allowed set — and it must state the fix is a *proposal requiring human approval* and never claim it executed anything.

- **`remediation`** (`agents.yaml:68-74`) — runs *after* human approval + the NeMo execution rail; its actions (`restart_service, rollback_deploy, scale_service`) are HITL-gated. This is downstream of the pattern proper.

### The loop — `app/incident.py`

`incident_events()` (`incident.py:77-138`) is the orchestrator loop, streaming SSE the whole way:

1. **Input rails** (`incident.py:82-89`) — refuse unsafe/off-topic alerts before spending any tokens.
2. **Orchestrator plan** (`incident.py:92-94`) — emits an `orchestrator.plan` step.
3. **Workers** (`incident.py:97-110`) — the fan-out loop:
   ```python
   for i, name in enumerate(G.plan_workers(alert), start=3):
       w = AC.worker(name)
       async for ev in _worker(f"s{i}", w, alert):
           if ev.get("type") == "tool.result":
               summary = ev.get("summary")   # the worker's real finding
           yield ev
       findings[name] = summary or f"{name}: no result"
   ```
   `_worker()` (`incident.py:36-56`) runs one sub-agent: it picks the worker's primary tool (`tools[0]`), calls it through `guarded_tool()` (retries + circuit-breaker, from `resilience.py`), and yields the result as that worker's finding. A worker with a `rag` block gets a `retrieve runbooks` step label; the others get `mcp.<tool>` — the label drives both the UI and the stage-timing bucket in `metrics.py`. If the retriever returns a `"<path> — <passage>"`, the path becomes a **citation** (`incident.py:108-110`).
4. **Synthesis** (`incident.py:113-122`) — builds the reducer prompt from `synthesis.role` + the collected `findings`, then streams it through the *output* guardrails via `guarded_reply()`, with a hard-coded `fallback` if the model is unavailable.
5. **HITL gate** (`incident.py:124-134`) — the proposed `rollback_deploy` mutates prod, so `execution_gate()` fires, the fix is stashed in `PENDING`, and the flow *stops* for a human. `resume_events()` (`incident.py:141-156`) picks it up on approve/reject.

### The graph shape — `app/graph.py`

`build_graph()` (`graph.py:44-75`) wires the LangGraph structure — the **fan-out/fan-in edges** are the pattern made explicit:

```python
g.add_edge(START, "orchestrator")
for w in workers:
    g.add_edge("orchestrator", w["name"])   # orchestrator fans OUT to every worker
    g.add_edge(w["name"], "synthesis")       # workers fan back IN to synthesis
g.add_edge("synthesis", "remediation")
g.add_edge("remediation", END)
```

`interrupt_before=["remediation"]` (`graph.py:73`) is the HITL pause baked into the graph. If `langgraph` isn't installed, `build_graph()` returns `None` and `incident.py` runs its own sequential fallback (`graph.py:44-50`) — the app works either way. See [05-langgraph.md] for the graph engine itself.

### Honest note: selection is deterministic today

The orchestrator's job is to pick a *subset* of workers. **Right now it doesn't.** `plan_workers()` (`graph.py:28-31`) is literally:

```python
def plan_workers(alert):
    """Decide which worker sub-agents to run. For now: all of them.
    M1: the orchestrator LLM will pick a smaller, smarter subset."""
    return [w["name"] for w in AC.AGENTS.get("workers", [])]
```

It returns **every** worker, every time. The LLM-driven selection — the whole point of having a `gpt-4o` orchestrator with few-shot routing exemplars — is **future work (M1)**, not shipped. The few-shot plans in `agents.yaml:24-28` and the strong orchestrator model are the *scaffolding* for it, wired but not yet exercised. Likewise the worker node bodies in `graph.py:34-41` are placeholders (`findings[name] = ""`) until M1 fills in real tool-calling. Be precise about this in an interview: the *structure* (fan-out, isolated workers, synthesis, HITL) is genuine; the *intelligent routing* is stubbed as "run all."

## 4. Build it up — variations

Four upgrades over the Section 2 skeleton. Each is a small, self-contained change.

### 4a. Parallel fan-out with graceful degradation

Real workers fail. One down worker must not sink the whole diagnosis — collect partials. Use `return_exceptions=True` so `gather` doesn't cancel siblings on the first error (mirrors the Copilot's "graceful degrade" in `incident.py:53-54`).

```python
async def orchestrate(alert, chosen):
    tasks = [WORKERS[n](alert) for n in chosen]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    findings = {}
    for name, r in zip(chosen, results):
        findings[name] = (f"{name} unavailable — using partial data"
                          if isinstance(r, Exception) else r)
    return synthesize(alert, findings)
```

### 4b. LLM-driven routing (the M1 upgrade)

Replace the deterministic `plan()` with an LLM that returns a subset. This is exactly what `graph.py:plan_workers` is a placeholder for.

```python
ROSTER = {
    "logs":    "query and summarize service logs; error signatures, first-seen ts",
    "metrics": "pull error-rate/latency/saturation + deploy markers",
    "code":    "search recent diffs for implicated code paths",
    "runbook": "retrieve a runbook only if one is genuinely relevant",
}

async def plan_llm(alert: str) -> list[str]:
    system = ("You are an incident commander. Given the alert, return a JSON list "
              "of worker names to dispatch. Pick the CHEAPEST SUFFICIENT set — omit "
              "a worker whose output won't change the diagnosis.\n"
              f"Available workers: {ROSTER}\n"
              # few-shot, straight from config/agents.yaml:24-28
              'Alert: "disk usage on db-primary at 94%" -> ["metrics","runbook"]')
    raw = await call_llm(system=system, user=alert)   # your LLM client
    chosen = json.loads(raw)
    return [w for w in chosen if w in ROSTER]          # validate against the roster
```

Always validate the returned names against the real roster — an LLM will occasionally invent a worker. This is where the `gpt-4o` orchestrator model and the few-shot exemplars finally earn their keep: a disk alert dispatches 2 workers instead of 4, cutting cost and latency.

### 4c. Worker specialization — isolated context per worker

The core discipline: each worker gets **only** its own system prompt and **only** its tools — never the full roster's tools, never another worker's context. This is why the Copilot gives `log_analyzer` just `[query_logs]` and `code_searcher` just `[search_code, recent_diffs]` (`agents.yaml:35,47`).

```python
async def run_worker(spec: dict, alert: str) -> str:
    # spec = one entry from agents.yaml workers[]
    messages = [{"role": "system", "content": spec["role"]},   # ONLY this worker's prompt
                {"role": "user", "content": alert}]
    return await call_llm(messages=messages, tools=spec["tools"])  # ONLY its tools
```

Isolation keeps each prompt small and on-task, and means a prompt-injected log line can't reach the code-search tools. It also makes workers independently swappable and testable.

### 4d. A judge / critic worker

Add a worker that doesn't gather evidence but *evaluates* the synthesis — a self-check before a human ever sees it. Runs *after* synthesis (it depends on the output), so it's a second stage, not part of the fan-out.

```python
async def critic(alert: str, draft: str, findings: dict) -> dict:
    system = ("You are a skeptical reviewer. Given the findings and the proposed "
              "root cause, return JSON {approve: bool, confidence: 0-1, gaps: [...]}. "
              "Reject if the root cause isn't supported by at least two findings.")
    return json.loads(await call_llm(system=system,
                                     user=f"FINDINGS:{findings}\nDRAFT:{draft}"))

# in the loop:
draft = synthesize(alert, findings)
review = await critic(alert, draft, findings)
if not review["approve"] or review["confidence"] < 0.6:
    # re-run with the critic's gaps fed back, or dispatch the missing worker
    ...
```

A critic worker catches confident-but-unsupported root causes — the multi-agent analog of the LLM-as-judge in [12-evaluation-and-regression.md]. Keep it cheap and give it veto power, not authorship.

## 5. Gotchas & pitfalls

- **Don't fake decomposition.** The cardinal sin. If two workers ask the same question of the same tool with the same context, you have one worker wearing a costume. The Copilot's four workers hit four *different* backends (logs, metrics, code, runbooks) — that's the bar. Before adding a worker, ask: "does its output change the diagnosis?" (the orchestrator prompt literally says this, `agents.yaml:21-22`). If no, cut it.

- **Coordination has a cost.** Every worker is its own LLM call (input + output tokens), plus the orchestrator's planning call, plus the synthesizer's reduce call. A 4-worker incident is ~6 model calls. Multi-agent burns *more* tokens than a single agent — you're buying parallelism and isolation, not efficiency. Anthropic's own data puts multi-agent token use at roughly 15× a single chat. Only spend it where the task genuinely decomposes.

- **Latency floors at the slowest worker.** Fan-out with `gather` means wall-clock = max(worker times), not sum — good. But one slow worker (a timing-out log query) stalls synthesis. Add per-worker timeouts (`asyncio.wait_for`) and degrade gracefully (4a) rather than block forever.

- **Context isolation per worker is a feature, not an accident.** Give each worker only its prompt and its tools (4c). Benefits: smaller/cheaper prompts, no cross-contamination, injection blast-radius contained to one worker, and independent testability. The moment you pass the whole shared blackboard into every worker, you've lost the isolation you paid for.

- **The synthesizer is a real component, not a `str.join`.** Merging findings — resolving contradictions, weighting confidence, picking one root cause — is genuine reasoning. Give it a strong model (`synthesis` uses `gpt-4o`, `agents.yaml:61`) and a prompt that forces evidence + confidence (`agents.yaml:62-66`).

- **Validate LLM routing output.** When the orchestrator picks workers (4b), it will occasionally return a nonexistent name or an empty list. Filter against the real roster; fall back to a safe default set if the plan is empty.

- **Keep mutation out of the pattern.** Workers should be read-only investigators. Anything that touches prod (`rollback_deploy`) belongs *after* synthesis, behind the HITL gate (`incident.py:124-134`, `graph.py:73`). Never let a worker mutate.

- **Be honest about what's wired vs. stubbed.** In this repo, selection is deterministic (`plan_workers` returns all) and worker node bodies are placeholders. Claiming "LLM-driven routing" of the current code would be wrong. Structure real; routing pending (M1).

## ✅ Best Practices

- **Decompose only when the split is real.** Reach for multi-agent when sub-tasks are independent, heterogeneous, and individually verifiable — otherwise ship a single agent with all the tools.
- **Give each worker a narrow role and isolated context.** Hand a worker only its own system prompt and its own 1–2 tools, never the full roster — that keeps prompts small, contains injection blast-radius, and makes workers independently testable.
- **Fan out independent work in parallel.** Dispatch workers concurrently with `asyncio.gather` (with `return_exceptions=True` and per-worker `asyncio.wait_for` timeouts) so wall-clock is the slowest worker, not the sum.
- **Keep orchestrator routing cheap and bounded.** Have the planner pick the cheapest sufficient subset of workers, prime it with few-shot alert→plan exemplars, and validate returned names against the real roster with a safe default fallback.
- **Synthesize deliberately with a strong model.** Treat the reducer as genuine reasoning — give it a capable model and a prompt that forces it to weigh evidence, resolve contradictions, and attach a confidence to one chosen root cause.
- **Add a verifier/critic stage for high-stakes output.** Run a skeptical post-synthesis reviewer with veto power (not authorship) that rejects root causes unsupported by multiple findings, and cap re-dispatch retries so it can't loop.
- **Budget the coordination cost explicitly.** Instrument model calls, tokens, and stage latency per run, and periodically benchmark the multi-agent path against a single-agent baseline to confirm the overhead still earns its keep.
- **Make the roster config-driven.** Keep worker prompts, models, and tools as data (YAML) that hot-reloads without a redeploy, so adding or tuning a specialist never requires touching the orchestration code.

## 6. Exercises

1. **Run and time it.** Run the Section 2 example. Add a fourth worker that sleeps 2s. Confirm total wall-clock is ~2s (the max), not ~5s (the sum). Now make one worker `raise` and add the 4a graceful-degrade handling so synthesis still runs on the survivors.

2. **Make the orchestrator LLM pick a subset.** Replace `plan()` (Section 2) / `plan_workers()` (`graph.py:28-31`) with the 4b `plan_llm()` against a real model. Feed it the two few-shot exemplars from `agents.yaml:24-28`. Verify that "disk usage on db-primary at 94%" dispatches only `[metric_fetcher, runbook_retriever]` and that a 5xx alert dispatches all four. Add roster validation so an invented worker name is dropped.

3. **Add a new worker.** Add a `dependency_checker` worker to `config/agents.yaml` (a `role`, a `tools:` list, following the `agents.yaml:31-54` shape) that checks upstream service health. Trace what else must change for it to run: it auto-joins the fan-out because `graph.py:57-58,64-66` and `incident.py:98` both iterate the roster — no code edit needed. Confirm that, and note it in your answer as evidence of config-driven design.

4. **Add a critic worker.** Implement 4d as a post-synthesis stage. Give it veto power: if it rejects, re-dispatch the single worker whose evidence was weakest and re-synthesize once. Cap the retries at 1 so you can't loop forever.

5. **Cost/latency ledger.** Instrument the Section 2 loop to count model calls and (mock) tokens per run. Compare: (a) run-all-workers vs. (b) LLM-routed subset from exercise 2, across five different alerts. Quantify the savings. This is the argument you'd make in an interview for *why* routing matters.

6. **Single-agent baseline (the honesty check).** Build a *single* agent with all six tools and one prompt that does the whole incident diagnosis. Compare its output quality, token cost, and latency against the multi-agent version on three alerts. Write two sentences on when the multi-agent overhead is justified and when it isn't — this is the Section 1 test applied to real numbers.

## 7. Connections

- **[05-langgraph.md]** — the graph engine that owns the fan-out/fan-in *shape*. `app/graph.py` builds the `orchestrator → [workers] → synthesis → remediation` edges and the `interrupt_before` HITL pause; this tutorial is the pattern that graph encodes.
- **[02-tool-and-function-calling.md]** — each worker *is* a tool-calling agent over its own narrow toolset (`agents.yaml` `tools:`). Understand single-agent tool calling first; a worker is one of those with a tighter scope. The "single agent vs. multi-agent" decision in Section 1 hinges on this.
- **[12-evaluation-and-regression.md]** — the critic/judge worker (4d) is LLM-as-judge applied inside the workflow; and you evaluate a multi-agent system per-worker (is each finding right?) *and* end-to-end (is the synthesis right?). Regression-test the router's selections too.

## 8. Further reading

- Anthropic — **"How we built our multi-agent research system"** (engineering blog): orchestrator–worker in production, the ~15× token-cost finding, and when parallel sub-agents win. anthropic.com/engineering/multi-agent-research-system
- Anthropic — **"Building effective agents"**: the canonical taxonomy of workflows vs. agents; the *orchestrator–workers* and *evaluator–optimizer* (critic) patterns, and the "don't add agentic complexity unless it pays" principle. anthropic.com/engineering/building-effective-agents
- Anthropic docs — **Agent SDK / subagents**: spawning isolated sub-agents with their own context and tools — the productized form of Section 4c's context isolation.
- LangGraph docs — **multi-agent / supervisor and map-reduce (`Send`) patterns**: how the fan-out in `app/graph.py` maps onto a real graph runtime (see [05-langgraph.md]).
