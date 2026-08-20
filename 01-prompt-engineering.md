# 01 · Prompt Engineering

> After this you can design system/role prompts, add few-shot exemplars, template prompts from data, and drive all of it from YAML instead of hard-coding strings — the exact pattern the On-Call Copilot uses to turn worker findings into a guarded root-cause diagnosis.

## 1. Mental model — what it is and why it matters

A chat LLM sees an ordered list of messages, each with a `role`:

- **`system`** — the standing instructions: who the model is, what it may/may not do, output shape. Highest-leverage text you write; it applies to every turn.
- **`user`** — the actual request/data for this turn.
- **`assistant`** — the model's replies (and, in few-shot, *fake* prior replies you supply as worked examples).

Prompt engineering is the discipline of shaping those messages so a general model behaves like a *specific* component. Four levers, roughly in order of power:

1. **System / role prompt** — set identity + constraints once. ("You are the on-call incident commander…")
2. **Few-shot exemplars** — show 1–5 input→output pairs instead of describing the format in prose. The model pattern-matches the shape.
3. **Structured / templated output** — demand JSON (or a fixed skeleton) so downstream code can parse the result, not scrape prose.
4. **Guardrailed generation** — wrap the call so unsafe input never reaches the model and unsafe output never reaches the user.

Why it matters for an AI engineer: the model is fixed; the prompt is your entire API surface. And in a real system the prompt is **data, not code** — you want to edit behavior without a redeploy. The On-Call Copilot puts every prompt, exemplar, and tool list in `config/agents.yaml` and reloads it at runtime, so a prompt tweak is a YAML edit + `POST /api/agents/reload`, not a code change.

## 2. Smallest working example

A standalone system + few-shot call. It runs against OpenAI if you have a key; otherwise it prints the exact message list you'd send, so you can still see the structure.

```bash
pip install openai>=1.0
export OPENAI_API_KEY=sk-...      # optional; omit to see the message structure only
python triage.py
```

```python
# triage.py — classify an alert's severity with a system prompt + few-shot exemplars
import os, json

messages = [
    # 1) SYSTEM: identity + constraints + exact output contract
    {"role": "system", "content": (
        "You are an on-call triage assistant. Classify each alert's severity as "
        "exactly one of: SEV1, SEV2, SEV3. Reply with ONLY the label, nothing else."
    )},
    # 2) FEW-SHOT: worked examples as fake user/assistant turns (teach the shape)
    {"role": "user", "content": "disk usage on db-primary at 94%"},
    {"role": "assistant", "content": "SEV2"},
    {"role": "user", "content": "5xx rate on checkout-api spiked to 12% at 02:14 UTC"},
    {"role": "assistant", "content": "SEV1"},
    {"role": "user", "content": "nightly batch job finished 3 min late"},
    {"role": "assistant", "content": "SEV3"},
    # 3) THE REAL REQUEST
    {"role": "user", "content": "latency p99 on search-api climbed from 200ms to 1.4s"},
]

if not os.getenv("OPENAI_API_KEY"):
    print("No API key — here is the exact message list you'd send:")
    print(json.dumps(messages, indent=2))
else:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,          # deterministic classification
        max_tokens=4,           # a label is tiny — cap it
        messages=messages,
    )
    print(resp.choices[0].message.content)   # → SEV2 (a p99 latency regression, not a full outage)
```

**Observe:** with the three exemplars the model returns a bare label (`SEV2`), no preamble, no explanation. Delete the few-shot turns and it will often add a sentence of reasoning — proving the exemplars, not the prose, enforced the format. `temperature=0` makes it repeatable.

## 3. How the On-Call Copilot uses it

Every prompt in the project is config, loaded once by `app/agent_config.py` and looked up by name:

```python
# app/agent_config.py
AGENTS = _load("agents.yaml")   # orchestrator + workers + synthesis + remediation
def worker(name):
    return next((w for w in AGENTS.get("workers", []) if w.get("name") == name), None)
def reload():                   # POST /api/agents/reload — re-read YAML, no restart
    global AGENTS, TOOLS
    AGENTS, TOOLS = _load("agents.yaml"), _load("tools.yaml")
```

**Role prompts live in `config/agents.yaml`.** Each agent is a named block whose `role:` is its system prompt. Note how tightly scoped they are — the synthesis role even pre-specifies the output sections and a hard safety constraint:

```yaml
# config/agents.yaml
synthesis:
  name: synthesis
  model: gpt-4o
  role: >
    Combine the workers' findings into: (1) a single most-likely ROOT CAUSE with a
    confidence, (2) the evidence for it, and (3) a concrete PROPOSED REMEDIATION as
    one of the allowed remediation actions. Be explicit that the remediation is a
    PROPOSAL requiring human approval. Never claim you executed anything.
```

Workers get narrower roles — e.g. the runbook retriever's role encodes an anti-hallucination rule directly in the prompt:

```yaml
  - name: runbook_retriever
    role: >
      Decide whether a runbook or past postmortem is relevant. If so, retrieve the
      most relevant passages ... and cite them. If nothing is relevant, say so —
      do not force a citation.
```

**Few-shot exemplars are config too.** The orchestrator block carries example `alert → plan` pairs so the planner can pattern-match which workers to dispatch:

```yaml
# config/agents.yaml — orchestrator
  few_shot:
    - alert: "5xx rate on checkout-api spiked to 12% at 02:14 UTC"
      plan: ["metric_fetcher", "log_analyzer", "code_searcher", "runbook_retriever"]
    - alert: "disk usage on db-primary at 94%"
      plan: ["metric_fetcher", "runbook_retriever"]
```

Be honest about the current state: these exemplars are staged in YAML for the planner, but the code that would feed them to an orchestrator LLM is an M1 placeholder — `app/graph.py:plan_workers()` currently returns *all* workers (`[w["name"] for w in AC.AGENTS.get("workers", [])]`) rather than an LLM-chosen subset. The prompt-engineering scaffolding is in place ahead of the model call that consumes it.

**Prompt assembly is templating from data.** `app/incident.py` builds the synthesis prompt at runtime by concatenating the config role with the workers' collected findings, then sends it as a `system` + `user` message pair:

```python
# app/incident.py — step 4, SYNTHESIS
syn = AC.AGENTS.get("synthesis", {})
prompt = (syn.get("role", "Give a root cause and a remediation that needs approval.")
          + "\n\nFindings:\n" + "\n".join(f"- {k}: {v}" for k, v in findings.items()))
fallback = ("Likely root cause: a recent checkout-api deploy caused the 5xx spike. "
            "Proposed remediation (needs approval): roll back to the previous deploy.")
async for ev in guarded_reply([{"role": "system", "content": prompt},
                               {"role": "user", "content": alert}],
                              fallback_text=fallback, citations=citations, final=False):
    yield ev
```

Three things worth internalizing here:
- The **system message is composed** (`role` from YAML + a rendered `Findings:` block), so behavior comes from config but the *content* comes from live worker results.
- There's a **`.get("role", ...)` default** — if the YAML block is missing, prompting still has a sane fallback string.
- The whole call goes through `guarded_reply` (see §4) — the model never speaks to the user directly.

## 4. Build it up

### 4a. Few-shot exemplars from config (the orchestrator pattern)
Turn the YAML `few_shot` pairs into real few-shot messages the way an LLM planner would consume them:

```python
import app.agent_config as AC   # or yaml.safe_load(open("config/agents.yaml"))
orch = AC.AGENTS["orchestrator"]

msgs = [{"role": "system", "content": orch["role"].strip()}]
for ex in orch.get("few_shot", []):
    msgs.append({"role": "user", "content": ex["alert"]})
    msgs.append({"role": "assistant", "content": str(ex["plan"])})   # the exemplar output
msgs.append({"role": "user", "content": "latency p99 on search-api climbed to 1.4s"})
# → send msgs to the model; it emits a worker list shaped like the exemplars
```

The exemplars teach the *output shape* (a JSON-ish list of worker names) far more reliably than describing it in prose.

### 4b. Structured output — force parseable JSON
When code has to consume the result, ask for JSON and (on OpenAI) enforce it:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini", temperature=0,
    response_format={"type": "json_object"},     # model MUST return valid JSON
    messages=[
        {"role": "system", "content":
         'Return JSON: {"root_cause": str, "confidence": "low|med|high", '
         '"remediation": "restart_service|rollback_deploy|scale_service"}. '
         'Never invent a remediation outside that enum.'},
        {"role": "user", "content": "findings:\n- metric_fetcher: 5xx spiked after 02:10 deploy"},
    ],
)
import json; data = json.loads(resp.choices[0].message.content)   # safe to parse
```

Note the enum echoes `remediation.actions` in `agents.yaml` (`[restart_service, rollback_deploy, scale_service]`) — the prompt and the real allowed-action list stay in lockstep.

### 4c. Prompt templating — compose system text from live data
This is exactly `incident.py`'s move, generalized. Keep the invariant instruction in config, render the variable part per request:

```python
ROLE = AC.AGENTS["synthesis"]["role"]           # invariant, from YAML
def build_synthesis_prompt(findings: dict) -> str:
    body = "\n".join(f"- {k}: {v}" for k, v in findings.items())
    return f"{ROLE}\n\nFindings:\n{body}"

prompt = build_synthesis_prompt({"metric_fetcher": "5xx after deploy",
                                 "runbook_retriever": "runbooks/checkout.md — roll back"})
```

Separating the stable role (config) from the rendered data (code) is what lets a non-engineer tune tone/constraints in YAML without touching Python.

### 4d. Guardrailed generation — never call the model raw
The project never calls the LLM directly for the user-facing reply; it routes through `app/rails.py:guarded_reply`, which runs input rails → LLM → output rails and degrades to a scripted fallback if generation fails:

```python
# app/rails.py — guarded_reply (abridged)
if config.RAILS is not None and messages:
    try:
        content, activated = rails_generate(config.RAILS, messages)   # input→LLM→output rails
        for a in activated:
            yield {"type": "rail.fired", "rail_type": ..., "name": a.get("name", "rail"),
                   "stop": a.get("stop", False), ...}
        async for ev in say(content or fallback_text):                # stream guarded text
            yield ev
        ...
        return
    except Exception as e:
        print(f"[guardrails] generate failed ({e}) → demo fallback")
# demo fallback — emit configured output rails, then the scripted reply
for r in config.DEMO.get("output_rails", []):
    yield {"type": "rail.fired", "rail_type": "output", "name": r.get("name", "output rail"), ...}
async for ev in say(fallback_text):
    yield ev
```

Prompt-engineering takeaway: a good prompt is necessary but not sufficient. `guarded_reply` is the belt-and-suspenders layer — even a perfect synthesis prompt gets wrapped so a jailbroken input or a leaked-secret output is caught by rails, and a model outage yields `fallback_text` instead of a stack trace. See [08-nemo-guardrails.md] for the rail runtime itself.

## 5. Gotchas & pitfalls

- **Put prompts in config, not string literals.** The whole project pivots on this: `agents.yaml` + `reload()` = tune behavior with no redeploy. Hard-coded prompts rot.
- **Always give the assembled prompt a default.** `syn.get("role", "Give a root cause…")` in `incident.py` means a missing/renamed YAML key degrades instead of crashing.
- **Few-shot teaches shape, not facts.** Use 2–5 exemplars to lock output *format*; don't stuff knowledge you actually need at runtime — that's what retrieval (the `runbook_retriever`) is for.
- **`temperature=0` for anything parsed or classified.** Classification, planning, JSON extraction want determinism. Save higher temperature for prose.
- **Cap `max_tokens` to the job.** `defaults.max_tokens: 512` for workers in `agents.yaml`; a severity label needs 4. Over-budgeting costs money and invites rambling.
- **Constraints belong *in* the prompt.** The synthesis role literally says "Never claim you executed anything" and the retriever says "do not force a citation." Encode the non-negotiables as text, then back them with rails.
- **Structured output needs enforcement.** Asking for JSON in prose isn't enough — use `response_format={"type":"json_object"}` (OpenAI) or tool/function schemas, and still `try/except` the `json.loads`.
- **Keep the prompt's enums in sync with real code.** The remediation enum in a prompt must match `remediation.actions` in `agents.yaml`; drift means the model proposes actions that don't exist.
- **Order matters:** system first, then few-shot pairs, then the real user turn. Exemplars after the real request confuse the model.
- **Don't trust the model to gate itself.** Even with "requires human approval" in the synthesis prompt, the actual stop is `execution_gate` + the HITL pause in `incident.py`, not the prompt. Prompts advise; code enforces.

## ✅ Best Practices

- **Keep every prompt in `config/agents.yaml`, not in code.** Treat prompts as data so behavior tunes via a YAML edit + `POST /api/agents/reload`, never a redeploy.
- **Version your prompt config in git.** Commit `agents.yaml` changes with the same review discipline as code, so any role tweak is diffable, attributable, and revertable.
- **Scope each role narrowly and pin its output contract in the system text.** Give one agent one job and state the exact output sections up front, the way the `synthesis` role pre-specifies root cause / evidence / remediation.
- **Reach for few-shot deliberately to lock output shape.** Add 2–5 exemplars (like the orchestrator's `alert → plan` pairs) when you need a stable format; skip them when a plain instruction already suffices.
- **Demand structured output whenever code consumes the result.** Request JSON and enforce it (`response_format={"type":"json_object"}` or a tool schema) so downstream parsing never scrapes prose.
- **Encode behavioral constraints as explicit prompt text.** Write the non-negotiables ("Never claim you executed anything", "requires human approval") directly into the role so intent is legible and testable.
- **Test prompts against a golden set before shipping.** Keep a fixed suite of input→expected pairs and re-run it after every role edit to catch regressions the way deleting the §2 exemplars would surface them.
- **Keep the base system prompt stable; render only the variable part per request.** Hold the invariant role in config and template just the live data (findings, alert) into the user turn, so tuning tone or constraints never means touching Python.

## 6. Exercises

1. **Run §2 both ways.** Execute `triage.py` without a key (see the message list), then with a key. Now delete the three few-shot turns and rerun — observe the model start explaining itself. That regression *is* the lesson.
2. **Edit a role in config.** Change `synthesis.role` in `config/agents.yaml` to also emit a one-line "blast radius" estimate. Trace how `incident.py` picks it up via `AC.AGENTS.get("synthesis")` — confirm no Python changed.
3. **Wire up the orchestrator few-shot.** Using the code in §4a, load `orchestrator.few_shot` from the real YAML and build the message list. Send it to a model and get back a worker plan. Compare to `graph.plan_workers()`, which currently returns *all* workers.
4. **Add structured output.** Rewrite the synthesis call to demand the JSON schema from §4b and `json.loads` it. Validate `remediation` against `agents.yaml`'s `remediation.actions` list; reject anything off-enum.
5. **Template a new worker prompt.** Add a `dependency_checker` worker block to `agents.yaml` with a tight `role`, then write the `build_*_prompt()` helper (§4c style) that renders its findings into a system message.
6. **Break the guardrail path.** Feed `guarded_reply` a `messages` list but simulate `rails_generate` raising (or set `config.RAILS = None`). Confirm you get `fallback_text` and the configured output rails, not an exception — then explain why the prompt alone couldn't have saved you.

## 7. Connections

- [02-tool-and-function-calling.md] — few-shot planning (§4a) becomes real when the orchestrator emits tool calls; workers' `tools:` in `agents.yaml` map to those functions.
- [08-nemo-guardrails.md] — the input/output rails wrapping every prompt in `guarded_reply`; where "prompt advises, code enforces" is implemented.
- [06-rag-and-retrieval.md] — the `runbook_retriever` role and why few-shot is the wrong tool for injecting runtime facts.
- [07-multi-agent-orchestration.md] — how `agents.yaml` roles compose into the orchestrator → workers → synthesis graph in `app/graph.py`.

## 8. Further reading

- OpenAI — Prompt engineering guide: https://platform.openai.com/docs/guides/prompt-engineering
- OpenAI — Structured Outputs / JSON mode: https://platform.openai.com/docs/guides/structured-outputs
- Anthropic — Prompt engineering overview: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Anthropic — Use examples (multishot / few-shot): https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/multishot-prompting
- Anthropic — System prompts / giving Claude a role: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts
