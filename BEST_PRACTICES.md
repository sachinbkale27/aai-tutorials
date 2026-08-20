# General Best Practices — Agentic AI Engineering

Cross-cutting principles that apply to **any** agentic/LLM system, independent of this curriculum's
On-Call Copilot. The per-tutorial `## ✅ Best Practices` sections are component-specific; this is the
universal layer — the judgment an interviewer probes for and production teams rely on.

---

## 1. Start simple — earn complexity
- **A single well-prompted LLM call beats an agent** for most tasks; add tools, then loops, then multiple agents only when the simpler design demonstrably fails.
- **Don't fake decomposition.** Multi-agent orchestration is justified only when the task genuinely splits into independent sub-tasks — otherwise it's cost and latency for no benefit.
- **Prefer the smallest capable model** for each step; reserve the frontier model for the hard reasoning and route cheap sub-tasks (routing, extraction) to smaller ones.
- **Make control flow explicit** (a graph/state machine) rather than hoping the model drives itself; deterministic control + model-driven reasoning is the reliable combination.

## 2. Prompting
- **Put prompts in config, not code**, and version them — a prompt change is a behavior change that deserves review and rollback.
- **Show, don't just tell:** few-shot exemplars and structured (JSON/schema) outputs beat long prose instructions for consistency.
- **Set explicit behavioral constraints and a clear role**; state what NOT to do, not only what to do.
- **Pin models and test prompts against a golden set** — model upgrades silently change behavior.

## 3. Tools & function calling
- **Write precise tool descriptions and typed schemas** — the model chooses tools from the docs you give it; vague descriptions cause wrong calls.
- **Keep tools idempotent where possible**, validate/parse arguments defensively, and never blindly execute model-provided arguments against production.
- **Bound the tool-call loop** (max iterations, total timeout) so a confused agent can't loop forever.
- **Return tool errors to the model as messages** so it can recover, rather than crashing the run.

## 4. Retrieval & knowledge (RAG)
- **Let the agent decide whether to retrieve** — always-retrieve wastes tokens and injects noise on questions that don't need it.
- **Always cite sources** so answers are auditable and users can verify.
- **Invest in retrieval quality** (chunking with overlap, hybrid search, reranking, metadata filters) before blaming the LLM — most RAG failures are retrieval failures.
- **Keep the corpus fresh** with an ETL/re-index pipeline; stale knowledge is a silent correctness bug.

## 5. Evaluation — the real differentiator
- **Write the golden set first; it IS the spec.** You can't improve what you can't measure.
- **Grade the trajectory, not just the final answer** — an agent can reach a right answer via the wrong path and will break when the pattern shifts.
- **Use precision/recall/FPR, not accuracy, for rare events** (attacks, failures); accuracy is a trap under class imbalance.
- **Validate LLM-as-judge against human labels** (agreement / Cohen's κ) before trusting it at scale.
- **Gate CI on eval thresholds** including cost/latency drift — a regression suite is the only thing that scales trust across changes.

## 6. Safety & guardrails
- **Defense in depth:** layer input rails (block bad requests), output rails (catch bad responses), and execution rails (gate actions) — no single check is enough.
- **Bias safety checks toward recall.** A missed attack is often catastrophic and irreversible; a false block is recoverable. Accept some over-blocking.
- **Human-in-the-loop for irreversible/high-stakes actions** — assist, don't auto-act, where a wrong move is expensive.
- **Least privilege:** give agents the narrowest tool/credential scope that works; treat every tool as an attack surface.
- **Guard against prompt injection** — untrusted content (retrieved docs, user input, tool output) can carry instructions; never let it silently escalate privileges.

## 7. Reliability & production
- **Only retry idempotent operations**; retrying a non-idempotent mutation can double-charge or double-act.
- **Exponential backoff with jitter**, capped attempts, and a total timeout — naive retries cause thundering herds.
- **Circuit breakers per dependency** so one flaky service fails fast instead of hanging the whole system; add half-open trials to recover.
- **Degrade gracefully** — return partial results with a clear caveat rather than crashing when a tool or model is down.
- **Define fallback chains** (a cheaper model, a cached answer, a simpler path) for when the primary route fails.

## 8. Observability & cost
- **Trace everything** — one span tree per request across nodes/tools/model calls; you can't debug an agent you can't see.
- **Track tokens and cost per call**, attributed to model and feature — LLM cost is a first-class production metric, not an afterthought.
- **Instrument with OpenTelemetry** (vendor-neutral) so the backend (Datadog/Grafana/Honeycomb) stays swappable without code changes.
- **Set latency budgets and watch percentiles (p95/p99)**, not averages — tail latency is what users feel.
- **Optimize cost with caching, model tiering, cheap-check-first ordering, and streaming** — measure before optimizing.

## 9. Config & deployment
- **Behavior in config, logic in code** — declarative YAML for prompts/tools/policy; don't push branching logic into config.
- **Secrets in environment variables, never in config files or prompts**, and never log them.
- **Validate configs at load** (schema/pydantic) so a typo fails fast at startup, not mid-incident.
- **Make everything reproducible** — pin dependencies, containerize, and script the environment so "works on my machine" isn't a risk.
- **Provide a real-vs-fallback path** so the system still runs (degraded) when a key/model/service is absent.

## 10. Data & security
- **Handle PII deliberately** — detect/redact sensitive data on input, retrieval, and output; know what leaves your boundary.
- **Keep an audit trail** of every agent action (especially mutations and approvals) for compliance and debugging.
- **Assume anything sent to an external LLM/API may be logged** — don't send secrets or regulated data without a contract that permits it.

## 11. Engineering discipline & judgment
- **Be honest about what's real.** Don't overclaim "production" or fabricated metrics — a savvy reviewer will find the gap, and honesty is more credible than polish.
- **Isolate dependency-heavy tools** (evaluators, experimental libs) in separate venvs rather than risking a pinned working stack.
- **Document decisions, not just code** — record *why* you chose LangGraph over CrewAI, this threshold, that model; the "why" is what you defend later.
- **Iterate against real distributions** — grow eval sets and prompts from production traffic, not hand-picked happy paths.

## 12. Team & lifecycle
- **Version prompts, tools, and eval sets** like code — they're the real behavior surface.
- **Monitor in production** (quality, cost, latency, guardrail hit-rate) — offline eval is necessary but not sufficient; behavior drifts with traffic and model updates.
- **Plan for model migration** — pin versions, keep an eval harness, and re-run it on every model/provider change.
- **Close the loop:** feed production failures back into the golden set so the same bug can't regress twice.

---

## 13. 💡 A novel practice — the Reversibility Ladder

Most systems decide "does this action need approval?" **ad-hoc, per action** — a scattered pile of
`if action == "delete": require_approval()` checks. That doesn't scale and it's fragile: the day
someone adds a new dangerous tool and forgets the guard, you have an incident. The novel move is to
make **safety a property of the tool's *reversibility*, declared once, and enforced automatically.**

Tag every tool with **how hard it is to undo**, and bind each tier to a mandatory safeguard the
framework applies for you — no per-call decision:

| Tier | Reversibility | Examples | Auto-attached safeguard |
|---|---|---|---|
| **0** | Read-only | query logs, fetch metrics, search | trace only |
| **1** | Fully reversible | create a draft, add a label, cache write | log + allow |
| **2** | Costly-reversible | send a message, scale a service, spend a token budget | soft-gate: confirm or budget check |
| **3** | Irreversible / high blast-radius | delete data, deploy, transfer money, `rm -rf` | **mandatory HITL + audit trail** |

**Why this is better than ad-hoc gating:**
- **Adding a tool can't silently skip its guard.** You declare its reversibility; the right safeguard is *derived*, not remembered. A new `drop_table` tool tagged Tier 3 is HITL-gated the moment it exists.
- **Fail-safe by default.** An untagged/unknown tool defaults to the **most restrictive** tier — the system errs toward asking a human, never toward acting.
- **It unifies four concerns onto one axis** — tool design, guardrails, human-in-the-loop, and cost all fall out of a single reversibility tag instead of four separate policies.
- **It's auditable and explainable.** "Why did this pause?" has a one-word answer: its tier. Compliance and postmortems love that.

**How to implement it:** add a `reversibility: 0..3` field to your tool manifest (config), and have the
execution layer look up the tier and apply the mapped safeguard before any call — exactly the shape of a
config-driven execution gate ([07-human-in-the-loop.md], [13-config-driven-design.md]). **The On-Call
Copilot now implements exactly this**: `config/tools.yaml` tags each tool `reversibility: 0..3`, and
`app/rails.py:execution_gate` derives the oversight mode (fail-safe default 3 for untagged tools) — it
replaced the ad-hoc `require_approval` execution rails. Idea → shipped, tested code.

> **The fail-safe default lives in code, not YAML.** `reversibility()` returns 3 for any tool that's untagged, mistyped, or missing from the manifest — because a default you can delete or typo in config isn't a *guarantee*. Forgetting to tag a tool makes it **stricter, never looser** — the safeguard can't be disabled by omission.

### Other frontier ideas worth stealing
- **Production traces as a self-growing eval set.** Every real run is a future test case — capture inputs/trajectory so any trace can be *replayed and re-judged* offline against a new model/prompt. Your eval set grows itself.
- **Calibrated abstention.** Design and *reward* agents for saying "I'm not sure — escalating." Measure the calibration of their self-reported confidence and route low-confidence work to humans or a stronger model, instead of forcing a confident guess.
- **Heterogeneous verification.** Use a **different model family** as the critic/judge than the generator, so verification doesn't inherit the generator's blind spots (correlated errors are the silent killer of self-critique).
- **Behavioral diffs, not just score diffs.** On every prompt/model change, diff *which tools and reasoning path* the agent used across versions — catch silent drift that an unchanged pass/fail score hides.

---

*Pair this with the per-component `## ✅ Best Practices` sections in tutorials 01–15 for the specific patterns.*
