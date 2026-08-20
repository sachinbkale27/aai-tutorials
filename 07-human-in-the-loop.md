# 07 · Human-in-the-Loop (HITL)

> Pause an agent before an irreversible or mutating action, hand the decision to a human, and only execute after an explicit approval.

---

## 1. Mental model

An LLM agent is a *proposer of actions*. Most actions it proposes are cheap and reversible: retrieve a doc, summarize logs, draft a message. But some actions **mutate the real world** — roll back a deploy, restart a service, issue a refund, drop a table. For those, the agent should **stop and ask a human first**.

Why gate mutating actions specifically? **Asymmetric cost.**

- The cost of a *wrong autonomous action* (rolled back the wrong service during peak traffic, refunded $50,000 to the wrong account) is large, sometimes unbounded.
- The cost of *asking for approval* is a few seconds of a human's attention.

When the downside is that lopsided, you pay the tiny approval tax every time to cap the catastrophic tail. This is the core HITL principle: **assist, don't act**. The agent does 100% of the *thinking* (diagnosis, root cause, drafting the fix) and 0% of the *committing* until a human signs off. The human is reduced to a one-bit decision (approve / reject) on a fully-prepared proposal — which is exactly where human judgment is cheap and valuable.

Two things make HITL a real engineering problem rather than an `input()` call:

1. **State handoff across a pause.** The agent must *serialize* everything it needs to resume (the proposed action, its arguments, the context it relied on), return control, and later reconstruct that state when the human responds — possibly minutes later, possibly in a different process.
2. **Policy.** *Which* actions need approval, and under what thresholds, must be **configuration, not code branches** — so a non-engineer can change "refunds over $500 need approval" without a deploy.

The rest of this tutorial builds that up, then shows exactly how the On-Call Copilot in this repo implements it.

---

## 2. Smallest working example

A standalone, runnable agent that **proposes an action, pauses, and only executes after approval**. The pause/resume is simulated with CLI input, but the *shape* — stash state → return → resume with a decision — is identical to the production flow.

```python
# hitl_min.py — run: python hitl_min.py
"""Smallest HITL loop: propose -> pause -> (human decides) -> execute/skip."""

# ---- 1. The "tools" the agent can call. Some mutate, some don't. ----
def read_metrics(service):            # safe, reversible
    return f"{service}: 5xx error rate 8% (baseline 0.2%)"

def rollback_deploy(service, to):     # MUTATING — must be gated
    return f"rolled back {service} -> {to}"

# ---- 2. Approval policy as DATA, not code. ----
NEEDS_APPROVAL = {"rollback_deploy"}   # the set of gated actions

# ---- 3. The pause store: everything needed to resume, keyed by a run id. ----
PENDING = {}   # run_id -> {"action": str, "args": dict, "reason": str}

def agent_turn(run_id, alert):
    """Diagnose, then PROPOSE a fix. Returns either a final result or a pause."""
    finding = read_metrics("checkout-api")        # cheap action: just do it
    proposed = {"action": "rollback_deploy",
                "args": {"service": "checkout-api", "to": "v41"}}

    if proposed["action"] in NEEDS_APPROVAL:
        # STASH the state and RETURN control — do NOT execute.
        PENDING[run_id] = {**proposed, "reason": finding}
        return {"status": "paused",
                "ask": f"Approve {proposed['action']}({proposed['args']})? because {finding}"}

    # (unreached here, but this is the auto-path for ungated actions)
    result = rollback_deploy(**proposed["args"])
    return {"status": "done", "result": result}

def resume(run_id, decision):
    """Called AFTER the human decides. Reconstructs state from PENDING."""
    fix = PENDING.pop(run_id, None)
    if fix is None:
        return {"status": "error", "result": "no pending action for this run"}
    if decision != "approve":
        return {"status": "rejected", "result": "not applied — escalating to on-call"}
    # approved -> NOW we execute the previously-stashed action.
    result = rollback_deploy(**fix["args"])
    return {"status": "done", "result": result}

if __name__ == "__main__":
    run_id = "run-1"
    turn = agent_turn(run_id, "checkout-api 5xx spike")
    print("AGENT:", turn["status"])
    if turn["status"] == "paused":
        print("AGENT ASKS:", turn["ask"])
        # ---- the pause/resume boundary: a human decides here ----
        decision = input("approve / reject > ").strip()
        print("RESULT:", resume(run_id, decision))
```

Run it:

```
$ python hitl_min.py
AGENT: paused
AGENT ASKS: Approve rollback_deploy({'service': 'checkout-api', 'to': 'v41'})? because checkout-api: 5xx error rate 8% (baseline 0.2%)
approve / reject > approve
RESULT: {'status': 'done', 'result': 'rolled back checkout-api -> v41'}
```

The load-bearing ideas, all present in ~40 lines:

- `agent_turn` **never executes** the mutating tool — it only *proposes* and returns `"paused"`.
- `PENDING[run_id]` is the **serialized pause state**: the whole point is that `resume` runs later with no access to `agent_turn`'s locals, so everything needed must be in the store.
- `resume` is a **separate entry point** keyed by `run_id`. In production the human's decision arrives over a different HTTP request; the run id is what re-links it to the stashed proposal.
- Policy (`NEEDS_APPROVAL`) is data. Adding `"restart_service"` to the set gates it with zero logic change.

---

## 3. How the On-Call Copilot uses it

The On-Call Copilot (an SRE incident-response agent) implements exactly this pattern, but production-grade: config-driven policy, an SSE event contract, an HTTP resume route, and a React approval card. The mapping to the toy example is one-to-one.

### The policy: `execution_gate`

`app/rails.py` holds the approval policy — separate from the NeMo Guardrails *input/output* rails, because this is **action gating**, not content filtering:

```python
# app/rails.py
def execution_gate(action: str, args: dict | None):
    """App-level action gating (HITL). Returns (requires_approval, reason, rail_name)."""
    for rail in config.DEMO.get("execution_rails", []):
        if rail.get("action") != action:
            continue
        if rail.get("require_approval") == "always":
            return True, f"{action} always requires human approval", "execution_gate"
        thr = rail.get("require_approval_over")
        amt = (args or {}).get("amount")
        if thr is not None and amt is not None and amt > thr:
            return True, f"{action} ${amt} exceeds the ${thr} auto-approve limit", "refund_gate"
        return False, "", "execution_gate"
    return False, "", "execution_gate"
```

This is pure policy evaluation: no side effects, returns a `(requires_approval, reason, rail_name)` triple. Two policy shapes are supported: **`require_approval: always`** (gate unconditionally) and **`require_approval_over: N`** (gate only when a numeric argument exceeds a threshold — the auto-approve limit pattern).

> **Now shipped — the Reversibility Ladder.** The copilot has since *generalized* this gate: instead of per-action `require_approval` rails, each tool carries a `reversibility: 0..3` tag in `config/tools.yaml`, and `execution_gate` **derives** the oversight mode from the tier — 0–1 → out-of-the-loop, 2 → human-on-the-loop (supervise), 3 → human-in-the-loop (approve first) — with a **fail-safe default of 3** for untagged tools. The `require_approval` shape below is the simpler pattern it grew out of. See `BEST_PRACTICES.md §13`.

The policy itself lives in `config/guardrails/demo_triggers.yml` — editable without touching code:

```yaml
# config/guardrails/demo_triggers.yml
execution_rails:
  - action: rollback_deploy
    require_approval: always
  - action: restart_service
    require_approval: always
  - action: scale_service
    require_approval_over: 10   # scaling within ±10 replicas is auto; larger needs approval
```

Note `rollback_deploy` is `require_approval: always` — a deploy rollback is never auto-run. `scale_service` is threshold-gated: small scaling is autonomous, large scaling needs a human. This is the config-driven approval policy in one file.

### The flow: gate → stash → pause → resume

`app/incident.py` is where the gate is *applied* mid-stream. After the agent has run its workers and synthesized a root cause + proposed fix, it hits the HITL gate (stage 5):

```python
# app/incident.py — inside incident_events()
# 5) HITL GATE — the proposed fix mutates prod, so ask a human first.
fix = {"action": "rollback_deploy",
       "args": {"service": "checkout-api", "to_version": "prev"},
       "draft": "Proposed: roll checkout-api back to the previous known-good deploy."}
needs_approval, reason, rail = execution_gate(fix["action"], fix["args"])
if needs_approval:
    yield {"type": "rail.fired", "rail_type": "execution", "name": rail, "stop": True, "reason": reason}
    PENDING[conv_id] = {**fix, "citations": citations}   # stash fix (+ source) until resume
    yield {"type": "hitl.required", "action": fix["action"], "args": fix["args"],
           "draft": fix["draft"], "reason": reason}
    return                                       # stop here — the human decides next
```

Three things happen, matching the toy example precisely:

1. **`PENDING[conv_id] = {...}`** — the module-level dict `PENDING = {}` (keyed by conversation id) is the serialized pause state. It stashes the action, its args, the human-readable draft, *and* the `citations` (the runbook the diagnosis relied on) so the audit trail survives the pause.
2. **`yield {"type": "hitl.required", ...}`** — an SSE event tells the UI to render an approval card. `draft` is the plain-English description; `reason` is why it's gated.
3. **`return`** — the generator *ends*. The agent has released control. Nothing executes.

The human's decision comes back over a **separate HTTP request**, routed in `app/main.py`:

```python
# app/main.py
@app.post("/api/incidents/{conv_id}/resume")
async def incident_resume(conv_id: str, req: Request):
    body = await req.json()
    rec = M.TraceRecorder(conv_id, "resume")
    async def gen():
        async for ev in incident_resume_events(conv_id, body.get("decision"), body.get("args")):
            rec.record(ev)
            yield sse(ev)
        rec.finalize()
    return StreamingResponse(gen(), media_type="text/event-stream")
```

The `conv_id` in the URL re-links this request to the stashed `PENDING[conv_id]`. `resume_events` in `app/incident.py` reconstructs state and either executes or escalates:

```python
# app/incident.py
async def resume_events(conv_id, decision, args):
    await asyncio.sleep(0.1)
    yield {"type": "hitl.resolved", "decision": decision}
    fix = PENDING.pop(conv_id, {}) or {}             # the fix we stashed in incident_events()
    citations = fix.get("citations", [])

    if decision == "reject":
        async for ev in say("Understood — I won't run that remediation. Escalating to on-call."):
            yield ev
        yield {"type": "message.final", "text": "", "citations": citations}
        return

    # approved → run it
    async for ev in _remediate(fix.get("action", "restart_service"), args or fix.get("args", {}), citations):
        yield ev
```

Key details:

- `PENDING.pop(conv_id, ...)` both *reads and removes* the stash — a resolved approval can't be replayed (see idempotency in §5).
- On **reject**, no mutation runs; it escalates and emits `message.final`.
- On **approve**, `_remediate(...)` finally calls the mutating tool (`rollback_deploy`) via `guarded_tool(...)`. Note `args or fix.get("args")` — the resume request can *override* the args (the UI's "Edit" button), letting a human tweak the proposal before approving.
- `yield {"type": "hitl.resolved", "decision": decision}` is the closing SSE event that flips the UI out of the pending state.

### The SSE contract

The pause/resume is visible to the browser as two events (see [14-sse-streaming.md]):

- **`hitl.required`** — emitted at the pause. Carries `action`, `args`, `draft`, `reason`. The UI renders the approval card and stops the "thinking" spinner.
- **`hitl.resolved`** — emitted at resume start. Carries `decision`. The UI clears the card and resumes streaming the remediation (or the escalation message).

### The UI: `HitlCard`

`src/components/HitlCard.jsx` renders the `hitl.required` payload and collects the one-bit (plus optional edit) decision:

```jsx
// src/components/HitlCard.jsx
export default function HitlCard({ hitl, onResolve }) {
  const [amount, setAmount] = useState(hitl.args?.amount ?? '');
  return (
    <div className="panel hitl">
      <h4>④ Human approval needed</h4>
      <p className="muted">{hitl.reason}</p>
      <div className="hitl-action"><b>{hitl.action}</b> {JSON.stringify(hitl.args)}</div>
      <p className="draft">“{hitl.draft}”</p>
      <div className="hitl-actions">
        <button className="approve" onClick={() => onResolve('approve')}>Approve</button>
        <button className="edit" onClick={() => onResolve('edit', { args: { ...hitl.args, amount: Number(amount) } })}>
          Edit&nbsp;$<input value={amount} onChange={(e) => setAmount(e.target.value)} />
        </button>
        <button className="reject" onClick={() => onResolve('reject')}>Reject</button>
      </div>
    </div>
  );
}
```

`onResolve('approve' | 'reject' | 'edit', {args})` is what POSTs to `/api/incidents/{id}/resume`. The **Edit** path demonstrates the "human amends the proposal" pattern — it sends modified `args` that `resume_events` picks up via `args or fix.get("args")`.

---

## 4. Build it up

Four variations that move from the app's current pattern toward a fully-checkpointed graph.

### 4a. Threshold-based approval ("auto-approve limit")

The `require_approval_over` shape already in `execution_gate`. Small, low-blast-radius actions run autonomously; large ones gate. This is how you keep the agent *useful* — gating everything trains humans to rubber-stamp (approval fatigue), which defeats the point.

```yaml
execution_rails:
  - action: scale_service
    require_approval_over: 10     # ±10 replicas auto; more needs a human
  - action: issue_refund
    require_approval_over: 500    # refunds ≤ $500 auto; larger needs a human
```

`execution_gate` reads `args["amount"]` and compares. The threshold is per-action, in config.

### 4b. Always-approve for destructive actions

For actions where *any* execution is irreversible, `require_approval: always` — no threshold, no exceptions:

```yaml
execution_rails:
  - action: rollback_deploy
    require_approval: always
  - action: drop_table
    require_approval: always
```

Pair this with an **input-side** destructive-command block (also in `demo_triggers.yml`): the `destructive command block` input rail refuses to even *plan* `drop database` / `rm -rf` / `truncate table`. Defense in depth: refuse dangerous intents at input, gate mutating actions at execution.

### 4c. LangGraph `interrupt_before` + a checkpointer

The app's `PENDING` dict is a hand-rolled pause store. LangGraph provides this natively. `app/graph.py` already declares the pause point structurally:

```python
# app/graph.py
# interrupt_before pauses the graph right before it touches prod — that's the
# human-in-the-loop gate (fully wired in M4 once a checkpointer is added).
try:
    return g.compile(interrupt_before=["remediation"])
except Exception:
    return g.compile()
```

`interrupt_before=["remediation"]` tells LangGraph: run every node up to `remediation`, then **stop and persist state**. With a checkpointer wired in, the full loop looks like:

```python
from langgraph.checkpoint.memory import MemorySaver   # or SqliteSaver / PostgresSaver in prod

graph = g.compile(interrupt_before=["remediation"], checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": conv_id}}   # thread_id == our conv_id

# 1) run until the interrupt — graph pauses before `remediation`
graph.invoke({"alert": alert}, cfg)

# 2) inspect what's pending (the equivalent of reading PENDING[conv_id])
state = graph.get_state(cfg)
print(state.next)            # -> ('remediation',)  the node about to run
print(state.values["remediation"])   # the proposed fix

# 3) human decides. Optionally amend state before resuming (the "Edit" path):
graph.update_state(cfg, {"approved": True})

# 4) resume: passing None continues from the checkpoint, runs `remediation`
graph.invoke(None, cfg)
```

The correspondence is exact:

| Hand-rolled (`incident.py`) | LangGraph |
|---|---|
| `PENDING[conv_id] = {...}` | checkpointer persists state at the interrupt |
| `conv_id` key | `thread_id` in config |
| `return` after `hitl.required` | `interrupt_before` stops the graph |
| `resume_events(conv_id, decision, args)` | `update_state(...)` + `invoke(None, cfg)` |
| args override via `args or fix["args"]` | `update_state` before resuming |

The advantage of the LangGraph version: state is durably checkpointed (survives a process restart if you use `SqliteSaver`/`PostgresSaver`), and you get time-travel/replay for free. See [05-langgraph.md].

### 4d. Timeouts & escalation

A pending approval that no one answers is an incident of its own. Add a deadline and an escalation path:

```python
import time
PENDING[conv_id] = {**fix, "citations": citations,
                    "expires_at": time.time() + 900}   # 15-minute SLA

async def resume_events(conv_id, decision, args):
    fix = PENDING.pop(conv_id, {}) or {}
    if fix.get("expires_at", 0) < time.time():
        yield {"type": "hitl.resolved", "decision": "expired"}
        async for ev in say("Approval window expired — auto-escalated to secondary on-call."):
            yield ev
        return
    ...
```

A background sweeper scans `PENDING` for expired entries and pages the next tier. The policy choice — *fail closed* (expire → do nothing, escalate) vs *fail open* (expire → auto-run) — should itself be config. For mutating prod actions, **fail closed** is almost always correct.

---

## 5. Gotchas & pitfalls

- **Persisting pause state.** `PENDING = {}` is an in-memory dict — it dies with the process and doesn't survive multiple workers. Fine for a demo, wrong for prod. Use a checkpointer/DB (Redis, Postgres, LangGraph's `SqliteSaver`/`PostgresSaver`) keyed by the run/thread id so a pause can be resumed after a restart or from a different instance.

- **Idempotency / no double-execution.** The `PENDING.pop(conv_id)` pattern is deliberately atomic: pop-then-act means a second resume for the same id finds nothing and can't re-run the mutation. Never `PENDING[conv_id]` (read) then later `del` — a concurrent resume could slip between. Also make the *mutating tool itself* idempotent where possible (dedupe key), because networks retry.

- **Validate the decision, don't trust the client.** The resume route accepts `decision` and `args` from the browser. Re-check the policy on resume — never assume the paused action is still safe, and never let the client send an arbitrary `action` to execute. The server should only run the action it stashed.

- **Who can approve.** The demo has no authz. In production the approver's identity matters: the person who *triggered* the agent often must not be the one who approves the mutating action (separation of duties), and high-blast-radius actions may need two approvers. Bind approval to an authenticated user, not just "whoever holds the conv_id."

- **Audit trail.** Every gate decision and resolution should be logged: what was proposed, why it was gated (`reason`), who approved/rejected, when, and with what final args. The app threads `citations` (the runbook the diagnosis relied on) through the pause so the *evidence* is attached to the decision. Note `TraceRecorder(conv_id, "resume")` in `main.py` records the resume stream — that's your audit source.

- **Avoid approval fatigue.** If everything gates, humans reflexively click approve and the gate becomes theater. Thresholds (4a) exist so only genuinely risky actions interrupt a human. Gate on blast radius, not on principle.

- **Make the proposal legible.** The human's decision is only as good as what they see. `draft` (plain English), `action`+`args` (exact call), and `reason` (why gated) are the minimum. A human approving `rollback_deploy {"service":"checkout-api","to_version":"prev"}` with the 5xx evidence attached is making a real decision; approving an opaque token is not.

---

## ✅ Best Practices

- **Gate only irreversible or mutating actions.** Let the agent run reads, retrievals, and drafts autonomously; reserve the human interrupt for actions that change prod, move money, or can't be undone.
- **Drive approvals from config, not code.** Express *what* gates and *at what threshold* as data (`require_approval: always`, `require_approval_over: N`) so a non-engineer can change policy without a deploy.
- **Persist the paused state durably.** Key the stash by run/thread id in a real store (Postgres, Redis, a LangGraph checkpointer) so a pause survives a restart and can resume from any worker.
- **Give the human enough context to decide.** Attach the plain-English draft, the exact `action`+`args`, the reason it gated, and the supporting evidence (runbook, metrics) so approval is a real judgment, not a rubber stamp.
- **Keep the mutating tool idempotent.** Pass a server-side dedupe/idempotency key so a retried or double-clicked approval executes the action at most once.
- **Record an audit trail for every gate.** Log what was proposed, why it gated, who approved or rejected, when, and with what final args — bind it to the run id for later review.
- **Add a timeout with escalation.** Give each pending approval an SLA and a `fail-closed` escalation path so an unanswered request pages the next tier instead of stalling silently.
- **Define who is allowed to approve.** Bind approval to an authenticated identity, enforce separation of duties (initiator ≠ approver) for high-blast-radius actions, and require two approvers where the downside warrants it.

## 6. Exercises

1. **Add a `$`-threshold gate.** Add an `issue_refund` action with `require_approval_over: 500` to `demo_triggers.yml`. Extend the `_remediate`/incident flow to propose a refund with an `amount` arg, and confirm `execution_gate` returns the `refund_gate` rail with the right reason when `amount > 500` and auto-approves when `amount <= 500`. Verify the `hitl.required` event carries the amount.

2. **Implement reject → escalate.** Extend `resume_events` so a `reject` doesn't just say "escalating" but emits a structured `escalation` event (`{type: "escalation", tier: "secondary", action, reason}`) and stashes the rejected proposal in an `ESCALATED` store for a human dashboard. Add the matching branch to `HitlCard`.

3. **Do it with a LangGraph checkpointer.** Wire `MemorySaver()` into `graph.py`'s `g.compile(interrupt_before=["remediation"], checkpointer=...)`. Drive it end to end: `invoke` to the interrupt, `get_state` to read the pending fix, `update_state({"approved": True})`, `invoke(None, cfg)` to resume. Confirm the pause survives reading state from a *fresh* graph object built with the same checkpointer + `thread_id`.

4. **Add a timeout + sweeper.** Give each `PENDING` entry an `expires_at` (per 4d). Write an async background task that scans `PENDING` every 30s and, for expired entries, emits an escalation and removes them. Decide fail-open vs fail-closed and justify it for `rollback_deploy`.

5. **Enforce separation of duties.** Add an authenticated `user_id` to both the incident-start and resume requests. Reject a resume where `approver_id == initiator_id` for any `require_approval: always` action, with a clear error surfaced in `HitlCard`.

6. **Harden idempotency.** Simulate a double-click: fire two concurrent POSTs to `/resume` for the same `conv_id`. Confirm exactly one executes the mutation. Then add a server-side dedupe key to `rollback_deploy` so even a retried tool call is safe.

---

## 7. Connections

- **[05-langgraph.md]** — `interrupt_before` + checkpointers are the graph-native version of `PENDING`/`resume_events`. The gate in this tutorial is the same pause point declared in `app/graph.py`.
- **[08-nemo-guardrails.md]** — HITL (`execution_rails`) is *action* gating; guardrails (`input_rails`/`output_rails`) are *content* gating. They live side by side in `demo_triggers.yml` and `rails.py`. Defense in depth: refuse dangerous intent at input, gate mutating actions at execution.
- **[14-sse-streaming.md]** — the pause is made visible to the browser via the `hitl.required` / `hitl.resolved` SSE events and the `/resume` streaming route. The UI is event-driven, not request/response.

---

## 8. Further reading

- LangGraph — *Human-in-the-loop* concept guide: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- LangGraph — *How to add human-in-the-loop* (`interrupt` / `interrupt_before`, `Command(resume=...)`): https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
- LangGraph — *Persistence & checkpointers* (`MemorySaver`, `SqliteSaver`, `PostgresSaver`): https://langchain-ai.github.io/langgraph/concepts/persistence/
- LangGraph — *Time travel* (inspect/replay/fork paused state): https://langchain-ai.github.io/langgraph/concepts/time-travel/

*This repo's files to study alongside: `app/rails.py` (`execution_gate`), `app/incident.py` (`PENDING`, `incident_events`, `resume_events`, `_remediate`), `config/guardrails/demo_triggers.yml` (`execution_rails`), `app/main.py` (`/api/incidents/{id}/resume`), `src/components/HitlCard.jsx`, `app/graph.py` (`interrupt_before`).*
