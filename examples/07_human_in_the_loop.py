#!/usr/bin/env python3
"""Human-in-the-Loop (HITL): propose -> PAUSE -> (human decides) -> execute/escalate.

Demonstrates the core HITL pattern from 07-human-in-the-loop.md, Section 2:
an agent diagnoses an incident and PROPOSES a mutating action, but never runs
it directly. Instead it stashes the pending action in a serialized pause store
(keyed by a run id), returns control, and only executes after a separate
`resume(run_id, decision)` entry point receives an explicit "approve". The
"reject" path runs no mutation and escalates to on-call.

The human decision is simulated programmatically (read from argv) so the whole
thing runs non-interactively.

How to run:
    python examples/07_human_in_the_loop.py          # default: approve -> executes
    python examples/07_human_in_the_loop.py reject    # reject  -> escalates, no mutation
"""

import sys

# ---- 1. The "tools" the agent can call. Some mutate, some don't. ----
def read_metrics(service):            # safe, reversible: just observe
    return f"{service}: 5xx error rate 8% (baseline 0.2%)"

def rollback_deploy(service, to):     # MUTATING -- must be gated behind approval
    return f"rolled back {service} -> {to}"

# ---- 2. Approval policy as DATA, not code branches. ----
NEEDS_APPROVAL = {"rollback_deploy"}  # the set of gated (mutating) actions

# ---- 3. The pause store: everything needed to resume, keyed by run id. ----
# resume() runs later with no access to agent_turn()'s locals, so the whole
# proposal must live here (the serialized pause state).
PENDING = {}   # run_id -> {"action": str, "args": dict, "reason": str}

def agent_turn(run_id, alert):
    """Diagnose, then PROPOSE a fix. Returns a final result OR a pause -- never mutates."""
    finding = read_metrics("checkout-api")           # cheap action: just do it
    proposed = {"action": "rollback_deploy",
                "args": {"service": "checkout-api", "to": "v41"}}

    if proposed["action"] in NEEDS_APPROVAL:
        # STASH the state and RETURN control -- do NOT execute the mutating tool.
        PENDING[run_id] = {**proposed, "reason": finding}
        return {"status": "paused",
                "ask": f"Approve {proposed['action']}({proposed['args']})? because {finding}"}

    # Auto-path for ungated actions (unreached here, since rollback is gated).
    result = rollback_deploy(**proposed["args"])
    return {"status": "done", "result": result}

def resume(run_id, decision):
    """Called AFTER the human decides. Reconstructs state from PENDING and acts."""
    # pop() both reads and removes -> a resolved approval can't be replayed.
    fix = PENDING.pop(run_id, None)
    if fix is None:
        return {"status": "error", "result": "no pending action for this run"}
    if decision != "approve":
        # reject -> escalate; NO mutation runs.
        return {"status": "rejected",
                "result": "not applied -- escalating to secondary on-call"}
    # approved -> NOW we execute the previously-stashed mutating action.
    result = rollback_deploy(**fix["args"])
    return {"status": "done", "result": result}

def human_decision():
    """Simulate the human's one-bit decision non-interactively (from argv)."""
    arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "approve"
    return "reject" if arg == "reject" else "approve"

if __name__ == "__main__":
    run_id = "run-1"
    turn = agent_turn(run_id, "checkout-api 5xx spike")
    print("AGENT:", turn["status"])

    if turn["status"] == "paused":
        print("AGENT ASKS:", turn["ask"])
        # ---- the pause/resume boundary: a human decides here ----
        decision = human_decision()
        print("HUMAN DECISION:", decision)
        outcome = resume(run_id, decision)
        print("OUTCOME:", outcome)
    else:
        print("OUTCOME:", turn)
