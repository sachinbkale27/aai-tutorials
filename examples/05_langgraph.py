"""
05 · LangGraph — standalone runnable
====================================

Demonstrates modeling an agent as a state machine with LangGraph:

    1. STATE      -> one TypedDict shared "blackboard" every node reads/writes.
    2. NODES      -> plain functions state -> partial-state (return only what
                     you changed; LangGraph merges it in).
    3. EDGES      -> declare who runs after whom: START -> classify -> answer -> END.
    4. COMPILE + INVOKE -> run the graph and watch the state flow (stream()).
    5. INTERRUPT  -> compile with a MemorySaver checkpointer + interrupt_before,
                     so the graph PAUSES before a node, you inspect the frozen
                     state, then RESUME exactly where it stopped.

No LLM, no API key, no network — the node logic is pure Python so this runs
anywhere langgraph is installed.

Deps:
    pip install langgraph

How to run:
    python examples/05_langgraph.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ── 1) STATE: the shared blackboard. total=False lets nodes fill keys ───────
#    incrementally — classify writes `category`, answer writes `answer`.
class State(TypedDict, total=False):
    question: str
    category: str
    answer: str


# ── 2) NODES: each is state -> partial state. Return ONLY what you changed. ──
def classify(state: State):
    q = state["question"].lower()
    category = "math" if any(c.isdigit() for c in q) else "general"
    print(f"  [classify] question={state['question']!r} -> category={category!r}")
    return {"category": category}


def answer(state: State):
    if state["category"] == "math":
        ans = "Looks like a math question — compute it."
    else:
        ans = "Here's a general answer."
    print(f"  [answer]   category={state['category']!r} -> answer={ans!r}")
    return {"answer": ans}


# ── 3) BUILD: register nodes, then wire the edges into a linear flow. ────────
def build_graph() -> StateGraph:
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("answer", answer)
    g.add_edge(START, "classify")     # entry
    g.add_edge("classify", "answer")  # linear flow
    g.add_edge("answer", END)         # exit
    return g


def main() -> None:
    g = build_graph()

    # ── 4) COMPILE + INVOKE: pass an initial state, get the merged state back.
    print("=" * 60)
    print("PART A — compile + invoke")
    print("=" * 60)
    app = g.compile()
    final = app.invoke({"question": "what is 2 + 2?\n"})
    print(f"final state: {final}\n")

    # ── STREAM: watch each node's output as it happens (glass-box UIs). ──────
    print("PART B — stream (per-node deltas as state flows)")
    print("-" * 60)
    for step in app.stream({"question": "tell me a joke"}):
        print(f"  step: {step}")
    print()

    # ── 5) INTERRUPT: a checkpointer (state survives the pause) + a thread_id
    #    (which conversation to resume). interrupt_before stops BEFORE `answer`.
    print("=" * 60)
    print("PART C — interrupt_before + MemorySaver (pause -> resume)")
    print("=" * 60)
    app = g.compile(checkpointer=MemorySaver(), interrupt_before=["answer"])
    cfg = {"configurable": {"thread_id": "conv-1"}}

    # Runs classify, then STOPS just before `answer` and returns.
    print("invoke() — runs until just before 'answer', then pauses:")
    app.invoke({"question": "what is 2 + 2?"}, cfg)

    # Inspect the frozen checkpoint: state is preserved, next node is queued.
    snap = app.get_state(cfg)
    print(f"\n  PAUSED. next node to run  = {snap.next}")
    print(f"  preserved state.category  = {snap.values['category']!r}")
    print(f"  answer not yet computed?  = {'answer' not in snap.values}")

    # RESUME: invoking with None means "continue from the checkpoint."
    print("\ninvoke(None, cfg) — thaws the graph and finishes:")
    final = app.invoke(None, cfg)
    print(f"\n  RESUMED. final answer = {final['answer']!r}")

    print("\nDone — graph ran, paused before 'answer', and resumed cleanly.")


if __name__ == "__main__":
    main()
