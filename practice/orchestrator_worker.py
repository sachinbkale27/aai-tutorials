import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send

class State(TypedDict, total=False):
    message: str
    frameworks: list
    findings: Annotated[dict, lambda a, b: {**a, **b}]   # REDUCER merges parallel writes
    risk: str
    approved: bool


def choose_frameworks(param):
    pass


def orchestrator(state):                       # pick which framework workers to run
    picked = choose_frameworks(state["message"])
    return {"frameworks": picked}

def fan_out(state):                            # Send API: dynamic map-reduce fan-out
    return [Send("worker", {"message": state["message"], "framework": fw})
            for fw in state["frameworks"]]


def classify_this(param, param1):
    pass


def re_reason_with_rag(payload):
    pass


def worker(payload):                           # classification-first -> RAG fallback
    tier, conf = classify_this(payload["message"], payload["framework"])
    if conf < 0.70:
        tier = re_reason_with_rag(payload)
    return {"findings": {payload["framework"]: {"tier": tier}}}


def worst_tier(param):
    pass


def synthesize(state):                         # fan-in: reducer already merged findings
    return {"risk": worst_tier(state["findings"])}

def route_review(state):                       # only HIGH risk pauses for a human
    return "review" if state["risk"] == "HIGH" else "auto"

def final_decision(state):
    return {"risk": state["findings"]}

def save_graph_image(graph, path=None):
    """Render the graph structure to a PNG (mermaid.ink API), or print the mermaid
    source if rendering isn't available (e.g. no network). Defaults to writing next
    to this script regardless of the current working directory."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "orchestrator_worker.png")
    try:
        with open(path, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print(f"[graph] saved diagram to {path}")
    except Exception as e:
        print(f"[graph] PNG render unavailable ({e}); mermaid source:\n")
        print(graph.get_graph().draw_mermaid())

g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("synthesize", synthesize)
g.add_node("review", lambda s: s)              # the PAUSE happens before this node
g.add_node("final", final_decision)
g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fan_out, ["worker"])   # fan-out
g.add_edge("worker", "synthesize")                             # fan-in
g.add_conditional_edges("synthesize", route_review, {"review": "review", "auto": "final"})
g.add_edge("review", "final"); g.add_edge("final", END)
app = g.compile(checkpointer=MemorySaver(), interrupt_before=["review"])  # conditional HITL
save_graph_image(app)