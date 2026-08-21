"""
The graph — pure wiring.
========================

Owns: the topology, and nothing else. Every node body is one line delegating to
      another module, which is what makes the SHAPE the thing you read here.
Depends on: cache, llm.

    START -> cache_lookup --(hit)---------------------------> END
                          \\--(miss)--> llm -> cache_write --> END

The conditional edge is the entire point: on a hit the `llm` node is never
entered. That skip is visible in the topology, in stream() events, and in your
traces — instead of being an `if` buried inside a node body.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

import cache
import llm


class State(TypedDict, total=False):
    question: str
    answer: str
    cache: str            # "hit" | "miss" — also what the conditional edge reads
    similarity: float
    matched: str


def build():
    """Compile the cached-agent graph."""

    def cache_lookup(state: State):
        result = cache.lookup(state["question"])
        hit = result.answer is not None
        return {"cache": "hit" if hit else "miss",
                "similarity": result.similarity,
                "matched": result.matched,
                **({"answer": result.answer} if hit else {})}

    def call_llm(state: State):
        return {"answer": llm.answer(state["question"])}

    def cache_write(state: State):
        cache.write(state["question"], state["answer"])
        return {}

    g = StateGraph(State)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("llm", call_llm)
    g.add_node("cache_write", cache_write)

    g.add_edge(START, "cache_lookup")
    g.add_conditional_edges("cache_lookup", lambda s: s["cache"],
                            {"hit": END, "miss": "llm"})
    g.add_edge("llm", "cache_write")     # only a SUCCESSFUL llm run writes
    g.add_edge("cache_write", END)
    return g.compile()
