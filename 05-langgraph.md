# 05 · LangGraph

> Model an agent as a state machine — a typed shared state flowing through nodes and edges — so control flow, fan-out, and human pauses become explicit graph structure instead of tangled `if`/`await` code.

---

## 1. Mental model — graph-as-state-machine for agents

An agent is really a **control-flow problem wearing an LLM hat**. You have steps that must run in an order, some that fan out in parallel, some that loop until a condition holds, and some that must *stop and wait for a human*. If you write that with plain `async`/`await` and `if` branches, three things rot fast:

1. **State is implicit.** "What has the agent learned so far?" is smeared across local variables, closures, and function arguments. Nobody can point at *the* state.
2. **Control flow is invisible.** The shape of the workflow (who runs before whom, where it branches) is buried in the order of statements. You can't draw it, diff it, or reason about it.
3. **Pausing is painful.** To stop before a dangerous action and resume later, you need to *serialize everything the agent knows* and rebuild it on resume. Ad-hoc code makes you hand-roll that.

LangGraph fixes all three by making you declare a **graph**:

- **State** — one typed object (a `TypedDict`) that every node reads and writes. This is the agent's "blackboard." One place, one schema.
- **Nodes** — plain functions `state -> partial_state`. A node returns *only the keys it changed*; LangGraph merges them in.
- **Edges** — declare who runs after whom. `add_edge(A, B)` means "after A, go to B." Multiple edges out of one node = **fan-out** (parallel). Multiple edges into one node = **fan-in** (join).
- **START / END** — the sentinel entry and exit nodes.
- **Reducers** — merge rules for state keys, so two parallel nodes writing the same key don't clobber each other.
- **Conditional edges** — a routing function picks the next node at runtime (branching, looping).
- **Checkpointers + interrupts** — persist state at each step so the graph can **pause before a node** and **resume later** — the backbone of human-in-the-loop.

**Why it beats ad-hoc control flow:** the workflow becomes *data you can inspect* (`graph.get_graph().draw_mermaid()`), the state is one auditable object, and pause/resume is a library feature instead of a bespoke serialization headache. You trade a little ceremony for a workflow you can draw on a whiteboard and a machine can execute the same way.

Contrast at a glance:

| Concern | Ad-hoc async code | LangGraph |
|---|---|---|
| Where's the state? | scattered locals | one `TypedDict` |
| What's the flow? | read the code top-to-bottom | declared edges, drawable |
| Fan-out to N workers | manual `asyncio.gather` | N edges out of one node |
| Pause before prod action | hand-rolled serialize/restore | `interrupt_before` + checkpointer |
| Resume after a week | you rebuild all context | reload from checkpoint by thread id |

---

## 2. Smallest working example — a standalone runnable graph

```bash
pip install langgraph
```

A three-node graph: **classify → answer → END**, with a typed state, streaming, and a first taste of interrupt. Save as `mini_graph.py` and run it.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1) STATE — the shared blackboard. total=False lets nodes fill keys incrementally.
class State(TypedDict, total=False):
    question: str
    category: str
    answer: str

# 2) NODES — each is state -> partial state. Return ONLY what you changed.
def classify(state: State):
    q = state["question"].lower()
    category = "math" if any(c.isdigit() for c in q) else "general"
    return {"category": category}

def answer(state: State):
    if state["category"] == "math":
        return {"answer": "Looks like a math question — compute it."}
    return {"answer": "Here's a general answer."}

# 3) BUILD — register nodes, then wire edges.
g = StateGraph(State)
g.add_node("classify", classify)
g.add_node("answer", answer)

g.add_edge(START, "classify")     # entry
g.add_edge("classify", "answer")  # linear flow
g.add_edge("answer", END)         # exit

app = g.compile()

# 4) INVOKE — pass the initial state, get the final merged state back.
final = app.invoke({"question": "what is 2 + 2?"})
print(final)
# {'question': 'what is 2 + 2?', 'category': 'math', 'answer': 'Looks like a math question — compute it.'}
```

**Streaming** — watch each node's output as it happens (this is what powers glass-box UIs):

```python
for step in app.stream({"question": "tell me a joke"}):
    print(step)
# {'classify': {'category': 'general'}}
# {'answer': {'answer': "Here's a general answer."}}
```

`stream()` yields `{node_name: partial_state}` after each node runs — perfect for pushing progress events to a UI.

**An interrupt** — stop the graph *before* a node, inspect state, then resume. This needs a **checkpointer** (state has to survive the pause) and a **thread id** (which conversation to resume):

```python
from langgraph.checkpoint.memory import MemorySaver

app = g.compile(checkpointer=MemorySaver(), interrupt_before=["answer"])
cfg = {"configurable": {"thread_id": "conv-1"}}

# Run until just before "answer", then it stops and returns.
app.invoke({"question": "what is 2 + 2?"}, cfg)

snap = app.get_state(cfg)
print(snap.next)             # ('answer',)  — paused here
print(snap.values["category"])  # 'math'    — state is preserved

# Resume: invoke with None means "continue from the checkpoint."
final = app.invoke(None, cfg)
print(final["answer"])       # 'Looks like a math question — compute it.'
```

That five-line pause/resume is the entire reason LangGraph exists for agents: the state machine froze, you looked at it, and it thawed exactly where it stopped.

---

## 3. How the On-Call Copilot uses it

The project is an **On-Call Copilot**: an alert comes in, an orchestrator plans which diagnostic workers to run, workers fan out, a synthesis step proposes a root cause and a fix, and a human must approve before anything touches production.

`app/graph.py` owns the **structure** of exactly that workflow. Read it top to bottom:

**The state schema** (`app/graph.py:17-25`) — the shared blackboard for the whole incident:

```python
class IncidentState(TypedDict, total=False):
    alert: str          # the incoming alert text
    plan: list          # which workers the orchestrator chose to run
    findings: dict      # worker name -> what it found
    root_cause: str     # synthesis conclusion
    remediation: dict   # the proposed fix: {action, args, draft}
    approved: bool      # did a human approve the fix?
    result: str         # final outcome text
```

Every node reads from and writes to this one object. `total=False` means nodes fill keys as they go — the orchestrator writes `plan`, workers write `findings`, synthesis writes `root_cause`.

**Fan-out / fan-in edges** (`app/graph.py:63-68`) — the orchestrator-worker topology as literal edges:

```python
g.add_edge(START, "orchestrator")
for w in workers:
    g.add_edge("orchestrator", w["name"])  # orchestrator fans OUT to every worker
    g.add_edge(w["name"], "synthesis")     # workers fan IN to synthesis
g.add_edge("synthesis", "remediation")
g.add_edge("remediation", END)
```

The workers come from `agents.yaml` (via `agent_config`), so the graph's width is **config-driven** — add a worker in YAML and the graph grows an edge. The shape is: `START → orchestrator → [workers…] → synthesis → remediation → END`.

**The named-helper closure trick** (`app/graph.py:34-41`, used at `:58`) — this is the subtle bit interviewers love:

```python
def _worker_node(name):
    """Named helper (not an inline lambda) so each node reliably captures its own `name`."""
    def node(state):
        findings = dict(state.get("findings") or {})
        findings[name] = ""   # M1: fill with the worker's real tool-calling result
        return {"findings": findings}
    return node

for w in workers:
    g.add_node(w["name"], _worker_node(w["name"]))
```

Why not `g.add_node(w["name"], lambda s: {...w["name"]...})`? Because a `lambda` inside a `for` loop **captures the variable `w`, not its value** — by the time any node runs, the loop has finished and every lambda sees the *last* `w`. Calling `_worker_node(w["name"])` binds `name` as a fresh function argument on each iteration, so each closure keeps its own name. (Section 5 shows this bug live.) Note the node also copies the dict (`dict(state.get("findings") or {})`) before mutating — it treats state as immutable and returns a fresh value.

**The interrupt gate** (`app/graph.py:70-75`):

```python
# interrupt_before pauses the graph right before it touches prod — the HITL gate
# (fully wired in M4 once a checkpointer is added).
try:
    return g.compile(interrupt_before=["remediation"])
except Exception:
    return g.compile()
```

`remediation` is the node that would mutate production (roll back a deploy). `interrupt_before=["remediation"]` declares "stop here and wait." The comment is honest that this only becomes a *real* pause once a checkpointer is attached (see Section 5) — without one, there's no persisted state to resume from.

**The langgraph-missing fallback** (`app/graph.py:44-49`, `:78`):

```python
def build_graph():
    try:
        from langgraph.graph import StateGraph, START, END
    except Exception as e:
        print(f"[graph] langgraph not installed ({e}) → incident.py runs the fallback path")
        return None
    ...

GRAPH = build_graph()   # None if langgraph absent — the app still runs
```

The import is *inside* the function and guarded. If langgraph isn't installed, `build_graph()` returns `None` and the app degrades gracefully instead of crashing on import. This is deliberate: LangGraph is an optional structural dependency, not a hard runtime one.

### Structure vs. execution — the honest part

Here's the split you must state plainly in an interview: **`app/graph.py` currently owns the STRUCTURE, but `app/incident.py` drives the EXECUTION.** The node bodies in `graph.py` are placeholders — `lambda s: {"root_cause": ""}`, a worker node that writes `findings[name] = ""`. Nothing calls `GRAPH.invoke()` in the request path.

The real work happens sequentially in `app/incident.py`. Look at `incident_events()` (`app/incident.py:77-134`): it runs input rails, then the orchestrator step, then loops over `G.plan_workers(alert)` calling `_worker(...)` one at a time (`app/incident.py:98-107`), then synthesis via `guarded_reply`, then the HITL gate (`app/incident.py:124-134`). The human pause is implemented **by hand**: it stashes the fix in a module-level `PENDING` dict (`app/incident.py:131`), yields `hitl.required`, and `return`s. Resume is a *separate* function, `resume_events()` (`app/incident.py:141-157`), that pops `PENDING` and runs the remediation.

So today:
- `graph.py` = the **map** (state schema, topology, where the interrupt goes). It reuses `plan_workers()` (`app/graph.py:28-31`) — that one function *is* shared with the live path.
- `incident.py` = the **engine** (the actual streaming execution, the real tool calls via `guarded_tool`, the real guardrails, the hand-rolled pause via `PENDING`).

**The graph becoming the execution engine is future work.** The plan (per the module docstrings, marked "M1"/"M4") is to move the real node logic into `graph.py`'s node functions, attach a checkpointer, and let `GRAPH` drive the flow — at which point `interrupt_before=["remediation"]` becomes the actual HITL pause and the `PENDING` dict disappears. Section 6 makes that an exercise. Being clear about this map-vs-engine gap is more impressive than pretending the graph runs the show.

---

## 4. Build it up — variations

### 4a. Conditional edges (branching / routing)

Static `add_edge` always goes the same way. `add_conditional_edges` lets a **router function** pick the next node from the current state — this is how you branch on a classification or loop until done.

```python
def route(state: State) -> str:
    return "math_solver" if state["category"] == "math" else "general_answer"

g.add_conditional_edges(
    "classify",              # after this node...
    route,                   # ...call this to decide...
    {"math_solver": "math_solver", "general_answer": "general_answer"},  # label -> node
)
```

For the Copilot, a conditional edge is the natural home for the orchestrator's *real* planning: instead of `plan_workers()` returning all workers, a router could inspect the alert and send only to `logs_analyzer` for a log-shaped alert, or short-circuit straight to `synthesis` for a known issue.

### 4b. A checkpointer for real interrupt/resume

The checkpointer is what turns `interrupt_before` from a declaration into a working pause. It persists a snapshot of state after every node, keyed by `thread_id`.

```python
from langgraph.checkpoint.memory import MemorySaver

app = g.compile(checkpointer=MemorySaver(), interrupt_before=["remediation"])
cfg = {"configurable": {"thread_id": conv_id}}

app.invoke({"alert": alert}, cfg)      # runs, then STOPS before remediation
snap = app.get_state(cfg)
# ... show snap.values["remediation"] to the human, get approval ...

app.update_state(cfg, {"approved": True})  # inject the human's decision into state
app.invoke(None, cfg)                       # None = resume from the checkpoint
```

`MemorySaver` is in-process (dies with the server). For production you'd use a persistent checkpointer (`langgraph.checkpoint.sqlite.SqliteSaver` or the Postgres one) so an incident paused Friday can resume Monday from a different process. This is *exactly* the mechanism that would replace `incident.py`'s hand-rolled `PENDING` dict.

### 4c. Cycles / loops

Edges can point backward, so a graph can loop — the classic "keep calling tools until the model is satisfied" agent loop. A conditional edge decides continue-vs-stop:

```python
def should_continue(state) -> str:
    return "act" if state.get("needs_more_data") else "finish"

g.add_node("think", think)
g.add_node("act", act)
g.add_edge("act", "think")                      # loop back
g.add_conditional_edges("think", should_continue,
                        {"act": "act", "finish": END})
```

Guard against infinite loops with a recursion cap: `app.invoke(state, {"recursion_limit": 25})` raises if exceeded. A worker that retries diagnostics until confident, or a synthesis step that requests one more log pull, would live here.

### 4d. Fine-grained streaming events

`stream()` takes a `stream_mode` to control granularity — useful for SSE UIs:

```python
for ev in app.stream({"alert": alert}, stream_mode="updates"):
    print(ev)   # {node: partial_state} after each node — good for step-by-step UI

for ev in app.stream({"alert": alert}, stream_mode="values"):
    print(ev)   # the FULL accumulated state after each step
```

`updates` gives deltas (what changed) — the natural source for the `step.start`/`step.end` events the Copilot's React UI already renders. `values` gives the whole state each time. There's also `stream_mode="messages"` for token-level LLM streaming inside nodes.

---

## 5. Gotchas & pitfalls

**Lambda closure capture in loops — the #1 bug.** This is *why* `graph.py` uses a named helper:

```python
# WRONG — every node captures the same `w`; all see the LAST worker.
for w in workers:
    g.add_node(w["name"], lambda s: {"findings": {**s.get("findings", {}), w["name"]: ""}})

# RIGHT — a factory binds the value as an argument (app/graph.py:34-41).
def _worker_node(name):
    def node(state):
        findings = dict(state.get("findings") or {})
        findings[name] = ""
        return {"findings": findings}
    return node
for w in workers:
    g.add_node(w["name"], _worker_node(w["name"]))
```

(A `lambda s, name=w["name"]: ...` default-arg trick also works, but the named helper is clearer and testable.)

**State reducers — parallel writes clobber each other.** When two nodes run in parallel (fan-out) and both write the *same* key, the default behavior is "last write wins" — or LangGraph raises `InvalidUpdateError` for concurrent updates. If your workers all append to `findings`, you need a **reducer** that says "merge, don't replace":

```python
from typing import Annotated
import operator

class IncidentState(TypedDict, total=False):
    findings: Annotated[dict, lambda a, b: {**a, **b}]   # merge dicts
    log_lines: Annotated[list, operator.add]             # concatenate lists
```

The `Annotated[type, reducer]` reducer runs whenever a node returns that key. The Copilot's placeholder worker nodes dodge this by each returning the *whole* merged dict, but the moment workers truly run in parallel you'd add a merge reducer to `findings`. **In an interview: "how do parallel nodes safely write shared state?" → reducers.**

**A checkpointer is mandatory for real interrupts.** `interrupt_before=[...]` without a checkpointer doesn't give you a resumable pause — there's no persisted snapshot to come back to. This is exactly the honest caveat in `graph.py:70-71` ("fully wired in M4 once a checkpointer is added"). Compile with `checkpointer=` and always pass a `thread_id`.

**Nodes return partial state, not the whole thing.** Return `{"plan": [...]}`, not a full `IncidentState`. LangGraph merges (applying reducers). Returning `{}` or `None` means "no change."

**Treat state as immutable.** Don't mutate `state["findings"]` in place — copy then modify, as `_worker_node` does (`dict(state.get("findings") or {})`). In-place mutation breaks checkpointing and time-travel, which snapshot the state object.

**Guard the optional import.** Importing langgraph at module top level makes it a hard dependency and crashes the app if it's missing. `graph.py` imports *inside* `build_graph()` and returns `None` on failure — copy that pattern for optional structural deps.

**`add_node` name collisions.** Node names must be unique; adding two nodes with the same name (or a name that collides with `START`/`END`) raises. Since Copilot worker names come from YAML, a duplicate name in `agents.yaml` would surface here.

---

## ✅ Best Practices

- **Keep state minimal and typed.** Model `IncidentState` as a lean `TypedDict` holding only what nodes actually read or write — bloated state is harder to checkpoint, diff, and reason about.
- **Use reducers for concurrent writes.** Annotate any key that fan-out nodes touch (`Annotated[dict, merge]`, `Annotated[list, operator.add]`) so parallel workers merge instead of clobbering or raising `InvalidUpdateError`.
- **Attach a checkpointer for durable pause/resume.** Compile with `checkpointer=` (persistent `SqliteSaver`/Postgres in prod) and a stable `thread_id` so an interrupt survives process restarts and can resume days later.
- **Gate irreversible steps with `interrupt_before`.** Put the HITL pause immediately before any node that mutates production (like `remediation`), then inject the decision via `update_state` and resume with `invoke(None, cfg)`.
- **Keep nodes small, pure, and side-effect-light.** Each node should be a `state -> partial_state` function that returns only the keys it changed and treats incoming state as immutable — copy-then-modify rather than mutating in place.
- **Stream events for observability.** Drive UIs and traces from `stream(..., stream_mode="updates")` deltas so every node transition is a visible `step.start`/`step.end` event instead of a black box.
- **Make graph structure config-driven.** Build nodes and edges from config (e.g. workers from `agents.yaml`) so adding capacity is a data change, and use a factory like `_worker_node(name)` to bind per-node values safely.
- **Prefer explicit edges over hidden control flow.** Encode branching and loops as `add_conditional_edges` routers and real edges you can draw with `draw_mermaid()`, rather than smuggling flow decisions into node bodies.

---

## 6. Exercises

1. **Run and draw the mini graph.** Build the Section 2 graph, then print `app.get_graph().draw_mermaid()`. Paste it into a Mermaid viewer and confirm the shape matches `START → classify → answer → END`.

2. **Prove the closure bug.** Build a 3-worker graph two ways — once with an inline `lambda` in the loop, once with `_worker_node`. Invoke both and inspect `findings`. Watch the lambda version write only the last worker's name; explain why in one sentence.

3. **Add a real conditional edge to the orchestrator.** Replace `plan_workers()`'s "run all" with a router that reads the alert text and picks a subset of workers (e.g. only `runbook_retriever` + `logs_analyzer` for a "5xx spike"). Wire it with `add_conditional_edges` and confirm only the chosen workers run.

4. **Make the interrupt real.** Compile `graph.py`'s graph with `MemorySaver()` and `interrupt_before=["remediation"]`. Invoke with an alert, assert `app.get_state(cfg).next == ("remediation",)`, `update_state` with `{"approved": True}`, resume with `invoke(None, cfg)`, and confirm `result == "done"`.

5. **Add a fan-out reducer.** Give `findings` an `Annotated[dict, merge]` reducer, make two worker nodes genuinely write different keys of `findings`, and confirm both survive (no clobber). Then remove the reducer and observe what breaks.

6. **(Capstone) Make the graph the execution engine.** Port the real logic from `incident.py` into `graph.py`'s node bodies: the orchestrator node calls the planner, worker nodes call `guarded_tool`, the synthesis node calls `guarded_reply`. Attach a persistent checkpointer, drive the request path with `GRAPH.stream(...)`, and delete the hand-rolled `PENDING` dict — let `interrupt_before=["remediation"]` + the checkpointer be the HITL pause. Map each `stream_mode="updates"` event to the existing SSE `step.start`/`step.end` contract so the UI is unchanged. This is the "structure becomes engine" migration the code calls future work.

---

## 7. Connections

- **[06-orchestrator-worker-multi-agent.md](06-orchestrator-worker-multi-agent.md)** — the fan-out/fan-in edges here (`orchestrator → workers → synthesis`) *are* the multi-agent topology; that tutorial fills in what each worker actually does.
- **[07-human-in-the-loop.md](07-human-in-the-loop.md)** — `interrupt_before=["remediation"]` + a checkpointer is the graph-native version of the approval gate; that tutorial covers the current hand-rolled `PENDING`/`resume_events` pause and how the checkpointer replaces it.
- **[02-tool-and-function-calling.md](02-tool-and-function-calling.md)** — today the node bodies are placeholders; the capstone fills them with the real tool-calling loop from that tutorial (`guarded_tool`).

---

## 8. Further reading

- **LangGraph docs** — <https://langchain-ai.github.io/langgraph/> (start with the Quickstart, then "Low Level Concepts").
- **StateGraph & reducers** — <https://langchain-ai.github.io/langgraph/concepts/low_level/> (state, nodes, edges, `Annotated` reducers).
- **Persistence & checkpointers** — <https://langchain-ai.github.io/langgraph/concepts/persistence/> (threads, `MemorySaver`, SQLite/Postgres savers).
- **Human-in-the-loop / interrupts** — <https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/> (`interrupt_before`, `update_state`, resume).
- **Streaming** — <https://langchain-ai.github.io/langgraph/concepts/streaming/> (`stream_mode` = `updates` / `values` / `messages`).
