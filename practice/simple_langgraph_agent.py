import os
from typing import Annotated, TypedDict

import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.errors import NodeError
from langgraph.graph import add_messages, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from dotenv import load_dotenv

# ── Timeout policy ───────────────────────────────────────────────────────────
# Sync LangGraph nodes can't be cancelled mid-run, so timeouts are enforced at
# the client level: the HTTP request and the LLM call each get a hard deadline.
API_TIMEOUT = 15   # seconds — per weather-API HTTP request
LLM_TIMEOUT = 30   # seconds — per OpenAI chat completion

# WMO weather-code -> human description (subset of the common ones)
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog", 51: "light drizzle", 61: "light rain",
    63: "moderate rain", 65: "heavy rain", 71: "light snow", 73: "moderate snow",
    75: "heavy snow", 80: "rain showers", 95: "thunderstorm",
}

@tool
def get_weather2(city: str) -> str:
    """Get the current weather for the given city"""
    try:
        # 1) city name -> latitude/longitude (no key required)
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            #"http://localhost:8089/search",
            params={"name": city, "count": 1}, timeout=API_TIMEOUT,
        ).json()
        if not geo.get("results"):
            return f"No location found for '{city}'."
        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        # 2) coordinates -> current weather
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,weather_code",
                "temperature_unit": "fahrenheit",
            }, timeout=API_TIMEOUT,
        ).json()["current"]

        temp = wx["temperature_2m"]
        desc = _WMO.get(wx["weather_code"], "unknown conditions")
        return f"It is {temp}°F in {loc['name']}. {desc}."
    except requests.RequestException as e:
        return f"Weather service error for '{city}': {e}"

@tool
def get_weather(city: str) -> str:
    """Get the current weather for the given city"""
    api_key = os.getenv("OWM_API_KEY")
    if not api_key:
        return "Weather service unavailable: OWM_API_KEY is not set."
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "imperial"},
            timeout=10,
        )
        if resp.status_code == 404:
            return f"No weather data found for '{city}'."
        resp.raise_for_status()
        data = resp.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"It is {temp}°F in {city}. {desc}."
    except requests.RequestException as e:
        return f"Weather service error for '{city}': {e}"

Tools = [get_weather2]
load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[list, add_messages] # reducer to append messages do not overwrite

llm = ChatOpenAI(model="gpt-4o-2024-11-20",
        temperature=0, timeout=LLM_TIMEOUT, max_retries=0).bind_tools(Tools)

# Only these tool names require a human's OK before they run.
NEEDS_APPROVAL = {"get_weather2"}

def agent(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def route_after_agent(state: State) -> str:
    """Agent asked for a tool -> tools; otherwise finish."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

# ── Retry policy ─────────────────────────────────────────────────────────────
# On a transient failure (connection error, HTTP 5xx, timeout) retry with
# exponential backoff. The default retry_on already skips non-transient errors
# like ValueError/TypeError, so bad LLM output isn't retried pointlessly.
RETRY = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True)

def tool_error(exc: Exception) -> str:
    """ERROR HANDLING: return a message (delivered as a ToolMessage) instead of
    letting a tool exception crash the graph — the agent can then apologise/adapt."""
    return f"Weather tool failed after retries: {exc}. Tell the user it is unavailable."

def on_call_llm_failed(state: State, error: NodeError) -> State:
    print(f"call_llm failed after retries:%s", error.error)
    return f"llm_unavailable after retrying : {error.error} times"

def build_graph(interrupt: bool = True) -> CompiledStateGraph:
    g = StateGraph(State)
    # retry_policy retries the whole node attempt on transient errors. (Node-level
    # timeouts need async nodes in LangGraph 1.x; here timeouts live on the clients.)
    g.add_node("agent", agent, retry_policy=RETRY, error_handler=on_call_llm_failed,)
    g.add_node(
        "tools",
        ToolNode(Tools, handle_tool_errors=tool_error),   # graceful tool errors
        retry_policy=RETRY,
    )
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    # INTERRUPT: pause BEFORE tools so the driver can ask a human first. Eval passes
    # interrupt=False so tools run automatically (no blocking input() prompt).
    return g.compile(checkpointer=MemorySaver(),
                     interrupt_before=["tools"] if interrupt else None)

def save_graph_image(graph, path=None):
    """Render the graph structure to a PNG (mermaid.ink API), or print the mermaid
    source if rendering isn't available (e.g. no network). Defaults to writing next
    to this script regardless of the current working directory."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "weather_agent_graph.png")
    try:
        with open(path, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print(f"[graph] saved diagram to {path}")
    except Exception as e:
        print(f"[graph] PNG render unavailable ({e}); mermaid source:\n")
        print(graph.get_graph().draw_mermaid())

def run(graph, inp, cfg):
    """Stream one segment, printing the latest message at each step."""
    for chunk in graph.stream(inp, config=cfg, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

def request_approval(calls) -> bool:
    """Show the pending tool call(s) and return True if the human approves."""
    print("\n[HUMAN REVIEW] The agent wants to call an external API tool:")
    for c in calls:
        print(f"    - {c['name']}({c['args']})")
    ok = input("Approve this action? (Y/N): ").strip().lower() in ("yes", "y")
    print(f"[DECISION] {'approved' if ok else 'rejected'}")
    return ok

def print_state_history(graph, cfg):
    """Print every checkpoint the graph recorded, oldest -> newest. get_state_history
    yields StateSnapshots newest-first, so we reverse to read the run top to bottom."""
    print("\n" + "=" * 60)
    print("STATE HISTORY (checkpoints)")
    print("=" * 60)
    for i, snap in enumerate(reversed(list(graph.get_state_history(cfg)))):
        step = snap.metadata.get("step") if snap.metadata else None
        msgs = snap.values.get("messages", [])
        last = msgs[-1].content if msgs else ""
        last = (last[:60] + "…") if len(last) > 60 else last
        print(f"#{i} step={step} next={snap.next} messages={len(msgs)} last={last!r}")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found.")
        return

    graph = build_graph()
    save_graph_image(graph)                      # <- writes weather_agent_graph.png
    cfg = {"configurable": {"thread_id": "1"}}
    query = {"messages": [{"role": "user", "content": "what's the weather in LA?"}]}

    # Top-level ERROR HANDLING: retries/timeouts handle transient faults inside the
    # graph; this catches anything that still escapes so the program exits cleanly.
    try:
        run(graph, query, cfg)                   # runs agent, then PAUSES before tools

        # HUMAN-IN-THE-LOOP driver: each time it interrupts before tools, decide.
        while (state := graph.get_state(cfg)).next:
            calls = getattr(state.values["messages"][-1], "tool_calls", [])
            gated = [c for c in calls if c["name"] in NEEDS_APPROVAL]
            if gated and not request_approval(gated):
                # Denied: answer each call as if 'tools' ran (keeps history valid)
                # and point next back to the agent, so the resume skips the API.
                denials = [ToolMessage(content="Tool call denied by human reviewer.",
                                       tool_call_id=c["id"]) for c in calls]
                graph.update_state(cfg, {"messages": denials}, as_node="tools")
            run(graph, None, cfg)                # resume: run tools, or continue from agent

        print_state_history(graph, cfg)          # <- dump the checkpoint history
    except Exception as e:
        print(f"\n[ERROR] agent run failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

