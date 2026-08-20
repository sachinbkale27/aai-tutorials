# 02 · Tool & Function Calling
> Give an LLM a menu of typed functions, let it decide which to call with what arguments, run them, and feed the results back — the loop that turns a chat model into an agent.

## 1. Mental model — what it is and why it matters

An LLM only emits text. **Tool calling** (a.k.a. function calling) is the protocol that lets that text
*request an action* in a structured, machine-parseable way, so your code can execute it and return the result.

The mental model is a four-beat loop:

```
1. REQUEST   You send messages + a list of tool SCHEMAS (JSON Schema descriptions of functions).
2. DECIDE    The model replies with either normal text OR one/more `tool_calls`
             (each = a function name + a JSON arguments blob it invented from the schema).
3. EXECUTE   YOUR code runs the real function. The model never runs anything — it only asks.
4. SUBMIT    You append the tool result as a `role:"tool"` message and call the API again.
             The model now "sees" the result and either calls more tools or writes the final answer.
```

Key ideas an interviewer will probe:

- **The model does not execute anything.** It emits a *request*. You are the runtime. This separation is the
  entire security/HITL story (see [07-human-in-the-loop.md]) — you can inspect, gate, or reject a call before running it.
- **Schemas are the contract.** The model picks tools and fills arguments *purely from the JSON Schema* — the
  `name`, `description`, and parameter descriptions are prompt engineering (see [01-prompt-engineering.md]). Vague
  descriptions → wrong tool / bad args.
- **It's a loop, not a single call.** One turn can trigger many tool calls; you keep looping until the model
  stops requesting tools and returns content.
- **Arguments are model-generated JSON** → treat them as untrusted input. Validate before executing.

Why it matters: tool calling is the primitive under *every* agent framework. MCP ([03-model-context-protocol.md])
standardizes *where tools live*; LangGraph ([05-langgraph.md]) standardizes *how the loop is orchestrated*; but
underneath both is exactly this request → tool_calls → execute → submit cycle.

## 2. Smallest working example

First, understand the pure structure **without any API key** — a tool schema is just a dict:

```python
# A tool schema = JSON Schema wrapped in OpenAI's function envelope. No key needed to read this.
get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",                       # must match your real function name
        "description": "Get current weather for a city.",  # the model reads this to decide WHEN to call
        "parameters": {                              # ← standard JSON Schema
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
```

Now the full loop. Deps + key:

```bash
pip install openai        # v1.x SDK
export OPENAI_API_KEY=sk-...
```

```python
import json
from openai import OpenAI

client = OpenAI()

# 1. The real function the schema points at.
def get_weather(city, unit="celsius"):
    return {"city": city, "temp": 21, "unit": unit}   # pretend this hit a weather API

TOOLS = [get_weather_schema]                # (schema from above)
DISPATCH = {"get_weather": get_weather}     # name → callable

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

# ---- STEP 1+2: REQUEST → the model DECIDES ----
resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
msg = resp.choices[0].message
messages.append(msg)                        # keep the assistant's tool-call turn in history

# ---- STEP 3: EXECUTE every requested call ----
if msg.tool_calls:
    for call in msg.tool_calls:
        fn   = DISPATCH[call.function.name]
        args = json.loads(call.function.arguments)   # model-generated JSON → validate in real code!
        result = fn(**args)
        # ---- STEP 4: SUBMIT the result, tagged with the SAME tool_call_id ----
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,        # links the result back to the request
            "content": json.dumps(result),
        })
    # Call again so the model can turn the tool result into a natural-language answer.
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
    print(final.choices[0].message.content)   # "It's currently 21°C in Paris."
else:
    print(msg.content)                        # model answered directly, no tool needed
```

The three things that make it work: the `tools=` list on every call, appending the assistant's
`tool_calls` message *and* the matching `role:"tool"` messages, and looping back for the final answer.

## 3. How the On-Call Copilot uses it

The On-Call Copilot (`~/projects/nvidia-aai`) already has the two hard halves of tool calling built —
the **schemas** and the **call layer** — but does *not yet* let the LLM drive tool selection. Be precise
about this in an interview; it's the seam where M1 plugs in.

**Tool manifest → schemas.** Tools are declared config-first in
[`config/tools.yaml`](../nvidia-aai/config/tools.yaml). Each entry is a `name`, `description`, and a
`parameters` map — i.e. everything you need to *generate* the OpenAI function schema from §2:

```yaml
# config/tools.yaml
- name: query_logs
  server: sre
  description: Search service logs in a time window for error signatures.
  parameters:
    service: {type: string, description: service name}
    window:  {type: string, description: e.g. "02:00-02:30 UTC"}
    pattern: {type: string, description: optional regex/keyword filter}
  # ...
- name: rollback_deploy
  server: sre
  mutating: true          # ← flags the HITL-gated tools (07-human-in-the-loop.md)
  description: Roll a service back to its previous known-good deploy.
  parameters: {service: {type: string}, to_version: {type: string}}
```

The file's own header comment states its purpose: *"app/incident.py / graph.py → generate OpenAI
tool-calling schemas from these."* The `mutating: true` flag is what the execution rail / HITL gate
key off (M4). This YAML *is* the `TOOLS` list from §2 — just not yet emitted into function envelopes.

**Tool implementations.** Each `name` maps to a plain Python function in
[`mcp_server/tools.py`](../nvidia-aai/mcp_server/tools.py) that reads mock infra data and returns a
short human-readable string:

```python
# mcp_server/tools.py
def query_logs(service="checkout-api", pattern=None, **extra):
    logs = _incident(service).get("logs", [])
    if pattern:
        logs = [l for l in logs if pattern.lower() in l.lower()]
    return f"{len(logs)} log lines for {service}; first error: {logs[0]}"
```

Note `**extra` on every function: callers can pass extra args without breaking — a pragmatic guard for
the fact that model-generated arguments are unpredictable.

**The call layer** is [`app/tools.py`](../nvidia-aai/app/tools.py), whose `call_raw(name, args)` is the
dispatcher — exactly the `DISPATCH[name](**args)` step from §2:

```python
# app/tools.py
def call_raw(name, args=None):
    """Invoke a tool; RAISES on failure so the resilience layer can retry/trip."""
    if name in FAILING:                 # demo fault injection ($RESILIENCE_FAIL)
        raise RuntimeError(f"{name} is unavailable")
    fn = getattr(_t, name, None)        # _t = mcp_server.tools
    if not fn:
        raise ValueError(f"unknown tool: {name}")
    return fn(**(args or {}))
```

### The honest gap: deterministic today, LLM-driven at M1

Here is what the project does **not** do yet. In [`app/incident.py`](../nvidia-aai/app/incident.py) the
worker sub-agents choose their tool **deterministically** — there is no `tool_calls`, no model decision:

```python
# app/incident.py — _worker(): the tool is picked by config, NOT by the LLM
tools   = w.get("tools") or []
primary = tools[0] if tools else w["name"]           # ← first tool in agents.yaml, hardcoded
args    = {"service": "checkout-api", "query": alert} # ← fixed args, not model-generated
result, ev_list = guarded_tool(primary, args)        # calls call_raw under retries/breaker
```

So the current pipeline is: orchestrator fans out to workers → **each worker calls its one assigned
tool directly** → synthesis. The LLM is *not* in the request→`tool_calls`→execute→submit loop. Even
`app/graph.py` marks the worker node body with a TODO: `findings[name] = ""  # M1: fill with the
worker's real tool-calling result`.

**What M1 changes, and why the seam already exists.** To make it real, M1 will:
1. Render `config/tools.yaml` into the OpenAI schema envelope (§2's `tools=` list).
2. Give each worker an LLM turn: send the alert + its tool subset, read `resp.choices[0].message.tool_calls`.
3. Route each `tool_call` through the *unchanged* `call_raw` / `guarded_tool` dispatcher.
4. Submit the `role:"tool"` result and loop for the finding.

Nothing in `tools.yaml` or `tools.py` needs to change — the schemas and the call layer are exactly the
two seams a real tool-calling loop plugs into. The only missing piece is the *model turn* that produces
`tool_calls` instead of `primary = tools[0]`. That's Exercise 6.

## 4. Build it up

### 4a. Multiple tools — the model routes
Give the model several schemas; it picks based on `description`s.

```python
TOOLS = [get_weather_schema, get_time_schema, search_web_schema]
resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
# "What time is it in Tokyo?" → the model returns a tool_call for get_time, not get_weather.
```
This is precisely the On-Call Copilot's target: hand a worker its subset of `tools.yaml` and let the
model route to `query_logs` vs `fetch_metrics` vs `list_deploys`.

### 4b. Parallel tool calls
One assistant turn can contain **many** `tool_calls` when requests are independent. The model does this
automatically. You execute them all (ideally concurrently) and submit *one `role:"tool"` message per call*:

```python
# "Weather in Paris AND Tokyo?" → msg.tool_calls has TWO entries.
for call in msg.tool_calls:                       # could run these with asyncio.gather / threads
    result = DISPATCH[call.function.name](**json.loads(call.function.arguments))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
# Every tool_call_id MUST get a reply before the next create(), or the API errors.
```
To *disable* this and force one-at-a-time: `parallel_tool_calls=False`.

### 4c. Forcing / restricting tool choice
`tool_choice` controls the DECIDE step:

```python
tool_choice="auto"      # default: model decides text vs tool(s)
tool_choice="required"  # MUST call at least one tool (no plain-text answer)
tool_choice="none"      # never call a tool this turn
tool_choice={"type": "function", "function": {"name": "get_weather"}}  # force THIS tool
```
`required` is useful for a worker whose job is *always* "call your diagnostic tool" — it removes the
chance the model chats instead of acting.

### 4d. Structured JSON output (not a tool call)
When you want *typed data back* rather than an action, use **Structured Outputs** — the model's final
message conforms to a schema you supply, guaranteed:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract severity + service from: checkout-api p95 spiking"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "incident",
            "strict": True,                 # ← hard schema conformance
            "schema": {
                "type": "object",
                "properties": {
                    "service":  {"type": "string"},
                    "severity": {"type": "string", "enum": ["SEV1", "SEV2", "SEV3"]},
                },
                "required": ["service", "severity"],
                "additionalProperties": False,
            },
        },
    },
)
data = json.loads(resp.choices[0].message.content)   # {"service": "checkout-api", "severity": "SEV2"}
```
Tool calling and structured outputs share the same JSON-Schema machinery. Rule of thumb: **tools = do
something**; **structured output = return typed data**. The Copilot's *synthesis* step is a natural fit
for structured output (a typed diagnosis object); the *workers* are a fit for tool calling.

## 5. Gotchas & pitfalls

- **Every `tool_call_id` must get a `role:"tool"` reply** before the next `create()`. Miss one (e.g. a
  tool threw) and the API rejects the whole request. Return an error string as the tool content instead
  of dropping it — the project's `call_raw` *raises*, so the resilience layer (§ [09-resilience.md]) must
  catch and turn failures into a submitted result or graceful-degrade message.
- **Append the assistant's tool-call message to history.** A common bug: you execute the call but forget
  to append `msg` itself, so the follow-up request has tool results with no matching request. Order
  matters: assistant(tool_calls) → tool → tool → …
- **Arguments are untrusted, model-generated JSON.** Validate types/enums before executing. `json.loads`
  can also fail on malformed args — wrap it. The project's `**extra` sink is a blunt version of this.
- **Descriptions are prompt engineering.** "Search logs" vs "Search service logs in a time window for
  error signatures" changes which tool the model picks. Invest in `description` fields — that's why
  `tools.yaml` writes full sentences.
- **Cap the loop.** A model can loop tool calls indefinitely. Bound iterations (e.g. max 5 rounds) to
  avoid runaway cost.
- **`strict: True` + `additionalProperties: False`** for reliable schemas. Without strict mode the model
  may hallucinate extra keys or skip required ones.
- **Don't over-tool.** 20+ tools in one request degrades selection accuracy and burns tokens. Give each
  worker only its relevant subset — which is exactly what per-worker tool lists in `agents.yaml` enable.
- **Mutating tools need a gate.** The `mutating: true` flag in `tools.yaml` exists so a human approves
  `rollback_deploy`/`restart_service` before execution — never let the model's `tool_call` auto-run an
  irreversible action ([07-human-in-the-loop.md]).

## ✅ Best Practices

- **Write tool descriptions as precise, full sentences.** The model routes purely from `name` + `description`, so state exactly *when* to call it and what it does — treat every schema field as prompt engineering.
- **Type every parameter with JSON Schema + `enum`s.** Constrain strings to enums, mark `required`, and set `additionalProperties: False` so the model can't invent keys or free-text where a fixed set belongs.
- **Validate and parse arguments defensively.** Wrap `json.loads` and coerce/check types against your schema *before* dispatch — never pass model-generated JSON straight into a function that touches infra.
- **Keep tools idempotent where possible.** Design reads and retry-safe writes so a duplicated or re-run `tool_call` (from the loop or a retry) can't double-charge, double-deploy, or corrupt state.
- **Return tool errors as messages, not exceptions.** Catch failures and submit a short error string as the `role:"tool"` result so the model can recover or degrade — never drop a `tool_call_id`.
- **Cap the tool-call loop with a hard iteration budget.** Bound rounds (e.g. max 5) and short-circuit on repeated identical calls to stop runaway cost and infinite tool loops.
- **Use Structured Outputs for extraction, tools for actions.** When you want typed data back rather than a side effect, reach for `response_format` json_schema instead of a dummy tool.
- **Keep secrets out of tool signatures.** Inject API keys, tokens, and credentials from server-side config inside the implementation — never expose them as schema parameters the model fills.

## 6. Exercises

1. **Run the loop.** Type out §2 end-to-end with your own `OPENAI_API_KEY`. Print `msg.tool_calls` before
   executing so you *see* the model-generated arguments JSON.
2. **Schema from YAML.** Write a `to_openai_schema(tool_dict)` that converts one `config/tools.yaml`
   entry into the `{"type":"function","function":{...}}` envelope from §2. Handle `required` (hint: which
   params have no default?) and `additionalProperties: False`.
3. **Multi-tool routing.** Register `query_logs`, `fetch_metrics`, and `list_deploys` (real functions
   from `mcp_server/tools.py`) as three schemas. Ask *"why did checkout-api error-rate spike at 02:00?"*
   and confirm the model routes to the right tool(s). Add `tool_choice="required"`.
4. **Parallel calls.** Ask a question that needs two services' logs at once. Verify `msg.tool_calls` has
   two entries; execute them with `asyncio.gather`; submit both results. Then set
   `parallel_tool_calls=False` and observe the difference.
5. **Structured synthesis.** After gathering tool results, do one final call with `response_format`
   json_schema to emit a typed `{root_cause, confidence, recommended_action}` object. This mirrors the
   Copilot's synthesis step.
6. **Wire the LLM to actually choose the tool (the M1 step).** In a copy of `app/incident.py`'s `_worker`,
   replace `primary = tools[0]` and the fixed `args` with a real model turn: render the worker's tools
   from `tools.yaml` into schemas, call the API, read `resp.choices[0].message.tool_calls`, route each
   through the *existing* `call_raw`, submit the `role:"tool"` result, and return the model's finding.
   You should not need to modify `app/tools.py` or `mcp_server/tools.py` at all — prove it.

## 7. Connections

- **[03-model-context-protocol.md]** — MCP is tool calling *standardized across a client/server
  boundary*. The same `mcp_server/tools.py` functions are already exposed over MCP; the schema/loop you
  learned here is what an MCP client wires into the model.
- **[05-langgraph.md]** — LangGraph's prebuilt agent *is* this loop as a graph (model node ⇄ tool node
  until no more `tool_calls`). Understanding the raw loop makes the framework legible.
- **[01-prompt-engineering.md]** — tool `name`/`description`/param docs are prompt engineering; they
  directly determine selection quality.
- **[07-human-in-the-loop.md]** — the `mutating: true` flag gates model-requested actions behind human
  approval before execution.
- **[09-resilience.md]** — `call_raw` raises; retries + circuit-breaker sit between the tool_call and a
  submitted result.

## 8. Further reading

- OpenAI — Function calling guide: https://platform.openai.com/docs/guides/function-calling
- OpenAI — Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- JSON Schema (the parameter language): https://json-schema.org/understanding-json-schema/
- OpenAI Python SDK: https://github.com/openai/openai-python
- Model Context Protocol (next tutorial's standard): https://modelcontextprotocol.io
