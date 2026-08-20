"""
02 · Tool & Function Calling — the request → tool_calls → execute → submit loop.

Demonstrates the primitive under every agent framework: you hand the model a
menu of typed functions (JSON Schema), it DECIDES which to call with what args,
YOUR code EXECUTES the real function, and you SUBMIT the result back as a
role:"tool" message so the model can turn it into a final answer.

The four beats (see 02-tool-and-function-calling.md §1):
  1. REQUEST  — send messages + tool SCHEMAS.
  2. DECIDE   — model replies with normal text OR one/more `tool_calls`.
  3. EXECUTE  — you run the real Python function (the model never runs anything).
  4. SUBMIT   — append a role:"tool" result (same tool_call_id) and call again.

Setup:
    pip install "openai>=1.0"

Needs a key for the LIVE loop:
    export OPENAI_API_KEY=sk-...      # optional — omit to see the mechanics offline

If OPENAI_API_KEY is unset, this script prints the tool schema and a MOCKED
tool_calls → execute → result walkthrough (no network), then exits cleanly.

Run:
    python examples/02_tool_and_function_calling.py
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
# --- The tool SCHEMA: JSON Schema wrapped in OpenAI's function envelope. ---
# The model picks tools and fills arguments purely from these strings, so the
# name/description/param docs are prompt engineering — invest in them.
get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",                              # must match the real function
        "description": "Get the current weather for a city.",
        "parameters": {                                     # standard JSON Schema
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


# --- The REAL function the schema points at. ---
def get_weather(city, unit="celsius", **extra):
    # Pretend this hit a weather API. **extra absorbs unexpected model-generated keys.
    return {"city": city, "temp": 21, "unit": unit}


TOOLS = [get_weather_schema]
DISPATCH = {"get_weather": get_weather}     # name → callable


def execute_tool_calls(tool_calls, messages):
    """STEP 3+4: run each requested call and SUBMIT a result per tool_call_id."""
    for call in tool_calls:
        fn = DISPATCH[call.function.name]
        args = json.loads(call.function.arguments)   # model-generated JSON — validate in real code!
        print(f"  → model asked to call {call.function.name}({args})")
        result = fn(**args)
        # Every tool_call_id MUST get a role:"tool" reply before the next create().
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,       # links the result back to the request
            "content": json.dumps(result),
        })
        print(f"  ← executed, result: {result}")


def live_loop():
    """Runs when OPENAI_API_KEY is present: the real request → execute → submit cycle."""
    from openai import OpenAI

    client = OpenAI()
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]

    # STEP 1+2: REQUEST → the model DECIDES.
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=TOOLS)
    msg = resp.choices[0].message
    messages.append(msg)                   # keep the assistant's tool-call turn in history

    if msg.tool_calls:
        execute_tool_calls(msg.tool_calls, messages)
        # Loop back so the model turns the tool result into a natural-language answer.
        final = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=TOOLS)
        print("\nFinal answer:", final.choices[0].message.content)
    else:
        print("\nModel answered directly (no tool needed):", msg.content)


def offline_walkthrough():
    """Runs without a key: show the schema + a MOCKED tool_calls loop so mechanics are visible."""
    print("=== Tool schema (what the model reads to decide) ===")
    print(json.dumps(get_weather_schema, indent=2))

    # Fake the shape of resp.choices[0].message.tool_calls using tiny stand-in objects.
    class _Fn:
        name = "get_weather"
        arguments = json.dumps({"city": "Paris", "unit": "celsius"})

    class _Call:
        id = "call_mock_001"
        function = _Fn()

    print("\n=== Mocked tool_calls → execute → submit ===")
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    execute_tool_calls([_Call()], messages)

    print("\n=== Messages now sent back to the model ===")
    print(json.dumps(messages, indent=2))
    print("\n(Set OPENAI_API_KEY to run the real request → answer loop.)")


if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY"):
        live_loop()
    else:
        offline_walkthrough()
