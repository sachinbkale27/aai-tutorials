"""
01 · Prompt Engineering — system + few-shot prompting via the OpenAI SDK.

Demonstrates the two highest-leverage prompt-engineering levers:
  1. A SYSTEM message that sets identity + constraints + an exact output contract.
  2. FEW-SHOT exemplars (fake user/assistant turns) that teach the output *shape*
     far more reliably than describing it in prose.

The task: classify an on-call alert's severity as exactly SEV1 / SEV2 / SEV3.

Setup:
    pip install "openai>=1.0"

Needs a key for a LIVE call:
    export OPENAI_API_KEY=sk-...      # optional — omit to see the message structure only

If OPENAI_API_KEY is unset, this script prints the fully-assembled message list
(system + few-shot + user) and exits cleanly, so it runs fine offline.

Run:
    python examples/01_prompt_engineering.py
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
# The message list is the model's entire API surface. Order matters:
# system first, then few-shot pairs, then the single real user turn.
messages = [
    # 1) SYSTEM — identity, constraints, and the exact output contract.
    {
        "role": "system",
        "content": (
            "You are an on-call triage assistant. Classify each alert's severity as "
            "exactly one of: SEV1, SEV2, SEV3. Reply with ONLY the label, nothing else."
        ),
    },
    # 2) FEW-SHOT — worked examples as fake user/assistant turns. These teach the
    #    output shape (a bare label, no prose) better than any prose instruction.
    {"role": "user", "content": "disk usage on db-primary at 94%"},
    {"role": "assistant", "content": "SEV2"},
    {"role": "user", "content": "5xx rate on checkout-api spiked to 12% at 02:14 UTC"},
    {"role": "assistant", "content": "SEV1"},
    {"role": "user", "content": "nightly batch job finished 3 min late"},
    {"role": "assistant", "content": "SEV3"},
    # 3) THE REAL REQUEST — the alert we actually want classified.
    {"role": "user", "content": "latency p99 on search-api climbed from 200ms to 1.4s"},
]


def main() -> None:
    # Detect the absence of a key and degrade gracefully instead of crashing.
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found — here is the exact message list you'd send:\n")
        print(json.dumps(messages, indent=2))
        print("\nSet OPENAI_API_KEY to run live (e.g. export OPENAI_API_KEY=sk-...).")
        return

    # Live path — only imported when we actually have a key.
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        temperature=0,   # deterministic: classification wants repeatability
        max_tokens=4,    # a severity label is tiny — cap it to save money + rambling
        messages=messages,
    )
    # Expect a bare label like "SEV2" — the few-shot exemplars enforce that shape.
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
