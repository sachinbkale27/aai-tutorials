"""
The expensive call the cache exists to avoid.
=============================================

Owns: answering a question the slow way.
Depends on: config.

Isolated in its own module so `graph.py` stays pure wiring — and so you can see
exactly what a cache hit skips.
"""

import os
import time

import config

_CANNED = {
    "restart":  "Run `kubectl rollout restart deploy/api -n prod`, then watch readiness.",
    "slo":      "The API SLO is 99.9% availability over a 30-day rolling window.",
    "rollback": "Use `kubectl rollout undo deploy/api -n prod` to revert.",
}


def answer(question: str) -> str:
    """Real model call when a key is present, canned+slow otherwise."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            resp = OpenAI().chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Terse SRE assistant. One sentence."},
                    {"role": "user", "content": question},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:                     # rate limit, bad key, no network
            print(f"    [llm] {type(e).__name__} -> canned answer")

    time.sleep(0.4)                                # stand in for model latency
    normalised = question.lower().replace("roll back", "rollback")
    return next((a for kw, a in _CANNED.items() if kw in normalised),
                "No runbook entry matched; escalate to the on-call lead.")
