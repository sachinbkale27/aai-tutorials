"""
LangSmith evaluation for the weather agent
==========================================

Evaluates simple_langgraph_agent's graph on a dataset of weather questions,
including edge cases (unknown city, off-topic question).

Metrics:
  • correctness        — LLM-as-judge: does the answer satisfy the expectation?
  • temp_appropriate   — heuristic: reports a temperature when (and only when) it should
  • weather_temp_coverage (summary) — % of real-weather examples that returned a temp

Two modes, chosen automatically:
  • LANGSMITH_API_KEY set -> langsmith.evaluate(): creates/reuses the dataset, runs the
    agent on every example, scores it, and uploads the experiment to smith.langchain.com
  • no key            -> LOCAL DRY-RUN: runs the same target + evaluators in-process.

Both modes call OpenAI (agent LLM + judge LLM) and Open-Meteo, so OPENAI_API_KEY and
network access are required either way.

Run:  python practice/eval_weather_agent.py
"""

import os
import re
import uuid
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from simple_langgraph_agent import build_graph

load_dotenv(override=True)

DATASET_NAME = "weather-agent-eval"

# ── Dataset: question -> reference expectation ───────────────────────────────
# kind: "weather" (real city), "unknown" (nonexistent place), "offtopic" (not weather)
# expected: natural-language description the LLM judge grades against.
EXAMPLES = [
    {"inputs": {"question": "what's the weather in Chicago?"},
     "outputs": {"kind": "weather", "city": "Chicago",
                 "expected": "Reports the current weather (a temperature) for Chicago."}},
    {"inputs": {"question": "what's the weather in Tokyo?"},
     "outputs": {"kind": "weather", "city": "Tokyo",
                 "expected": "Reports the current weather (a temperature) for Tokyo."}},
    {"inputs": {"question": "how's the weather in London right now?"},
     "outputs": {"kind": "weather", "city": "London",
                 "expected": "Reports the current weather (a temperature) for London."}},
    {"inputs": {"question": "what's the weather in Nowhereville12345?"},
     "outputs": {"kind": "unknown", "city": "Nowhereville12345",
                 "expected": "Says it could not find that location / has no weather for it. "
                             "Does NOT invent a temperature."}},
    {"inputs": {"question": "who won the 2018 FIFA World Cup?"},
     "outputs": {"kind": "offtopic", "city": None,
                 "expected": "Does not report any weather/temperature. Either answers the "
                             "trivia or says it only handles weather questions."}},
]

# ── Target: run the agent non-interactively and return its final answer ──────
# interrupt=False so the tools node runs automatically (no human input() prompt).
# A fresh thread_id per call keeps the in-memory checkpointer's runs isolated.
_graph = build_graph(interrupt=False)

def run_agent(inputs: dict) -> dict:
    query = {"messages": [{"role": "user", "content": inputs["question"]}]}
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = _graph.invoke(query, cfg)
    return {"answer": result["messages"][-1].content}

# ── LLM-as-judge ─────────────────────────────────────────────────────────────
class Grade(TypedDict):
    score: int        # 1 if the answer meets the expectation, else 0
    reasoning: str

_judge = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0).with_structured_output(Grade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: grade the answer against the reference expectation."""
    prompt = (
        "You are grading a weather assistant. Any weather numbers came from a trusted "
        "live weather API — assume they are accurate; do NOT penalize the answer for "
        "being unverifiable. Judge only whether the ANSWER meets the EXPECTATION's "
        "intent. Score 1 if it does, else 0.\n\n"
        f"QUESTION: {inputs['question']}\n"
        f"EXPECTATION: {reference_outputs.get('expected', 'A helpful, accurate answer.')}\n"
        f"ANSWER: {outputs['answer']}"
    )
    grade = _judge.invoke(prompt)
    return {"key": "correctness", "score": float(grade["score"]), "comment": grade["reasoning"]}

_TEMP_RE = re.compile(r"\d+(\.\d+)?\s*(°\s*F|degree)", re.I)

def temp_appropriate(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Heuristic: a temperature should appear for real-weather questions, and should
    NOT appear for unknown-city / off-topic questions."""
    has_temp = bool(_TEMP_RE.search(outputs["answer"]))
    want_temp = reference_outputs.get("kind") == "weather"
    return {"key": "temp_appropriate", "score": float(has_temp == want_temp)}

EVALUATORS = [correctness, temp_appropriate]

# ── Summary evaluator: aggregate across the whole run ────────────────────────
def weather_temp_coverage(runs, examples) -> dict:
    """% of real-weather examples whose answer actually reported a temperature."""
    hits = total = 0
    for run, example in zip(runs, examples):
        if example.outputs.get("kind") != "weather":
            continue
        total += 1
        answer = (run.outputs or {}).get("answer", "")
        hits += bool(_TEMP_RE.search(answer))
    score = hits / total if total else 0.0
    return {"key": "weather_temp_coverage", "score": score}

# ── Mode 1: real LangSmith evaluation ───────────────────────────────────────
def run_langsmith_eval():
    from langsmith import Client, evaluate

    client = Client()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        print(f"[langsmith] creating dataset '{DATASET_NAME}'")
        ds = client.create_dataset(DATASET_NAME, description="Weather agent smoke tests")
        client.create_examples(dataset_id=ds.id, examples=EXAMPLES)
    else:
        # Reuse the dataset, but append any examples it doesn't have yet (keyed by
        # question) so newly-added edge cases show up without duplicating old rows.
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        seen = {ex.inputs["question"] for ex in client.list_examples(dataset_id=ds.id)}
        missing = [e for e in EXAMPLES if e["inputs"]["question"] not in seen]
        if missing:
            client.create_examples(dataset_id=ds.id, examples=missing)
        print(f"[langsmith] reusing '{DATASET_NAME}' (+{len(missing)} new example(s))")

    evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        summary_evaluators=[weather_temp_coverage],
        experiment_prefix="weather-agent",
        max_concurrency=2,
    )
    print("\n[langsmith] experiment done — view it at https://smith.langchain.com")

# ── Mode 2: local dry-run (no LangSmith account needed) ─────────────────────
class _Obj:  # tiny stand-in so the summary evaluator can read .outputs locally
    def __init__(self, outputs): self.outputs = outputs

def run_local_eval():
    print("[local] no LANGSMITH_API_KEY — running evaluators in-process.\n")
    per_metric = {ev.__name__: 0.0 for ev in EVALUATORS}
    runs, examples = [], []
    for ex in EXAMPLES:
        outputs = run_agent(ex["inputs"])
        runs.append(_Obj(outputs)); examples.append(_Obj(ex["outputs"]))
        print(f"Q: {ex['inputs']['question']}")
        print(f"A: {outputs['answer']}")
        for ev in EVALUATORS:
            res = ev(ex["inputs"], outputs, ex["outputs"])
            per_metric[ev.__name__] += res["score"]
            note = f"  ({res['comment']})" if res.get("comment") else ""
            print(f"   {res['key']}: {res['score']}{note}")
        print()

    n = len(EXAMPLES)
    print("── average scores ──")
    for name, total in per_metric.items():
        print(f"   {name}: {total / n:.2f}")
    summary = weather_temp_coverage(runs, examples)
    print(f"   {summary['key']} (summary): {summary['score']:.2f}")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required to run the agent.")
        return
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        run_langsmith_eval()
    else:
        run_local_eval()

if __name__ == "__main__":
    main()
