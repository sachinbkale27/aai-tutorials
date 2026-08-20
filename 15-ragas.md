# 15 · RAGAS (RAG Evaluation)

> After this you can score a RAG pipeline with RAGAS's core metrics — faithfulness, answer relevancy, and context precision/recall — understand what each actually measures, wire it into a regression suite, and navigate RAGAS's real-world version traps.

---

## 1. Mental model — what RAGAS is and why it exists

Generic LLM eval (tutorial [12-evaluation-and-regression.md]) asks "was the final answer right?". **RAGAS** ("RAG Assessment") asks the sharper, RAG-specific question: **did the *retrieval* and the *grounding* actually work?** A RAG system can produce a right-sounding answer for two very different reasons — because it retrieved the right context and used it, or because the base model already knew it and **ignored the context entirely** (which will fail the moment the knowledge shifts). RAGAS separates those.

It decomposes RAG quality into a small set of **LLM-judged** metrics, each isolating one failure mode:

| Metric | Question it answers | Failure it catches | Needs a `reference`? |
|---|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Hallucination / making things up | No |
| **Response Relevancy** | Is the answer actually about the question? | Evasive / off-topic answers | No |
| **Context Precision** | Are the *relevant* chunks ranked high in what was retrieved? | Retriever returns junk near the top | Yes (reference or ground-truth) |
| **Context Recall** | Did retrieval fetch *everything* needed? | Retriever misses key chunks | Yes (reference) |

Two things to internalize:
- **Faithfulness ≠ correctness.** An answer can be faithful to the context but the context itself can be wrong. Faithfulness measures *grounding*, not *truth*.
- **The metrics are computed by an LLM judge** (plus embeddings for some). That makes them scalable but **noisy and non-deterministic** — treat them as signal, validate the judge (see gotchas).

RAGAS's data model is four fields per sample: **`user_input`** (the question), **`retrieved_contexts`** (what the retriever returned), **`response`** (the generated answer), and optionally **`reference`** (ground truth, needed for the context-*recall*/precision-with-reference metrics).

## 2. Smallest working example

Runnable file: [`examples/15_ragas.py`](examples/15_ragas.py). It runs a **real** RAGAS evaluation when RAGAS + an API key are present, and falls back to a dependency-free illustration otherwise.

Deps (RAGAS is version-sensitive — see §5):
```bash
pip install "ragas==0.2.10" "langchain-community==0.3.14" "langchain-openai>=0.2"
export OPENAI_API_KEY=sk-...      # metrics are LLM-judged
python examples/15_ragas.py
```

The core of a real run (RAGAS ≥ 0.2 API, verified):
```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness, ResponseRelevancy,
    LLMContextPrecisionWithReference, LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dataset = EvaluationDataset.from_list([
    {
        "user_input": "What is the standard return window?",
        "retrieved_contexts": ["Our standard return window is 30 days from delivery."],
        "response": "You can return items within 30 days of delivery.",
        "reference": "30 days from delivery.",
    },
    # ... a hallucinated answer here scores LOW on faithfulness
])

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))   # the judge
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

result = evaluate(
    dataset,
    metrics=[Faithfulness(), ResponseRelevancy(),
             LLMContextPrecisionWithReference(), LLMContextRecall()],
    llm=llm, embeddings=emb,
)
print(result)          # {'faithfulness': 0.9x, 'answer_relevancy': ..., ...}
print(result.to_pandas())   # per-sample scores
```
**What to observe:** the grounded answer scores ~1.0 faithfulness; the "90 days" hallucination scores low — RAGAS caught that the answer isn't supported by the context, without you writing a rule.

## 3. How the On-Call Copilot uses it

The runbook retriever ([04-agentic-rag.md]) is exactly the kind of component RAGAS grades — and the project now wires it in for real, as a **two-phase pipeline**, because RAGAS's deps conflict with the copilot's pinned `nemoguardrails`/langchain (they can't share a venv):

- **Phase 1 — `eval/run_eval.py` (project venv).** `export_ragas_dataset()` runs each golden incident and captures, per case: `user_input` (the alert), `retrieved_contexts` (the runbook passage `search_runbooks` returned), `response` (the synthesized diagnosis), and `reference` (ground-truth root cause from `incidents.yaml`). It writes `eval/ragas_dataset.json`:
```python
records.append({
    "user_input": c["alert"],
    "retrieved_contexts": [o["context"]] if o.get("context") else [],
    "response": o["response"],
    "reference": c.get("reference") or " ".join(c.get("expect_root_cause", [])),
})
```
- **Phase 2 — `eval/ragas_eval.py` (isolated `.venv-ragas`).** Loads that JSON and runs `evaluate(...)` with the four metrics, writing `eval/ragas_results.json`. See `eval/README_ragas.md`.

This turns the project's *trajectory* eval ("did it call the right tools?") into a *quality* eval ("was the retrieved runbook actually faithful and sufficient?") — complementary, not redundant. The decoupling (export in one venv, score in another) is the key pattern for using a dependency-heavy evaluator against a system with its own pinned stack.

Design note: RAGAS **complements — doesn't replace** — the guardrail precision/recall and trajectory metrics in [12-evaluation-and-regression.md].

## 4. Build it up

1. **Per-sample scores.** `result.to_pandas()` gives a row per sample — use it to find *which* questions fail, not just the aggregate.
2. **Reference-free subset.** If you don't have ground-truth references, run only `Faithfulness` + `ResponseRelevancy` (they need no reference) — a cheap first pass.
3. **Compare two retrievers.** Run the same questions through keyword vs vector retrieval, evaluate both, and let **context precision/recall** decide which retriever wins — objective, not vibes.
4. **Gate CI on it.** Add a threshold (e.g., mean faithfulness ≥ 0.85) to the regression suite from [12-evaluation-and-regression.md] so a retrieval regression fails the build.
5. **Swap the judge.** Use a cheaper/stronger judge model (or a local one via a different wrapper) and see how scores move — this exposes judge sensitivity.

## 5. Gotchas & pitfalls

- **The version trap (you WILL hit this).** Every RAGAS release imports `langchain_community.chat_models.vertexai`; newer `langchain-community` (≥ 0.4) **removed** that module, so even `import ragas` raises `ModuleNotFoundError`. Fix: pin **`langchain-community==0.3.14`** (still ships it). Install RAGAS in a **clean venv** — its pins conflict with pinned `nemoguardrails`/`langchain` in projects like the On-Call Copilot (that's why the project keeps `ragas` commented in requirements).
- **Metrics are LLM-judged → non-deterministic and cost tokens.** Two runs can differ. Average over samples, and budget for the judge calls (one incident × 4 metrics = several LLM calls).
- **Validate the judge before trusting it at scale.** Spot-check RAGAS scores against human labels on a sample (agreement / Cohen's κ). An unvalidated LLM-judge is a confident-but-unproven oracle.
- **Faithfulness measures grounding, not truth.** If your source docs are wrong, a faithful answer is still wrong. Pair with source-quality checks.
- **Context recall needs a good `reference`.** Weak references → meaningless recall. Garbage-in.
- **Don't over-index on one number.** A high faithfulness with low context recall means "honest but under-informed" — a different fix than the reverse.

## ✅ Best Practices

- **Isolate RAGAS in its own pinned venv.** Give it a dedicated `.venv-ragas` with exact pins (`ragas==0.2.10`, `langchain-community==0.3.14`) so its dependency graph never has to reconcile with your app's stack.
- **Decouple dataset export from scoring.** Export samples to a plain JSON in your app's venv, then score that file in the RAGAS venv — the two phases share data, not dependencies.
- **Validate the LLM judge against human labels first.** Hand-label a small sample and confirm agreement (e.g., Cohen's κ) before you let RAGAS scores drive any decision at scale.
- **Run reference-free metrics first when you lack ground truth.** Start with `Faithfulness` + `ResponseRelevancy` (no `reference` needed) as a cheap pass before investing in labeled references for context precision/recall.
- **Average scores over multiple runs and samples.** The judge is non-deterministic, so report means across samples (and repeated runs for critical gates) rather than trusting a single number.
- **Use per-sample scores to locate failures.** Drive off `result.to_pandas()` to find *which* questions fail and why, instead of reacting only to the aggregate.
- **Gate CI on an explicit faithfulness threshold.** Wire a floor (e.g., mean faithfulness ≥ 0.85) into the regression suite so a retrieval or grounding regression fails the build automatically.
- **Budget the judge token cost up front.** Each sample fans out to several LLM calls per metric, so size your eval set and judge model against a real token/cost budget before running the full suite.

## 6. Exercises

1. **Run it live.** Install the pinned combo + a key and run `examples/15_ragas.py`; confirm the hallucinated sample scores low faithfulness.
2. **Add a third sample** where retrieval fetched an *irrelevant* chunk — watch context precision drop while faithfulness stays high.
3. **Reference-free pass:** re-run with only `Faithfulness` + `ResponseRelevancy` and note it needs no `reference`.
4. **Run the copilot's two-phase pipeline** (already wired): `python -m eval.run_eval` to export the dataset, then score it with `eval/ragas_eval.py` in a `.venv-ragas` + a key (see `eval/README_ragas.md`). Then extend it to also score the guardrail-blocked cases.
5. **Retriever bake-off:** score keyword vs Chroma retrieval on the same questions; report which wins on context recall.
6. **Judge validation:** hand-label faithfulness on 10 samples, compare to RAGAS, and compute agreement — decide whether you'd trust it unsupervised.

## 7. Connections
- [04-agentic-rag.md] — the retriever RAGAS grades.
- [12-evaluation-and-regression.md] — RAGAS is one metric family inside the broader eval/regression harness; trajectory + guardrail metrics live there.
- [08-nemo-guardrails.md] — guardrails' hallucination/self-check rails address the same failure faithfulness measures, from the enforcement side.
- [10-opentelemetry.md] — trace the judge calls' token/cost so RAGAS runs are observable.

## 8. Further reading
- RAGAS docs: https://docs.ragas.io
- RAGAS metrics reference (faithfulness, answer relevancy, context precision/recall)
- "LLM-as-a-judge" reliability and bias literature (validate before trusting)
