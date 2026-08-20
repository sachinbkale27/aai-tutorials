"""
15 · RAGAS — evaluating RAG quality (faithfulness, relevancy, context precision/recall).

What this shows:
  • The RAGAS data model: (user_input, retrieved_contexts, response, reference).
  • The four core metrics and what each measures.
  • A REAL RAGAS run when ragas + OPENAI_API_KEY are available.
  • A dependency-free "toy" illustration of faithfulness so the idea is visible
    offline (clearly a crude proxy, NOT the real LLM-judged metric).

Deps (RAGAS is VERY version-sensitive — see the gotcha below):
    pip install "ragas==0.2.10" "langchain-community==0.3.14" "langchain-openai>=0.2"
  RAGAS metrics are LLM-judged, so a live run also needs:  export OPENAI_API_KEY=sk-...

⚠️  Version gotcha (real, hit while building this): every ragas release imports
    `langchain_community.chat_models.vertexai`, which newer langchain-community
    (>= 0.4) REMOVED — so even `import ragas` throws ModuleNotFoundError. Fix by
    pinning langchain-community==0.3.14 (still ships that module), in a CLEAN venv
    (ragas's pins clash with pinned nemoguardrails/langchain in some projects).

Run:  python 15_ragas.py       (falls back to the offline demo without ragas/key)
"""
import os

# A tiny RAG sample to evaluate: one grounded answer, one hallucinated answer.
SAMPLES = [
    {
        "user_input": "What is the standard return window?",
        "retrieved_contexts": ["Our standard return window is 30 days from delivery."],
        "response": "You can return items within 30 days of delivery.",
        "reference": "30 days from delivery.",
    },
    {
        "user_input": "What is the standard return window?",
        "retrieved_contexts": ["Our standard return window is 30 days from delivery."],
        "response": "You can return items within 90 days, no questions asked.",  # hallucinated
        "reference": "30 days from delivery.",
    },
]

METRIC_DOC = """RAGAS core metrics (all LLM-judged):
  - Faithfulness             : is the answer grounded in the retrieved context? (anti-hallucination)
  - ResponseRelevancy        : is the answer relevant to the question?
  - LLMContextPrecision...   : are the RELEVANT chunks ranked high among what was retrieved?
  - LLMContextRecall         : did retrieval fetch everything needed (vs the reference)?
"""


def real_ragas():
    """Run actual RAGAS if it imports AND a key is set. Returns True if it ran."""
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (
            Faithfulness, ResponseRelevancy,
            LLMContextPrecisionWithReference, LLMContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except Exception as e:
        print(f"[ragas] not importable ({type(e).__name__}: {e})")
        print("        -> install the pinned combo above; running the offline demo instead.\n")
        return False

    if not os.getenv("OPENAI_API_KEY"):
        print("[ragas] imported OK, but OPENAI_API_KEY is not set (metrics are LLM-judged).")
        print("        -> set the key to score live; running the offline demo instead.\n")
        return False

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    dataset = EvaluationDataset.from_list(SAMPLES)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(),
                 LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm, embeddings=emb,
    )
    print("REAL RAGAS scores:\n", result)
    return True


def toy_offline():
    """Dependency-free ILLUSTRATION of faithfulness (crude word-overlap proxy, NOT real RAGAS)."""
    print("Offline toy illustration (word-overlap proxy — NOT real RAGAS):\n")
    for s in SAMPLES:
        ctx = " ".join(s["retrieved_contexts"]).lower()
        words = [w for w in s["response"].lower().replace(".", "").replace(",", "").split() if len(w) > 3]
        supported = sum(1 for w in words if w in ctx)
        score = supported / len(words) if words else 0.0
        verdict = "grounded" if score >= 0.5 else "likely hallucinated"
        print(f"  Q: {s['user_input']}")
        print(f"     answer: {s['response']}")
        print(f"     faithfulness~{score:.2f} -> {verdict}\n")


if __name__ == "__main__":
    print(METRIC_DOC)
    if not real_ragas():
        toy_offline()
    print("Done. Swap the toy proxy for real RAGAS by installing the pinned combo + setting a key.")
