"""
04 · Agentic RAG — standalone runnable
======================================

Demonstrates the RAG pipeline (chunk -> embed -> store -> retrieve -> ground)
plus the "agentic" twist: the agent DECIDES whether retrieval is even worth it,
and declines to cite anything when nothing is relevant.

Two backends, one contract — both return a "<source> — <passage>" citation:
    * PRIMARY:  Chroma vector store (semantic match, handles paraphrase).
    * FALLBACK: pure-stdlib keyword overlap (used if `import chromadb` fails),
                so this script runs anywhere with zero dependencies.

The agentic gate has two layers (mirrors the On-Call Copilot's runbook_retriever):
    1. a cheap relevance pre-check   -> skip retrieval for non-knowledge queries.
    2. a distance/score threshold    -> decline to cite a weak nearest chunk.

Deps:
    pip install chromadb        # optional; without it the keyword fallback runs
How to run:
    python examples/04_agentic_rag.py
"""

# ── Toy corpus: 3 in-code "runbook" docs (source filename -> passage) ───────
DOCS = {
    "checkout-5xx.md": "Elevated 5xx error rate on checkout-api. Errors cluster on "
                       "/checkout/submit and correlate with a recent deploy. Most "
                       "common cause: a bad deploy. Remediation: roll back.",
    "db-disk.md":      "db-primary disk usage above 90 percent. Risk of write failures "
                       "and cascading outages. Cause: unrotated WAL and logs. "
                       "Remediation: rotate logs and scale the volume. Restarting does "
                       "NOT free disk.",
    "latency.md":      "p99 latency regression on api-gateway after a config change. "
                       "Check connection-pool saturation and thread starvation.",
}


# ── Backend A: Chroma vector store (semantic retrieval) ─────────────────────
def chroma_search(query, top_k=2):
    """Ingest DOCS into an in-memory Chroma collection, embed, return top-k
    as (source, passage, distance). Raises if chromadb is unavailable."""
    import chromadb

    client = chromadb.EphemeralClient()                  # in-memory; no disk, no service
    col = client.get_or_create_collection("runbooks")    # default embedder: all-MiniLM-L6-v2
    if col.count() == 0:                                  # lazy ingest (chunk == whole doc here)
        col.add(
            documents=list(DOCS.values()),
            ids=list(DOCS.keys()),
            metadatas=[{"file": k} for k in DOCS],        # source filename rides with each vector
        )
    res = col.query(query_texts=[query], n_results=top_k)
    return [
        (m["file"], d, dist)
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ── Backend B: keyword overlap (stdlib fallback) ────────────────────────────
def keyword_search(query, top_k=2):
    """Rank DOCS by shared-word overlap. Returns (source, passage, pseudo_distance)
    where a smaller number = better, so the threshold logic is uniform."""
    q = set(query.lower().split())
    scored = [
        (fname, passage, len(q & set(passage.lower().split())))
        for fname, passage in DOCS.items()
    ]
    scored.sort(key=lambda t: t[2], reverse=True)         # most shared words first
    # Convert overlap-count -> a "distance" so downstream code is backend-agnostic.
    return [(f, p, 1.0 / (1 + s)) for f, p, s in scored[:top_k]]


# ── The agentic decision: whether to retrieve, and whether to cite ──────────
KNOWLEDGE_HINTS = {"error", "5xx", "disk", "latency", "deploy", "runbook", "space",
                   "database", "db", "outage", "logs", "how", "why", "fix", "cause"}


def looks_like_knowledge_question(query):
    """Cheap gate: does this query plausibly need the knowledge base at all?"""
    return bool(set(query.lower().split()) & KNOWLEDGE_HINTS)


def agentic_rag(query, search_fn, threshold):
    """Emulate the runbook_retriever worker: gate -> retrieve -> threshold -> cite."""
    if not looks_like_knowledge_question(query):
        return "No runbook needed — query doesn't look like a knowledge question."

    hits = search_fn(query)                               # top-k ranked candidates
    source, passage, dist = hits[0]                       # rank k, cite the single best
    if dist > threshold:                                  # nearest chunk still too far
        return "No relevant runbook found."              # honest empty result, not a forced guess
    return f"{source} — {passage[:120]}"                  # the "<source> — <passage>" citation


# ── Demo ────────────────────────────────────────────────────────────────────
def main():
    # Pick the backend + a matching "too far" threshold for each.
    try:
        import chromadb  # noqa: F401
        backend, search_fn, threshold = "Chroma (semantic vectors)", chroma_search, 1.3
    except Exception:
        backend, search_fn, threshold = "keyword overlap (fallback)", keyword_search, 0.99

    print(f"Backend: {backend}\n")

    queries = [
        "database is running out of disk space",   # paraphrase -> should hit db-disk.md
        "5xx errors after a deploy",               # should hit checkout-5xx.md
        "what is the capital of France",           # off-topic -> agent should decline / skip
    ]
    for q in queries:
        print(f"Q: {q}")
        print(f"   -> {agentic_rag(q, search_fn, threshold)}\n")


if __name__ == "__main__":
    main()
