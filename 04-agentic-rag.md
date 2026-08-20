# 04 · Agentic RAG

> Ground an agent's answers in a document corpus by retrieving the most relevant passages at query time — and let the agent *decide* whether retrieval is even needed.

---

## 1. Mental model

**Plain RAG** (Retrieval-Augmented Generation) is a five-step pipeline that injects external knowledge into an LLM's context so it answers from *your* documents instead of its (stale, hallucination-prone) parametric memory:

```
chunk  →  embed  →  store  →  retrieve  →  ground
 │         │         │          │            │
split docs  turn each  put vectors  embed the   stuff the top-k
into small  chunk into  in an index  query, find  chunks into the
passages    a vector    (vector DB)  nearest      prompt so the LLM
                                     chunks       answers from them
```

- **Chunk** — documents are split into passages (paragraphs, sliding windows, sections). Chunk size is a real tuning knob: too big and retrieval is imprecise; too small and you lose context.
- **Embed** — each chunk → a dense vector (e.g. 384–1536 floats) from an embedding model. Semantically similar text → nearby vectors.
- **Store** — vectors go into an index (Chroma, FAISS, pgvector, Pinecone…) that supports fast nearest-neighbour search.
- **Retrieve** — the query is embedded the same way; the store returns the top-k nearest chunks (cosine / L2 distance).
- **Ground** — those chunks are pasted into the prompt with a "answer using these sources" instruction. The model cites them.

### What makes it "agentic"

Plain RAG **always retrieves** — every query hits the vector store, whether or not it needs to. **Agentic RAG** hands the *retrieve-or-not* (and *what to retrieve*) decision to the agent itself:

| | Plain RAG | Agentic RAG |
|---|---|---|
| Retrieval trigger | Always, unconditionally | Agent decides per-query |
| Query used | The raw user query | Agent may rewrite / expand it |
| Empty result | Forces a low-relevance answer | Agent can say "nothing relevant" and skip |
| Cost | 1 embed + 1 search every time | 0 if the agent judges it unnecessary |

In an agent graph, retrieval becomes just another **tool** the LLM can choose to call. "Should I retrieve?" is a routing decision — the same shape as "should I call the calculator?". This is exactly how the On-Call Copilot in this repo frames it (§3): a `runbook_retriever` worker whose *whole job* is to decide whether a runbook is relevant before citing one.

---

## 2. Smallest working example

Two versions of the same idea: a real vector store, and a zero-dependency keyword fallback. Both answer "which runbook covers a database disk problem?" over two toy docs.

### 2a. Real vector store (Chroma)

```bash
pip install chromadb
```

```python
# rag_chroma.py — ingest 3 docs, embed, query top-k
import chromadb

DOCS = {
    "checkout-5xx.md": "Elevated 5xx error rate on checkout-api. Errors cluster on "
                       "/checkout/submit and often correlate with a recent deploy. "
                       "Most common cause: a bad deploy. Remediation: roll back.",
    "db-disk.md":      "db-primary disk usage above 90%. Risk of write failures and "
                       "cascading outages. Cause: unrotated WAL/logs. Remediation: "
                       "rotate logs and scale the volume. Restarting does NOT free disk.",
    "latency.md":      "p99 latency regression on api-gateway after a config change. "
                       "Check connection-pool saturation and thread starvation.",
}

client = chromadb.EphemeralClient()                 # in-memory; no disk
col = client.get_or_create_collection("runbooks")   # default embedder: all-MiniLM-L6-v2
col.add(
    documents=list(DOCS.values()),
    ids=list(DOCS.keys()),
    metadatas=[{"file": k} for k in DOCS],
)

res = col.query(query_texts=["database is running out of disk space"], n_results=2)
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{dist:.3f}] {meta['file']}: {doc[:70]}...")
```

**Run it:**
```bash
python rag_chroma.py
```

**Observe:** `db-disk.md` comes back **first** with the *smallest* distance — even though the query says "running out of disk space" and the doc says "disk usage above 90% / unrotated WAL". No shared keywords beyond "disk", yet the embedding model matches them *semantically*. That is the whole point of vectors over keyword match. (First run downloads the ~80 MB MiniLM model.)

### 2b. Keyword fallback (no dependencies)

```python
# rag_keyword.py — same corpus, pure stdlib, ranks by shared-word overlap
DOCS = { ... }  # same dict as above

def keyword_search(query, docs):
    q = set(query.lower().split())
    ranked = sorted(
        docs.items(),
        key=lambda kv: len(q & set(kv[1].lower().split())),
        reverse=True,
    )
    return ranked[0]

fname, passage = keyword_search("database disk usage high", DOCS)
print(f"{fname}: {passage[:70]}...")
```

**Observe:** the keyword version only wins when the query *literally shares words* with the doc ("disk", "usage"). Ask it "running out of space" and it degrades — no overlap with "disk usage above 90%". Vectors handle the paraphrase; keyword search doesn't. This is precisely the tradeoff the real code makes (§3): use Chroma if available, degrade to keyword overlap if not.

---

## 3. How the On-Call Copilot uses it

The Copilot exposes retrieval as an SRE tool, `search_runbooks`, in
[`mcp_server/tools.py`](../nvidia-aai/mcp_server/tools.py). The corpus is two Markdown runbooks in
[`data/runbooks/`](../nvidia-aai/data/runbooks/): `checkout-5xx.md` (5xx spike → roll back the bad deploy)
and `db-disk-pressure.md` (disk >90% → scale the volume, don't restart).

### The retrieval tool

```python
# mcp_server/tools.py
def search_runbooks(query="", top_k=4, **extra):
    """Agentic RAG retrieval: uses a Chroma vector store if chromadb is installed,
    else a simple keyword search over the runbook files. Returns '<path> — <passage>'."""
    try:
        return _chroma_search(query, top_k)
    except Exception:
        return _keyword_search(query)
```

The `try/except` **is** the graceful-degradation strategy: attempt the vector path, fall back to keyword overlap on *any* failure (import error, corrupt index, etc.).

> **Honesty note.** The task brief assumed `chromadb` might not be installed. In *this* checkout it **is** (`chromadb==1.1.1`), so `_chroma_search` runs and you get real vector retrieval. On a machine without it, `import chromadb` raises inside `_chroma_search`, the `except` catches it, and `_keyword_search` runs instead — same interface, weaker matching. Both return the identical `"<path> — <passage>"` string, so nothing downstream can tell which ran.

### Chunking, ingest, and query — all in one function

```python
# mcp_server/tools.py
def _paragraphs():
    """Yield (filename, paragraph) for every non-empty block across all runbooks."""
    for f in sorted(RUNBOOKS.glob("*.md")):
        for para in f.read_text().split("\n\n"):   # ← the CHUNKING strategy: split on blank lines
            if para.strip():
                yield f.name, " ".join(para.split())

def _chroma_search(query, top_k):
    import chromadb
    col = chromadb.PersistentClient(path=str(DATA / ".chroma")).get_or_create_collection("runbooks")
    if col.count() == 0:                       # first run: ingest the runbook corpus
        docs, ids, metas = [], [], []
        for i, (fname, para) in enumerate(_paragraphs()):
            docs.append(para); ids.append(str(i)); metas.append({"file": fname})
        col.add(documents=docs, ids=ids, metadatas=metas)   # embeds with Chroma's default model
    res = col.query(query_texts=[query], n_results=top_k)
    return f"runbooks/{res['metadatas'][0][0]['file']} — {res['documents'][0][0][:200]}"
```

Note the real design choices, all visible in ~10 lines:
- **Chunk = paragraph.** `text.split("\n\n")` — one blank-line-delimited block per chunk. Simple and effective for short Markdown runbooks.
- **`metadatas={"file": fname}`** — the source filename rides along with each vector so the answer can cite *which* runbook it came from.
- **Lazy ingest.** `if col.count() == 0` ingests only on the first ever call; a `PersistentClient` keeps the index on disk in `data/.chroma/` across process restarts.
- **`top_k=4`** is retrieved but only the **top-1** (`[0][0]`) is returned in the summary string — the code *ranks* 4 but *cites* the single best.

The keyword fallback is the same contract with a bag-of-words ranker:

```python
# mcp_server/tools.py
def _keyword_search(query):
    q = set(query.lower().split())
    best, best_score = None, 0
    for fname, para in _paragraphs():
        score = len(q & set(para.lower().split()))   # shared-word count = relevance
        if score > best_score:
            best, best_score = (fname, para), score
    if not best:
        return "No relevant runbook found."          # ← honest empty result, not a forced guess
    return f"runbooks/{best[0]} — {best[1][:200]}"
```

### The "<path> — <passage>" citation format

Both backends return the **same** string shape:

```
runbooks/checkout-5xx.md — A bad deploy. The 5xx rate jumping within minutes of a deploy marker almost always means the new version is the cause.
```

That `<path> — <passage>` convention is the contract the synthesis step relies on downstream: the path is a **checkable citation** (an engineer can open the file) and the passage is the **grounding evidence**. Truncating to `[:200]` keeps the tool output compact for the prompt.

### The whether-to-retrieve decision

This is the agentic half, and it lives in **config, not code** — the `runbook_retriever` worker in
[`config/agents.yaml`](../nvidia-aai/config/agents.yaml):

```yaml
  - name: runbook_retriever      # ← agentic RAG (M2): decides whether/what to retrieve
    role: >
      Decide whether a runbook or past postmortem is relevant. If so, retrieve the
      most relevant passages from the runbook knowledge base and cite them. If
      nothing is relevant, say so — do not force a citation.
    tools: [search_runbooks]
    rag:
      collection: runbooks       # Chroma collection (data/runbooks/)
      top_k: 4
```

Read the role prompt carefully — it is a masterclass in agentic RAG framing:
- *"Decide **whether** a runbook is relevant"* — retrieval is conditional, not automatic.
- *"If nothing is relevant, **say so** — do not force a citation."* — the agent is explicitly allowed to return nothing. This kills the classic RAG failure mode where the model dresses up an irrelevant top-1 chunk as an answer.
- `rag.collection: runbooks` and `rag.top_k: 4` — the retrieval config is declarative, sitting next to the prompt, matching the `top_k=4` default in the tool.

And the orchestrator decides whether to even *dispatch* this worker (a second, coarser layer of "should we retrieve?"):

```yaml
  few_shot:
    - alert: "5xx rate on checkout-api spiked to 12% at 02:14 UTC"
      plan: ["metric_fetcher", "log_analyzer", "code_searcher", "runbook_retriever"]
    - alert: "disk usage on db-primary at 94%"
      plan: ["metric_fetcher", "runbook_retriever"]
```

So there are **two** whether-to-retrieve gates: the orchestrator picks *whether to run the retriever at all*, and the retriever picks *whether to cite anything it finds*.

---

## 4. Build it up

### 4a. Real embeddings via OpenAI (swap the embedder)

Chroma's default (`all-MiniLM-L6-v2`, local, 384-dim) is fine for toy corpora. For production quality, plug in a hosted embedding model:

```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="...", model_name="text-embedding-3-small",   # 1536-dim
)
col = client.get_or_create_collection("runbooks", embedding_function=openai_ef)
```

The rest of the pipeline is unchanged — this is the value of the vector-store abstraction. (Rule: the *same* embedding function must be used at ingest and at query time, or the vectors live in different spaces and retrieval is garbage.)

### 4b. Metadata filters (retrieve within a subset)

The `metadatas={"file": fname}` we stored enables filtered search — "only retrieve from DB runbooks":

```python
res = col.query(
    query_texts=["disk filling up"],
    n_results=4,
    where={"file": "db-disk-pressure.md"},   # metadata filter (pre-filter before ANN search)
)
```

In a bigger system you'd store `{"service": "checkout-api", "type": "postmortem", "date": ...}` and filter by service — dramatically cutting false matches.

### 4c. Agentic decision to skip retrieval

Make the "should I retrieve?" gate explicit with a cheap classifier step before the search:

```python
def maybe_retrieve(query, col, top_k=4):
    # 1. cheap relevance gate — skip retrieval entirely for non-knowledge queries
    if not looks_like_a_knowledge_question(query):     # e.g. an LLM yes/no, or a keyword heuristic
        return None                                    # ← retrieved 0 chunks, spent 0 on search

    # 2. retrieve, then a RELEVANCE THRESHOLD gate
    res = col.query(query_texts=[query], n_results=top_k)
    dist = res["distances"][0][0]
    if dist > 0.6:                                      # nearest chunk is still too far
        return None                                    # "nothing relevant" — mirrors the YAML role
    return res["documents"][0][0], res["metadatas"][0][0]["file"]
```

This is the code embodiment of the YAML role: *decide whether relevant → if not, say so → don't force a citation.* The distance threshold turns "top-k always returns something" into "top-k returns something **good enough**".

### 4d. Citing sources cleanly

Return structured citations instead of a truncated blob, so the UI can render clickable links:

```python
def cited_answer(query, col, top_k=4):
    res = col.query(query_texts=[query], n_results=top_k)
    return [
        {"source": m["file"], "passage": d, "distance": s}
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
# → [{"source": "db-disk-pressure.md", "passage": "...", "distance": 0.31}, ...]
```

The repo's `"<path> — <passage>"` string is the flattened, prompt-friendly version of this.

---

## 5. Gotchas & pitfalls

- **Chunking is the highest-leverage knob.** `split("\n\n")` works for tidy Markdown; real docs need sentence/token-window chunking with overlap (e.g. 512 tokens, 64-token stride) so a fact split across a boundary isn't lost. Too-large chunks dilute the embedding; too-small chunks lose context.
- **Embed query and corpus with the *same* model.** Mixing embedders (MiniLM ingest, OpenAI query) silently returns nonsense — no error, just bad results.
- **`top_k` returns *something* no matter what** — an empty/irrelevant corpus still yields a "nearest" chunk. Always pair `top_k` with a **relevance/distance threshold** (§4c) or the agent will confidently cite an irrelevant runbook. This is the exact failure the YAML role guards against.
- **Vector ≠ keyword.** Vectors nail paraphrase ("out of space" ≈ "disk >90%") but can miss exact identifiers (error codes, function names) that keyword/BM25 catches. Production systems often do **hybrid** (vector + keyword) retrieval and re-rank.
- **Stale index.** The `if col.count() == 0` lazy-ingest here **never re-ingests** after the first run — edit a runbook and the index is stale until you wipe `data/.chroma/`. Real systems need an ingestion/refresh job.
- **Eval & class imbalance.** Evaluate retrieval separately from generation: **hit-rate / recall@k** (was the right chunk in the top-k?) and **MRR** (how high was it ranked?). With imbalanced corpora (100 checkout docs, 2 DB docs) a naive retriever over-returns the majority class — measure per-class recall, not just overall. This ties directly to [12-evaluation-and-regression.md].
- **Don't force citations.** An agent that always cites *something* trains users to distrust every citation. "Nothing relevant" is a valid, valuable answer — both `_keyword_search` (`"No relevant runbook found."`) and the YAML role enforce this.

---

## ✅ Best Practices

- **Chunk with overlap and tune the size.** Use a sliding token window (e.g. 512 tokens, 64-token stride) and A/B the size against your eval set so facts that straddle a boundary survive retrieval.
- **Add a relevance/decision gate.** Put a cheap classifier or distance threshold in front of the store (§4c) so the agent skips retrieval for non-knowledge queries and declines when the nearest chunk is too far.
- **Always cite sources.** Return the `<path> — <passage>` (or structured `{source, passage, distance}`) so every claim is traceable to an openable document and reviewers can verify it.
- **Use hybrid search plus reranking.** Combine dense vectors with keyword/BM25 to catch exact identifiers (error codes, function names), then rerank the merged candidates with a cross-encoder before grounding.
- **Filter by metadata before the ANN search.** Store `{service, type, date, ...}` on each vector and pass a `where` filter so a `checkout-api` query can't surface a DB runbook — this slashes false matches at scale.
- **Cache embeddings and reuse the same embedder.** Persist vectors keyed by content hash so unchanged chunks aren't re-embedded, and pin one embedding model for both ingest and query.
- **Measure retrieval quality continuously.** Track recall@k / hit-rate and MRR on a labeled query set (and RAGAS context precision/recall) so a regression in retrieval is caught before it reaches answers.
- **Keep the corpus fresh with an ETL job.** Run a scheduled ingest that re-chunks and re-embeds changed docs (or wipes stale indexes) so edits show up instead of serving yesterday's index.

---

## 6. Exercises

1. **Swap keyword → Chroma and diff the answers.** With `chromadb` installed, call `search_runbooks("database running out of space")`. Then force the fallback (temporarily `raise` at the top of `_chroma_search`, or uninstall chromadb) and call it again. The vector path should match `db-disk-pressure.md` on the paraphrase; the keyword path may not. Explain why.
2. **Add a relevance threshold.** Extend `_chroma_search` to read `res["distances"]` and return `"No relevant runbook found."` when the best distance exceeds a threshold you pick. Query it with an off-topic string ("how do I reset my password") and confirm it now declines instead of citing a runbook.
3. **Measure retrieval hit-rate.** Write 6 test queries with the runbook you *expect* each to retrieve (`{"5xx after deploy": "checkout-5xx.md", "disk 94%": "db-disk-pressure.md", ...}`). Run both backends, compute `hits / total` for each, and report **recall@1** for keyword vs. vector.
4. **Improve chunking.** Replace `split("\n\n")` with a version that keeps the Markdown `##` section heading attached to each chunk (so "## Remediation" travels with its steps). Re-run exercise 3 — did hit-rate change?
5. **Add metadata filtering.** Store `{"service": ...}` metadata at ingest and add a `service=` param to `search_runbooks` that passes a `where` filter. Verify a `checkout-api` query can't return the DB runbook.
6. **Wire the whether-to-retrieve gate.** Implement §4c's `maybe_retrieve` as a pre-check in `search_runbooks` that returns `"No runbook needed."` for queries that don't look like knowledge questions, and log how often retrieval is skipped over a batch of 10 mixed queries.

---

## 7. Connections

- **[03-model-context-protocol.md]** — `search_runbooks` is exposed as an MCP tool by `mcp_server/server.py`; retrieval is *just another tool* the agent can call. That tutorial covers the transport; this one covers what's behind this particular tool.
- **[06-orchestrator-worker-multi-agent.md]** — the `runbook_retriever` is one worker node in the orchestrator→worker graph. The orchestrator's whether-to-*dispatch* decision (see the `few_shot` plans) is the outer layer of the whether-to-*retrieve* decision covered here.
- **[12-evaluation-and-regression.md]** — retrieval needs its *own* metrics (recall@k, MRR, per-class hit-rate) upstream of end-to-end answer eval; the class-imbalance caveat in §5 lands there.

---

## 8. Further reading

- **Chroma docs** — collections, embedding functions, `where` metadata filters, persistent vs. ephemeral clients: <https://docs.trychroma.com>
- **RAGAS** — reference-free RAG evaluation (context precision/recall, faithfulness, answer relevance): <https://docs.ragas.io>
- **OpenAI embeddings guide** — model choices, dimensions, cost, the "same embedder at ingest & query" rule: <https://platform.openai.com/docs/guides/embeddings>
- **Sentence-Transformers** — the `all-MiniLM-L6-v2` model behind Chroma's default embedder: <https://www.sbert.net>
- **"Retrieval-Augmented Generation" (Lewis et al., 2020)** — the original RAG paper: <https://arxiv.org/abs/2005.11401>
