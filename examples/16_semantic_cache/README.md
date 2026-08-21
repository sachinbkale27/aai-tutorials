# 16 · LangGraph + Redis semantic cache

A semantic cache in front of the expensive node of a LangGraph agent, so a
*paraphrase* of an already-answered question is served from Redis instead of
costing another model call.

```
START -> cache_lookup --(hit)---------------------------> END
                      \--(miss)--> llm -> cache_write --> END
```

The conditional edge is the point: on a hit, `llm` is **never entered**.

## Run

```bash
python examples/16_semantic_cache/demo.py
```

Works with nothing installed but `langgraph`, `redis`, and `numpy`. For the real
thing:

```bash
# vector search needs the RediSearch module — plain `redis:7` will NOT work
docker run -d --name aai-redis -p 6379:6379 redis/redis-stack-server:latest

export OPENAI_API_KEY=...        # real embeddings + a real LLM
python examples/16_semantic_cache/demo.py
```

Without Redis it falls back to an in-memory list; without a key it falls back to
a toy lexical embedder and canned answers. Both fallbacks print why, and the
hit/miss pattern is identical — so you can read the flow offline.

## The modules

Read them in this order; each one only knows about the ones below it.

| Module | Owns | Depends on |
|---|---|---|
| `demo.py` | the three scenarios (cold / paraphrase / must-miss) | everything |
| `graph.py` | the topology — nodes and the conditional edge, nothing else | `cache`, `llm` |
| `cache.py` | **the threshold decision**: is the neighbour the same question? | `embeddings`, `store`, `config` |
| `llm.py` | the expensive call the cache exists to avoid | `config` |
| `store.py` | vector storage + KNN (Redis, or a list) | `config` |
| `embeddings.py` | text → vector, **and that backend's threshold** | — |
| `config.py` | every tunable: TTL, namespace, index names | — |

The dependency chain is one-directional and acyclic: `demo → graph → cache →
{embeddings, store} → config`. Two seams are worth noticing:

- **`store.py` knows nothing about thresholds or questions.** Swapping it for
  pgvector/Qdrant/Milvus should require touching no other file. That is the test
  of whether the seam is in the right place.
- **`cache.py` has no LangGraph import.** It is the module you lift into a real
  service — see tutorial 16 §3 and its capstone exercise.

**Why the threshold lives in `embeddings.py`, not `config.py`:** it is a property
of the embedding model, not a global setting. The same paraphrase pair scores
~0.95 with real embeddings and ~0.86 with the toy one, so a single hardcoded
`0.92` would give a 100% hit rate with one backend and **0%** with the other.
Keeping them in one file means they cannot drift apart.

## A note on imports

These are plain sibling modules, not a package — `16_semantic_cache` starts with
a digit, so it is not a valid Python identifier and `python -m
examples.16_semantic_cache.demo` cannot work. Running `demo.py` directly is
fine (Python puts the script's directory on `sys.path`), from any working
directory. To reuse the code in your own project, copy `cache.py`,
`store.py`, and `embeddings.py` into a properly-named package.

## Try breaking it

- Set `THRESHOLD` in `embeddings.py` to `0.92` for the toy backend → section B's
  hit rate goes to zero.
- Set it to `0.25` → section C's on-call question gets answered from the
  "restart api" entry. A confidently wrong answer: the failure mode the
  threshold exists to prevent.
- Bump `PROMPT_VERSION` in `config.py` to `v2` and re-run without clearing Redis
  → every question misses, because the namespace moved and the old entries are
  now unreachable rather than deleted.

Full write-up, including the production concerns this example skips:
[`16-langgraph-redis-semantic-caching.md`](../../16-langgraph-redis-semantic-caching.md).
