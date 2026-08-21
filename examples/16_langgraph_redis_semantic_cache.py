"""
16 · LangGraph + Redis (semantic caching) — standalone runnable
==============================================================

Demonstrates putting a **semantic cache** in front of the expensive node of a
LangGraph agent, so a *paraphrase* of an earlier question is answered from
Redis instead of the LLM:

    1. EMBEDDER      -> text -> unit vector. Real embeddings if OPENAI_API_KEY
                        is set; a toy LEXICAL embedder otherwise (see caveat).
    2. VECTOR STORE  -> Redis RediSearch KNN (COSINE) if a server is reachable;
                        an in-memory numpy store otherwise. Same interface.
    3. SEMANTIC CACHE-> lookup(question) does a 1-NN search and accepts the hit
                        only if similarity >= THRESHOLD. write() stores the
                        answer with a TTL, namespaced by model + prompt version.
    4. GRAPH         -> START -> cache_lookup -> (hit? END : llm -> cache_write)
                        The conditional edge is the whole trick: a cache hit
                        SKIPS the LLM node entirely.
    5. DRIVE IT      -> cold misses, paraphrase hits, a threshold rejection,
                        exact-vs-semantic contrast, and LangGraph's built-in
                        exact node cache (CachePolicy) for comparison.

Degrades gracefully — with no Redis and no API key it still runs end to end and
prints the same structure, so you can read the flow offline.

Deps:
    pip install langgraph redis numpy
    # optional, for REAL embeddings + a real LLM call:
    pip install openai   # and export OPENAI_API_KEY=...

Optional Redis (recommended — this is the point of the example):
    docker run -d --name aai-redis -p 6379:6379 redis/redis-stack-server:latest
    # RediSearch is required for vector search; plain `redis:7` will NOT work.

How to run:
    python examples/16_langgraph_redis_semantic_cache.py
    REDIS_URL=redis://localhost:6379 python examples/16_langgraph_redis_semantic_cache.py
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Iterable, TypedDict

import numpy as np
from langgraph.graph import StateGraph, START, END

# ── Tunables ────────────────────────────────────────────────────────────────
# The similarity threshold is the single most important knob in a semantic
# cache: it trades hit rate against wrong answers. Too low -> confidently serves
# a cached answer to a different question. Too high -> paraphrases miss and you
# pay full price.
#
# It is NOT a universal constant — it belongs to the embedding model. Different
# embedders spread their scores differently, so each one below carries its own
# calibrated default (see `Embedder.threshold`). Re-calibrate whenever you swap
# embedders: label ~50 real query pairs same/different, then pick the cutoff
# that maximises hits subject to zero false hits.
TTL_SECONDS = 3600        # cache entries self-expire; stale answers are worse than none
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_VERSION = "v1"     # bump to invalidate every entry built with the old prompt
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# ════════════════════════════════════════════════════════════════════════════
# 1) EMBEDDER — text -> unit vector
# ════════════════════════════════════════════════════════════════════════════
class OpenAIEmbedder:
    """Real embeddings. Similarity here is genuinely semantic."""

    name = "openai:text-embedding-3-small"
    dim = 1536
    threshold = 0.92      # calibrated: paraphrases land ~0.93-0.99, distinct pairs <0.85

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI()

    def __call__(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


class LexicalEmbedder:
    """
    Offline fallback: hash word + character-trigram features into a fixed vector.

    HONEST CAVEAT: this measures *lexical overlap*, not meaning. "restart the api"
    and "reboot the service" are near-zero similarity here even though a real
    embedder scores them high. The demo questions below are worded so overlap is
    high enough to show the mechanism. Set OPENAI_API_KEY for the real thing.
    """

    name = "lexical-hash-toy"
    dim = 512
    threshold = 0.80      # lower on purpose: overlap scores are flatter than real
                          # embeddings, so 0.92 would reject every real paraphrase.
                          # Same calibration exercise, different model, different cutoff.

    def _features(self, text: str) -> Iterable[str]:
        t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        words = [w for w in t.split() if w not in _STOPWORDS]
        yield from words
        joined = " ".join(words)
        for i in range(len(joined) - 2):          # char trigrams add fuzziness
            yield f"#{joined[i:i + 3]}"

    def __call__(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        return v / (np.linalg.norm(v) + 1e-9)


_STOPWORDS = {"the", "a", "an", "is", "do", "i", "to", "of", "for", "in", "on",
              "what", "how", "s", "whats", "can", "you", "me", "my", "it"}


def build_embedder():
    if os.getenv("OPENAI_API_KEY"):
        try:
            emb = OpenAIEmbedder()
            emb("warmup")                          # fail fast on a bad key
            return emb
        except Exception as e:
            print(f"[embedder] OpenAI unavailable ({type(e).__name__}) -> lexical fallback")
    return LexicalEmbedder()


# ════════════════════════════════════════════════════════════════════════════
# 2) VECTOR STORE — Redis KNN, or an in-memory stand-in with the same interface
# ════════════════════════════════════════════════════════════════════════════
class RedisVectorStore:
    """
    RediSearch vector index over plain hashes.

    Each entry is a HASH at `{prefix}{sha}` holding question / answer / ns / the
    float32 embedding blob. The index is created once over that key prefix, and
    a KNN query returns the nearest neighbours by COSINE *distance*.
    """

    kind = "redis"

    def __init__(self, url: str, dim: int, index: str = "aai:semcache:idx",
                 prefix: str = "aai:semcache:") -> None:
        import redis
        from redis.commands.search.field import NumericField, TagField, TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        self.r = redis.Redis.from_url(url)
        self.r.ping()                              # raises if unreachable
        if not any(m[b"name"] == b"search" for m in self.r.execute_command("MODULE", "LIST")):
            raise RuntimeError("RediSearch module missing — use redis/redis-stack-server")

        self.dim, self.index, self.prefix = dim, index, prefix
        try:
            self.r.ft(index).info()                # already there? reuse it
        except Exception:
            self.r.ft(index).create_index(
                (
                    TextField("question"),
                    TextField("answer"),
                    TagField("ns"),                # namespace = model + prompt version
                    NumericField("created_at"),
                    VectorField(
                        "embedding",
                        "HNSW",                    # ANN; use "FLAT" for exact brute force
                        {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                    ),
                ),
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )

    def add(self, key: str, question: str, answer: str, ns: str,
            vec: np.ndarray, ttl: int) -> None:
        rkey = f"{self.prefix}{key}"
        self.r.hset(rkey, mapping={
            "question": question,
            "answer": answer,
            "ns": ns,
            "created_at": int(time.time()),
            "embedding": vec.astype(np.float32).tobytes(),
        })
        self.r.expire(rkey, ttl)                   # TTL evicts; the index follows

    def search(self, ns: str, vec: np.ndarray, k: int = 1):
        from redis.commands.search.query import Query

        # `(@ns:{...})=>[KNN k @embedding $vec AS score]` — pre-filter by namespace,
        # then nearest-neighbour. dialect(2) is required for vector syntax.
        q = (
            Query(f"(@ns:{{{ns}}})=>[KNN {k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("question", "answer", "score")
            .dialect(2)
        )
        res = self.r.ft(self.index).search(
            q, query_params={"vec": vec.astype(np.float32).tobytes()}
        )
        # COSINE *distance* -> similarity is 1 - distance.
        return [
            (1.0 - float(d.score), d.question.decode() if isinstance(d.question, bytes) else d.question,
             d.answer.decode() if isinstance(d.answer, bytes) else d.answer)
            for d in res.docs
        ]

    def size(self) -> int:
        return int(self.r.ft(self.index).info()["num_docs"])

    def clear(self) -> None:
        for k in self.r.scan_iter(match=f"{self.prefix}*"):
            self.r.delete(k)


class MemoryVectorStore:
    """Offline stand-in: brute-force cosine over a list. Same interface, no server."""

    kind = "memory"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._rows: list[tuple[str, str, str, np.ndarray, float]] = []

    def add(self, key, question, answer, ns, vec, ttl):
        self._rows.append((question, answer, ns, vec.astype(np.float32), time.time() + ttl))

    def search(self, ns, vec, k=1):
        now = time.time()
        self._rows = [r for r in self._rows if r[4] > now]        # manual TTL sweep
        scored = [(float(np.dot(vec, r[3])), r[0], r[1]) for r in self._rows if r[2] == ns]
        scored.sort(key=lambda t: -t[0])
        return scored[:k]

    def size(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()


def build_store(dim: int):
    try:
        store = RedisVectorStore(REDIS_URL, dim)
        print(f"[store] Redis vector index ready at {REDIS_URL}")
        return store
    except Exception as e:
        print(f"[store] Redis unavailable ({type(e).__name__}: {e}) -> in-memory fallback")
        return MemoryVectorStore(dim)


# ════════════════════════════════════════════════════════════════════════════
# 3) SEMANTIC CACHE — the lookup/write policy on top of the store
# ════════════════════════════════════════════════════════════════════════════
class SemanticCache:
    def __init__(self, embedder, store, threshold=None, ttl=TTL_SECONDS) -> None:
        self.embed, self.store = embedder, store
        # Default to the embedder's own calibrated cutoff, not a global constant.
        self.threshold = embedder.threshold if threshold is None else threshold
        self.ttl = ttl
        # The namespace is why a model swap or prompt edit can't serve stale answers:
        # it's part of the key space, so old entries become unreachable, not wrong.
        # Sanitised to [A-Za-z0-9_] so it needs no RediSearch TAG escaping.
        self.ns = re.sub(r"[^A-Za-z0-9_]", "_", f"{MODEL}_{PROMPT_VERSION}_{embedder.name}")
        self.hits = self.misses = 0

    def lookup(self, question: str):
        vec = self.embed(question)
        neighbours = self.store.search(self.ns, vec, k=1)
        if not neighbours:
            self.misses += 1
            return None, 0.0, None
        sim, matched_q, answer = neighbours[0]
        if sim >= self.threshold:
            self.hits += 1
            return answer, sim, matched_q
        self.misses += 1
        return None, sim, matched_q          # near-miss: report the score we rejected

    def write(self, question: str, answer: str) -> None:
        key = hashlib.sha256(f"{self.ns}|{question}".encode()).hexdigest()[:24]
        self.store.add(key, question, answer, self.ns, self.embed(question), self.ttl)


class ExactCache:
    """The naive baseline: hash the normalised string. Zero paraphrase tolerance."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}
        self.hits = self.misses = 0

    def lookup(self, question: str):
        k = hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()
        hit = self._d.get(k)
        self.hits += bool(hit)
        self.misses += not hit
        return hit

    def write(self, question: str, answer: str) -> None:
        self._d[hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()] = answer


# ════════════════════════════════════════════════════════════════════════════
# 4) THE GRAPH — cache_lookup -> (hit? END : llm -> cache_write)
# ════════════════════════════════════════════════════════════════════════════
class State(TypedDict, total=False):
    question: str
    answer: str
    cache: str            # "hit" | "miss"
    similarity: float     # score of the best neighbour, hit or not
    matched: str          # which cached question matched (for auditing)
    llm_ms: float         # time actually spent in the LLM node


_CANNED = {
    "restart": "Run `kubectl rollout restart deploy/api -n prod`, then watch readiness probes.",
    "slo":     "The API SLO is 99.9% availability over a 30-day rolling window.",
    "rollback": "Use `kubectl rollout undo deploy/api -n prod` to revert to the previous ReplicaSet.",
}


def call_llm(question: str) -> str:
    """The expensive node. Real call if a key is present, else a slow canned answer."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            resp = OpenAI().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a terse SRE assistant. One sentence."},
                    {"role": "user", "content": question},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [llm] API call failed ({type(e).__name__}) -> canned answer")
    time.sleep(0.4)                                   # stand in for real model latency
    normalised = question.lower().replace("roll back", "rollback")
    for kw, ans in _CANNED.items():
        if kw in normalised:
            return ans
    return "No runbook entry matched; escalate to the on-call lead."


def build_app(cache: SemanticCache):
    # ── NODE: cache_lookup — the only node that touches Redis on the read path.
    def cache_lookup(state: State):
        answer, sim, matched = cache.lookup(state["question"])
        if answer is not None:
            return {"cache": "hit", "answer": answer, "similarity": sim, "matched": matched}
        return {"cache": "miss", "similarity": sim, "matched": matched or ""}

    # ── NODE: llm — skipped entirely on a hit. That skip is the whole payoff.
    def llm(state: State):
        t0 = time.perf_counter()
        answer = call_llm(state["question"])
        return {"answer": answer, "llm_ms": (time.perf_counter() - t0) * 1000}

    # ── NODE: cache_write — populate on the way out, never on the way in.
    def cache_write(state: State):
        cache.write(state["question"], state["answer"])
        return {}

    # ── ROUTER: a conditional edge is what makes the cache a *graph* concern
    #    rather than an if-statement buried inside a node.
    def route(state: State) -> str:
        return "hit" if state["cache"] == "hit" else "miss"

    g = StateGraph(State)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("llm", llm)
    g.add_node("cache_write", cache_write)

    g.add_edge(START, "cache_lookup")
    g.add_conditional_edges("cache_lookup", route, {"hit": END, "miss": "llm"})
    g.add_edge("llm", "cache_write")
    g.add_edge("cache_write", END)
    return g.compile()


def ask(app, question: str) -> State:
    t0 = time.perf_counter()
    out = app.invoke({"question": question})
    wall = (time.perf_counter() - t0) * 1000
    tag = "HIT " if out["cache"] == "hit" else "MISS"
    detail = f"sim={out['similarity']:.3f}"
    if out["cache"] == "hit":
        detail += f"  matched={out['matched']!r}"
    print(f"  [{tag}] {question!r}\n         {detail}  wall={wall:6.1f}ms")
    print(f"         -> {out['answer'][:72]}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# 5) DRIVE IT
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    embedder = build_embedder()
    store = build_store(embedder.dim)
    print(f"[setup] embedder={embedder.name} dim={embedder.dim} store={store.kind}")

    cache = SemanticCache(embedder, store)
    print(f"[setup] threshold={cache.threshold} (calibrated for this embedder)")
    if isinstance(embedder, LexicalEmbedder):
        print("[setup] NOTE: toy lexical embedder — it scores word overlap, not meaning.")
        print("        Set OPENAI_API_KEY for real embeddings (and threshold 0.92).")
    print()

    cache.store.clear()                    # deterministic demo run
    app = build_app(cache)

    print("=" * 74)
    print("PART A — cold cache: every question is a miss and pays for the LLM")
    print("=" * 74)
    seeds = [
        "How do I restart the api service in prod?",
        "What is the api availability SLO?",
        "How do I roll back the api deployment?",
    ]
    for q in seeds:
        ask(app, q)
    print(f"\n  cache now holds {store.size()} entries\n")

    print("=" * 74)
    print("PART B — paraphrases: different strings, same intent -> served from Redis")
    print("=" * 74)
    for q in [
        "How do I restart the api service in production?",
        "What is the availability SLO for the api?",
        "How do I roll back an api deployment?",
    ]:
        ask(app, q)
    print()

    print("=" * 74)
    print("PART C — the threshold earns its keep: a DIFFERENT question must miss")
    print("=" * 74)
    ask(app, "Who is the on-call engineer for the billing service tonight?")
    print("\n  A cache that answered that one from the 'restart api' entry would be")
    print("  worse than no cache at all — that's the failure mode THRESHOLD prevents.\n")

    print("=" * 74)
    print("PART D — exact-match cache on the same traffic, for contrast")
    print("=" * 74)
    exact = ExactCache()
    for q in seeds:
        exact.write(q, "cached")
    for q in [
        "How do I restart the api service in production?",   # one word added
        "how do I restart the api service in prod?",         # only case differs
    ]:
        print(f"  exact[{'HIT ' if exact.lookup(q) else 'MISS'}] {q!r}")
    print("\n  Exact matching catches only the re-typed-identically case. Every real")
    print("  paraphrase falls through to the model. Semantic caching is the fix.\n")

    print("=" * 74)
    print("PART E — LangGraph's BUILT-IN node cache (exact, not semantic)")
    print("=" * 74)
    demo_builtin_node_cache()

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    total = cache.hits + cache.misses
    saved = cache.hits * 0.4                     # ~one skipped LLM call each
    print(f"  semantic cache : {cache.hits} hits / {total} lookups "
          f"({cache.hits / total:.0%} hit rate)")
    print(f"  exact cache    : {exact.hits} hits / {exact.hits + exact.misses} lookups")
    print(f"  LLM calls avoided: {cache.hits}  (~{saved:.1f}s and {cache.hits} billed "
          f"completions not spent)")
    print(f"  store: {store.kind}, {store.size()} entries, ttl={TTL_SECONDS}s, ns={cache.ns}")
    print("\nDone — the conditional edge after cache_lookup is what skipped the LLM.")


def demo_builtin_node_cache() -> None:
    """
    LangGraph ships node-level caching: `add_node(..., cache_policy=CachePolicy(...))`
    plus `compile(cache=...)`. It keys on a hash of the node INPUT, so it is an
    EXACT cache — great for deterministic/expensive nodes, blind to paraphrase.
    Use both: built-in for node memoization, semantic for question-level reuse.
    """
    try:
        from langgraph.cache.memory import InMemoryCache
        from langgraph.types import CachePolicy
    except Exception as e:
        print(f"  [skip] this langgraph build has no node cache API ({type(e).__name__})\n")
        return

    calls = {"n": 0}

    def expensive(state: State):
        calls["n"] += 1
        time.sleep(0.2)
        return {"answer": f"computed for {state['question']!r}"}

    g = StateGraph(State)
    g.add_node("expensive", expensive,
               cache_policy=CachePolicy(key_func=lambda s: s["question"], ttl=60))
    g.add_edge(START, "expensive")
    g.add_edge("expensive", END)

    cache_backend = InMemoryCache()
    # Swap in Redis with:  from langgraph.cache.redis import RedisCache
    #                      RedisCache(redis.Redis.from_url(REDIS_URL))
    try:
        app = g.compile(cache=cache_backend)
    except TypeError as e:
        print(f"  [skip] compile(cache=...) unsupported here ({e})\n")
        return

    q = "How do I restart the api service in prod?"
    for label in ("first call ", "second call"):
        t0 = time.perf_counter()
        app.invoke({"question": q})
        print(f"  {label}: {(time.perf_counter() - t0) * 1000:6.1f}ms  "
              f"node executions so far = {calls['n']}")
    print(f"  node ran {calls['n']}x for 2 invocations -> the second was cached.")
    print("  But note: change one word and it runs again. Exact, not semantic.\n")


if __name__ == "__main__":
    main()
