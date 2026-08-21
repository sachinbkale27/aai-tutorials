"""
16 · LangGraph + Redis (semantic caching) — standalone runnable
==============================================================

A *paraphrase* of a question you already answered should not cost another LLM
call. So put a vector-similarity cache in front of the expensive node:

    START -> cache_lookup --(hit)---------------------------> END
                          \--(miss)--> llm -> cache_write --> END

The conditional edge is the whole trick: on a hit, `llm` is NEVER entered.

Four moving parts, in order down this file:

    1. embed()         text -> unit vector.
    2. Redis           one hash per cache entry + a KNN search over the vectors.
    3. THRESHOLD       "how similar counts as the same question?" — the knob that
                       trades hit rate against serving a WRONG answer.
    4. the graph       cache_lookup -> conditional edge -> llm -> cache_write.

Degrades gracefully so it runs anywhere: no Redis falls back to an in-memory
list, no OPENAI_API_KEY falls back to a toy embedder and canned answers.

Deps:
    pip install langgraph redis numpy       # + openai for real embeddings

Redis (recommended — vector search needs the RediSearch module, so plain
`redis:7` will NOT work):
    docker run -d --name aai-redis -p 6379:6379 redis/redis-stack-server:latest

How to run:
    python examples/16_langgraph_redis_semantic_cache.py
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import TypedDict

import numpy as np
from langgraph.graph import StateGraph, START, END

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL_SECONDS = 3600          # entries self-expire; a stale answer is worse than none
INDEX, PREFIX = "aai:semcache:idx", "aai:semcache:"

# Set by setup() once we know which embedder we got.
USE_OPENAI = False
DIM = 512
THRESHOLD = 0.80
NS = ""
R = None                    # redis client, or None -> _MEM fallback
_MEM: list[tuple[str, str, np.ndarray, float]] = []


# ════════════════════════════════════════════════════════════════════════════
# 1) EMBED — text to a unit vector
# ════════════════════════════════════════════════════════════════════════════
_STOPWORDS = {"the", "a", "an", "is", "do", "i", "to", "of", "for", "in", "on",
              "what", "how", "s", "whats", "can", "you", "me", "my", "it"}


def embed(text: str) -> np.ndarray:
    """Real embeddings when a key is available, else a toy lexical stand-in."""
    if USE_OPENAI:
        from openai import OpenAI

        resp = OpenAI().embeddings.create(model="text-embedding-3-small", input=text)
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    else:
        # TOY FALLBACK — hashes words and character trigrams into a fixed vector.
        # HONEST CAVEAT: this scores *word overlap*, not meaning. "restart the api"
        # and "reboot the service" look unrelated to it. The demo questions below
        # are worded so overlap is high enough to show the mechanism working.
        v = np.zeros(DIM, dtype=np.float32)
        words = [w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
                 if w not in _STOPWORDS]
        joined = " ".join(words)
        for feat in words + [f"#{joined[i:i + 3]}" for i in range(len(joined) - 2)]:
            h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            v[int.from_bytes(h, "big") % DIM] += 1.0
    return v / (np.linalg.norm(v) + 1e-9)          # normalise -> cosine is a dot


# ════════════════════════════════════════════════════════════════════════════
# 2) THE CACHE — write on the way out, 1-NN lookup on the way in
# ════════════════════════════════════════════════════════════════════════════
def cache_write(question: str, answer: str) -> None:
    vec = embed(question).astype(np.float32)
    if R is None:
        _MEM.append((question, answer, vec, time.time() + TTL_SECONDS))
        return
    key = PREFIX + hashlib.sha256(f"{NS}|{question}".encode()).hexdigest()[:24]
    R.hset(key, mapping={"question": question, "answer": answer,
                         "ns": NS, "embedding": vec.tobytes()})
    R.expire(key, TTL_SECONDS)                     # TTL evicts; the index follows


def cache_lookup(question: str) -> tuple[str | None, float, str]:
    """Return (answer_or_None, similarity_of_best_match, matched_question)."""
    vec = embed(question).astype(np.float32)

    if R is None:                                  # brute-force cosine over a list
        live = [row for row in _MEM if row[3] > time.time()]
        _MEM[:] = live
        best = max(((float(np.dot(vec, r[2])), r[0], r[1]) for r in live),
                   default=None, key=lambda t: t[0])
    else:
        from redis.commands.search.query import Query

        # `(@ns:{...})=>[KNN 1 @embedding $vec AS score]` — filter by namespace,
        # then nearest neighbour. dialect 2 is what parses the vector syntax.
        q = (Query(f"(@ns:{{{NS}}})=>[KNN 1 @embedding $vec AS score]")
             .sort_by("score").return_fields("question", "answer", "score").dialect(2))
        docs = R.ft(INDEX).search(q, query_params={"vec": vec.tobytes()}).docs
        # COSINE gives DISTANCE (0.0 == identical), so similarity is 1 - distance.
        best = (1.0 - float(docs[0].score), _s(docs[0].question), _s(docs[0].answer)) \
            if docs else None

    if best is None:
        return None, 0.0, ""
    sim, matched_q, answer = best
    # THE decision. Below the threshold we return nothing and pay for the LLM —
    # which is the cheap kind of wrong. Above it we'd serve `answer` to whatever
    # was asked, so a threshold that is too low invents wrong answers.
    return (answer if sim >= THRESHOLD else None), sim, matched_q


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else x


# ════════════════════════════════════════════════════════════════════════════
# 3) THE GRAPH — a conditional edge that skips the expensive node
# ════════════════════════════════════════════════════════════════════════════
class State(TypedDict, total=False):
    question: str
    answer: str
    cache: str            # "hit" | "miss"
    similarity: float
    matched: str


_CANNED = {
    "restart":  "Run `kubectl rollout restart deploy/api -n prod`, then watch readiness.",
    "slo":      "The API SLO is 99.9% availability over a 30-day rolling window.",
    "rollback": "Use `kubectl rollout undo deploy/api -n prod` to revert.",
}


def call_llm(question: str) -> str:
    """The expensive node's body — a real call when possible, canned otherwise."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            resp = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": "Terse SRE assistant. One sentence."},
                          {"role": "user", "content": question}])
            return resp.choices[0].message.content.strip()
        except Exception as e:                     # rate limit, bad key, no network
            print(f"    [llm] {type(e).__name__} -> canned answer")

    time.sleep(0.4)                                # stand in for model latency
    normalised = question.lower().replace("roll back", "rollback")
    return next((a for kw, a in _CANNED.items() if kw in normalised),
                "No runbook entry matched; escalate to the on-call lead.")


def build_app():
    def lookup_node(state: State):
        answer, sim, matched = cache_lookup(state["question"])
        hit = answer is not None
        return {"cache": "hit" if hit else "miss", "similarity": sim,
                "matched": matched, **({"answer": answer} if hit else {})}

    def llm_node(state: State):
        return {"answer": call_llm(state["question"])}

    def write_node(state: State):
        cache_write(state["question"], state["answer"])
        return {}

    g = StateGraph(State)
    g.add_node("cache_lookup", lookup_node)
    g.add_node("llm", llm_node)
    g.add_node("cache_write", write_node)

    g.add_edge(START, "cache_lookup")
    g.add_conditional_edges("cache_lookup", lambda s: s["cache"],
                            {"hit": END, "miss": "llm"})   # a hit never enters `llm`
    g.add_edge("llm", "cache_write")
    g.add_edge("cache_write", END)
    return g.compile()


# ════════════════════════════════════════════════════════════════════════════
# 4) SETUP + DRIVE IT
# ════════════════════════════════════════════════════════════════════════════
def setup() -> None:
    global USE_OPENAI, DIM, THRESHOLD, NS, R

    if os.getenv("OPENAI_API_KEY"):
        try:
            USE_OPENAI, DIM, THRESHOLD = True, 1536, 0.92
            embed("warmup")                        # fail fast on a bad/limited key
        except Exception as e:
            print(f"[embed] OpenAI embeddings unavailable ({type(e).__name__}) -> toy embedder")
            USE_OPENAI, DIM, THRESHOLD = False, 512, 0.80

    # THRESHOLD belongs to the EMBEDDER, not to the cache: real embeddings put
    # paraphrases at ~0.95, the toy one only reaches ~0.86. Re-calibrate on every
    # embedder swap or your hit rate silently collapses to zero.
    print(f"[embed] {'openai:text-embedding-3-small' if USE_OPENAI else 'toy-lexical'} "
          f"dim={DIM} threshold={THRESHOLD}")

    # Everything that shapes the answer goes in the namespace, so bumping any of
    # it makes old entries unreachable — that IS the invalidation strategy.
    # Sanitised to [A-Za-z0-9_] because RediSearch TAG fields split on `-` `.` `:`.
    NS = re.sub(r"[^A-Za-z0-9_]", "_",
                f"{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}_v1_"
                f"{'openai' if USE_OPENAI else 'lexical'}")

    try:
        import redis
        from redis.commands.search.field import TagField, TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        R = redis.Redis.from_url(REDIS_URL)
        R.ping()
        if not any(m[b"name"] == b"search" for m in R.execute_command("MODULE", "LIST")):
            raise RuntimeError("RediSearch missing — use redis/redis-stack-server")

        for key in R.scan_iter(match=f"{PREFIX}*"):     # deterministic demo run
            R.delete(key)
        try:
            R.ft(INDEX).dropindex()
        except Exception:
            pass
        R.ft(INDEX).create_index(
            (TextField("question"), TextField("answer"), TagField("ns"),
             # FLAT = exact brute force; fine below ~10k entries. Swap to "HNSW"
             # (approximate, so it can miss a true neighbour) when you outgrow it.
             VectorField("embedding", "FLAT",
                         {"TYPE": "FLOAT32", "DIM": DIM, "DISTANCE_METRIC": "COSINE"})),
            definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH))
        print(f"[store] Redis vector index ready at {REDIS_URL}\n")
    except Exception as e:
        print(f"[store] Redis unavailable ({type(e).__name__}: {e}) -> in-memory list\n")
        R = None


def ask(app, question: str) -> None:
    t0 = time.perf_counter()
    out = app.invoke({"question": question})
    ms = (time.perf_counter() - t0) * 1000
    hit = out["cache"] == "hit"
    print(f"  [{'HIT ' if hit else 'MISS'}] {question!r}")
    print(f"         sim={out['similarity']:.3f}  {ms:7.1f}ms"
          + (f"  matched={out['matched']!r}" if hit else ""))
    print(f"         -> {out['answer'][:70]}")


def main() -> None:
    setup()
    app = build_app()

    print("=" * 72)
    print("A — cold cache: every question misses and pays for the LLM")
    print("=" * 72)
    for q in ["How do I restart the api service in prod?",
              "What is the api availability SLO?",
              "How do I roll back the api deployment?"]:
        ask(app, q)

    print("\n" + "=" * 72)
    print("B — paraphrases: different strings, same intent -> served from cache")
    print("=" * 72)
    for q in ["How do I restart the api service in production?",
              "What is the availability SLO for the api?",
              "How do I roll back an api deployment?"]:
        ask(app, q)

    print("\n" + "=" * 72)
    print("C — the threshold earns its keep: a DIFFERENT question must MISS")
    print("=" * 72)
    ask(app, "Who is the on-call engineer for the billing service tonight?")
    print("\n  Answering that from the 'restart api' entry would be worse than having")
    print("  no cache at all. A semantic cache that never says no is a bug generator.")

    print("\nDone — the conditional edge after cache_lookup is what skipped the LLM.")


if __name__ == "__main__":
    main()
