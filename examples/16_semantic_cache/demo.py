"""
16 · LangGraph + Redis semantic caching — run this file.
========================================================

Owns: the three scenarios that show the cache working, and failing safely.
Depends on: everything else (this is the top of the dependency chain).

    python examples/16_semantic_cache/demo.py

Optional Redis (vector search needs the RediSearch module, so plain `redis:7`
will NOT work):

    docker run -d --name aai-redis -p 6379:6379 redis/redis-stack-server:latest

Runs fine without it — and without OPENAI_API_KEY — by falling back to an
in-memory store and a toy embedder. See embeddings.py for that caveat.
"""

import time

import cache
import embeddings
import graph
import store


def ask(app, question: str) -> None:
    started = time.perf_counter()
    out = app.invoke({"question": question})
    elapsed_ms = (time.perf_counter() - started) * 1000

    hit = out["cache"] == "hit"
    print(f"  [{'HIT ' if hit else 'MISS'}] {question!r}")
    print(f"         sim={out['similarity']:.3f}  {elapsed_ms:7.1f}ms"
          + (f"  matched={out['matched']!r}" if hit else ""))
    print(f"         -> {out['answer'][:70]}")


def main() -> None:
    backend = store.connect(embeddings.dim())
    print(f"[embed] {embeddings.name()}  dim={embeddings.dim()}  "
          f"threshold={embeddings.threshold()}")
    print(f"[store] {backend}  ns={cache.namespace()}\n")

    store.reset()                       # deterministic run
    app = graph.build()

    print("=" * 72)
    print("A — cold cache: every question misses and pays for the LLM")
    print("=" * 72)
    for q in ["How do I restart the api service in prod?",
              "What is the api availability SLO?",
              "How do I roll back the api deployment?"]:
        ask(app, q)
    print(f"\n  cache now holds {store.size()} entries")

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
