"""
Vector storage + nearest-neighbour search.
==========================================

Owns: talking to Redis (or a list, when Redis is absent). Knows nothing about
      thresholds, questions, or graphs — hand it vectors, get back neighbours.
Depends on: config.

Swapping this module for pgvector/Qdrant/Milvus should require touching
nothing else — that is the test of whether the seam is in the right place.
"""

import time

import numpy as np

import config

_redis = None                                        # None -> in-memory fallback
_mem: list[tuple[str, str, str, np.ndarray, float]] = []


def connect(dim: int) -> str:
    """Open Redis and ensure the vector index exists. Returns 'redis' or 'memory'."""
    global _redis
    try:
        import redis
        from redis.commands.search.field import TagField, TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        client = redis.Redis.from_url(config.REDIS_URL)
        client.ping()

        # Vector search lives in the RediSearch module. Plain `redis:7` connects
        # happily and then fails on create_index — check explicitly so the error
        # says what is actually wrong.
        if not any(m[b"name"] == b"search"
                   for m in client.execute_command("MODULE", "LIST")):
            raise RuntimeError("RediSearch missing — use redis/redis-stack-server")

        try:
            client.ft(config.INDEX).info()           # already exists? reuse it
        except Exception:
            client.ft(config.INDEX).create_index(
                (
                    TextField("question"),
                    TextField("answer"),
                    TagField("ns"),
                    # FLAT = exact brute force, fine below ~10k entries (and a
                    # short TTL often keeps you there forever). Switch to "HNSW"
                    # when you outgrow it: it is approximate, so it can miss a
                    # true neighbour — a false MISS, which is the safe direction.
                    VectorField("embedding", "FLAT",
                                {"TYPE": "FLOAT32", "DIM": dim,
                                 "DISTANCE_METRIC": "COSINE"}),
                ),
                definition=IndexDefinition(prefix=[config.PREFIX],
                                           index_type=IndexType.HASH),
            )
        _redis = client
        return "redis"
    except Exception as e:
        print(f"[store] Redis unavailable ({type(e).__name__}: {e}) -> in-memory list")
        _redis = None
        return "memory"


def write(key: str, question: str, answer: str, ns: str, vec: np.ndarray) -> None:
    """Store one entry with a TTL. Redis expiry also drops it from the index."""
    if _redis is None:
        _mem.append((question, answer, ns, vec, time.time() + config.TTL_SECONDS))
        return
    rkey = config.PREFIX + key
    _redis.hset(rkey, mapping={"question": question, "answer": answer,
                               "ns": ns, "embedding": vec.tobytes()})
    _redis.expire(rkey, config.TTL_SECONDS)


def nearest(ns: str, vec: np.ndarray) -> tuple[float, str, str] | None:
    """1-NN within a namespace -> (similarity, question, answer), or None."""
    if _redis is None:
        live = [row for row in _mem if row[4] > time.time()]   # manual TTL sweep
        _mem[:] = live
        scored = [(float(np.dot(vec, r[3])), r[0], r[1]) for r in live if r[2] == ns]
        return max(scored, key=lambda t: t[0]) if scored else None

    from redis.commands.search.query import Query

    # `(@ns:{...})=>[KNN 1 @embedding $vec AS score]` — pre-filter by namespace,
    # then nearest neighbour. dialect 2 is what parses the `=>[KNN ...]` syntax:
    # the SERVER default is still 1, though recent redis-py sends 2 for you. Pin
    # it here rather than inherit a client default you do not control.
    q = (Query(f"(@ns:{{{ns}}})=>[KNN 1 @embedding $vec AS score]")
         .sort_by("score")
         .return_fields("question", "answer", "score")
         .dialect(2))
    docs = _redis.ft(config.INDEX).search(q,
                                          query_params={"vec": vec.tobytes()}).docs
    if not docs:
        return None
    # COSINE returns a DISTANCE: 0.0 means identical. Similarity is 1 - distance.
    # Invert this by accident and your threshold test silently reverses.
    return (1.0 - float(docs[0].score), _text(docs[0].question), _text(docs[0].answer))


def reset() -> None:
    """Drop every entry, so the demo starts from a known-empty cache."""
    if _redis is None:
        _mem.clear()
        return
    for key in _redis.scan_iter(match=f"{config.PREFIX}*"):
        _redis.delete(key)


def size() -> int:
    if _redis is None:
        return len(_mem)
    return int(_redis.ft(config.INDEX).info()["num_docs"])


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value
