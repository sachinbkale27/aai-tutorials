"""
Text -> unit vector.
====================

Owns: the embedding backend AND its similarity threshold — those two belong
      together, which is the whole point of this module.
Depends on: nothing but numpy (+ openai when a key is present).

WHY THRESHOLD LIVES HERE: it is a property of the embedding model, not of the
cache. The same paraphrase pair scores ~0.95 with real embeddings and ~0.86
with the toy one below. A single hardcoded 0.92 would give a 100% hit rate
with one backend and 0% with the other — same code, same questions. So the
backend carries its own calibrated cutoff and they can never drift apart.
"""

import hashlib
import os
import re

import numpy as np

_TOY_DIM = 512
_STOPWORDS = {"the", "a", "an", "is", "do", "i", "to", "of", "for", "in", "on",
              "what", "how", "s", "whats", "can", "you", "me", "my", "it"}

# Resolved once, on first use, by _backend().
_RESOLVED: str | None = None


def _backend() -> str:
    """Pick 'openai' or 'toy' exactly once, verifying the key actually works."""
    global _RESOLVED
    if _RESOLVED is None:
        _RESOLVED = "toy"
        if os.getenv("OPENAI_API_KEY"):
            try:
                _openai_embed("warmup")          # fail fast on a bad/limited key
                _RESOLVED = "openai"
            except Exception as e:
                print(f"[embed] OpenAI embeddings unavailable "
                      f"({type(e).__name__}) -> toy embedder")
    return _RESOLVED


def name() -> str:
    return "openai" if _backend() == "openai" else "lexical"


def dim() -> int:
    """Vector width. The Redis index is created with this, so it must be stable."""
    return 1536 if _backend() == "openai" else _TOY_DIM


def threshold() -> float:
    """Cosine similarity at which two questions count as the same question.

    Calibrated per backend. To re-calibrate for your own traffic: label ~50 real
    query pairs same-intent/different-intent and pick the cutoff that maximises
    hit rate SUBJECT TO zero false hits. The costs are asymmetric — a false miss
    costs one API call, a false hit serves a wrong answer to a user.
    """
    return 0.92 if _backend() == "openai" else 0.80


def embed(text: str) -> np.ndarray:
    """text -> L2-normalised float32 vector."""
    v = _openai_embed(text) if _backend() == "openai" else _toy_embed(text)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def _openai_embed(text: str) -> np.ndarray:
    from openai import OpenAI

    resp = OpenAI().embeddings.create(model="text-embedding-3-small", input=text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def _toy_embed(text: str) -> np.ndarray:
    """Offline stand-in: hash words + character trigrams into a fixed vector.

    HONEST CAVEAT: this measures WORD OVERLAP, not meaning. "restart the api"
    and "reboot the service" look unrelated to it, though a real embedder scores
    them high. The demo questions are worded so overlap is enough to show the
    mechanism. Set OPENAI_API_KEY to see genuine semantic matching.
    """
    v = np.zeros(_TOY_DIM, dtype=np.float32)
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
             if w not in _STOPWORDS]
    joined = " ".join(words)
    trigrams = [f"#{joined[i:i + 3]}" for i in range(len(joined) - 2)]
    for feature in words + trigrams:
        h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        v[int.from_bytes(h, "big") % _TOY_DIM] += 1.0
    return v
