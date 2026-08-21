"""
The cache policy.
=================

Owns: the ONE decision that makes this a semantic cache — is the nearest
      neighbour close enough to count as the same question?
Depends on: embeddings, store, config.

This is the module you would lift into a real service (tutorial 16 §3 and its
capstone exercise): it has no LangGraph import and no demo code.
"""

import hashlib
from typing import NamedTuple

import config
import embeddings
import store


class Result(NamedTuple):
    answer: str | None      # None == miss; the caller must do the real work
    similarity: float       # best neighbour's score, hit or miss
    matched: str            # which cached question it matched (for auditing)


def namespace() -> str:
    return config.namespace(embeddings.name())


def lookup(question: str) -> Result:
    """Nearest neighbour, accepted only if it clears the threshold."""
    best = store.nearest(namespace(), embeddings.embed(question))
    if best is None:
        return Result(None, 0.0, "")

    similarity, matched_question, answer = best

    # THE decision. Below the threshold we return a miss and pay for the real
    # call — the cheap kind of wrong. Above it we hand back `answer` for whatever
    # was asked, so a threshold set too low does not slow things down, it
    # INVENTS WRONG ANSWERS. Always report the rejected score: logging near-miss
    # similarities gives you a threshold-tuning dataset for free.
    if similarity >= embeddings.threshold():
        return Result(answer, similarity, matched_question)
    return Result(None, similarity, matched_question)


def write(question: str, answer: str) -> None:
    """Store an answer.

    Only ever call this on a clean success path. Caching an error, a timeout, a
    guardrail refusal, or a partial result turns a transient blip into a
    persistent wrong answer for the whole TTL — and semantic matching then
    spreads it to every paraphrase. Validate BEFORE writing, because a cache hit
    bypasses the output rails that would otherwise have caught it.
    """
    ns = namespace()
    key = hashlib.sha256(f"{ns}|{question}".encode()).hexdigest()[:24]
    store.write(key, question, answer, ns, embeddings.embed(question))
