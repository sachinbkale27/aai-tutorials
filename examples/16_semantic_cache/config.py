"""
Tunables, in one place.
=======================

Owns: every knob you would otherwise hunt for across the other modules.
Depends on: nothing (this is the leaf of the dependency chain).

In a real service these would live in YAML and be hot-reloaded — see
tutorial 13 (config-driven design) and the proposed `config/cache.yaml`
in tutorial 16 §3.
"""

import os
import re

# ── Where the cache lives ───────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
INDEX = "aai:semcache:idx"      # RediSearch index name
PREFIX = "aai:semcache:"        # every cache entry is a hash under this prefix

# ── How long an answer stays valid ──────────────────────────────────────────
# Similarity cannot detect staleness: "who is on call tonight?" is 0.99-similar
# to itself asked yesterday, and yesterday's answer is wrong. TTL is the only
# defense. Set it from how fast the underlying truth changes.
TTL_SECONDS = 3600

# ── What the answer depends on ──────────────────────────────────────────────
# Anything that changes the answer belongs in the namespace, so bumping it makes
# every old entry UNREACHABLE. That is the invalidation strategy — you never
# hunt down and delete affected entries, you just move to a new namespace and
# let TTL clean up the corpses.
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_VERSION = "v1"           # bump when you edit the system prompt


def namespace(embedder_name: str) -> str:
    """model + prompt version + embedder -> a RediSearch-safe TAG value.

    Sanitised to [A-Za-z0-9_] on purpose: RediSearch TAG fields treat `-`, `.`,
    `:` and spaces as separators, so a raw `gpt-4o-mini` would need escaping
    (`gpt\\-4o\\-mini`) and silently match nothing if you forgot.

    Add the tenant and user scope here for multi-tenancy — serving tenant A's
    answer to tenant B is a data-leak incident, not a cache bug.
    """
    return re.sub(r"[^A-Za-z0-9_]", "_",
                  f"{CHAT_MODEL}_{PROMPT_VERSION}_{embedder_name}")
