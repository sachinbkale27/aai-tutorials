# 16 · LangGraph + Redis (semantic caching)

> Put a vector-similarity cache in front of the expensive node of your graph, so a *paraphrase* of a question you already answered is served from Redis in single-digit milliseconds instead of costing another model call.

---

## 1. Mental model — cache on *meaning*, not on *bytes*

Every cache answers one question: **"have I done this before?"** For HTTP or a database, "before" means *byte-identical key* — `GET /users/42` is either the same request or it isn't. That definition collapses the moment your key is natural language:

```
"How do I restart the api service in prod?"
"How do I restart the api service in production?"
```

Two different strings. One intent. One answer. An exact-match cache sees a 100% miss rate and you pay full price — latency *and* tokens — for a question you have already answered.

A **semantic cache** changes the key from the string to its **embedding**, and changes the match from equality to **nearest-neighbour above a similarity threshold**:

| | Exact cache | Semantic cache |
|---|---|---|
| Key | `sha256(question)` | embedding vector of the question |
| Match | equality | cosine similarity ≥ threshold |
| Lookup | `O(1)` hash get | ANN search (Redis KNN) |
| Paraphrase | miss | **hit** |
| Extra cost per lookup | none | one embedding call |
| New failure mode | none | **a wrong hit** — a confidently served answer to a *different* question |

That last row is the whole engineering problem. An exact cache can only ever be slow; a semantic cache can be *wrong*. Everything else in this tutorial — thresholds, namespaces, TTLs — exists to buy back the correctness you traded away for the hit rate.

**Why Redis.** You need three things in one place: a vector index with KNN search (RediSearch), TTL-based expiry so entries can't go stale forever, and sub-millisecond reads shared across every process. Redis gives you all three, and the cache is a plain hash you can `HGETALL` and inspect while debugging. (Redis is not the only option — pgvector, Qdrant, Milvus all work — but the latency budget of a *cache* strongly favours the thing already in your stack running in-memory.)

**Why this belongs in the graph, not inside a node.** The point of a cache is to **not run the expensive work**. In LangGraph that's a routing decision, so it belongs in the topology as a conditional edge:

```
START → cache_lookup ─┬─(hit)──────────────────────────→ END
                      └─(miss)→ llm → cache_write → END
```

The `llm` node isn't "called with a cache in front of it" — on a hit it is **never entered**. That's visible in the graph, in the `stream()` events, and in your traces, instead of being buried in an `if` inside a node body. Free consequences: the hit/miss decision is one auditable state field, and a cache hit shows up as a *shorter path through the graph* in your observability stack.

---

## 2. Smallest working example

```bash
pip install langgraph redis numpy
docker run -d --name aai-redis -p 6379:6379 redis/redis-stack-server:latest
```

> **RediSearch is required.** Vector search lives in the RediSearch module. Plain `redis:7` will connect fine and then fail on `create_index` — use `redis/redis-stack-server` (or Redis 8+, which bundles the query engine).

**The index** — one hash per entry, one vector field, cosine distance:

```python
import numpy as np, redis, time
from redis.commands.search.field import TextField, TagField, NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

r = redis.Redis.from_url("redis://localhost:6379")
DIM, IDX, PREFIX = 1536, "semcache:idx", "semcache:"

r.ft(IDX).create_index(
    (TextField("question"), TextField("answer"), TagField("ns"),
     NumericField("created_at"),
     VectorField("embedding", "HNSW",                    # "FLAT" = exact brute force
                 {"TYPE": "FLOAT32", "DIM": DIM, "DISTANCE_METRIC": "COSINE"})),
    definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
)
```

**Write** — a plain hash plus a TTL. RediSearch drops expired keys from the index for you:

```python
def cache_write(question, answer, ns, vec, ttl=3600):
    key = f"{PREFIX}{hashlib.sha256(f'{ns}|{question}'.encode()).hexdigest()[:24]}"
    r.hset(key, mapping={"question": question, "answer": answer, "ns": ns,
                         "created_at": int(time.time()),
                         "embedding": vec.astype(np.float32).tobytes()})
    r.expire(key, ttl)
```

**Lookup** — pre-filter by namespace, then 1-NN. Note the two things that bite everyone:

```python
def cache_lookup(question, ns, vec, threshold=0.92):
    q = (Query(f"(@ns:{{{ns}}})=>[KNN 1 @embedding $vec AS score]")
         .sort_by("score").return_fields("question", "answer", "score")
         .dialect(2))                                    # (1) dialect 2 is REQUIRED
    res = r.ft(IDX).search(q, query_params={"vec": vec.astype(np.float32).tobytes()})
    if not res.docs:
        return None
    d = res.docs[0]
    similarity = 1.0 - float(d.score)                    # (2) COSINE returns DISTANCE
    return d.answer if similarity >= threshold else None
```

1. **`dialect(2)`** — the `=>[KNN ...]` syntax requires query dialect 2, and the *server* default is still `1` (`FT.CONFIG GET DEFAULT_DIALECT`). Recent `redis-py` sets `Query._dialect = 2` for you, so this line is often redundant — keep it anyway, so your query doesn't depend on a client default or a server config you don't control.
2. **`DISTANCE_METRIC: COSINE` returns a distance, not a similarity.** `0.0` means identical. Similarity is `1 - distance`. Getting this backwards inverts your threshold and your cache will either never hit or always hit.

**The graph** — the conditional edge is the whole trick:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    question: str; answer: str; cache: str; similarity: float

def lookup_node(state):
    hit = cache_lookup(state["question"], NS, embed(state["question"]))
    return {"cache": "hit", "answer": hit} if hit else {"cache": "miss"}

def llm_node(state):
    return {"answer": call_llm(state["question"])}       # the expensive node

def write_node(state):
    cache_write(state["question"], state["answer"], NS, embed(state["question"]))
    return {}

g = StateGraph(State)
g.add_node("cache_lookup", lookup_node)
g.add_node("llm", llm_node)
g.add_node("cache_write", write_node)

g.add_edge(START, "cache_lookup")
g.add_conditional_edges("cache_lookup", lambda s: s["cache"],
                        {"hit": END, "miss": "llm"})      # a hit NEVER enters `llm`
g.add_edge("llm", "cache_write")
g.add_edge("cache_write", END)
app = g.compile()
```

**What it looks like running** — `examples/16_langgraph_redis_semantic_cache.py`, cold cache then paraphrases, against a real Redis Stack and a real `gpt-4o-mini`:

```
PART A — cold cache: every question is a miss and pays for the LLM
  [MISS] 'How do I restart the api service in prod?'      sim=0.000  wall=1650.6ms
  [MISS] 'What is the api availability SLO?'               sim=0.128  wall=1946.9ms

PART B — paraphrases: different strings, same intent -> served from Redis
  [HIT ] 'How do I restart the api service in production?' sim=0.867  wall=   3.2ms
         matched='How do I restart the api service in prod?'
  [HIT ] 'What is the availability SLO for the api?'       sim=0.857  wall=   2.3ms

PART C — the threshold earns its keep: a DIFFERENT question must miss
  [MISS] 'Who is the on-call engineer for the billing service tonight?'  sim=0.291
```

**1650ms → 3.2ms**, and a completion not billed. Part C is the important one: the cache *declined*. A semantic cache that never says no is a bug generator.

The example degrades gracefully — no Redis falls back to an in-memory numpy store, no `OPENAI_API_KEY` falls back to a toy lexical embedder — so you can read the flow offline. See the threshold caveat in Section 5.

---

## 3. How the On-Call Copilot would use it

**Honest status first: the Copilot does not have a semantic cache today.** There is no Redis dependency in `~/projects/nvidia-aai`; caching isn't in `app/resilience.py` and isn't in `config/`. Everything in this section is *proposed wiring* — described against the real files it would touch, so you can implement it rather than hand-wave it. Say it that way in an interview; claiming a cache that isn't there is the fastest way to get caught.

**Where it goes.** The synthesis step is the expensive one. `app/incident.py:112-121` builds a prompt from the worker findings and streams the answer through `guarded_reply` (`app/rails.py:110`):

```python
# app/incident.py:114-121 (today — always calls the model)
syn = AC.AGENTS.get("synthesis", {})
prompt = (syn.get("role", "...") + "\n\nFindings:\n"
          + "\n".join(f"- {k}: {v}" for k, v in findings.items()))
async for ev in guarded_reply([{"role": "system", "content": prompt},
                               {"role": "user", "content": alert}], ...):
    yield ev
```

Alert text is *exactly* the shape semantic caching is built for: "checkout-api 5xx spike, 12% error rate" and "checkout api throwing 5xx, errors at 12%" are the same incident phrased two ways, and during a flapping alert you get them dozens of times in an hour. The cache key should be the **alert + the findings digest**, not the alert alone — same alert with different metrics is a different situation and must not share an answer.

**Two cache layers, at different tiers:**

1. **Tool-result cache** in `guarded_tool` (`app/resilience.py:77`) — read-only diagnostics (`query_logs`, `fetch_metrics`, `search_runbooks`) are cacheable for 30–60s; that's a *huge* win when three workers all pull the same window. This one should be **exact-keyed** on `(tool, args)` — tool args are structured, so there is no paraphrase problem and no reason to accept a fuzzy match.
2. **Synthesis cache** — semantic, keyed on the alert + findings digest, short TTL (minutes).

The tier split in `config/resilience.yaml` already encodes exactly the right instinct — read-only diagnostics get retries, mutating actions get `retries: 0`:

```yaml
  # config/resilience.yaml:19-24 — the existing tiering
  search_runbooks: {retries: 1}
  restart_service: {retries: 0}   # mutating prod — do NOT blindly retry
```

**Never cache a mutating tool.** Apply the same rule: `restart_service`, `rollback_deploy`, `scale_service` are uncacheable at any TTL. A cached "rollback succeeded" is a fabricated production event.

**Config-driven, matching the existing pattern.** `app/resilience.py:21-30` loads YAML into `CFG` at import; a cache would follow suit with `config/cache.yaml`:

```yaml
# config/cache.yaml (proposed) — policy as data, mirroring resilience.yaml
semantic:
  enabled: true
  threshold: 0.94          # higher than the 0.92 default: incidents are high-stakes
  ttl_s: 300               # 5 min — an incident's context goes stale fast
  embed_model: text-embedding-3-small
tools:
  query_logs:      {ttl_s: 30}
  fetch_metrics:   {ttl_s: 30}
  search_runbooks: {ttl_s: 600}   # runbooks change on the order of days
  restart_service: {cacheable: false}   # mutating — never
  rollback_deploy: {cacheable: false}
  scale_service:   {cacheable: false}
```

**Observability.** `app/metrics.py` already has `TraceRecorder` (`:47`) and `snapshot()` (`:164`). A cache is invisible-and-therefore-untrustworthy without `cache.hit` / `cache.miss` / `similarity` / `tokens_saved` on the span, plus a hit-rate gauge in the snapshot. And the SSE contract (`app/sse.py`, tutorial 14) needs a `cache.hit` event — otherwise the glass-box UI shows an instant answer with no visible steps and looks broken. **A cached answer must be labelled as cached in the UI.**

---

## 4. Build it up — variations

### 4a. Namespaces: the invalidation strategy that actually works

The dangerous stale-cache case isn't an old answer — it's an answer generated by a **different model or a different prompt** being served as if it were current. Trying to *delete* affected entries is hopeless. Instead make the old entries **unreachable** by putting every input that shapes the answer into the namespace:

```python
ns = re.sub(r"[^A-Za-z0-9_]", "_", f"{model}_{prompt_version}_{embedder_name}")
```

Bump `PROMPT_VERSION`, and every pre-existing entry is instantly invisible; TTL cleans up the corpse. This also makes an embedder swap safe — vectors from two different models are not comparable, and a shared namespace would silently compare nonsense.

The `re.sub` isn't cosmetic: RediSearch `TAG` fields treat `-`, `.`, `:` and spaces as separators, so a raw namespace like `gpt-4o-mini` needs escaping (`gpt\-4o\-mini`) or your filter silently matches nothing. Sanitising to `[A-Za-z0-9_]` sidesteps the whole class of bug.

**Multi-tenancy is the same mechanism, and it's a security control:** put the tenant (and, for anything user-scoped, the user or role) in the namespace. A semantic cache that serves tenant A's answer to tenant B is a data-leak incident, not a cache bug.

### 4b. HNSW vs FLAT, and what "approximate" costs you

```python
VectorField("embedding", "FLAT", {...})    # exact: scans every vector
VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536,
                                  "DISTANCE_METRIC": "COSINE",
                                  "M": 16, "EF_CONSTRUCTION": 200,
                                  "EF_RUNTIME": 10})
```

`FLAT` is exact and perfectly fine below ~10k entries — a cache with a short TTL often never gets bigger. `HNSW` is approximate: it can **miss the true nearest neighbour**, which in cache terms is a false miss (you pay for a call you'd already answered). That's the safe direction to be wrong, and you buy it back with a higher `EF_RUNTIME` at query time. Start `FLAT`, move to `HNSW` when `num_docs` justifies it.

### 4c. LangGraph's built-in node cache — exact, and complementary

LangGraph ships node-level memoization. It is **not** semantic — it keys on a hash of the node input — but it's the right tool for deterministic expensive nodes:

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.redis import RedisCache          # shared across processes
from langgraph.types import CachePolicy

g.add_node("retrieve", retrieve,
           cache_policy=CachePolicy(key_func=lambda s: s["question"], ttl=60))
app = g.compile(cache=RedisCache(redis.Redis.from_url(REDIS_URL)))
```

Verified in the example (`langgraph 1.2.4`): the second identical invocation skips the node entirely (207ms → 1.2ms, node body executed once for two invocations). Change one word and it runs again — that's the line between the two mechanisms:

- **Built-in `CachePolicy`** → per-node memoization, exact key, near-zero effort. Use it on retrieval and tool nodes.
- **Your semantic cache node** → question-level reuse across paraphrases. Use it in front of the LLM.

They compose; the built-in one is not a substitute for the semantic one.

### 4d. Cache the embedding too

Easy thing to miss: on the read path you pay an **embedding call on every request, including hits**. That's real latency (10–50ms) and real money added to your supposedly-free hit. Put a small exact cache in front of the embedder itself — keyed on the normalised question — and hits become pure Redis:

```python
@functools.lru_cache(maxsize=4096)       # process-local; or a Redis STRING with a TTL
def embed_cached(normalised_question: str) -> bytes: ...
```

Also worth caching at other layers: retrieval results (tutorial 04), tool results (Section 3), and rendered runbook passages.

### 4e. Streaming and negative caching

**Streaming.** A cache hit returns a complete answer instantly, which breaks a token-by-token UI. Either store the chunk list and replay it with a small delay, or emit a `cache.hit` event and render the answer whole — deliberately, with a "cached" badge. Silence is the bad option.

**Never cache failures.** An error, a timeout, a guardrail refusal, or a degraded partial finding must **not** be written. Caching an error turns a transient blip into a persistent wrong answer for the whole TTL — and semantic matching then spreads it to every paraphrase. Write only on a clean success path (in the graph above: `cache_write` is only reachable from a successful `llm`).

---

## 5. Gotchas & pitfalls

**The threshold is a property of the embedder, not a constant.** This is the one that surprises people, and the example demonstrates it live. The same paraphrase pair scored **0.867** with the toy lexical embedder and would score ~0.95+ with real embeddings. A hardcoded `0.92` gives a 43% hit rate with one embedder and 0% with the other — same code, same questions. So:

```python
class OpenAIEmbedder:  threshold = 0.92   # paraphrases 0.93-0.99, distinct pairs <0.85
class LexicalEmbedder: threshold = 0.80   # flatter score distribution
```

**Calibrate, don't guess:** label ~50 real query pairs as same-intent/different-intent, plot the two score distributions, and pick the cutoff that maximises hit rate **subject to zero false hits**. Re-calibrate on every embedder change. Asymmetric costs: a false miss costs one API call, a false hit costs a wrong answer to a user.

**COSINE gives distance; similarity is `1 - distance`.** Silently inverted logic, and nothing errors. Measured on RediSearch 2.10: an identical vector scores `5.96e-08`, a near-duplicate `1.8e-04` — i.e. ~0, not ~1. Invert this and a "similarity ≥ 0.92" test rejects every true match while accepting the unrelated ones.

**Query dialect 2 — set it explicitly even though it often works without.** The `=>[KNN ...]` syntax needs dialect 2; the server default is `1`, but `redis-py` 8 already sends `DIALECT 2` on every `Query`. So omitting `.dialect(2)` works *on your machine* and can break against an older client or a differently-configured server. Pin it rather than inherit it.

**Plain Redis has no vector search.** `redis:7` connects, `create_index` fails. Check for the module explicitly rather than letting it surface as a confusing search error:

```python
if not any(m[b"name"] == b"search" for m in r.execute_command("MODULE", "LIST")):
    raise RuntimeError("RediSearch missing — use redis/redis-stack-server")
```

**Similarity cannot detect staleness.** "Who is on call tonight?" is 0.99-similar to itself asked yesterday, and yesterday's answer is *wrong*. No threshold saves you — this is what TTL is for, and why time-sensitive or personalised questions need a short TTL or a bypass. Classify the question before caching it.

**Dimension and dtype mismatches.** The index fixes `DIM`; writing a vector of a different length errors, and writing `float64` bytes into a `FLOAT32` field yields garbage distances rather than an error. Always `.astype(np.float32).tobytes()`.

**Unnormalised vectors.** Cosine is scale-invariant *in theory*; in practice normalise on write and on query (`v / np.linalg.norm(v)`) so your thresholds stay comparable and you can use inner-product metrics interchangeably.

**Cache poisoning.** One hallucinated or guardrail-bypassing answer, once written, is served to every paraphrase for the whole TTL — and it now bypasses the output rails that would have caught it. Run the answer through validation **before** `cache_write`, and give yourself a targeted-purge path (`FT.SEARCH` by namespace → delete) for when a bad answer lands.

**A hit skips your rails, your tools, and your traces.** That's the point, but it means everything you attached to the LLM path — output guardrails (tutorial 08), citations, spans — is skipped too. Either validate before writing (preferred) or re-run output rails on cached answers.

**Unbounded growth.** No TTL means the index grows forever, and RAM is the expensive kind. Always `expire()`. Watch `FT.INFO num_docs` and put a bound on it.

---

## ✅ Best Practices

- **Calibrate the threshold per embedder against labelled pairs.** Treat it as a tuned parameter with an owner and a test, not a magic number — optimise hit rate subject to *zero* false hits, and re-tune whenever the embedding model changes.
- **Put everything that shapes the answer in the namespace.** Model, prompt version, embedder, tenant, and user scope. Invalidation then becomes "bump a version" instead of "find and delete the affected entries," and cross-tenant leakage becomes structurally impossible.
- **Always set a TTL, and set it from how fast the truth changes.** Runbooks: minutes to hours. Live metrics: tens of seconds. Anything person- or time-specific: very short or not cached. Similarity cannot detect staleness.
- **Never cache errors, refusals, partial results, or mutating actions.** Only the clean success path writes. Make that a property of the graph — `cache_write` reachable only from a successful expensive node — rather than a rule people have to remember.
- **Validate before you write, not after you read.** A cached answer bypasses your output rails. Run guardrails and citation checks *before* `cache_write` so the cache can only ever hold already-validated content.
- **Make the cache a graph edge, not an `if` in a node.** A conditional edge after `cache_lookup` keeps the skip visible in the topology, in `stream()` events, and in traces — and keeps the expensive node genuinely un-entered.
- **Instrument hit rate, similarity distribution, and tokens/latency saved.** An uninstrumented cache is indistinguishable from a broken one. Alert on a hit-rate cliff (embedder or prompt drift) and log the similarity of every *rejected* near-miss — that's your threshold-tuning dataset, free.
- **Label cached answers in the UI.** Emit a `cache.hit` event so a glass-box UI can show a "cached" badge instead of an unexplained instant answer with no visible steps.
- **Start `FLAT`, graduate to `HNSW`.** Exact search is cheap below ~10k entries, and a short TTL often keeps you there permanently. Adopt ANN when `num_docs` justifies it, and remember its errors are false misses.
- **Cache the embedding too.** You pay an embedding call on every lookup, hits included; a small exact cache in front of the embedder makes hits pure Redis.

---

## 6. Exercises

1. **Run it both ways.** Run `examples/16_langgraph_redis_semantic_cache.py` with and without Redis (`REDIS_URL=redis://localhost:9999` forces the in-memory fallback). Confirm the hit/miss pattern is identical and explain why the store is swappable but the *threshold* is not.

2. **Break it with the threshold.** Set the lexical embedder's `threshold` to `0.92`, re-run, and watch Part B's hit rate go to zero. Then set it to `0.25` and find the question in Part C that gets a **wrong** answer. Write down the two costs you just traded.

3. **Prove the distance/similarity trap.** Store one entry, query with the identical text, and print the raw `d.score`. Confirm it's ≈0, not ≈1. Then "fix" the code to treat `score` as similarity and describe the resulting cache behaviour in one sentence.

4. **Add a namespace bump.** Populate the cache, then change `PROMPT_VERSION` from `v1` to `v2` and re-run without clearing Redis. Confirm every question misses, and that `FT.INFO num_docs` shows the old entries still present but unreachable.

5. **Instrument it.** Add `cache.hit` / `cache.miss` counters and a `similarity` histogram via tutorial 10's OpenTelemetry setup, then log the similarity of every rejected near-miss. Run 20 questions and use the collected scores to pick a better threshold than the default.

6. **Cache the embedder.** Add an `lru_cache` in front of `embed()` and measure the hit-path latency before and after. Report how much of your "free" cache hit was actually an embedding call.

7. **(Capstone) Wire it into the Copilot.** Implement `config/cache.yaml` and a `app/cache.py` following the `app/resilience.py` loader pattern (`:21-30`). Add an exact tool-result cache inside `guarded_tool` (`app/resilience.py:77`) honouring the per-tool TTLs and refusing to cache the three mutating tools, and a semantic cache around the synthesis call (`app/incident.py:114-121`) keyed on **alert + findings digest**. Emit a `cache.hit` SSE event (tutorial 14) so the UI badges cached answers, add hit rate to `metrics.snapshot()` (`app/metrics.py:164`), and verify with a replayed flapping alert that the second occurrence skips the model. Then write down what you'd need to observe in production before raising the TTL.

---

## 7. Connections

- **[05-langgraph.md](05-langgraph.md)** — the conditional edge that skips the LLM node is the routing primitive from that tutorial (`add_conditional_edges`); this is its highest-leverage use.
- **[09-resilience.md](09-resilience.md)** — a cache is the fourth defense alongside retries, breakers, and fallback: during an outage a warm cache *is* your graceful-degrade path. The `config/resilience.yaml` tool tiering is also the model for `config/cache.yaml`.
- **[04-agentic-rag.md](04-agentic-rag.md)** — the same embed-then-KNN mechanics, different purpose: RAG retrieves *context* to build an answer, a semantic cache retrieves the *answer itself*. Same index, one threshold apart.
- **[10-opentelemetry.md](10-opentelemetry.md)** / **[11-observability-stack.md](11-observability-stack.md)** — hit rate, similarity distribution, and tokens saved are the spans and metrics that make the cache trustworthy.
- **[08-nemo-guardrails.md](08-nemo-guardrails.md)** — a cache hit bypasses the output rails, so validate *before* writing; this is where cache poisoning is prevented.
- **[13-config-driven-design.md](13-config-driven-design.md)** — thresholds and TTLs are policy, so they belong in YAML where they can be tuned and reviewed, not in code.
- **[14-sse-streaming.md](14-sse-streaming.md)** — an instant cached answer breaks a token-streaming UI; the `cache.hit` event is the fix.

---

## 8. Further reading

- **Redis vector search** — <https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/> (index schema, `FLAT` vs `HNSW`, distance metrics).
- **Redis KNN query syntax** — <https://redis.io/docs/latest/develop/interact/search-and-query/query/vector-search/> (the `=>[KNN k @field $vec]` form and `DIALECT 2`).
- **RedisVL semantic cache** — <https://docs.redisvl.com/en/latest/user_guide/llmcache_03.html> (a batteries-included `SemanticCache` if you'd rather not hand-roll the index).
- **LangGraph node caching** — <https://langchain-ai.github.io/langgraph/how-tos/node-caching/> (`CachePolicy`, `compile(cache=...)`, the Redis/in-memory backends).
- **LangChain LLM caching** — <https://python.langchain.com/docs/how_to/llm_caching/> (`set_llm_cache`, including the Redis semantic variant).
- **HNSW** — Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW graphs* — <https://arxiv.org/abs/1603.09320> (why ANN errors are false misses).
