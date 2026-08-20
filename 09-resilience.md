# 09 · Resilience (Retries, Circuit Breaker, Fallback)

> After this you can wrap any flaky external call so that transient failures retry with backoff, a persistently-broken dependency trips a circuit breaker and gets skipped, mutations are never blindly retried, and the whole thing *degrades* to a partial answer instead of crashing the incident — the exact pattern `app/resilience.py` uses.

## 1. Mental model — external calls fail; four defenses

Anything that leaves your process — an HTTP API, a database, an LLM endpoint, an MCP tool — can fail in ways your own code cannot. The three failure *shapes* you must design for:

- **Transient** — a dropped packet, a 503 during a deploy, a 1-in-1000 timeout. Retrying the *same* call a moment later usually works.
- **Persistent** — the dependency is genuinely down (bad deploy, dead region, rate-limit ban). Retrying just wastes time and *adds load to a service that's already on fire*.
- **Slow** — the call never errors, it just hangs. Unbounded, one slow dependency exhausts your worker threads and takes down everything (a **bulkhead**/**timeout** problem).

Four defenses, each answering a different question:

| Defense | Question it answers | Behavior |
|---|---|---|
| **Retry (with backoff)** | "Was this a blip?" | Call again, waiting longer each time. |
| **Circuit breaker** | "Is this dependency *down*?" | After N consecutive failures, stop calling it for a cooldown — fail fast. |
| **Fallback** | "Is there a plan B?" | Try a cheaper/secondary provider (e.g. a smaller model). |
| **Graceful degrade** | "Can I finish with less?" | Return a partial result and keep the larger task alive. |

They compose in exactly that order: *retry* handles the blip; if retries keep failing the *breaker* opens so future calls are skipped; a *fallback* may substitute another provider; and if nothing works you *degrade*. The On-Call Copilot's rule: **one broken tool must never crash the whole incident** — the agent keeps investigating with whatever findings it did get.

**When NOT to retry: mutations.** Retry is only safe for **idempotent** operations — reading logs, fetching metrics, a GET. If a call *changes prod state* — `restart_service`, `rollback_deploy`, `scale_service` — a retry might fire the action *twice*. Did the first `restart` fail, or did it succeed and only the *response* got lost? You cannot tell, so you must not retry. In `config/resilience.yaml` these tools get `retries: 0`.

## 2. Smallest working example

A standalone, runnable script: a flaky function wrapped with **tenacity** retries plus a tiny hand-rolled circuit breaker. It reproduces the retry → trip → skip sequence using an injected fault, mirroring how the real app works.

```bash
pip install tenacity
python flaky.py
```

```python
# flaky.py — retries (tenacity) + a minimal circuit breaker, end to end.
import time
from tenacity import Retrying, stop_after_attempt, wait_exponential

# ── 1. a fault-injected "external" call that RAISES on failure ────────────────
FAIL = True  # flip to inject a persistent outage

def call_raw():
    """The bare call. RAISES so the layer above can decide to retry/trip."""
    if FAIL:
        raise RuntimeError("service unavailable")
    return {"ok": True}

# ── 2. a tiny circuit breaker: closed → open → half-open ──────────────────────
class Breaker:
    def __init__(self, fail_max, reset_after_s):
        self.fail_max, self.reset_after_s = fail_max, reset_after_s
        self.fails, self.opened_at = 0, None

    def is_open(self):
        if self.opened_at is None:
            return False                                  # CLOSED
        if time.monotonic() - self.opened_at >= self.reset_after_s:
            self.fails, self.opened_at = 0, None          # cooled down → HALF-OPEN trial
            return False
        return True                                       # OPEN → fail fast

    def on_success(self): self.fails, self.opened_at = 0, None
    def on_failure(self):
        self.fails += 1
        if self.fails >= self.fail_max:
            self.opened_at = time.monotonic()             # trip → OPEN

# ── 3. guard: breaker check → retries → degrade ───────────────────────────────
def guarded(breaker, retries=2, backoff_ms=100):
    if breaker.is_open():
        print("  circuit OPEN — skipping call, degrading")
        return None
    attempts = retries + 1
    try:
        for attempt in Retrying(stop=stop_after_attempt(attempts),
                                wait=wait_exponential(multiplier=backoff_ms / 1000),
                                reraise=True):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n > 1:
                    print(f"  retry #{n} …")
                result = call_raw()
                breaker.on_success()
                return result
    except Exception as e:
        breaker.on_failure()
        print(f"  failed after {attempts} attempt(s): {e}")
        if breaker.is_open():
            print(f"  >> circuit TRIPPED after {breaker.fail_max} consecutive failures")
        return None

# ── 4. drive it: watch retry → trip → skip ────────────────────────────────────
br = Breaker(fail_max=2, reset_after_s=30)
for i in range(1, 5):
    print(f"call {i}:")
    guarded(br)
```

Expected output — the first two calls each burn their retries, the breaker trips, and every later call is skipped instantly (no retries, no waiting):

```
call 1:
  retry #2 …
  retry #3 …
  failed after 3 attempt(s): service unavailable
call 2:
  retry #2 …
  retry #3 …
  failed after 3 attempt(s): service unavailable
  >> circuit TRIPPED after 2 consecutive failures
call 3:
  circuit OPEN — skipping call, degrading
call 4:
  circuit OPEN — skipping call, degrading
```

The value of the breaker is stark here: without it, calls 3 and 4 would each wait through three failing attempts with exponential backoff. With it, they return in microseconds. Set `FAIL = False` and every call returns `{"ok": True}` on the first try; the breaker stays closed.

## 3. How the On-Call Copilot uses it

The real implementation is the same three pieces, made **config-driven** and **observable**. Three files:

### `config/resilience.yaml` — the policy, as data

```yaml
defaults:
  retries: 2              # extra attempts after the first (so 3 tries total)
  backoff_ms: 100         # exponential backoff base between retries
  breaker:
    fail_max: 2           # open the circuit after N consecutive failed calls
    reset_after_s: 30     # cooldown, then allow one trial call (half-open)

tools:
  query_logs:      {retries: 2}   # read-only diagnostics — safe to retry
  search_code:     {retries: 1}
  restart_service: {retries: 0}   # mutating prod action — do NOT retry
  rollback_deploy: {retries: 0}
  scale_service:   {retries: 0}

models:
  chain: [gpt-4o, gpt-4o-mini]    # LLM fallback order (applied in M1)
```

Policy is data, not code: to make `search_runbooks` more forgiving you edit YAML, no redeploy. Note the deliberate split — read-only tools get `retries: 1–2`; the three prod-mutating tools get `retries: 0` for exactly the idempotency reason from §1.

### `app/resilience.py` — `guarded_tool(name, args)` → `(result, events)`

Every tool call routes through one function. Its signature is the whole design:

```python
def guarded_tool(name, args):
    """Circuit-breaker + retries. Returns (result_or_None, events)."""
    pol, br, events = _policy(name), _breaker(name), []

    # 1) circuit already open → skip the call entirely and degrade
    if br.is_open():
        events.append(_ev("circuit-open", name, f"{name} circuit is open — skipping, degrading"))
        return None, events

    # 2) attempt the call, retrying per policy (tenacity)
    attempts = pol.get("retries", 0) + 1
    wait = wait_exponential(multiplier=pol.get("backoff_ms", 100) / 1000)
    try:
        for attempt in Retrying(stop=stop_after_attempt(attempts), wait=wait, reraise=True):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n > 1:                    # this is a retry → surface it as a UI step
                    events.append({"type": "step.start", "id": f"retry-{name}-{n}", ...})
                    events.append({"type": "step.end",   "id": f"retry-{name}-{n}", "ok": True})
                result = tools.call_raw(name, args)
                br.on_success()
                return result, events
    except Exception as e:                   # 3) all attempts failed → breaker may open
        br.on_failure()
        events.append(_ev("degraded", name, f"{name} failed after {attempts} attempt(s): {e}"))
        if br.is_open():
            events.append(_ev("circuit-tripped", name, ...))
        return None, events
```

Two design points worth internalizing for an interview:

- **`result` is `None` on any failure, never an exception.** The caller doesn't `try/except` — it checks `if result is None:` and degrades (uses a partial finding). Failure is a *value*, not control flow. This is what keeps one dead tool from unwinding the whole incident.
- **`events` makes resilience *observable*.** Every retry, trip, and degrade is appended as an SSE-ready dict, so the trace/UI can literally *show* "retry query_logs (#2)" and "circuit tripped." Resilience that you can't see is resilience you can't trust. Note `_ev` uses its own `"type": "resilience"` event so these don't skew guardrail metrics.

The breaker is **one instance per tool** (`_BREAKERS.setdefault(name, ...)` in `_breaker()`): `query_logs` tripping must not stop `fetch_metrics`. The `_Breaker` class is the same closed/open/half-open logic as §2 — `is_open()` returns `False` once `reset_after_s` has elapsed, resetting the counters so the next call is a half-open trial.

pybreaker isn't installed; the pattern is ~20 lines and clearer inline. That's a legitimate engineering call — know both sides (see the exercise to swap it in).

### `app/tools.py` — `call_raw` that RAISES, plus fault injection

```python
FAILING = set(filter(None, os.getenv("RESILIENCE_FAIL", "").split(",")))

def call_raw(name, args=None):
    """Invoke a tool; RAISES on failure so the resilience layer can retry/trip."""
    if name in FAILING:
        raise RuntimeError(f"{name} is unavailable")
    fn = getattr(_t, name, None)
    if not fn:
        raise ValueError(f"unknown tool: {name}")
    return fn(**(args or {}))
```

The contract: **`call_raw` raises, `guarded_tool` decides.** The bare layer's only job is to fail loudly so the layer above can retry/trip/degrade. `$RESILIENCE_FAIL` is a demo lever — list tool names and they raise on every call:

```bash
RESILIENCE_FAIL=query_logs python -m app.some_flow   # watch retries → trip → degrade live
```

## 4. Build it up

### 4a. Exponential backoff **+ jitter** (thundering herd)

Plain exponential backoff synchronizes clients: if 500 workers all fail at `t=0`, they all retry at `t=100ms`, then all at `t=200ms` — a self-inflicted stampede. Add randomness so retries *spread out*:

```python
from tenacity import wait_exponential, wait_random
# full jitter: base backoff PLUS a random 0–100ms smear
wait = wait_exponential(multiplier=0.1, max=2) + wait_random(0, 0.1)
```

`wait_exponential(multiplier=0.1)` gives 0.1s, 0.2s, 0.4s…; the `+ wait_random` desynchronizes the herd. AWS's canonical write-up calls the fully-randomized version "full jitter."

### 4b. The half-open trial, explained

When a breaker's cooldown elapses you don't slam the dependency with all queued traffic — you send **one** trial call:

- **Half-open + success** → dependency recovered → close the breaker, resume normal traffic (`on_success()` zeroes the counters).
- **Half-open + failure** → still down → re-open immediately for another `reset_after_s`.

In `_Breaker`, `is_open()` implements the transition implicitly: once cooled down it resets state and returns `False`, so the *next* call is the trial; that call's success/failure decides whether the breaker stays closed or trips again. (A production breaker like pybreaker models `HALF_OPEN` as an explicit state and can require *several* consecutive successes before fully closing.)

### 4c. Model fallback chain

For LLM calls, the plan-B isn't a retry — it's a *different model*. `config/resilience.yaml` declares the order and `resilience.py` exposes it:

```python
def model_chain():
    return CFG.get("models", {}).get("chain", [])   # -> ["gpt-4o", "gpt-4o-mini"]
```

The pattern you'd wire up in M1 (when real LLM tool-calling lands — it's **config-only today**):

```python
def call_with_fallback(messages):
    last = None
    for model in model_chain():          # try gpt-4o, then gpt-4o-mini
        try:
            return client.chat(model=model, messages=messages)
        except (RateLimitError, Timeout, APIError) as e:
            last = e                       # this model failed → try the next
    raise last                            # whole chain exhausted
```

Fallback trades quality for availability: if the flagship is rate-limited, a smaller model answering *something* beats the primary answering *nothing*. Order the chain best-first.

### 4d. Timeouts & bulkheads

Retries and breakers don't help against a call that *hangs* — you need a **timeout** so a slow call becomes a fast failure the breaker can count:

```python
result = tools.call_raw(name, args, timeout=2.0)   # httpx/requests: a call that hangs now RAISES
```

A **bulkhead** isolates resource pools so one saturated dependency can't starve the rest — e.g. cap concurrent calls per tool with a `Semaphore(4)`, so a slow `query_logs` can't consume every worker and block `fetch_metrics`. The per-tool breaker in §3 is a bulkhead in spirit: failure is contained to one dependency.

## 5. Gotchas & pitfalls

- **Never retry a non-idempotent op.** `restart_service`/`rollback_deploy`/`scale_service` are `retries: 0` for a reason: a retried mutation can fire twice. If you *must* retry a write, make it idempotent first (idempotency keys, conditional `If-Match`), then retry the key — not the raw action.
- **One breaker per dependency, not one global.** A shared breaker means a flaky log service also blocks your healthy metrics service. Key breakers by `(dependency, maybe operation)`. The app does `_BREAKERS.setdefault(name, ...)`.
- **Surface degradation — don't swallow it.** Returning `None` silently is how a "green" dashboard hides a half-broken system. Emit an event (the `events` list), tag the response as partial, and count degrade events as a metric/alert. A silent fallback to gpt-4o-mini can quietly wreck answer quality — you want to *know* it's happening.
- **Beware thundering herd** (§4a): always jitter backoff, and stagger breaker cooldowns so all breakers don't reopen at the same instant.
- **Bound everything.** `stop_after_attempt` caps retries; `reset_after_s` caps the breaker; a timeout caps latency. An unbounded retry loop against a down service is a DoS you wrote yourself.
- **Retry only *retryable* errors.** A 400/validation error or `ValueError: unknown tool` will fail identically every time — retrying wastes attempts. Filter by exception type (`retry=retry_if_exception_type(Timeout)`) so you don't burn retries on deterministic failures. (The demo retries broadly for simplicity.)
- **Backoff base matters.** `backoff_ms: 100` with 2 retries adds ~0.3s worst case — fine for a background worker, maybe too slow for a user-facing request. Tune per call site.

## ✅ Best Practices

- **Retry only idempotent operations.** Wrap reads (`query_logs`, `fetch_metrics`, GETs) with retries; give mutating actions `retries: 0` — or add an idempotency key before you retry a write.
- **Exponential backoff *with* jitter.** Combine `wait_exponential` with `wait_random` so 500 failing clients don't all retry on the same tick and stampede the recovering dependency.
- **Bound every dimension.** Cap attempts (`stop_after_attempt`), per-call latency (a timeout), and total wait — an unbounded retry loop against a dead service is a self-inflicted DoS.
- **One circuit breaker per dependency.** Key breakers by `(dependency, operation)` so a tripped log service never blocks healthy metrics; use half-open trial calls to auto-recover when the cooldown elapses.
- **Degrade to partial results.** Return a value (e.g. `None`) instead of raising, so one dead tool leaves the larger task alive with whatever findings it did get.
- **Make resilience observable.** Emit structured events for every retry, trip, and degrade, tag partial responses, and alert on degrade-rate so a silent fallback can't quietly rot answer quality.
- **Define explicit fallback chains.** Order model/provider fallbacks best-first (`[gpt-4o, gpt-4o-mini]`) so a rate-limited flagship yields to a smaller model answering *something*.
- **Config-drive the policy.** Keep retries, backoff, and breaker thresholds in data (`config/resilience.yaml`) so tuning a dependency is an edit, not a redeploy.

## 6. Exercises

1. **Add jitter.** Change §2's `wait_exponential` to `wait_exponential(...) + wait_random(0, 0.1)`. Run 20 workers (threads) against `FAIL=True` and log each retry's timestamp; confirm retries no longer cluster on the same tick.
2. **Swap in pybreaker.** `pip install pybreaker`, replace `_Breaker` with `pybreaker.CircuitBreaker(fail_max=2, reset_timeout=30)`, and adapt `guarded_tool`. Compare: what does pybreaker give you that the hand-rolled class doesn't (explicit `HALF_OPEN` state, listeners, exclude-lists)? What did you lose in simplicity?
3. **Add a fallback model.** Implement `call_with_fallback` from §4c against a mock client that raises `RateLimitError` for `gpt-4o` and succeeds for `gpt-4o-mini`. Emit a `resilience` event `"model-fallback"` when you drop down the chain, so it shows in the trace.
4. **Visualize breaker state.** Extend `_Breaker` with a `state()` method returning `"closed" | "open" | "half-open"`. Drive it through a failing→cooldown→recovering sequence and print an ASCII timeline (`CCC-OOO-H-C`). Where exactly does half-open appear?
5. **Make a retry-then-degrade test.** With `RESILIENCE_FAIL=query_logs`, assert `guarded_tool("query_logs", {})` returns `(None, events)` where `events` contains a `"degraded"` event after exactly 3 attempts (`retries: 2` + 1). Then call it twice more and assert the second call trips the breaker and the third is skipped with `"circuit-open"`.
6. **Add per-error retry policy.** Introduce a non-retryable exception (e.g. `ValueError`) in `call_raw` and use `retry=retry_if_exception_type(RuntimeError)` so validation errors fail immediately while transient errors still retry. Verify the `ValueError` path burns zero retries.

## 7. Connections

- **[02-tool-and-function-calling.md]** — `guarded_tool` is the wrapper *around* the tool calls you learned to dispatch there. Every tool the model invokes flows through this resilience layer; `retries: 0` on mutations is the safety half of tool-calling.
- **[03-model-context-protocol.md]** — the same tools are served over MCP (`mcp_server/server.py`). Resilience lives on the *client* side of that boundary: a remote MCP tool is just another external call that can time out, so `call_raw` raising and `guarded_tool` retrying applies identically.
- **[10-opentelemetry.md]** — the `events` returned by `guarded_tool` are the raw material for tracing. Retries, trips, and degrades become spans/attributes; §5's "surface degradation" is exactly what OTel operationalizes into metrics and alerts.

## 8. Further reading

- **tenacity docs** — `Retrying`, `stop_after_attempt`, `wait_exponential`, `wait_random`, `retry_if_exception_type`, `reraise`: https://tenacity.readthedocs.io
- **Circuit Breaker pattern** — Martin Fowler's original write-up: https://martinfowler.com/bliki/CircuitBreaker.html
- **pybreaker** — the standard Python circuit-breaker library (states, listeners, storage): https://github.com/danielfm/pybreaker
- **AWS — "Exponential Backoff and Jitter"** — why full jitter beats plain backoff: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- **Release It! (Nygard)** — the book that popularized circuit breakers, bulkheads, and timeouts as stability patterns.
