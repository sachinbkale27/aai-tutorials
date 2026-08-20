"""09 · Resilience — retries + circuit breaker, end to end.

Demonstrates how to wrap a flaky "external" call so that:
  1. transient failures RETRY with exponential backoff (via tenacity), then
  2. after N consecutive failures a hand-rolled circuit breaker TRIPS (opens), and
  3. subsequent calls are SKIPPED instantly and the caller DEGRADES (returns None)
     instead of burning retries against a dependency that's already down.

The breaker models the classic three states: closed -> open -> half-open.

Deps:   pip install tenacity
Run:    python examples/09_resilience.py
No API key needed — failures are injected locally.
"""

import time

from tenacity import Retrying, stop_after_attempt, wait_exponential

# ── 1. a fault-injected "external" call that RAISES on failure ────────────────
FAIL = True  # flip to False and every call succeeds on the first try


def call_raw():
    """The bare call. RAISES so the layer above can decide to retry/trip/degrade."""
    if FAIL:
        raise RuntimeError("service unavailable")
    return {"ok": True}


# ── 2. a tiny circuit breaker: closed → open → half-open ──────────────────────
class Breaker:
    """Opens after `fail_max` consecutive failures; after `reset_after_s` a single
    half-open trial call is allowed, whose result decides re-close vs re-open."""

    def __init__(self, fail_max, reset_after_s):
        self.fail_max, self.reset_after_s = fail_max, reset_after_s
        self.fails, self.opened_at = 0, None

    def state(self):
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= self.reset_after_s:
            return "half-open"
        return "open"

    def is_open(self):
        if self.opened_at is None:
            return False  # CLOSED — normal traffic
        if time.monotonic() - self.opened_at >= self.reset_after_s:
            self.fails, self.opened_at = 0, None  # cooled down → HALF-OPEN trial
            return False
        return True  # OPEN → fail fast, skip the call

    def on_success(self):
        self.fails, self.opened_at = 0, None  # recovered → fully closed

    def on_failure(self):
        self.fails += 1
        if self.fails >= self.fail_max:
            self.opened_at = time.monotonic()  # trip → OPEN


# ── 3. guard: breaker check → retries → degrade ───────────────────────────────
def guarded(breaker, retries=2, backoff_ms=100):
    """Route one call through the breaker + retry policy. Returns result or None."""
    if breaker.is_open():
        print("  circuit OPEN — skipping call, degrading (returns None)")
        return None

    attempts = retries + 1  # first try + `retries` extra attempts
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=backoff_ms / 1000),
            reraise=True,
        ):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n == 1:
                    print(f"  attempt #{n} …")
                else:
                    print(f"  retry #{n} …")
                result = call_raw()
                breaker.on_success()
                return result
    except Exception as e:
        breaker.on_failure()  # all attempts failed → maybe trip the breaker
        print(f"  failed after {attempts} attempt(s): {e}")
        if breaker.is_open():
            print(f"  >> circuit TRIPPED after {breaker.fail_max} consecutive failures")
        return None


# ── 4. drive it: watch retry → trip → skip ────────────────────────────────────
if __name__ == "__main__":
    # short cooldown so the breaker stays open for the rest of this quick demo
    br = Breaker(fail_max=2, reset_after_s=30)
    for i in range(1, 5):
        print(f"call {i}: [breaker={br.state()}]")
        guarded(br)
    print("\nDone: calls 1-2 burned their retries, the breaker tripped,")
    print("and calls 3-4 were skipped instantly (fail fast) → degraded to None.")
