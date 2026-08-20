# 13 · Config-Driven Design

> Push *what the system does* into declarative YAML the code merely reads, so you can change behavior by editing a file and hot-reloading — no code change, no restart, no redeploy.

---

## 1. Mental model — behavior as data

Most of what a running agent "decides" is not really logic — it's **policy**: which model does the planner use, how many times do we retry a flaky tool, which OTLP endpoint do traces go to, what's the roster of workers. If you bake those into Python, every tuning change is a code change: edit, review, test, redeploy. That's slow, and worse, it hides intent — the *why* is buried in `if` branches across ten files.

**Config-driven design** flips it: the code contains only *mechanism* (the loop that retries, the graph that dispatches), and all the *policy* lives in declarative config that the code reads at runtime. Three payoffs:

1. **Config is documentation-of-intent.** `retries: 0` next to `restart_service` with a comment "do NOT blindly retry a mutation" tells the next engineer *why* in the place they'll look. The roster in `agents.yaml` is a readable spec of the whole agent system — you can hand it to a new hire.
2. **12-factor.** Factor III ("store config in the environment") and its spirit: config that varies between deploys (endpoints, keys, feature flags) is separated from code that doesn't. Same image, different behavior per environment.
3. **Fast, safe iteration.** Tuning a prompt or a backoff is a YAML edit + a reload call, reviewable as a small diff, revertable in one line. The blast radius is contained to data.

**When NOT to.** Config is not a programming language. If a "setting" needs loops, conditionals, or references to other settings to express, that's *logic* and it belongs in code. Red flags that you've over-abstracted into YAML: config keys that are really function names, deeply nested conditionals in data, a "config" that only one code path ever reads with one value. The test: *could a non-author reasonably change this value and predict the effect?* If yes, config. If changing it safely requires understanding the code, keep it in code and expose only the knob.

The other half of this component is the **loader/singleton pattern** and its cousin **real-or-fallback**. You load config *once* at import into a module-level singleton everyone reads, provide a `reload()` that re-reads it in place, and — for anything that might not be available (a real guardrails runtime, a live backend) — you try the real thing and fall back to a degraded stand-in instead of crashing.

---

## 2. Smallest working example

A tiny standalone program: a YAML config, a loader singleton with `reload()`, and a **defaults + overrides merge**. Save the two files, run, and watch behavior change by editing YAML.

`policy.yaml`:

```yaml
defaults:
  retries: 2
  timeout_s: 5
tools:
  send_email:  {retries: 0}      # override: never auto-retry a send
  fetch_stats: {timeout_s: 30}   # override: this one is slow
```

`loader.py`:

```python
import pathlib, yaml

HERE = pathlib.Path(__file__).parent

def _load():
    # empty dict if the file is missing/broken — degrade, don't crash
    try:
        return yaml.safe_load((HERE / "policy.yaml").read_text()) or {}
    except Exception as e:
        print(f"[policy] unavailable: {e}")
        return {}

CFG = _load()                      # singleton: loaded ONCE at import

def policy(tool):
    """Merge a tool's overrides ONTO the defaults."""
    p = dict(CFG.get("defaults", {}))          # copy defaults
    p.update(CFG.get("tools", {}).get(tool, {}))  # tool-specific wins
    return p

def reload():                      # re-read in place; callers see it via CFG
    global CFG
    CFG = _load()
    return {"tools": len(CFG.get("tools", {}))}

if __name__ == "__main__":
    print("send_email  ->", policy("send_email"))   # retries=0, timeout=5
    print("fetch_stats ->", policy("fetch_stats"))  # retries=2, timeout=30
    print("unknown     ->", policy("unknown"))      # pure defaults
```

Run it:

```
$ python loader.py
send_email  -> {'retries': 0, 'timeout_s': 5}
fetch_stats -> {'retries': 2, 'timeout_s': 30}
unknown     -> {'retries': 2, 'timeout_s': 5}
```

Now **change behavior without touching code**: edit `policy.yaml`, set `send_email: {retries: 1}`, and in a live process call `loader.reload()` — the next `policy("send_email")` returns `retries: 1`. That's the entire idea: `defaults` provide the baseline, `tools` provide per-item overrides, the merge combines them, and `reload()` swaps the singleton in place so long-lived readers pick up the change.

Two subtleties worth internalizing now, because the real codebase relies on both:

- **Copy before merge.** `dict(CFG.get("defaults", {}))` copies so `.update()` doesn't mutate the shared defaults dict. Forget the copy and the first override permanently corrupts the baseline for every later lookup.
- **Readers must reference the module, not import the value.** `from loader import CFG` captures the *old* dict; after `reload()` reassigns `CFG`, that importer still sees the stale one. Read `loader.CFG` (or go through an accessor like `policy()`) so the reassignment is visible.

---

## 3. How the On-Call Copilot uses it

The Copilot puts **four different behavior domains** in `config/*.yaml`, each with its own loader module, and exposes two of them as hot-reload HTTP routes. All paths below are under `/Users/sachinkale/projects/nvidia-aai`.

### The four config files (all behavior, no logic)

- `config/agents.yaml` — the **agent roster**: `defaults` (model/temperature/max_tokens), then `orchestrator`, `workers[]`, `synthesis`, `remediation`. Prompts (`role`), model choices, per-worker `tools`, few-shot exemplars, and RAG settings (`runbook_retriever.rag.collection: runbooks`, `top_k: 4`) all live here, not in code. The header comment says it outright: "prompts/models/tools/few-shot live here, NOT in code. Edit YAML → reload via POST /api/agents/reload."
- `config/tools.yaml` — the **tool manifest**: `servers` (which MCP server, its transport/command) and `tools[]` (name, description, `parameters` schema, and a `reversibility: 0..3` tag per tool). The OpenAI tool-calling schemas are *generated from this*, and the MCP server implements each `name`. That `reversibility` tag is the single source of truth: `execution_gate` **derives** the human-oversight mode from it (the Reversibility Ladder), with a fail-safe default of 3 for untagged tools.
- `config/resilience.yaml` — the **resilience policy**: `defaults` (retries, backoff, breaker `fail_max`/`reset_after_s`), per-tool `tools` overrides, and a `models.chain` fallback list. Note the intent-as-documentation: read-only diagnostics get `retries: 2`, but every mutating action gets `retries: 0` under the comment "do NOT blindly retry a mutation."
- `config/observability.yaml` — the **backend switchboard**: `service_name`, `otel_backend.endpoint`/`insecure`, `llm_tracing.provider`/`enabled`, `local_dev.phoenix`, `eval.ragas`. Because the app already exports OTLP, *which* backend to use is a config choice, not a code change.

### The four loader modules

Each file has a loader that follows the same shape — a `_load()` that returns `{}` on failure, a module-level singleton, and (for the mutable ones) a `reload()`:

- `app/agent_config.py` loads **two** singletons, `AGENTS` and `TOOLS`, at import (lines 21–22). It exposes `worker(name)` to look up one worker's block, and `reload()` (lines 30–34) which re-reads both files and returns counts. The docstring states the contract: "Editing the YAML and calling reload() updates them at runtime — no code change or restart needed."
- `app/config.py` owns the **runtime singletons** `DEMO`, `RAILS`, `RAILS_MODE`. This is the real-or-fallback module (see below). Its `reload()` (lines 44–50) re-reads the demo policy and re-instantiates the guardrails runtime, recomputing the mode.
- `app/obs_config.py` loads `config/observability.yaml` into `CFG`, then `apply()` (lines 25–34) **translates YAML into `OTEL_*` env vars** — the config-to-environment bridge — and `active()` (lines 37–45) reports which backends are on for the health dashboard.
- `app/resilience.py` loads `config/resilience.yaml` into `CFG` and does the **defaults+override merge** (below).

Wiring order matters in `app/config.py`: it calls `obs_config.apply()` (YAML → env) *before* `setup_observability()` reads that env (lines 17–21). Config translation has to happen before the thing that consumes it initializes.

### The defaults+override merge (resilience.py)

This is the exact same pattern as the smallest example, in production:

```python
# app/resilience.py, lines 33-37
def _policy(name):
    """Merge a tool's overrides onto the defaults (retries, backoff)."""
    p = dict(CFG.get("defaults", {}))
    p.update(CFG.get("tools", {}).get(name, {}))
    return p
```

So `guarded_tool("query_logs", …)` gets `retries: 2` (defaults + its own `{retries: 2}`) while `guarded_tool("restart_service", …)` gets `retries: 0` — the mutation override wins over the default `2`, and the breaker block (`fail_max`, `reset_after_s`) rides along from defaults. The `_breaker()` helper (lines 72–74) reads `CFG.get("defaults", {}).get("breaker", {})` the same way, with inline literal fallbacks (`fail_max=3`) so a missing config key still yields a working breaker.

### Real-or-fallback (config.py)

`app/config.py` builds the guardrails runtime as a singleton that is **either the real thing or a degraded stand-in**, and records which:

```python
# app/config.py, lines 38-41
DEMO = _load_demo()                 # keyword stand-ins + action-gating policy (fallback)
RAILS = _load_rails()               # real NeMo Guardrails runtime, or None
RAILS_MODE = "nemo-guardrails" if RAILS else "demo-fallback"
print(f"[guardrails] mode = {RAILS_MODE}")
```

`_load_rails()` (in `app/guardrails_runtime.py`) tries the official path — `RailsConfig.from_path(...)` + `LLMRails(...)` — and returns `None` if the package is missing, NIMs are unreachable, or the config is bad. So the whole app can boot and run in `demo-fallback` mode on a laptop with no GPU, then silently upgrade to `nemo-guardrails` mode once the real runtime is available — **no code path changes**, only which singleton is non-null. `RAILS_MODE` is surfaced on `/api/health` so you can *see* which mode you're in. This is the config-loader pattern generalized: the "config" being loaded is a whole runtime, and `{}`/`None` is the honest fallback.

### The hot-reload endpoints (main.py)

Two POST routes turn a YAML edit into live behavior with no restart:

```python
# app/main.py
@app.post("/api/agents/reload")          # lines 58-60
async def reload_agents():
    return {"reloaded": True, **AC.reload()}      # {"workers": N, "tools": M}

@app.post("/api/guardrails/reload")      # lines 128-130
async def reload_guardrails():
    return {"reloaded": True, "guardrails_mode": config.reload()}  # new mode string
```

Workflow: edit `config/agents.yaml` (say, tweak the orchestrator prompt or add a worker), `curl -X POST localhost:8010/api/agents/reload`, and the running graph now reads the new roster on its next incident — the response echoes the new worker/tool counts as confirmation. `resilience.py` and `obs_config.py` have loaders too but no reload route; their config is read at startup (resilience could add one trivially — see exercises).

---

## 4. Build it up

Four variations that take the bare loader toward production-grade. Each is small and composable with the pattern above.

### 4a. Schema validation with Pydantic

The raw loader trusts the YAML. One typo — `reties: 2` — silently becomes "use the default forever," and you debug it at 2am. Validate the parsed dict into typed models so mistakes fail loudly *at load time*:

```python
from pydantic import BaseModel, Field, ValidationError
import yaml, pathlib

class Breaker(BaseModel):
    fail_max: int = 3
    reset_after_s: int = 30

class Defaults(BaseModel):
    retries: int = Field(2, ge=0)          # ge=0 rejects negative retries
    backoff_ms: int = 100
    breaker: Breaker = Breaker()

class ToolPolicy(BaseModel):
    retries: int | None = None
    backoff_ms: int | None = None

class ResilienceCfg(BaseModel):
    defaults: Defaults = Defaults()
    tools: dict[str, ToolPolicy] = {}
    models: dict[str, list[str]] = {}

def load_validated(path):
    raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    try:
        return ResilienceCfg(**raw)        # typos / bad types raise here
    except ValidationError as e:
        raise SystemExit(f"bad resilience.yaml:\n{e}")
```

Now `reties: 2` raises "extra fields not permitted" (with `model_config = {'extra': 'forbid'}`) instead of being ignored. Field types and constraints (`ge=0`) turn config bugs into startup errors — the cheapest place to catch them. This is the single highest-value upgrade to the pattern.

### 4b. Environment interpolation (secrets stay in env)

Config files get committed; secrets must not. The common bridge is `${VAR}` placeholders that the loader expands from the environment:

```python
import os, re, yaml

_VAR = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")   # ${VAR} or ${VAR:default}

def _expand(x):
    if isinstance(x, str):
        return _VAR.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), x)
    if isinstance(x, dict):  return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):  return [_expand(v) for v in x]
    return x

def load_interpolated(path):
    return _expand(yaml.safe_load(open(path).read()) or {})
```

```yaml
llm_tracing:
  provider: langfuse
  host: ${LANGFUSE_HOST:http://localhost:3000}   # env override, sane default
  public_key: ${LANGFUSE_PUBLIC_KEY}             # secret — never hardcoded
```

The Copilot already does the *inverse* bridge in `app/obs_config.py`: instead of pulling env into YAML, `apply()` pushes YAML into env (`OTEL_EXPORTER_OTLP_ENDPOINT`, etc.), and critically only when the env isn't already set — `if endpoint and not os.getenv(...)`. That precedence rule (explicit env wins over YAML default) is the whole point of 12-factor config layering.

### 4c. Per-environment overrides

Same code, different behavior per deploy, via a base file plus an environment layer merged on top:

```python
def load_env(name, env=None):
    env = env or os.getenv("APP_ENV", "dev")
    base = yaml.safe_load(open(f"config/{name}.yaml").read()) or {}
    override_path = pathlib.Path(f"config/{name}.{env}.yaml")
    override = yaml.safe_load(override_path.read_text()) if override_path.exists() else {}
    return _deep_merge(base, override)

def _deep_merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out
```

`config/resilience.yaml` holds safe defaults; `config/resilience.prod.yaml` might set `defaults.retries: 3` and point the model chain at bigger models. Note `_deep_merge` recurses into nested dicts — a shallow `.update()` would replace the whole `defaults` block instead of merging inside it. This is exactly the defaults→override idea from §2, lifted one level up to whole files.

### 4d. Hot-reload watchers

The Copilot reloads on an explicit HTTP call. To reload automatically when a file changes on disk, watch the directory:

```python
from watchfiles import watch
import threading

def _watch_config(reload_fn, path="config"):
    for changes in watch(path):        # blocks, yields on each fs change
        print(f"[config] changed {changes} → reloading")
        reload_fn()

threading.Thread(target=_watch_config, args=(agent_config.reload,), daemon=True).start()
```

Trade-off: auto-reload is convenient in dev but risky in prod (a half-saved file mid-write can reload garbage). The explicit endpoint the Copilot uses is deliberately safer — reload happens exactly when *you* say so, atomically, and returns a confirmation you can assert on. A middle ground: watch for changes but debounce and validate (§4a) before swapping the singleton, so a broken file leaves the last-good config in place.

---

## 5. Gotchas & pitfalls

- **Validate configs (§4a).** Unvalidated YAML fails silently — a typo'd key is ignored and you fall back to a default you didn't intend. Pydantic (or at least a required-keys check) turns that into a loud startup error. Cheapest bug-catch you'll ever add.
- **Don't over-abstract.** Every knob you expose is a knob someone can set wrong. Config a value only when it genuinely varies across deploys or needs non-engineer tuning. A "config" with exactly one valid value is just code with extra indirection. If setting it correctly requires reading the code, it's logic — keep it in code.
- **Secrets in env, not YAML.** Config files get committed and shipped in images. API keys, tokens, DB passwords live in the environment (or a secret manager) and enter config via `${VAR}` interpolation (§4b). The Copilot keeps keys out of `observability.yaml` entirely — `llm_tracing.enabled: false` "until LANGFUSE_HOST / keys are set" in env.
- **Reload safety.** (1) Read `module.CFG`, never `from module import CFG` — the latter freezes the old object and never sees a reload (§2). (2) Copy defaults before merging so an override can't corrupt the baseline. (3) Make reload atomic: build the new config fully, then reassign the singleton in one statement — never mutate the live dict key-by-key while requests read it. (4) A reload can *fail*: if the new file is broken, keep serving the last-good config rather than swapping in `{}`. The Copilot's `_load()` returns `{}` on parse error, which is safe at *startup* but during a `reload()` would blank the roster — validate-then-swap avoids that.
- **Empty-dict fallback is a policy, not an accident.** `_load()` returning `{}` means "boot degraded rather than crash." Pair it with `active()`/`RAILS_MODE`-style reporting so a silent fallback is *visible* on a health endpoint, not a mystery.
- **Config precedence must be one documented order.** The Copilot's rule is: explicit env var > YAML value > inline literal default (`obs_config.apply()`, `_breaker()`). Pick an order, apply it everywhere, and write it down — mixed precedence is where "why is this endpoint wrong" tickets come from.

---

## ✅ Best Practices

- **Config holds behavior, code holds logic.** Put policy — models, prompts, retries, endpoints, rosters — in YAML, and keep every loop, conditional, and dispatch that acts on it in Python; the moment a config key needs branching to interpret, promote it back to code.
- **Validate at load time.** Parse each config into a Pydantic model (or schema) with `extra='forbid'` and field constraints so a typo or bad type fails loudly at startup, not silently at 2am.
- **Keep secrets in the environment, not YAML.** Commit only non-secret structure and reference credentials through `${VAR}` interpolation or a secret manager, so config files stay safe to ship in an image.
- **Layer defaults, then overrides.** Provide a full baseline in `defaults` and merge per-item and per-environment overrides on top (deep-merge for nested blocks) so each deploy expresses only its diffs.
- **Establish one documented precedence order.** Fix a single rule — explicit env var > YAML value > inline literal — apply it in every loader, and write it down so "why is this value wrong" is answerable.
- **Reload atomically with a validate-then-swap.** Build the new config fully, validate it, and reassign the singleton in one statement; on failure keep serving the last-good config instead of blanking it, and gate auto-reload behind debounced validation.
- **Always expose a real-vs-fallback signal.** When a loader can degrade (empty dict, `None` runtime), surface the active mode on a health endpoint so a silent fallback is observable rather than a mystery.
- **Treat config as documentation-of-intent and version it.** Comment the *why* next to each non-obvious value, keep configs in git with reviewable diffs, and use per-environment files so the roster reads as an auditable spec of the system.

## 6. Exercises

1. **Add Pydantic validation to `resilience.py`.** Define `ResilienceCfg`/`Defaults`/`ToolPolicy` models (§4a) with `extra='forbid'`, validate inside `_load()`, and confirm that renaming `retries` to `reties` in `config/resilience.yaml` now fails at startup instead of silently reverting to the default.

2. **Add a `/api/resilience/reload` route.** `app/resilience.py` has a loader but no reload path. Add a module-level `reload()` (mirror `agent_config.reload()`), wire a `@app.post("/api/resilience/reload")` in `app/main.py` returning the tool count, then edit a tool's `retries` in YAML and verify the merged policy changes without a restart. Watch out: `_BREAKERS` is a separate cache — decide whether reload should reset it.

3. **Add a whole new config file + loader + reload route.** Create `config/routing.yaml` (e.g. which model the orchestrator prefers per alert severity), an `app/routing_config.py` loader with `CFG`/`_load()`/`reload()`, and a `/api/routing/reload` endpoint. This exercises the full pattern end-to-end.

4. **Add per-environment overrides (§4c).** Implement `_deep_merge` + `load_env`, add a `config/resilience.prod.yaml` that bumps `defaults.retries` and swaps `models.chain`, and verify `APP_ENV=prod` merges it over the base while `APP_ENV=dev` uses the base alone.

5. **Add env interpolation to `obs_config.py`.** Support `${LANGFUSE_HOST:...}` in `observability.yaml` via `_expand` (§4b), then confirm the secret never appears in the committed file and that an explicit env var still wins (matching `apply()`'s existing `not os.getenv(...)` precedence).

6. **Prove the stale-import bug, then fix it.** In a scratch module do `from app.agent_config import AGENTS`, call `agent_config.reload()` after editing `agents.yaml`, and show the imported `AGENTS` is stale while `agent_config.AGENTS` is fresh. Write a one-line note on why every reader in the codebase goes through `AC.` / `config.`.

---

## 7. Connections

- **[08-nemo-guardrails.md]** — the real-or-fallback singleton in `config.py` (`RAILS` = real NeMo runtime *or* `None` → `DEMO`) is the config-loader pattern applied to a whole runtime; `RAILS_MODE` and `config.reload()` are its config-driven surface.
- **[09-resilience.md]** — `resilience.py`'s retries/breaker/degrade behavior is *entirely* driven by `config/resilience.yaml` through the `_policy()` defaults+override merge shown in §3; this tutorial is the config half of that component.
- **[06-orchestrator-worker-multi-agent.md]** — the worker roster, per-worker prompts, model choices, and tool assignments the orchestrator dispatches all come from `config/agents.yaml` via `agent_config.AGENTS`/`worker()`; the multi-agent structure is config data, not hardcoded nodes.

---

## 8. Further reading

- **pydantic-settings** — typed settings from env + files with validation and precedence built in; the natural upgrade from raw `yaml.safe_load` (§4a). https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **The Twelve-Factor App**, Factor III (Config) — the canonical argument for config-in-environment and strict config/code separation. https://12factor.net/config
- **Dynaconf** — a batteries-included config library doing per-environment layers, env interpolation, and secret backends (§4b–4c) so you don't hand-roll them. https://www.dynaconf.com/
- **watchfiles** — fast filesystem-watching for the hot-reload-on-change variant (§4d). https://watchfiles.helpmanual.io/
