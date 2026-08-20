"""
06 · Orchestrator–Worker Multi-Agent — CONFIG-DRIVEN, standalone runnable
=========================================================================

The graph's SHAPE is data: the worker roster, the orchestrator's plan, and the
synthesis/remediation roles are read from `config/agents.yaml` — not hardcoded.
Add a worker in YAML and this program grows a node with NO code change.

    1. ORCHESTRATOR (plan)  -> reads `orchestrator.few_shot` from YAML to decide
                               WHICH workers to dispatch for this alert.
    2. WORKERS (fan-out)     -> run in PARALLEL via asyncio.gather; the ROSTER
                               comes from `workers:` in YAML. One generic stub
                               stands in for every worker's LLM+tools call.
    3. SYNTHESIZER (reduce)  -> folds the findings into one answer, using the
                               `synthesis.role` prompt and `remediation.actions`
                               from YAML.

The "LLM call" is stubbed with asyncio.sleep, so this runs with no API key and
no network. The teaching point stays: N workers finish in ~max(latency)
wall-clock, not the sum, because they run concurrently.

Deps:   pip install pyyaml
Run:    python examples/06_orchestrator_worker.py
"""

import asyncio
import pathlib
import time

import yaml

# config/agents.yaml lives at the project root, one level up from examples/.
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "agents.yaml"


# ── WORKER BODIES: one method per specialist, each fetching its behavior from ─
# YAML. In production each is an LLM call driven by that worker's role/model/tools;
# here each looks itself up in config/agents.yaml by name and returns the configured
# `finding` after sleeping for the configured `latency`. Edit the YAML -> behavior
# changes with NO code edit.
def _block(name: str, cfg: dict) -> dict:
    """The workers[] entry for `name` from config, or {} if absent."""
    return next((w for w in cfg.get("workers", []) if w.get("name") == name), {})


async def _run(name: str, cfg: dict) -> str:
    w = _block(name, cfg)
    await asyncio.sleep(w.get("latency", 0.5))  # mock latency, from YAML
    return w.get("finding", f"({name}) no finding configured")


async def _log_analyzer(cfg: dict) -> str:
    return await _run("log_analyzer", cfg)


async def _metric_fetcher(cfg: dict) -> str:
    return await _run("metric_fetcher", cfg)


async def _code_searcher(cfg: dict) -> str:
    return await _run("code_searcher", cfg)


async def _runbook_retriever(cfg: dict) -> str:
    return await _run("runbook_retriever", cfg)


# Name -> method. Keys MUST match `workers[].name` in config/agents.yaml.
WORKER_IMPLS = {
    "log_analyzer": _log_analyzer,
    "metric_fetcher": _metric_fetcher,
    "code_searcher": _code_searcher,
    "runbook_retriever": _runbook_retriever,
}


# ── CONFIG: load the roster + orchestrator plan + synthesis role from YAML ───
def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def roster(cfg: dict) -> list[str]:
    """The worker names declared in YAML that we actually have a method for."""
    names = [w["name"] for w in cfg.get("workers", [])]
    known, unknown = [], []
    for n in names:
        (known if n in WORKER_IMPLS else unknown).append(n)
    if unknown:
        print(f"[config] no method for {unknown} — skipping those workers.")
    return known


# ── ORCHESTRATOR: decide WHICH workers to run, from the YAML few-shot plan ───
def plan(alert: str, cfg: dict, available: list[str]) -> list[str]:
    """A real orchestrator is an LLM prompted with `orchestrator.few_shot`. Here we
    reuse those exemplars deterministically: if the alert matches an exemplar, use
    its plan; otherwise dispatch every available worker."""
    for shot in cfg.get("orchestrator", {}).get("few_shot", []):
        if shot.get("alert", "").strip().lower() == alert.strip().lower():
            chosen = [w for w in shot.get("plan", []) if w in available]
            if chosen:
                return chosen
    return list(available)  # no matching exemplar -> run the whole roster


# ── SYNTHESIZER: reduce findings into a root cause + proposed remediation ────
def synthesize(alert: str, findings: dict[str, str], cfg: dict) -> str:
    joined = "\n".join(f"  - {name}: {finding}" for name, finding in findings.items())
    role = cfg.get("synthesis", {}).get("role", "").strip()
    actions = cfg.get("remediation", {}).get("actions", ["rollback_deploy"])
    # A real synthesizer is an LLM call over `role` + `joined`. Stubbed here; the
    # proposed action is drawn from the YAML remediation roster.
    remediation = "rollback_deploy" if "rollback_deploy" in actions else actions[0]
    header = f"[synthesis role loaded from YAML: {len(role)} chars]\n" if role else ""
    return (
        f"{header}"
        f"ALERT: {alert}\n"
        f"FINDINGS:\n{joined}\n"
        "ROOT CAUSE (stub): correlate the findings above.\n"
        f"PROPOSED REMEDIATION (needs human approval): {remediation}."
    )


# ── THE LOOP: plan -> fan-out (parallel) -> reduce ─────────────────────────
async def orchestrate(alert: str, cfg: dict) -> str:
    available = roster(cfg)
    chosen = plan(alert, cfg, available)
    print(f"ROSTER (from YAML): {available}")
    print(f"PLAN: dispatching {len(chosen)} workers -> {chosen}\n")

    t0 = time.perf_counter()
    # Fan-out: all chosen workers run concurrently, not one after another.
    # return_exceptions=True => one failing worker won't cancel its siblings.
    results = await asyncio.gather(
        *(WORKER_IMPLS[name](cfg) for name in chosen),
        return_exceptions=True,
    )

    findings: dict[str, str] = {}
    for name, result in zip(chosen, results):
        if isinstance(result, Exception):
            findings[name] = f"{name} unavailable ({result}) — using partial data"
        else:
            findings[name] = result
        print(f"WORKER[{name}]: {findings[name]}")

    elapsed = time.perf_counter() - t0
    print(f"\n[{len(chosen)} workers finished in {elapsed:.2f}s (concurrent fan-out)]\n")

    return synthesize(alert, findings, cfg)


if __name__ == "__main__":
    cfg = load_config()
    alert = "5xx rate on checkout-api spiked to 12% at 02:14 UTC"
    answer = asyncio.run(orchestrate(alert, cfg))
    print("SYNTHESIS:")
    print(answer)
