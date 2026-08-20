"""13 · Config-driven design — loader singleton + defaults/overrides merge + reload().

Demonstrates the config-driven pattern from the tutorial (Section 2), standalone:
  1. YAML config is loaded ONCE at import into a module-level singleton (CFG).
  2. policy(tool) merges a tool's per-item overrides ONTO the shared defaults —
     the same defaults+override merge the project's resilience.py `_policy()` uses.
  3. reload() re-reads the file in place so long-lived callers see edits without
     any code change or restart — behavior-as-data.

The script prints merged config for a couple of items, then edits the YAML on
disk, calls reload(), and prints again to show behavior changing with NO code
change. It restores the original YAML at the end so it is repeatable.

Deps:   pip install pyyaml
Run:    python examples/13_config_driven.py
No API key needed — everything runs locally off the YAML file.
"""

import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
CONFIG_PATH = HERE / "13_sample.yaml"


def _load():
    """Parse the YAML → dict. Return {} on missing/broken file: degrade, don't crash."""
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception as e:  # noqa: BLE001 — any load failure degrades to empty policy
        print(f"[policy] unavailable: {e}")
        return {}


# Singleton: the config is loaded exactly ONCE, at import time. Every reader
# goes through this module (via policy()) so a reload() reassignment is visible.
CFG = _load()


def policy(tool):
    """Merge one tool's overrides ONTO a COPY of the defaults; overrides win.

    Copy first — dict(...) — so .update() can't mutate the shared defaults dict.
    Forget the copy and the first override permanently corrupts the baseline.
    """
    p = dict(CFG.get("defaults", {}))               # copy the baseline
    p.update(CFG.get("tools", {}).get(tool, {}))    # per-item override wins
    return p


def reload():
    """Re-read the YAML in place; swap the singleton in ONE assignment (atomic)."""
    global CFG
    CFG = _load()
    return {"tools": len(CFG.get("tools", {}))}


def _demo():
    # ── read merged values straight off the singleton ────────────────────────
    print("=== initial config (loaded once at import) ===")
    print("send_email  ->", policy("send_email"))   # retries=0 override, timeout=5 default
    print("fetch_stats ->", policy("fetch_stats"))  # retries=2 default, timeout=30 override
    print("unknown     ->", policy("unknown"))      # pure defaults, no override exists

    # ── change behavior WITHOUT touching code: edit YAML, then reload() ───────
    original = CONFIG_PATH.read_text()              # stash so we can restore
    try:
        cfg = yaml.safe_load(original)
        cfg["tools"]["send_email"] = {"retries": 1}  # was 0 → now 1
        cfg["defaults"]["timeout_s"] = 9             # bump the shared baseline too
        CONFIG_PATH.write_text(yaml.safe_dump(cfg))

        print("\n=== after editing YAML + calling reload() (no code changed) ===")
        print("reload() ->", reload())               # re-reads the file in place
        print("send_email  ->", policy("send_email"))  # retries now 1, timeout now 9
        print("fetch_stats ->", policy("fetch_stats"))  # timeout override still wins over 9
        print("unknown     ->", policy("unknown"))      # picks up the new default timeout=9
    finally:
        CONFIG_PATH.write_text(original)             # restore so the demo is repeatable
        reload()

    print("\n=== restored original config ===")
    print("send_email  ->", policy("send_email"))    # back to retries=0, timeout=5


if __name__ == "__main__":
    _demo()
