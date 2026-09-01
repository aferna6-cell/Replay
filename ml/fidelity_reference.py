"""Frozen reference fingerprints for Simulator Fidelity Benchmark v1.

Records immutable Simulator v1 identity before any environment changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

from hsbg_coach.pace import FIRESTONE_PACE

from .experiment_contract import env_config, git_commit, runtime_fingerprint


def _sha256_dict(data: Dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

FIDELITY_BENCHMARK_VERSION = "Replay Simulator Fidelity Benchmark v1"
SIMULATOR_VERSION = "Simulator v1"

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA_STATS = os.path.join(_ROOT, "data", "stats")
_DATA_CARDS = os.path.join(_ROOT, "data", "cards")

REFERENCE_PATHS = {
    "firestone_pace": FIRESTONE_PACE,
    "firestone_final_boards": os.path.join(_DATA_STATS, "firestone_final_boards.json"),
    "firestone_hero_stats": os.path.join(_DATA_STATS, "firestone_hero_stats.json"),
    "firestone_comp_stats": os.path.join(_DATA_STATS, "firestone_comp_stats.json"),
    "firestone_card_stats": os.path.join(_DATA_STATS, "firestone_card_stats.json"),
    "bg_cards": os.path.join(_DATA_CARDS, "bg_cards.json"),
    "card2vec": os.path.join(_DATA_CARDS, "card2vec.json"),
}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_fingerprints() -> Dict[str, str]:
    out = {}
    for name, path in REFERENCE_PATHS.items():
        if os.path.isfile(path):
            out[name] = file_sha256(path)
    return out


def reference_at_exact(curve: Dict[int, float], turn: int) -> Optional[float]:
    """Return a reference curve value only when that turn is explicitly measured."""
    if not curve:
        return None
    return curve.get(turn)


def load_reference_metadata() -> Dict[str, Any]:
    """Firestone fetch metadata from pace file (shared envelope)."""
    if not os.path.isfile(FIRESTONE_PACE):
        return {}
    data = json.load(open(FIRESTONE_PACE, encoding="utf-8"))
    meta = {k: data[k] for k in ("_source", "_fetched", "_mmr", "_period",
                                  "_heroDataPoints") if k in data}
    fetched = meta.get("_fetched", "unknown")
    mmr = meta.get("_mmr", "?")
    period = meta.get("_period", "?")
    meta["reference_label"] = (
        f"Firestone {fetched} reference distribution "
        f"(top-{mmr}% MMR, {period})")
    return meta


def build_simulator_v1_contract(*, evaluation_seed: int = 0,
                                lobbies: int = 200) -> Dict[str, Any]:
    """Immutable Simulator v1 snapshot for fidelity baselines."""
    env = env_config()
    refs = reference_fingerprints()
    return {
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "simulator_module": "hsbg_coach.bg_env.BGEnv",
        "code_commit": git_commit(),
        "runtime": runtime_fingerprint(),
        "environment": env,
        "env_config_hash_sha256": _sha256_dict(env),
        "reference_data_fingerprints": refs,
        "reference_metadata": load_reference_metadata(),
        "evaluation": {
            "policy": "greedy",
            "lobbies": lobbies,
            "base_seed": evaluation_seed,
            "note": ("Fidelity rollouts seed lobbies as base_seed + i. "
                     "Not Replay Benchmark v1 agent-eval seeds."),
        },
        "combat_note": (
            "Combat accuracy is NOT re-measured in Fidelity Benchmark v1. "
            "Prior spot-checks found ~97% outcome agreement vs Firestone; "
            "Phase 2B+ should not rewrite combat unless spot-checks regress."
        ),
    }
