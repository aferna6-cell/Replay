"""Frozen reference fingerprints for Simulator Fidelity Benchmark v1.

Records immutable Simulator v1 identity before any environment changes.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Optional

from hsbg_coach.pace import FIRESTONE_PACE

from .experiment_contract import env_config, git_commit


def _sha256_dict(data: Dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _optional_package_version(distribution: str) -> Optional[str]:
    """Return installed distribution version without importing the package."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def fidelity_runtime_fingerprint() -> Dict[str, Any]:
    """Runtime provenance for simulator-only fidelity measurements.

    Fidelity jobs do not require the ML training stack. Record NumPy/Torch
    versions when their distributions are installed, but do not import either
    package merely to construct a simulator contract. Strict ML experiments
    continue to use ``experiment_contract.runtime_fingerprint`` and
    ``enforce_runtime_match``.
    """
    return {
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "torch_version": _optional_package_version("torch"),
        "numpy_version": _optional_package_version("numpy"),
        "torch_device": None,
        "torch_cuda_available": None,
        "provenance_scope": "simulator_fidelity",
        "ml_runtime_imported": False,
    }


def fidelity_env_config() -> Dict[str, Any]:
    """Environment provenance for simulator-only fidelity measurements.

    ``experiment_contract.env_config`` imports ``ml.rl_common`` to read the
    RL episode decision cap; that module imports NumPy. Simulator fidelity CI
    intentionally does not require the ML stack, so preserve the same frozen
    environment values when NumPy/Torch are absent instead of dropping the
    entire contract. Any other missing dependency still fails loudly.
    """
    try:
        return env_config()
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "torch"}:
            raise
        from hsbg_coach.bg_env import MAX_TURNS, N_ACTIONS

        return {
            "env": "hsbg_coach.bg_env.BGEnv",
            "n_players": 8,
            "field_size": 7,
            "agent_seat": 0,
            "n_actions": N_ACTIONS,
            "max_turns": MAX_TURNS,
            "max_decisions": 400,
        }


def fidelity_commit_provenance() -> Dict[str, Any]:
    """Return execution and source commits without conflating PR merge refs.

    GitHub ``pull_request`` workflows execute a synthetic merge commit, so
    ``git rev-parse HEAD`` is the execution identity, not the PR-head source
    identity. When a trustworthy pull-request event payload is available,
    record its head SHA separately. Local/push runs intentionally collapse
    source to execution so the contract remains deterministic and portable.
    """
    execution_commit = git_commit()
    source_commit = execution_commit
    source_kind = "execution_commit"

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.isfile(event_path):
        try:
            with open(event_path, encoding="utf-8") as f:
                event = json.load(f)
            candidate = event.get("pull_request", {}).get("head", {}).get("sha")
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
                source_commit = candidate.lower()
                source_kind = "pull_request_head"
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            pass

    return {
        "execution_commit": execution_commit,
        "source_commit": source_commit,
        "source_commit_kind": source_kind,
    }


FIDELITY_BENCHMARK_VERSION = "Replay Simulator Fidelity Benchmark v1"
SIMULATOR_VERSION = "Simulator v1"
SIMULATOR_V1_1_VERSION = "Simulator v1.1"

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
    return _build_simulator_contract(
        simulator_version=SIMULATOR_VERSION,
        scaling_mode="ratio",
        evaluation_seed=evaluation_seed,
        lobbies=lobbies,
    )


def git_working_tree_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL, text=True)
        return out.strip() == ""
    except Exception:
        return False


def build_simulator_v1_1_contract(*, evaluation_seed: int = 0,
                                  lobbies: int = 200,
                                  success_thresholds_sha256: Optional[str] = None,
                                  success_thresholds_path: Optional[str] = None
                                  ) -> Dict[str, Any]:
    """Simulator v1.1 — residual end-of-turn scaling only."""
    contract = _build_simulator_contract(
        simulator_version=SIMULATOR_V1_1_VERSION,
        scaling_mode="residual",
        evaluation_seed=evaluation_seed,
        lobbies=lobbies,
        parent_version=SIMULATOR_VERSION,
    )
    contract["working_tree_clean"] = git_working_tree_clean()
    if success_thresholds_sha256:
        contract["success_thresholds_sha256"] = success_thresholds_sha256
    if success_thresholds_path:
        contract["success_thresholds_path"] = success_thresholds_path
    return contract


def _build_simulator_contract(*, simulator_version: str, scaling_mode: str,
                              evaluation_seed: int, lobbies: int,
                              parent_version: Optional[str] = None
                              ) -> Dict[str, Any]:
    """Shared contract builder for fidelity simulator snapshots."""
    env = fidelity_env_config()
    refs = reference_fingerprints()
    commit_provenance = fidelity_commit_provenance()
    contract: Dict[str, Any] = {
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version": simulator_version,
        "scaling_mode": scaling_mode,
        "simulator_module": "hsbg_coach.bg_env.BGEnv",
        "code_commit": commit_provenance["execution_commit"],
        "execution_commit": commit_provenance["execution_commit"],
        "source_commit": commit_provenance["source_commit"],
        "source_commit_kind": commit_provenance["source_commit_kind"],
        "runtime": fidelity_runtime_fingerprint(),
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
    if parent_version:
        contract["parent_simulator_version"] = parent_version
    return contract
