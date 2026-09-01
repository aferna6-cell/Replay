"""Reusable experiment contract — freeze and validate matched A/B runs.

Records immutable fields (warm-start parameter hash, runtime fingerprint,
PPO recipe, environment config, DEV seed range, code commit) and provides
hard-fail gates so confounded experiments cannot proceed silently.

    from ml.experiment_contract import build_contract, ContractViolation
    contract = build_contract("results/exp/warm_start.pt")
    save_contract("results/exp/contract.json", contract)
    verify_warm_start("results/exp/warm_start.pt",
                      contract["expected_warm_start_parameter_sha256"])
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .model_fingerprint import checkpoint_fingerprint, checkpoint_parameter_sha256
from .seeds import DEV_SEED_END, DEV_SEED_START

_SEP = b"\x00"


class ContractViolation(RuntimeError):
    """Raised when a run violates the frozen experiment contract."""


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _sha256_dict(data: Mapping[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def runtime_fingerprint() -> Dict[str, Any]:
    """Current Python/Torch/device identity for the active process."""
    import numpy as np
    import torch

    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
    return {
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_device": device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def ppo_recipe() -> Dict[str, Any]:
    """Frozen PPO hyperparameters shared across matched runs."""
    from .train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,
                            LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)
    return {
        "iterations": 320,
        "episodes_per_iteration": 16,
        "shaping_initial": 1.0,
        "shaping_horizon": 40,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "gamma": GAMMA,
        "lam": LAM,
        "clip": CLIP,
        "entropy_coef": ENTROPY,
        "value_coef": VALUE_COEF,
        "ppo_epochs": PPO_EPOCHS,
        "minibatch": 256,
        "grad_clip_norm": 1.0,
        "league_every": LEAGUE_EVERY,
        "league_max": LEAGUE_MAX,
    }


def env_config() -> Dict[str, Any]:
    """Environment constants that must not drift between arms of an A/B run."""
    from hsbg_coach.bg_env import MAX_TURNS, N_ACTIONS
    from .benchmark import FIELD_SIZE
    from .rl_common import MAX_DECISIONS
    return {
        "env": "hsbg_coach.bg_env.BGEnv",
        "n_players": 8,
        "field_size": FIELD_SIZE,
        "agent_seat": 0,
        "n_actions": N_ACTIONS,
        "max_turns": MAX_TURNS,
        "max_decisions": MAX_DECISIONS,
    }


def build_contract(warm_start_path: str,
                   *,
                   kl_coef_values: Sequence[float] = (0.0, 0.1),
                   training_seeds: Sequence[int] = (0, 1, 2, 3),
                   primary_iterations: Sequence[int] = (0, 40, 80, 160, 320),
                   ) -> Dict[str, Any]:
    """Assemble a contract dict from the current runtime and warm-start file."""
    if not os.path.isfile(warm_start_path):
        raise ContractViolation(f"warm start not found: {warm_start_path}")
    warm_fp = checkpoint_fingerprint(warm_start_path)
    rt = runtime_fingerprint()
    recipe = ppo_recipe()
    env = env_config()
    return {
        "contract_version": 1,
        "expected_warm_start_parameter_sha256": warm_fp["parameter_sha256"],
        "warm_start_checkpoint_sha256": warm_fp["checkpoint_sha256"],
        "warm_start_path": warm_start_path,
        "code_commit": git_commit(),
        "runtime": rt,
        "runtime_fingerprint_sha256": _sha256_dict(rt),
        "ppo_recipe": recipe,
        "ppo_config_hash_sha256": _sha256_dict(recipe),
        "environment": env,
        "env_config_hash_sha256": _sha256_dict(env),
        "dev_seed_range": [DEV_SEED_START, DEV_SEED_END],
        "primary_dev_eval": {
            "field": "greedy",
            "games": 1000,
            "seed_range": [DEV_SEED_START, DEV_SEED_START + 999],
        },
        "secondary_dev_eval": {
            "field": "greedy4_random3",
            "games": 500,
            "seed_range": [DEV_SEED_START, DEV_SEED_START + 499],
        },
        "primary_iterations": list(primary_iterations),
        "training_seeds": list(training_seeds),
        "kl_coef_values": list(kl_coef_values),
        "single_variable_within_seed_pair": "kl_coef",
    }


def save_contract(path: str, contract: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(contract), f, indent=2)


def load_contract(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def enforce_runtime_match(contract: Mapping[str, Any]) -> None:
    """Fail if the active process differs from the recorded runtime."""
    current = runtime_fingerprint()
    recorded = contract["runtime"]
    for key in ("python_version", "torch_version", "numpy_version",
                "torch_device", "torch_cuda_available"):
        if current[key] != recorded[key]:
            raise ContractViolation(
                f"runtime mismatch on {key!r}: contract={recorded[key]!r} "
                f"current={current[key]!r}")
    if _sha256_dict(current) != contract["runtime_fingerprint_sha256"]:
        raise ContractViolation("runtime fingerprint hash mismatch")
    if git_commit() != contract.get("code_commit"):
        raise ContractViolation(
            f"code commit mismatch: contract={contract.get('code_commit')!r} "
            f"current={git_commit()!r}")


def verify_warm_start(path: str, expected_parameter_sha256: str) -> str:
    """Return the actual parameter hash; raise if it differs from expected."""
    actual = checkpoint_parameter_sha256(path)
    if actual != expected_parameter_sha256:
        raise ContractViolation(
            f"warm-start parameter_sha256 mismatch for {path}:\n"
            f"  expected: {expected_parameter_sha256}\n"
            f"  actual:   {actual}")
    return actual


def verify_checkpoint_parameter_sha256(path: str,
                                       expected_parameter_sha256: str,
                                       label: str = "") -> str:
    actual = checkpoint_parameter_sha256(path)
    if actual != expected_parameter_sha256:
        tag = f" ({label})" if label else ""
        raise ContractViolation(
            f"checkpoint parameter_sha256 mismatch{tag} for {path}:\n"
            f"  expected: {expected_parameter_sha256}\n"
            f"  actual:   {actual}")
    return actual


def verify_matched_iter0_pair(beta0_iter0: str, beta01_iter0: str,
                              expected_parameter_sha256: str) -> None:
    """Both arms of a seed pair must reproduce the warm start at iteration 0."""
    h0 = verify_checkpoint_parameter_sha256(
        beta0_iter0, expected_parameter_sha256, label="beta0 iter0")
    h1 = verify_checkpoint_parameter_sha256(
        beta01_iter0, expected_parameter_sha256, label="beta0.1 iter0")
    if h0 != h1:
        raise ContractViolation(
            f"iter0 hash mismatch within seed pair:\n"
            f"  beta0:  {h0}\n"
            f"  beta0.1: {h1}")


def verify_identical_placements(path_a: str, path_b: str,
                                  label: str = "") -> None:
    """Require byte-identical per-game placement sequences (paired DEV eval)."""
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    pa, pb = a.get("placements"), b.get("placements")
    if pa != pb:
        tag = f" ({label})" if label else ""
        diffs = sum(1 for x, y in zip(pa, pb) if x != y)
        raise ContractViolation(
            f"placement sequence mismatch{tag} between\n"
            f"  {path_a}\n  {path_b}\n"
            f"  games={len(pa)}, differing_positions={diffs}")


def verify_contract_fields_unchanged(contract: Mapping[str, Any],
                                     other: Mapping[str, Any],
                                     fields: Sequence[str]) -> None:
    for field in fields:
        if contract.get(field) != other.get(field):
            raise ContractViolation(
                f"contract field {field!r} differs: "
                f"{contract.get(field)!r} vs {other.get(field)!r}")
