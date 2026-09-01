"""Simulator Fidelity Phase 2D — build-aware recruit scoring A/B.

    python -m ml.fidelity_phase_2d
    python -m ml.fidelity_phase_2d --lobbies 200 --seed 0

Paired 200-lobby A/B on identical seeds (Simulator v1.1, residual scaling):
  control  = greedy_policy (raw attack + health buy ranking)
  treatment = build_aware_greedy_policy (path_value-augmented buy ranking)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, Optional

from hsbg_coach.bg_env import build_aware_greedy_policy, greedy_policy
from hsbg_coach.build_aware_policy import POLICY_CONFIG_FINGERPRINT

from .composition_diagnostic import METHODOLOGY_VERSION, aggregate_diagnostics
from .composition_trace import run_traced_rollouts
from .fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                               aggregate_turn_curves, real_composition_baseline,
                               run_fidelity_rollouts, summarize_divergence)
from .fidelity_paired import paired_turn_comparison, per_lobby_turn_means
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean, reference_fingerprints)
from .phase_2d_acceptance import (composition_mechanism_summary,
                                    evaluate_acceptance, macro_regression_summary,
                                    placement_summary)

DEFAULT_DIR = "results/sim_fidelity_phase_2d"
PHASE = "2D build-aware recruit scoring"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_config_hash() -> str:
    blob = json.dumps(POLICY_CONFIG_FINGERPRINT, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def build_phase_2d_contract(*, evaluation_seed: int = 0,
                            lobbies: int = 200) -> Dict:
    base = build_simulator_v1_1_contract(
        evaluation_seed=evaluation_seed, lobbies=lobbies)
    base["phase"] = PHASE
    base["phase_2c_methodology_version"] = METHODOLOGY_VERSION
    base["policy_config"] = POLICY_CONFIG_FINGERPRINT
    base["policy_config_hash_sha256"] = _policy_config_hash()
    base["arms"] = {
        "control": {
            "policy": "hsbg_coach.bg_env.greedy_policy",
            "buy_scoring": "attack + health",
        },
        "treatment": {
            "policy": "hsbg_coach.bg_env.build_aware_greedy_policy",
            "buy_scoring": POLICY_CONFIG_FINGERPRINT["buy_scoring"],
            "build_path_buy_divisor": POLICY_CONFIG_FINGERPRINT["build_path_buy_divisor"],
        },
    }
    base["evaluation_note"] = (
        "Paired A/B: identical lobby seeds, only recruit buy valuation differs.")
    return base


def _run_arm(lobbies: int, seed: int, policy: Callable, label: str) -> Dict:
    print(f"  [{label}] fidelity rollouts…")
    rows = run_fidelity_rollouts(lobbies, seed=seed, policy=policy,
                                 scaling_mode="residual")
    print(f"  [{label}] composition traces…")
    traces = run_traced_rollouts(lobbies, seed=seed, policy=policy,
                                 scaling_mode="residual")
    diagnostic = aggregate_diagnostics(traces)
    return {
        "label": label,
        "rows": rows,
        "traces": traces,
        "diagnostic": diagnostic,
        "turn_curves": aggregate_turn_curves(rows),
        "lobby_dynamics": aggregate_lobby_dynamics(rows),
        "composition": aggregate_composition(rows),
        "headline": summarize_divergence(aggregate_turn_curves(rows)),
        "per_lobby_stats": per_lobby_turn_means(rows),
        "placement": placement_summary(rows),
        "mechanism": composition_mechanism_summary(diagnostic),
    }


def run_phase_2d(*, lobbies: int = 200, seed: int = 0,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError(
            "Working tree is not clean. Commit Phase 2D implementation first, "
            "then rerun with a clean tree.")

    t0 = time.time()
    print(f"Paired Phase 2D A/B — {lobbies} lobbies, seeds {seed}–{seed + lobbies - 1}")
    control = _run_arm(lobbies, seed, greedy_policy, "control")
    treatment = _run_arm(lobbies, seed, build_aware_greedy_policy, "treatment")

    paired_stats = paired_turn_comparison(
        control["per_lobby_stats"], treatment["per_lobby_stats"])
    macro_delta = macro_regression_summary(
        control["turn_curves"], treatment["turn_curves"],
        control["lobby_dynamics"], treatment["lobby_dynamics"],
        control["headline"], treatment["headline"])
    acceptance = evaluate_acceptance(
        control["mechanism"], treatment["mechanism"], macro_delta)
    real = real_composition_baseline()
    contract = build_phase_2d_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["working_tree_clean"] = tree_clean

    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "research_question": (
            "Does adding existing build-path value to recruit decisions convert "
            "Phase 2C seeded composition opportunities into coherent boards "
            "without damaging Simulator v1.1 macro fidelity?"),
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "implementation_commit": impl_commit,
        "code_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "policy_config": POLICY_CONFIG_FINGERPRINT,
        "policy_config_hash_sha256": _policy_config_hash(),
        "contract": contract,
        "runtime_seconds": round(time.time() - t0, 2),
        "n_lobbies": lobbies,
        "evaluation_seed_base": seed,
        "real_final_winner_coverage_mean": real["real_final_winner_coverage_mean"],
        "control": _arm_report(control, include_traces=False),
        "treatment": _arm_report(treatment, include_traces=False),
        "paired_macro_comparison": paired_stats,
        "macro_regression_delta_treatment_minus_control": macro_delta,
        "placement_comparison": {
            "control": control["placement"],
            "treatment": treatment["placement"],
            "delta_mean_placement": (
                (treatment["placement"].get("mean_placement") or 0)
                - (control["placement"].get("mean_placement") or 0)),
            "note": "Secondary diagnostic — not an acceptance gate.",
        },
        "acceptance": acceptance,
    }
    _write_json(os.path.join(out_dir, "phase_2d_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def _arm_report(arm: Dict, *, include_traces: bool) -> Dict:
    out = {
        "label": arm["label"],
        "mechanism": arm["mechanism"],
        "composition": arm["composition"],
        "headline_divergence": arm["headline"],
        "turn_curves": arm["turn_curves"],
        "lobby_dynamics": arm["lobby_dynamics"],
        "placement": arm["placement"],
        "n_trace_events": len(arm["traces"]["events"]),
    }
    if include_traces:
        out["diagnostic"] = arm["diagnostic"]
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2D")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")
    print(f"Policy config hash: {_policy_config_hash()}")

    try:
        result = run_phase_2d(
            lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    acc = result["acceptance"]
    ctrl_s = result["control"]["mechanism"]["seeded_current_target"]
    treat_s = result["treatment"]["mechanism"]["seeded_current_target"]

    print(f"\nImplementation commit: {result['implementation_commit']}")
    print(f"Working tree clean: {result['working_tree_clean']}")
    print(f"\nSeeded current-target (control vs treatment):")
    print(f"  control:   {ctrl_s['legally_buyable_exposures']} exp, "
          f"{ctrl_s['fulfilled_exposures']} fulfilled, "
          f"{ctrl_s['rejected_exposures']} rejected")
    print(f"  treatment: {treat_s['legally_buyable_exposures']} exp, "
          f"{treat_s['fulfilled_exposures']} fulfilled, "
          f"{treat_s['rejected_exposures']} rejected")
    print(f"\nFinal winner coverage:")
    print(f"  control:   {ctrl_s.get('mean_max_core_pieces', 'n/a')} mean max core / "
          f"{result['control']['mechanism']['sim_final_winner_coverage_mean']:.4f} coverage")
    print(f"  treatment: {treat_s.get('mean_max_core_pieces', 'n/a')} mean max core / "
          f"{result['treatment']['mechanism']['sim_final_winner_coverage_mean']:.4f} coverage")
    print(f"\nAccept Phase 2D treatment: {acc['accept_phase_2d_treatment']}")
    print(f"  {acc['interpretation']}")

    print(f"\nSaved -> {args.out_dir}/phase_2d_report.json")
    return 0 if acc["accept_phase_2d_treatment"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
