"""Simulator Fidelity Phase 2E — seeded-core conversion stress test.

    python -m ml.fidelity_phase_2e
    python -m ml.fidelity_phase_2e --lobbies 200 --seed 1000

Oracle/stress A/B on fresh seeds 1000–1199 (not inspected in 2C/2D):
  control   = raw-stat greedy_policy
  treatment = seeded_core_stress_greedy_policy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, Optional

from hsbg_coach.bg_env import greedy_policy, seeded_core_stress_greedy_policy
from hsbg_coach.seeded_core_stress_policy import (PHASE_2E_EVAL_SEED_BASE,
                                                  POLICY_CONFIG_FINGERPRINT)

from .composition_diagnostic import METHODOLOGY_VERSION, aggregate_diagnostics
from .composition_trace import run_traced_rollouts
from .fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                               aggregate_turn_curves, real_composition_baseline,
                               run_fidelity_rollouts, summarize_divergence)
from .fidelity_paired import paired_turn_comparison, per_lobby_turn_means
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .phase_2d_acceptance import (composition_mechanism_summary,
                                  macro_regression_summary)
from .phase_2e_decision import evaluate_phase_2e_decision

DEFAULT_DIR = "results/sim_fidelity_phase_2e"
DEFAULT_SEED = PHASE_2E_EVAL_SEED_BASE
PHASE = "2E seeded-core conversion stress test"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_config_hash() -> str:
    blob = json.dumps(POLICY_CONFIG_FINGERPRINT, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def build_phase_2e_contract(*, evaluation_seed: int = DEFAULT_SEED,
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
            "policy": "hsbg_coach.bg_env.seeded_core_stress_greedy_policy",
            "note": "Oracle stress — not production policy",
        },
    }
    base["evaluation_note"] = (
        f"Fresh paired seeds {evaluation_seed}–{evaluation_seed + lobbies - 1}; "
        "seeds 0–199 reserved for Phase 2C/2D.")
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
        "mechanism": composition_mechanism_summary(diagnostic),
    }


def run_phase_2e(*, lobbies: int = 200, seed: int = DEFAULT_SEED,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError(
            "Working tree is not clean. Commit Phase 2E implementation first.")

    t0 = time.time()
    print(f"Paired Phase 2E stress test — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    control = _run_arm(lobbies, seed, greedy_policy, "control")
    treatment = _run_arm(lobbies, seed, seeded_core_stress_greedy_policy,
                         "treatment")

    paired_stats = paired_turn_comparison(
        control["per_lobby_stats"], treatment["per_lobby_stats"])
    macro_delta = macro_regression_summary(
        control["turn_curves"], treatment["turn_curves"],
        control["lobby_dynamics"], treatment["lobby_dynamics"],
        control["headline"], treatment["headline"])
    decision = evaluate_phase_2e_decision(
        control["mechanism"], treatment["mechanism"], macro_delta)
    real = real_composition_baseline()
    contract = build_phase_2e_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["working_tree_clean"] = tree_clean

    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "research_question": (
            "If we force conversion of seeded core opportunities (oracle stress), "
            "do coherent compositions emerge?"),
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
        "control": _arm_report(control),
        "treatment": _arm_report(treatment),
        "paired_macro_comparison": paired_stats,
        "macro_regression_delta_treatment_minus_control": macro_delta,
        "decision": decision,
        "phase_2d_context": (
            "Phase 2D showed path_adj/5 insufficient (0 seeded fulfillment). "
            "Phase 2E tests whether strong oracle conversion changes mechanism/outcome."),
    }
    _write_json(os.path.join(out_dir, "phase_2e_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def _arm_report(arm: Dict) -> Dict:
    return {
        "label": arm["label"],
        "mechanism": arm["mechanism"],
        "composition": arm["composition"],
        "headline_divergence": arm["headline"],
        "turn_curves": arm["turn_curves"],
        "lobby_dynamics": arm["lobby_dynamics"],
        "n_trace_events": len(arm["traces"]["events"]),
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2E")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")

    try:
        result = run_phase_2e(
            lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    dec = result["decision"]
    ctrl_s = result["control"]["mechanism"]["seeded_current_target"]
    treat_s = result["treatment"]["mechanism"]["seeded_current_target"]
    deltas = dec["deltas_treatment_minus_control"]

    print(f"\nSeeded current-target (control vs treatment):")
    print(f"  control:   {ctrl_s['fulfilled_exposures']}/{ctrl_s['legally_buyable_exposures']} fulfilled")
    print(f"  treatment: {treat_s['fulfilled_exposures']}/{treat_s['legally_buyable_exposures']} fulfilled")
    print(f"  delta fulfilled: {deltas['seeded_fulfilled_exposures']}")
    print(f"  delta reached 2+ core: {deltas['reached_2_core_states']}")
    print(f"  delta coverage: {deltas['final_winner_coverage']:+.4f}")
    print(f"\nDecision branch: {dec['decision_branch']}")
    print(f"  {dec['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/phase_2e_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
