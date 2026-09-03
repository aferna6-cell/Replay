"""Simulator Fidelity Phase 2G — seeded-core deployment / board-slot stress test.

    python -m ml.fidelity_phase_2g
    python -m ml.fidelity_phase_2g --lobbies 200 --seed 2000

Paired A/B on fresh seeds 2000–2199 (not inspected in 2E/2F):
  control   = seeded_core_stress_greedy_policy (Phase 2E buy oracle)
  treatment = seeded_core_deploy_stress_greedy_policy (+ board-slot sell for hand cores)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, Optional

from hsbg_coach.bg_env import (seeded_core_deploy_stress_greedy_policy,
                               seeded_core_stress_greedy_policy)
from hsbg_coach.seeded_core_deploy_policy import (PHASE_2G_EVAL_SEED_BASE,
                                                  POLICY_CONFIG_FINGERPRINT)

from .composition_diagnostic import METHODOLOGY_VERSION, aggregate_diagnostics
from .composition_trace import run_traced_rollouts
from .core_lifecycle_diagnostic import (METHODOLOGY_VERSION as PHASE_2F_VERSION,
                                        analyze_core_lifecycles,
                                        lifecycle_meets_fulfillment_count)
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
from .phase_2g_decision import evaluate_phase_2g_decision

DEFAULT_DIR = "results/sim_fidelity_phase_2g"
DEFAULT_SEED = PHASE_2G_EVAL_SEED_BASE
PHASE = "2G seeded-core deployment board-slot stress test"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_config_hash() -> str:
    blob = json.dumps(POLICY_CONFIG_FINGERPRINT, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def build_phase_2g_contract(*, evaluation_seed: int = DEFAULT_SEED,
                            lobbies: int = 200) -> Dict:
    base = build_simulator_v1_1_contract(
        evaluation_seed=evaluation_seed, lobbies=lobbies)
    base["phase"] = PHASE
    base["phase_2c_methodology_version"] = METHODOLOGY_VERSION
    base["phase_2f_methodology_version"] = PHASE_2F_VERSION
    base["policy_config"] = POLICY_CONFIG_FINGERPRINT
    base["policy_config_hash_sha256"] = _policy_config_hash()
    base["arms"] = {
        "control": {
            "policy": "hsbg_coach.bg_env.seeded_core_stress_greedy_policy",
            "note": "Phase 2E buy oracle only",
        },
        "treatment": {
            "policy": "hsbg_coach.bg_env.seeded_core_deploy_stress_greedy_policy",
            "note": "Phase 2E buy oracle + non-core sell to deploy hand cores",
        },
    }
    base["evaluation_note"] = (
        f"Fresh paired seeds {evaluation_seed}–{evaluation_seed + lobbies - 1}; "
        "seeds 1000–1199 reserved for Phase 2E/2F.")
    return base


def _run_arm(lobbies: int, seed: int, policy: Callable, label: str) -> Dict:
    print(f"  [{label}] fidelity rollouts…")
    rows = run_fidelity_rollouts(lobbies, seed=seed, policy=policy,
                                 scaling_mode="residual")
    print(f"  [{label}] composition traces…")
    traces = run_traced_rollouts(lobbies, seed=seed, policy=policy,
                                 scaling_mode="residual")
    diagnostic = aggregate_diagnostics(traces)
    lifecycle = analyze_core_lifecycles(traces)
    return {
        "label": label,
        "rows": rows,
        "traces": traces,
        "diagnostic": diagnostic,
        "lifecycle": lifecycle,
        "lifecycle_latch_parity_ok": lifecycle_meets_fulfillment_count(traces),
        "turn_curves": aggregate_turn_curves(rows),
        "lobby_dynamics": aggregate_lobby_dynamics(rows),
        "composition": aggregate_composition(rows),
        "headline": summarize_divergence(aggregate_turn_curves(rows)),
        "per_lobby_stats": per_lobby_turn_means(rows),
        "mechanism": composition_mechanism_summary(diagnostic),
    }


def run_phase_2g(*, lobbies: int = 200, seed: int = DEFAULT_SEED,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError(
            "Working tree is not clean. Commit Phase 2G implementation first.")

    t0 = time.time()
    print(f"Paired Phase 2G deployment stress — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    control = _run_arm(lobbies, seed, seeded_core_stress_greedy_policy, "control")
    treatment = _run_arm(lobbies, seed, seeded_core_deploy_stress_greedy_policy,
                         "treatment")

    paired_stats = paired_turn_comparison(
        control["per_lobby_stats"], treatment["per_lobby_stats"])
    macro_delta = macro_regression_summary(
        control["turn_curves"], treatment["turn_curves"],
        control["lobby_dynamics"], treatment["lobby_dynamics"],
        control["headline"], treatment["headline"])
    decision = evaluate_phase_2g_decision(
        control["mechanism"], treatment["mechanism"],
        control["lifecycle"], treatment["lifecycle"], macro_delta)
    real = real_composition_baseline()
    contract = build_phase_2g_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["working_tree_clean"] = tree_clean

    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "research_question": (
            "If we guarantee a board slot for oracle-acquired seeded cores, "
            "does acquisition become persistent 2+ core assembly?"),
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "phase_2f_methodology_version": PHASE_2F_VERSION,
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
        "phase_2f_context": (
            "Phase 2F showed 33/34 fulfilled cores stuck in hand on full boards. "
            "Phase 2G tests whether board-slot creation enables deployment."),
    }
    _write_json(os.path.join(out_dir, "phase_2g_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def _arm_report(arm: Dict) -> Dict:
    lc = arm["lifecycle"]
    return {
        "label": arm["label"],
        "mechanism": arm["mechanism"],
        "lifecycle": {
            "n_fulfilled_purchases": lc["n_fulfilled_purchases"],
            "funnel": lc["funnel"],
            "fate_totals": lc["fate_totals"],
            "board_full_summary": lc["board_full_summary"],
        },
        "lifecycle_latch_parity_ok": arm["lifecycle_latch_parity_ok"],
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

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2G")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")

    try:
        result = run_phase_2g(
            lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    dec = result["decision"]
    deltas = dec["deltas_treatment_minus_control"]
    ctrl_lc = result["control"]["lifecycle"]
    treat_lc = result["treatment"]["lifecycle"]

    print(f"\nLifecycle played (control vs treatment):")
    print(f"  control:   {ctrl_lc['funnel']['played']}/{ctrl_lc['n_fulfilled_purchases']}")
    print(f"  treatment: {treat_lc['funnel']['played']}/{treat_lc['n_fulfilled_purchases']}")
    print(f"  delta played: {deltas['played_count']} (rate {deltas['played_rate']:+.2f})")
    print(f"  delta reached 2+ core: {deltas['reached_2_core_states']}")
    print(f"  delta coverage: {deltas['final_winner_coverage']:+.4f}")
    print(f"\nDecision branch: {dec['decision_branch']}")
    print(f"  {dec['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/phase_2g_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
