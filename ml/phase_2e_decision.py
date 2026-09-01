"""Phase 2E decision tree from paired stress-test results."""

from __future__ import annotations

from typing import Dict, Optional

from ml.composition_diagnostic import METHODOLOGY_VERSION

# Frozen evaluation seeds (not used in Phase 2C/2D on 0–199).
PHASE_2E_EVAL_SEED_BASE = 1000

# Pre-specified decision thresholds — not tuned on seeds 1000–1199.
DECISION_THRESHOLDS = {
    "seeded_fulfilled_min_delta": 10,
    "reached_2_core_min_delta": 5,
    "coverage_improvement_min": 0.003,
    "macro_turn_14_stats_ratio_max_abs_delta": 0.25,
    "macro_game_length_max_abs_delta": 1.5,
}


def evaluate_phase_2e_decision(control_mechanism: Dict, treatment_mechanism: Dict,
                               macro_delta: Dict,
                               thresholds: Optional[Dict] = None) -> Dict:
    th = thresholds or DECISION_THRESHOLDS
    ctrl = control_mechanism.get("seeded_current_target") or {}
    treat = treatment_mechanism.get("seeded_current_target") or {}

    fulfill_delta = (treat.get("fulfilled_exposures", 0)
                     - ctrl.get("fulfilled_exposures", 0))
    reached_2_delta = (treat.get("reached_2_core", 0) - ctrl.get("reached_2_core", 0))
    cov_delta = ((treatment_mechanism.get("sim_final_winner_coverage_mean") or 0)
                 - (control_mechanism.get("sim_final_winner_coverage_mean") or 0))

    fulfillment_up = fulfill_delta >= th["seeded_fulfilled_min_delta"]
    assembly_up = reached_2_delta >= th["reached_2_core_min_delta"]
    coverage_up = cov_delta >= th["coverage_improvement_min"]
    macro_ok = (
        abs(macro_delta.get("turn_14_stats_ratio_delta") or 0)
        <= th["macro_turn_14_stats_ratio_max_abs_delta"]
        and abs(macro_delta.get("game_length_delta") or 0)
        <= th["macro_game_length_max_abs_delta"])

    if fulfillment_up and assembly_up and coverage_up:
        branch = "recruit_causal"
        next_step = (
            "Phase 2C hypothesis confirmed: forcing seeded conversion produces "
            "coherent boards. Design a realistic tempo-aware build policy using "
            "separate DEV calibration — not this oracle.")
    elif fulfillment_up and assembly_up and not coverage_up:
        branch = "card_effect_fidelity"
        next_step = (
            "Pieces assemble under oracle stress but final coverage stays flat — "
            "strong evidence for card-effect fidelity as next bottleneck.")
    elif fulfillment_up and not assembly_up:
        branch = "retention_triples_transitions"
        next_step = (
            "Seeded fulfillment rises but 2+ core assembly does not — investigate "
            "retention, triples/discovers, or target transitions.")
    elif not fulfillment_up:
        branch = "tracing_or_policy_path"
        next_step = (
            "Even oracle stress did not raise seeded fulfillment — investigate "
            "tracing correctness or policy integration path.")
    else:
        branch = "mixed"
        next_step = "Results do not match a clean decision-tree branch."

    return {
        "phase_2c_methodology": METHODOLOGY_VERSION,
        "evaluation_seed_base": PHASE_2E_EVAL_SEED_BASE,
        "thresholds": th,
        "deltas_treatment_minus_control": {
            "seeded_fulfilled_exposures": fulfill_delta,
            "reached_2_core_states": reached_2_delta,
            "final_winner_coverage": cov_delta,
        },
        "flags": {
            "fulfillment_up": fulfillment_up,
            "assembly_up": assembly_up,
            "coverage_up": coverage_up,
            "macro_regression_ok": macro_ok,
        },
        "decision_branch": branch,
        "recommended_next_step": next_step,
        "control_seeded": ctrl,
        "treatment_seeded": treat,
    }
