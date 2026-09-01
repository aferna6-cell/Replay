"""Phase 2G decision tree from paired deployment stress-test results."""

from __future__ import annotations

from typing import Dict, Optional

from ml.composition_diagnostic import METHODOLOGY_VERSION

PHASE_2G_EVAL_SEED_BASE = 2000

DECISION_THRESHOLDS = {
    "played_count_min_delta": 10,
    "played_rate_min_delta": 0.25,
    "reached_2_core_min_delta": 5,
    "coverage_improvement_min": 0.003,
    "macro_turn_14_stats_ratio_max_abs_delta": 0.25,
    "macro_game_length_max_abs_delta": 1.5,
}


def _played_rate(lifecycle: Optional[Dict]) -> float:
    if not lifecycle:
        return 0.0
    n = lifecycle.get("n_fulfilled_purchases") or 0
    if n <= 0:
        return 0.0
    return (lifecycle.get("funnel") or {}).get("played", 0) / n


def evaluate_phase_2g_decision(control_mechanism: Dict, treatment_mechanism: Dict,
                               control_lifecycle: Dict, treatment_lifecycle: Dict,
                               macro_delta: Dict,
                               thresholds: Optional[Dict] = None) -> Dict:
    th = thresholds or DECISION_THRESHOLDS
    ctrl_m = control_mechanism.get("seeded_current_target") or {}
    treat_m = treatment_mechanism.get("seeded_current_target") or {}

    ctrl_played = (control_lifecycle.get("funnel") or {}).get("played", 0)
    treat_played = (treatment_lifecycle.get("funnel") or {}).get("played", 0)
    played_delta = treat_played - ctrl_played
    played_rate_delta = _played_rate(treatment_lifecycle) - _played_rate(control_lifecycle)

    reached_2_delta = (treat_m.get("reached_2_core", 0) - ctrl_m.get("reached_2_core", 0))
    cov_delta = ((treatment_mechanism.get("sim_final_winner_coverage_mean") or 0)
                 - (control_mechanism.get("sim_final_winner_coverage_mean") or 0))

    played_up = (played_delta >= th["played_count_min_delta"]
                 or played_rate_delta >= th["played_rate_min_delta"])
    assembly_up = reached_2_delta >= th["reached_2_core_min_delta"]
    coverage_up = cov_delta >= th["coverage_improvement_min"]
    macro_ok = (
        abs(macro_delta.get("turn_14_stats_ratio_delta") or 0)
        <= th["macro_turn_14_stats_ratio_max_abs_delta"]
        and abs(macro_delta.get("game_length_delta") or 0)
        <= th["macro_game_length_max_abs_delta"])

    if not macro_ok:
        branch = "macro_regression"
        next_step = "Reject causal interpretation until macro fidelity is preserved."
    elif played_up and assembly_up and coverage_up:
        branch = "board_slot_causal"
        next_step = (
            "Guaranteeing board slots for oracle-acquired cores produces persistent "
            "assembly and coverage — design a realistic board-management policy "
            "(separate DEV calibration, not this oracle).")
    elif played_up and assembly_up and not coverage_up:
        branch = "retention_or_card_effects"
        next_step = (
            "Deployment oracle assembles 2+ cores but coverage stays flat — "
            "investigate retention or card-effect fidelity depending on survival paths.")
    elif played_up and not assembly_up:
        branch = "seed_target_identity"
        next_step = (
            "Played rate rises but persistent 2+ core assembly does not — investigate "
            "seed loss, target transitions, or identity/triple mechanics.")
    elif not played_up:
        branch = "deployment_oracle_insufficient"
        next_step = (
            "Board-slot sell oracle did not materially raise played rate — "
            "investigate deployment integration or hand/board tracing.")
    else:
        branch = "mixed"
        next_step = "Results do not match a clean decision-tree branch."

    return {
        "phase_2c_methodology": METHODOLOGY_VERSION,
        "evaluation_seed_base": PHASE_2G_EVAL_SEED_BASE,
        "thresholds": th,
        "deltas_treatment_minus_control": {
            "played_count": played_delta,
            "played_rate": played_rate_delta,
            "reached_2_core_states": reached_2_delta,
            "final_winner_coverage": cov_delta,
        },
        "flags": {
            "played_up": played_up,
            "assembly_up": assembly_up,
            "coverage_up": coverage_up,
            "macro_regression_ok": macro_ok,
        },
        "control_lifecycle_funnel": control_lifecycle.get("funnel"),
        "treatment_lifecycle_funnel": treatment_lifecycle.get("funnel"),
        "control_board_full_summary": control_lifecycle.get("board_full_summary"),
        "treatment_board_full_summary": treatment_lifecycle.get("board_full_summary"),
        "decision_branch": branch,
        "recommended_next_step": next_step,
        "control_seeded": ctrl_m,
        "treatment_seeded": treat_m,
    }
