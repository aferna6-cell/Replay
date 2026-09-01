"""Phase 2D acceptance criteria and metric extraction."""

from __future__ import annotations

from typing import Dict, Optional

from ml.composition_diagnostic import METHODOLOGY_VERSION

# Frozen Phase 2C control baseline (greedy, seeds 0–199, implementation 0781ddf).
PHASE_2C_CONTROL_BASELINE = {
    "methodology_version": "2c_v3",
    "implementation_commit": "0781ddf76cfa1884fffee145ecb2b4f3cfe8b3c3",
    "seeded_current_target": {
        "legally_buyable_exposures": 82,
        "fulfilled_exposures": 0,
        "rejected_exposures": 82,
        "fulfillment_rate": 0.0,
        "rejection_rate": 1.0,
    },
    "sim_final_winner_coverage_mean": 0.00854456420206345,
}

# Pre-specified Phase 2D acceptance thresholds (not tuned on eval seeds 0–199).
ACCEPTANCE_THRESHOLDS = {
    "seeded_fulfilled_min": 5,
    "seeded_fulfillment_rate_min": 0.05,
    "reached_2_core_states_min": 5,
    "coverage_improvement_min": 0.003,
    "macro_turn_14_stats_ratio_max_abs_delta": 0.20,
    "macro_tier_error_turn_10_max_abs_delta": 0.35,
    "macro_game_length_max_abs_delta": 1.0,
    "macro_alive_turn_10_max_abs_delta": 1.0,
}


def _funnel_view(diagnostic: Dict, view: str) -> Dict:
    return ((diagnostic.get("winner_decision_funnel") or {})
            .get(view, {}).get("aggregate_funnel") or {})


def composition_mechanism_summary(diagnostic: Dict) -> Dict:
    seeded = _funnel_view(diagnostic, "seeded_current_target")
    committed = _funnel_view(diagnostic, "committed_current_target")
    hindsight = _funnel_view(diagnostic, "final_target_hindsight")
    broad = _funnel_view(diagnostic, "broad_current_target")
    return {
        "methodology_version": diagnostic.get("methodology_version"),
        "sim_final_winner_coverage_mean": diagnostic.get(
            "sim_final_winner_coverage_mean"),
        "seeded_current_target": {
            "legally_buyable_exposures": seeded.get("legally_buyable_exposures", 0),
            "fulfilled_exposures": seeded.get("fulfilled_exposures", 0),
            "rejected_exposures": seeded.get("rejected_exposures", 0),
            "fulfillment_rate": seeded.get("fulfillment_rate", 0.0),
            "rejection_rate": seeded.get("rejection_rate", 0.0),
            "reached_2_core": seeded.get("reached_2_core", 0),
            "reached_4_core": seeded.get("reached_4_core", 0),
            "mean_max_core_pieces": seeded.get("mean_max_core_pieces", 0.0),
            "exposure_accounting_valid": seeded.get("exposure_accounting_valid"),
        },
        "committed_current_target": committed,
        "final_target_hindsight": {
            "legally_buyable_exposures": hindsight.get("legally_buyable_exposures", 0),
            "fulfilled_exposures": hindsight.get("fulfilled_exposures", 0),
            "rejected_exposures": hindsight.get("rejected_exposures", 0),
            "fulfillment_rate": hindsight.get("fulfillment_rate", 0.0),
        },
        "broad_current_target_exploratory": {
            "legally_buyable_exposures": broad.get("legally_buyable_exposures", 0),
            "fulfilled_exposures": broad.get("fulfilled_exposures", 0),
            "rejected_exposures": broad.get("rejected_exposures", 0),
        },
    }


def macro_regression_summary(control_curves: Dict, treatment_curves: Dict,
                             control_lobby: Dict, treatment_lobby: Dict,
                             control_headline: Dict,
                             treatment_headline: Dict) -> Dict:
    def _delta(key_ctrl, key_treat, turn: str):
        c = (control_curves.get(turn) or {}).get(key_ctrl)
        t = (treatment_curves.get(turn) or {}).get(key_treat)
        if c is None or t is None:
            return None
        return t - c

    return {
        "turn_10_stats_ratio_delta": (
            (treatment_headline.get("stats_ratio_turn_10") or 0)
            - (control_headline.get("stats_ratio_turn_10") or 0)),
        "turn_14_stats_ratio_delta": (
            (treatment_headline.get("stats_ratio_turn_14") or 0)
            - (control_headline.get("stats_ratio_turn_14") or 0)),
        "turn_10_tier_error_delta": _delta("tier_error", "tier_error", "10"),
        "turn_14_tier_error_delta": _delta("tier_error", "tier_error", "14"),
        "turn_10_alive_error_delta": _delta(
            "alive_error_vs_prior", "alive_error_vs_prior", "10"),
        "game_length_delta": (
            (treatment_lobby.get("avg_game_length") or 0)
            - (control_lobby.get("avg_game_length") or 0)),
        "control_avg_game_length": control_lobby.get("avg_game_length"),
        "treatment_avg_game_length": treatment_lobby.get("avg_game_length"),
    }


def placement_summary(rows) -> Dict:
    placements = [r["placement"] for r in rows if r.get("placement")]
    if not placements:
        return {"mean_placement": None, "n": 0}
    wins = sum(1 for p in placements if p == 1)
    return {
        "mean_placement": sum(placements) / len(placements),
        "win_rate": wins / len(placements),
        "n": len(placements),
    }


def evaluate_acceptance(control_mechanism: Dict, treatment_mechanism: Dict,
                        macro_delta: Dict,
                        thresholds: Optional[Dict] = None) -> Dict:
    th = thresholds or ACCEPTANCE_THRESHOLDS
    seeded = treatment_mechanism.get("seeded_current_target") or {}
    ctrl_cov = control_mechanism.get("sim_final_winner_coverage_mean") or 0.0
    treat_cov = treatment_mechanism.get("sim_final_winner_coverage_mean") or 0.0
    cov_delta = treat_cov - ctrl_cov

    mechanism = {
        "seeded_fulfilled": {
            "value": seeded.get("fulfilled_exposures", 0),
            "baseline": PHASE_2C_CONTROL_BASELINE["seeded_current_target"]["fulfilled_exposures"],
            "min_required": th["seeded_fulfilled_min"],
            "passed": seeded.get("fulfilled_exposures", 0) >= th["seeded_fulfilled_min"],
        },
        "seeded_fulfillment_rate": {
            "value": seeded.get("fulfillment_rate", 0.0),
            "min_required": th["seeded_fulfillment_rate_min"],
            "passed": seeded.get("fulfillment_rate", 0.0) >= th["seeded_fulfillment_rate_min"],
        },
        "reached_2_core_states": {
            "value": seeded.get("reached_2_core", 0),
            "min_required": th["reached_2_core_states_min"],
            "passed": seeded.get("reached_2_core", 0) >= th["reached_2_core_states_min"],
        },
    }
    outcome = {
        "final_winner_coverage_delta": {
            "control": ctrl_cov,
            "treatment": treat_cov,
            "delta": cov_delta,
            "min_improvement": th["coverage_improvement_min"],
            "passed": cov_delta >= th["coverage_improvement_min"],
        },
    }
    macro = {
        "turn_14_stats_ratio_delta": {
            "value": macro_delta.get("turn_14_stats_ratio_delta"),
            "max_abs_delta": th["macro_turn_14_stats_ratio_max_abs_delta"],
            "passed": abs(macro_delta.get("turn_14_stats_ratio_delta") or 0)
            <= th["macro_turn_14_stats_ratio_max_abs_delta"],
        },
        "turn_10_tier_error_delta": {
            "value": macro_delta.get("turn_10_tier_error_delta"),
            "max_abs_delta": th["macro_tier_error_turn_10_max_abs_delta"],
            "passed": abs(macro_delta.get("turn_10_tier_error_delta") or 0)
            <= th["macro_tier_error_turn_10_max_abs_delta"],
        },
        "game_length_delta": {
            "value": macro_delta.get("game_length_delta"),
            "max_abs_delta": th["macro_game_length_max_abs_delta"],
            "passed": abs(macro_delta.get("game_length_delta") or 0)
            <= th["macro_game_length_max_abs_delta"],
        },
        "turn_10_alive_error_delta": {
            "value": macro_delta.get("turn_10_alive_error_delta"),
            "max_abs_delta": th["macro_alive_turn_10_max_abs_delta"],
            "passed": abs(macro_delta.get("turn_10_alive_error_delta") or 0)
            <= th["macro_alive_turn_10_max_abs_delta"],
        },
    }

    mechanism_win = all(m["passed"] for m in mechanism.values())
    outcome_win = outcome["final_winner_coverage_delta"]["passed"]
    macro_ok = all(m["passed"] for m in macro.values())
    accept = mechanism_win and outcome_win and macro_ok

    interpretation = _interpret(accept, mechanism_win, outcome_win, macro_ok)

    return {
        "phase_2c_control_baseline": PHASE_2C_CONTROL_BASELINE,
        "thresholds": th,
        "phase_2c_methodology": METHODOLOGY_VERSION,
        "mechanism": mechanism,
        "outcome": outcome,
        "macro_regression": macro,
        "mechanism_win": mechanism_win,
        "outcome_win": outcome_win,
        "macro_regression_ok": macro_ok,
        "accept_phase_2d_treatment": accept,
        "interpretation": interpretation,
    }


def _interpret(accept: bool, mech: bool, out: bool, macro: bool) -> str:
    if accept:
        return ("Phase 2D treatment accepted: seeded fulfillment and composition "
                "coverage improved without material macro regression.")
    if mech and not out:
        return ("Mechanism improved but final coverage flat — next bottleneck "
                "likely card-effect fidelity (pieces assemble but lack value).")
    if not mech:
        return ("Build-aware buy valuation insufficient to convert seeded "
                "opportunities — treatment does not meet mechanism thresholds.")
    if not macro:
        return ("Composition metrics may have moved but macro fidelity regressed "
                "— reject treatment despite local comp gains.")
    return "Phase 2D treatment not accepted under pre-specified gates."
