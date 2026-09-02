"""Phase 2H decision tree and confirmation acceptance."""

from __future__ import annotations

from typing import Dict, Optional

from ml.composition_diagnostic import METHODOLOGY_VERSION

CONFIRMATION_THRESHOLDS = {
    "persistent_2_core_min_delta": 5,
    "played_rate_min_delta": 0.25,
    "coverage_improvement_min": 0.003,
    "macro_turn_14_stats_ratio_max_abs_delta": 0.25,
    "macro_tier_error_turn_10_max_abs_delta": 0.35,
    "macro_game_length_max_abs_delta": 1.5,
    "macro_alive_turn_10_max_abs_delta": 1.0,
}


def _played_rate(lifecycle: Optional[Dict]) -> float:
    if not lifecycle:
        return 0.0
    n = lifecycle.get("n_fulfilled_purchases") or 0
    if n <= 0:
        return 0.0
    return (lifecycle.get("funnel") or {}).get("played", 0) / n


def _committed_states(mechanism: Dict) -> int:
    committed = mechanism.get("committed_current_target") or {}
    if isinstance(committed, dict):
        return int(committed.get("n_lobby_archetype_states") or 0)
    return 0


def _macro_ok(macro_delta: Dict, th: Dict) -> bool:
    return (
        abs(macro_delta.get("turn_14_stats_ratio_delta") or 0)
        <= th["macro_turn_14_stats_ratio_max_abs_delta"]
        and abs(macro_delta.get("turn_10_tier_error_delta") or 0)
        <= th["macro_tier_error_turn_10_max_abs_delta"]
        and abs(macro_delta.get("game_length_delta") or 0)
        <= th["macro_game_length_max_abs_delta"]
        and abs(macro_delta.get("turn_10_alive_error_delta") or 0)
        <= th["macro_alive_turn_10_max_abs_delta"])


def evaluate_confirmation_acceptance(
        greedy_mechanism: Dict, treatment_mechanism: Dict,
        greedy_lifecycle: Dict, treatment_lifecycle: Dict,
        macro_delta: Dict,
        oracle_mechanism: Optional[Dict] = None,
        thresholds: Optional[Dict] = None) -> Dict:
    th = thresholds or CONFIRMATION_THRESHOLDS
    g_seed = greedy_mechanism.get("seeded_current_target") or {}
    t_seed = treatment_mechanism.get("seeded_current_target") or {}
    o_seed = (oracle_mechanism or {}).get("seeded_current_target") or {}

    reached_2_delta = t_seed.get("reached_2_core", 0) - g_seed.get("reached_2_core", 0)
    cov_delta = ((treatment_mechanism.get("sim_final_winner_coverage_mean") or 0)
                 - (greedy_mechanism.get("sim_final_winner_coverage_mean") or 0))
    played_delta = _played_rate(treatment_lifecycle) - _played_rate(greedy_lifecycle)
    treat_fulfilled = treatment_lifecycle.get("n_fulfilled_purchases") or 0
    meaningful_cohort = treat_fulfilled >= 5
    played_up = (meaningful_cohort
                 and played_delta >= th["played_rate_min_delta"])
    committed_delta = (_committed_states(treatment_mechanism)
                       - _committed_states(greedy_mechanism))
    macro_ok = _macro_ok(macro_delta, th)

    mechanism_up = (
        reached_2_delta >= th["persistent_2_core_min_delta"]
        and committed_delta > 0
        and played_up)
    coverage_up = cov_delta >= th["coverage_improvement_min"]
    accept = macro_ok and mechanism_up and coverage_up

    oracle_2core_lift = o_seed.get("reached_2_core", 0) - g_seed.get("reached_2_core", 0)
    oracle_cov_lift = ((oracle_mechanism or {}).get("sim_final_winner_coverage_mean") or 0) - (
        greedy_mechanism.get("sim_final_winner_coverage_mean") or 0)
    recovery = {
        "persistent_2_core_lift_recovered_fraction": (
            reached_2_delta / oracle_2core_lift if oracle_2core_lift else None),
        "coverage_lift_recovered_fraction": (
            cov_delta / oracle_cov_lift if oracle_cov_lift else None),
    }

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "thresholds": th,
        "deltas_treatment_minus_greedy": {
            "reached_2_core_states": reached_2_delta,
            "final_winner_coverage": cov_delta,
            "played_rate": played_delta,
            "committed_states": committed_delta,
            "treatment_fulfilled_purchases": treat_fulfilled,
            "meaningful_fulfilled_cohort": meaningful_cohort,
        },
        "flags": {
            "macro_regression_ok": macro_ok,
            "mechanism_up": mechanism_up,
            "coverage_up": coverage_up,
            "accept_phase_2h_policy": accept,
        },
        "oracle_recovery_fractions": recovery,
    }


def evaluate_phase_2h_decision(
        greedy_mechanism: Dict, treatment_mechanism: Dict,
        greedy_lifecycle: Dict, treatment_lifecycle: Dict,
        macro_delta: Dict,
        acceptance: Dict) -> Dict:
    flags = acceptance.get("flags") or {}
    deltas = acceptance.get("deltas_treatment_minus_greedy") or {}

    if not flags.get("macro_regression_ok"):
        branch = "macro_regression"
        next_step = "Reject policy — macro fidelity regressed vs raw greedy."
    elif flags.get("accept_phase_2h_policy"):
        branch = "accept_board_management_policy"
        next_step = (
            "Realistic tempo-aware policy recovers mechanism and coverage with "
            "clean macro — accept as candidate next simulator recruit/board policy.")
    elif flags.get("mechanism_up") and not flags.get("coverage_up"):
        branch = "card_effect_fidelity"
        next_step = (
            "Coherent boards emerge but coverage stays flat — card-effect fidelity "
            "is the leading next intervention.")
    elif ((greedy_lifecycle.get("n_fulfilled_purchases") or 0) >= 5
          or (treatment_lifecycle.get("n_fulfilled_purchases") or 0) >= 5) \
            and deltas.get("played_rate", 0) >= CONFIRMATION_THRESHOLDS[
                "played_rate_min_delta"] and not flags.get("mechanism_up"):
        branch = "retention_inadequate"
        next_step = (
            "Buy/deploy improve but persistent 2+ core assembly does not — "
            "retention/sell policy still inadequate.")
    elif not flags.get("mechanism_up"):
        branch = "transition_utility_inadequate"
        next_step = (
            "Transition utility barely moves mechanism — reformulate policy; "
            "do not touch card effects yet.")
    else:
        branch = "mixed"
        next_step = "Results do not match a clean decision-tree branch."

    return {
        "decision_branch": branch,
        "recommended_next_step": next_step,
        "acceptance": acceptance,
    }


def rank_calibration_candidate(row: Dict) -> tuple:
    """Frozen selection order for λ screen/replication (lower is better)."""
    macro_penalty = 0 if row.get("macro_ok") else 1
    return (
        macro_penalty,
        -row.get("reached_2_core", 0),
        -row.get("committed_states", 0),
        -row.get("coverage_mean", 0.0),
        row.get("action_deviation_rate", 1.0),
    )
