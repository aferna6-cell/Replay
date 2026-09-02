"""Phase 2J decision tree and confirmation acceptance."""

from __future__ import annotations

from typing import Dict, Optional

from ml.phase_2h_decision import (
    CONFIRMATION_THRESHOLDS as PHASE_2H_THRESHOLDS,
    _committed_states,
    _macro_ok,
    _played_rate,
)

METHODOLOGY_VERSION = "2j_v1"

CONFIRMATION_THRESHOLDS = {
    **PHASE_2H_THRESHOLDS,
    # Safety: reject policies that routinely sacrifice large board share.
    "mean_relative_tempo_loss_max": 0.35,
    "p95_relative_tempo_loss_max": 0.70,
}


def evaluate_confirmation_acceptance(
        greedy_mechanism: Dict, treatment_mechanism: Dict,
        greedy_lifecycle: Dict, treatment_lifecycle: Dict,
        macro_delta: Dict,
        oracle_mechanism: Optional[Dict] = None,
        policy_stats: Optional[Dict] = None,
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
    g_fulfilled = (g_seed.get("fulfilled_exposures")
                   or (greedy_lifecycle.get("n_fulfilled_purchases") or 0))
    t_fulfilled_exp = t_seed.get("fulfilled_exposures") or treat_fulfilled
    meaningful_cohort = treat_fulfilled >= 5
    played_up = (meaningful_cohort
                 and played_delta >= th["played_rate_min_delta"])
    committed_delta = (_committed_states(treatment_mechanism)
                       - _committed_states(greedy_mechanism))
    fulfillment_up = t_fulfilled_exp > g_fulfilled
    macro_ok = _macro_ok(macro_delta, th)

    ps = policy_stats or {}
    mean_rel = ps.get("mean_relative_tempo_loss")
    p95_rel = ps.get("p95_relative_tempo_loss")
    board_sacrifice_ok = True
    if mean_rel is not None and mean_rel > th["mean_relative_tempo_loss_max"]:
        board_sacrifice_ok = False
    if p95_rel is not None and p95_rel > th["p95_relative_tempo_loss_max"]:
        board_sacrifice_ok = False

    mechanism_up = (
        reached_2_delta >= th["persistent_2_core_min_delta"]
        and committed_delta > 0
        and played_up
        and fulfillment_up)
    coverage_up = cov_delta >= th["coverage_improvement_min"]
    accept = macro_ok and mechanism_up and coverage_up and board_sacrifice_ok

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
            "seeded_fulfilled_exposures": t_fulfilled_exp - g_fulfilled,
            "treatment_fulfilled_purchases": treat_fulfilled,
            "meaningful_fulfilled_cohort": meaningful_cohort,
        },
        "board_sacrifice": {
            "mean_relative_tempo_loss": mean_rel,
            "p95_relative_tempo_loss": p95_rel,
            "replacement_transitions": ps.get("replacement_transitions"),
            "mean_persistence_weight": ps.get("mean_persistence_weight"),
            "mean_raw_stat_sacrifice_completed": ps.get(
                "mean_raw_stat_sacrifice_completed"),
        },
        "flags": {
            "macro_regression_ok": macro_ok,
            "mechanism_up": mechanism_up,
            "coverage_up": coverage_up,
            "fulfillment_up": fulfillment_up,
            "board_sacrifice_ok": board_sacrifice_ok,
            "accept_phase_2j_policy": accept,
        },
        "oracle_recovery_fractions": recovery,
    }


def evaluate_phase_2j_decision(
        greedy_mechanism: Dict, treatment_mechanism: Dict,
        greedy_lifecycle: Dict, treatment_lifecycle: Dict,
        macro_delta: Dict,
        acceptance: Dict) -> Dict:
    flags = acceptance.get("flags") or {}
    deltas = acceptance.get("deltas_treatment_minus_greedy") or {}

    if not flags.get("macro_regression_ok"):
        branch = "macro_regression"
        next_step = "Reject policy — macro fidelity regressed vs raw greedy."
    elif not flags.get("board_sacrifice_ok"):
        branch = "large_board_strength_sacrifice"
        next_step = (
            "Reject — compositions may rise but relative board-strength "
            "sacrifice is too large.")
    elif flags.get("accept_phase_2j_policy"):
        branch = "accept_board_management_policy"
        next_step = (
            "Board-relative opportunity-cost policy recovers mechanism and "
            "coverage with clean macro — accept as candidate recruit policy.")
    elif flags.get("mechanism_up") and not flags.get("coverage_up"):
        branch = "card_effect_fidelity"
        next_step = (
            "Mechanism up strongly but coverage flat — card-effect fidelity "
            "is now the leading bottleneck.")
    elif ((greedy_lifecycle.get("n_fulfilled_purchases") or 0) >= 5
          or (treatment_lifecycle.get("n_fulfilled_purchases") or 0) >= 5) \
            and deltas.get("played_rate", 0) >= CONFIRMATION_THRESHOLDS[
                "played_rate_min_delta"] and not flags.get("mechanism_up"):
        branch = "retention_inadequate"
        next_step = (
            "Fulfillment/deploy up but persistence remains low — "
            "retention still insufficient.")
    elif not flags.get("mechanism_up"):
        branch = "replacement_cost_insufficient"
        next_step = (
            "Policy barely changes core conversion — replacement-cost reform "
            "was not enough; revisit build-value representation.")
    else:
        branch = "mixed"
        next_step = "Results do not match a clean decision-tree branch."

    return {
        "decision_branch": branch,
        "recommended_next_step": next_step,
        "acceptance": acceptance,
    }


def rank_calibration_candidate(row: Dict) -> tuple:
    """Frozen selection order for α screen/replication (lower is better)."""
    macro_penalty = 0 if row.get("macro_ok") else 1
    sacrifice_penalty = 0 if row.get("board_sacrifice_ok", True) else 1
    return (
        macro_penalty,
        sacrifice_penalty,
        -row.get("reached_2_core", 0),
        -row.get("committed_states", 0),
        -row.get("fulfilled_exposures", 0),
        -row.get("coverage_mean", 0.0),
        row.get("action_deviation_rate", 1.0),
    )
