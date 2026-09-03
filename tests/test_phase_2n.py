"""Tests for Phase 2N catalogue classification + measurement guards."""

from hsbg_coach.bg_env import (
    PHASE_2N_DEATH_RETURN, PHASE_2N_FREEZE_TOPUP, POOL_COPIES, build_pool, TRIBES,
)
from hsbg_coach.active_tavern_pool import active_tavern_card_ids
from ml.fidelity_phase_2n import (
    METHODOLOGY_VERSION,
    PHASE_2N_SEED,
    assert_seed_range_allowed,
    evaluate_phase_2n_decision,
)
from ml.phase_2n_regression import (
    assert_fingerprint_matches,
    build_simulator_v1_x_fingerprint,
    evaluate_macro_fidelity_vs_reference,
)
from ml.shop_pool_audit import (
    audit_active_pool_precision_recall,
    audit_catalogue_synchronization,
    audit_rule_mismatches,
)


def test_methodology_and_flags():
    assert METHODOLOGY_VERSION == "2n_v3"
    assert PHASE_2N_SEED == 11700
    assert PHASE_2N_DEATH_RETURN is True
    assert PHASE_2N_FREEZE_TOPUP is True
    assert POOL_COPIES[6] == 7


def test_rejects_prior_and_confirm_seeds():
    for seed in (8000, 9000, 10200, 11000, 11500):
        try:
            assert_seed_range_allowed(seed, 10)
            assert False, f"should reject {seed}"
        except RuntimeError:
            pass
    assert_seed_range_allowed(11700, 500)


def test_catalogue_clean_after_2n_a():
    cat = audit_catalogue_synchronization()
    assert cat.get("n_missing_from_kb", 0) == 0
    assert cat["status_counts"].get("MISSING_OR_INVALID_TIER", 0) == 0


def test_active_pool_precision_recall_gates():
    ap = audit_active_pool_precision_recall(list(TRIBES))
    assert ap["active_pool_recall"] == 1.0
    assert ap["active_pool_precision"] == 1.0
    assert ap["token_cards_in_build_pool"] == 0
    assert ap["duos_only_in_solo_build_pool"] == 0
    assert ap["out_of_scope_t7_in_build_pool"] == 0
    assert ap["removed_or_generated_only_in_build_pool"] == 0
    assert ap["foraging_bat_in_build_pool"] is False
    assert ap["gates_pass"] is True


def test_build_pool_intersects_active_manifest():
    active = active_tavern_card_ids()
    cat = build_pool(lobby_tribes=list(TRIBES))
    assert len(cat) == len(active)
    assert {m.card_id for m in cat} == active
    assert all(m.name != "Foraging Bat" for m in cat)


def test_no_actionable_rule_mismatches_after_2n():
    rules = audit_rule_mismatches()
    assert rules["phase_2n_actionable_ids"] == []
    assert rules["n_phase_2n_actionable"] == 0


def _base_analysis(**overrides):
    base = {
        "catalogue_synchronization": {
            "n_missing_from_kb": 0,
            "status_counts": {"IN_EXACT_CATALOGUE": 224},
        },
        "rule_mismatches": {"phase_2n_actionable_ids": []},
        "live_calibration": {
            "primary_deal_level": {
                "raw_ratio_obs_over_exp": 0.97,
                "lobby_clustered": {
                    "raw_obs_minus_exp": {"ci95": [-1.0, 1.0]},
                },
            },
        },
        "pool_conservation": {"conservation_ok": True},
        "active_pool_precision_recall": {
            "active_pool_recall": 1.0,
            "active_pool_precision": 1.0,
            "gates_pass": True,
        },
        "paired_regression_panel": {
            "phase_2j_mechanism_regression_pass": True,
            "macro_fidelity_pass": True,
            "panel_pass": True,
        },
        "headlines": {},
    }
    base.update(overrides)
    return base


def test_decision_accept_requires_full_2n_v3_gate():
    assert evaluate_phase_2n_decision(_base_analysis())[
        "decision_branch"] == "accept_simulator_v1_x_candidate"

    dirty_pool = _base_analysis(active_pool_precision_recall={
        "active_pool_recall": 1.0,
        "active_pool_precision": 0.5,
        "gates_pass": False,
    })
    assert evaluate_phase_2n_decision(dirty_pool)["decision_branch"] == (
        "active_pool_precision_incomplete")

    dirty_mech = _base_analysis(paired_regression_panel={
        "phase_2j_mechanism_regression_pass": False,
        "macro_fidelity_pass": True,
        "panel_pass": False,
    })
    assert evaluate_phase_2n_decision(dirty_mech)["decision_branch"] == (
        "phase_2j_mechanism_regression")

    dirty_macro = _base_analysis(paired_regression_panel={
        "phase_2j_mechanism_regression_pass": True,
        "macro_fidelity_pass": False,
        "panel_pass": False,
    })
    assert evaluate_phase_2n_decision(dirty_macro)["decision_branch"] == (
        "macro_fidelity_regression")


def test_fingerprint_roundtrip_and_mismatch_guard():
    fp = build_simulator_v1_x_fingerprint(implementation_commit="deadbeef")
    assert "fingerprint_sha256" in fp
    assert_fingerprint_matches(fp, fp)
    bad = dict(fp)
    bad["fingerprint_sha256"] = "0" * 64
    try:
        assert_fingerprint_matches(fp, bad)
        assert False, "expected mismatch"
    except RuntimeError as e:
        assert "fingerprint mismatch" in str(e)


def test_macro_fidelity_helper_runs_on_tiny_rollout():
    from hsbg_coach.bg_env import greedy_policy
    from ml.fidelity_metrics import aggregate_lobby_dynamics, aggregate_turn_curves, run_fidelity_rollouts
    rows = run_fidelity_rollouts(3, seed=11700, policy=greedy_policy,
                                 scaling_mode="residual")
    tc = aggregate_turn_curves(rows)
    ld = aggregate_lobby_dynamics(rows)
    out = evaluate_macro_fidelity_vs_reference(tc, ld, rows=rows)
    assert "macro_fidelity_pass" in out
    assert "gates" in out
    assert "nontermination" in out["gates"]
