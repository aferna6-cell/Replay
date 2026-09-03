"""Tests for Phase 2I seeded margin diagnostic (2i_v2)."""

import random

from hsbg_coach.bg_env import BGEnv
from hsbg_coach.tempo_board_audit_hook import _decode_chosen
from hsbg_coach.tempo_board_policy import PendingTransition, policies_for_lobby
from hsbg_coach.tempo_margin_audit import (
    ScoredTransition,
    TempoMarginAuditCollector,
    break_even_lambda,
    directional_break_even_bucket,
)
from ml.composition_diagnostic import aggregate_diagnostics
from ml.composition_trace import board_fingerprint
from ml.fidelity_phase_2i import (
    FROZEN_LAMBDA,
    PHASE_2I_SEED,
    run_margin_audit_rollouts,
)
from ml.seeded_margin_diagnostic import (
    FAILURE_CODES,
    METHODOLOGY_VERSION,
    TrackedExposure,
    _close_exposure,
    analyze_margin_exposures,
    classify_rejection,
    is_composition_progress_failure,
)


def _finals_fingerprint(lobbies: int, seed: int, *, with_audit: bool):
    if with_audit:
        out = run_margin_audit_rollouts(lobbies, seed, FROZEN_LAMBDA)
        finals = out["traces"]["player_finals"]
    else:
        finals = []
        for i in range(lobbies):
            policies = policies_for_lobby(FROZEN_LAMBDA, 8)
            env = BGEnv(seed=seed + i, scaling_mode="residual")
            env.play_scripted(list(policies))
            tracer_finals = []
            for seat, p in enumerate(env.players):
                # After Phase 2N death-return, eliminated seats have empty
                # board but last_board snapshot (same as RecruitTracer).
                src = p.board if p.board else p.last_board
                tracer_finals.append({
                    "lobby": i, "seat": seat, "placement": p.placement,
                    "final_board": [{"name": m.name, "attack": m.attack,
                                     "health": m.health, "golden": m.golden}
                                    for m in src],
                })
            finals.extend(tracer_finals)
    return sorted(
        (f["lobby"], f["seat"], f.get("placement"),
         board_fingerprint(f.get("final_board") or []))
        for f in finals)


def test_audit_enabled_matches_disabled_actions_and_placements():
    lobbies, seed = 2, PHASE_2I_SEED + 50
    plain_fp = _finals_fingerprint(lobbies, seed, with_audit=False)
    audit_fp = _finals_fingerprint(lobbies, seed, with_audit=True)
    assert plain_fp == audit_fp


def test_no_duplicate_exposure_per_shop_generation():
    out = run_margin_audit_rollouts(3, PHASE_2I_SEED, FROZEN_LAMBDA)
    analysis = analyze_margin_exposures(
        out["traces"], out["audit"], out["audit_event_links"])
    keys = [
        (r["lobby"], r["archetype_key"], r["core_name"],
         r["turn"], r["shop_generation"])
        for r in analysis["rejected_exposure_records"]
    ]
    fulfilled_count = analysis["funnel"]["fulfilled"]
    total = analysis["funnel"]["seeded_legally_buyable_exposures"]
    assert len(set(keys)) == analysis["funnel"]["rejected"]
    assert total == fulfilled_count + analysis["funnel"]["rejected"]


def test_fulfilled_not_classified_as_rejected():
    out = run_margin_audit_rollouts(4, PHASE_2I_SEED + 10, FROZEN_LAMBDA)
    analysis = analyze_margin_exposures(
        out["traces"], out["audit"], out["audit_event_links"])
    assert analysis["funnel"]["fulfilled"] >= 0
    rej_keys = {
        (r["lobby"], r["core_name"], r["turn"], r["shop_generation"])
        for r in analysis["rejected_exposure_records"]}
    assert analysis["funnel"]["rejected"] == len(rej_keys)


def test_every_rejected_has_one_primary_cause():
    out = run_margin_audit_rollouts(3, PHASE_2I_SEED + 20, FROZEN_LAMBDA)
    analysis = analyze_margin_exposures(
        out["traces"], out["audit"], out["audit_event_links"])
    for r in analysis["rejected_exposure_records"]:
        assert r["primary_cause"] in FAILURE_CODES
        assert isinstance(r["composition_progress_failure"], bool)


def test_exposure_counts_reconcile_with_2c_v3():
    out = run_margin_audit_rollouts(4, PHASE_2I_SEED + 30, FROZEN_LAMBDA)
    analysis = analyze_margin_exposures(
        out["traces"], out["audit"], out["audit_event_links"])
    diag = aggregate_diagnostics(out["traces"])
    seeded = diag["winner_decision_funnel"]["seeded_current_target"]["aggregate_funnel"]
    rec = analysis["reconciliation_2c_v3"]
    assert rec["tracked_exposures"] == seeded["legally_buyable_exposures"]
    assert rec["phase_2c_fulfilled"] == seeded["fulfilled_exposures"]
    assert rec["phase_2c_rejected"] == seeded["rejected_exposures"]
    assert rec["counts_match"] is True


def test_decisive_rejection_at_first_loss_of_buyability():
    exp = TrackedExposure(
        lobby=0, archetype_key="k", core_name="c", turn=3,
        shop_generation=1, core_have=1, tier=4, board_full_at_open=False,
        target_at_open="k", last_pre_buyable=True)
    exp.decision_audit_indices.append(7)
    _close_exposure(
        exp,
        reason="first_loss_of_buyability",
        audit_idx=exp.decision_audit_indices[-1],
        event_idx=10,
    )
    assert exp.closed
    assert exp.close_reason == "first_loss_of_buyability"
    assert exp.decisive_audit_index == 7
    code, _ = classify_rejection(exp, None)
    assert code in FAILURE_CODES


def test_directional_break_even_lambda_synthetic():
    current_lambda = 12.0
    core_slope = 1.0
    chosen_slope = 2.0
    lam = break_even_lambda(
        core_raw=10.0, core_build=1.0, repl_raw=5.0, repl_build=0.0,
        chosen_raw=15.0, chosen_build=2.0)
    assert lam is not None
    assert lam < current_lambda
    bucket = directional_break_even_bucket(
        lam,
        current_lambda=current_lambda,
        core_slope=core_slope,
        chosen_slope=chosen_slope,
    )
    assert bucket == "helpful_lower_lambda_only"

    lam_high = break_even_lambda(
        core_raw=10.0, core_build=0.5, repl_raw=5.0, repl_build=0.0,
        chosen_raw=15.0, chosen_build=0.0)
    assert lam_high is not None
    assert lam_high > current_lambda
    assert directional_break_even_bucket(
        lam_high,
        current_lambda=current_lambda,
        core_slope=0.5,
        chosen_slope=0.0,
    ) == "helpful_higher_lambda_le_24"

    assert break_even_lambda(
        core_raw=1.0, core_build=1.0, repl_raw=0.0, repl_build=0.0,
        chosen_raw=2.0, chosen_build=1.0) is None
    assert directional_break_even_bucket(
        None, current_lambda=current_lambda,
        core_slope=1.0, chosen_slope=1.0) == "no_lambda_effect"


def test_compound_sell_decodes_from_pending_not_first_transition():
    from hsbg_coach.bg_env import A_SELL0

    sell_action = A_SELL0 + 2
    transitions = [
        ScoredTransition(
            action_type="shop_sell_buy", candidate_name="CoreA",
            candidate_slot=0, raw_component=5.0, build_gain=1.0,
            build_component=12.0, replacement_name="Weak", replacement_slot=2,
            replacement_raw=100.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=-95.0, is_target_core=True,
            action_id=sell_action),
        ScoredTransition(
            action_type="hand_sell_play", candidate_name="HandMinion",
            candidate_slot=0, raw_component=20.0, build_gain=0.0,
            build_component=0.0, replacement_name="Weak", replacement_slot=2,
            replacement_raw=100.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=-80.0, is_target_core=False,
            action_id=sell_action),
    ]
    pending = PendingTransition(
        source="hand",
        stage="play",
        candidate_name="HandMinion",
        candidate_slot=0,
        replacement_slot=2,
        net_value=-42.0,
        build_gain=0.5,
        raw_sacrifice=100.0,
    )
    obs = {
        "board": [{"name": "Weak", "attack": 50, "health": 50}],
        "hand": [{"name": "HandMinion", "attack": 10, "health": 10}],
        "shop": [],
        "tavern_tier": 4,
    }
    mask = [False] * 200
    mask[sell_action] = True
    chosen = _decode_chosen(
        sell_action, obs, mask, transitions, pending, None,
        fit=None, tier=4, lambda_build=12.0)
    assert chosen.candidate_name == "HandMinion"
    assert chosen.net_value == -42.0
    assert chosen.action_type == "hand_sell_play"


def test_alternate_core_not_composition_failure():
    assert is_composition_progress_failure("E_ALTERNATE_CORE_SELECTED") is False
    assert is_composition_progress_failure("B_RAW_STAT_COMPETITOR_DOMINATES") is True


def test_audit_does_not_consume_rng():
    seed = PHASE_2I_SEED + 99
    env1 = BGEnv(seed=seed, scaling_mode="residual")
    p1 = policies_for_lobby(FROZEN_LAMBDA, 8)
    env1.play_scripted(list(p1))

    audit = TempoMarginAuditCollector()
    env2 = BGEnv(seed=seed, scaling_mode="residual")
    p2 = policies_for_lobby(FROZEN_LAMBDA, 8, audit=audit)
    env2.play_scripted(list(p2))
    assert env1.turn == env2.turn


def test_fresh_policy_instances_per_lobby():
    out = run_margin_audit_rollouts(3, PHASE_2I_SEED + 40, FROZEN_LAMBDA)
    assert len(out["audit"].snapshots) > 0
    lobbies_seen = {s.lobby for s in out["audit"].snapshots}
    assert len(lobbies_seen) >= 1


def test_methodology_version():
    assert METHODOLOGY_VERSION == "2i_v2"


def test_report_includes_rank_and_quartile_breakdowns():
    out = run_margin_audit_rollouts(4, PHASE_2I_SEED + 60, FROZEN_LAMBDA)
    analysis = analyze_margin_exposures(
        out["traces"], out["audit"], out["audit_event_links"])
    hm = analysis["headline_metrics"]
    assert "pct_core_ranked_first_with_build" in hm
    assert "pct_core_ranked_first_without_build" in hm
    assert "mean_chosen_minus_core_raw_gap" in hm
    assert "mean_core_raw_advantage" in hm
    assert "breakdown_by_core_frequency_quartile" in analysis
    be = analysis["break_even_lambda"]
    assert "pct_helpful_lower_lambda_only" in be
    assert "pct_helpful_higher_lambda_le_24" in be
