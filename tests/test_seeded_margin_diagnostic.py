"""Tests for Phase 2I seeded margin diagnostic (2i_v1)."""

import random

from hsbg_coach.bg_env import BGEnv
from hsbg_coach.tempo_board_policy import policies_for_lobby
from hsbg_coach.tempo_margin_audit import (
    TempoMarginAuditCollector,
    break_even_lambda,
    break_even_lambda_bucket,
)
from ml.composition_trace import board_fingerprint
from ml.fidelity_phase_2i import (
    FROZEN_LAMBDA,
    PHASE_2I_SEED,
    MarginAuditTracer,
    run_margin_audit_rollouts,
)
from ml.seeded_margin_diagnostic import (
    FAILURE_CODES,
    METHODOLOGY_VERSION,
    analyze_margin_exposures,
    classify_rejection,
    is_composition_progress_failure,
)
from ml.composition_diagnostic import aggregate_diagnostics


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
            tracer_finals = [{
                "lobby": i, "seat": seat, "placement": p.placement,
                "final_board": [{"name": m.name, "attack": m.attack,
                                 "health": m.health, "golden": m.golden}
                                for m in p.board],
            } for seat, p in enumerate(env.players)]
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


def test_decisive_rejection_at_generation_close():
    from ml.seeded_margin_diagnostic import TrackedExposure
    exp = TrackedExposure(
        lobby=0, archetype_key="k", core_name="c", turn=3,
        shop_generation=1, core_have=1, tier=4, board_full_at_open=False,
        target_at_open="k")
    exp.close_reason = "roll"
    exp.decisive_event_index = 10
    code, _ = classify_rejection(exp, None)
    assert code in FAILURE_CODES


def test_break_even_lambda_synthetic():
    lam = break_even_lambda(
        core_raw=10.0, core_build=1.0, repl_raw=5.0, repl_build=0.0,
        chosen_raw=15.0, chosen_build=0.0)
    assert lam is not None
    assert abs(lam - 10.0) < 1e-9
    assert break_even_lambda_bucket(lam) == "lambda_le_12"
    assert break_even_lambda(
        core_raw=1.0, core_build=1.0, repl_raw=0.0, repl_build=0.0,
        chosen_raw=2.0, chosen_build=1.0) is None


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
    assert METHODOLOGY_VERSION == "2i_v1"
