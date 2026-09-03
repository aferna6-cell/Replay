"""Simulator Fidelity Phase 2J — board-relative opportunity-cost policy.

    python -m ml.fidelity_phase_2j fit-prior
    python -m ml.fidelity_phase_2j calibrate
    python -m ml.fidelity_phase_2j confirm --alpha 1.0
    python -m ml.fidelity_phase_2j full

Persistence prior fit: DEV 7000–7299 (greedy only).
Screen α ∈ {0.5,1,2}: DEV 7300–7399.
Replication top-2: DEV 7400–7799.
Confirmation: 8000–8199 (once).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

from hsbg_coach.bg_env import BGEnv, greedy_policy, seeded_core_deploy_stress_greedy_policy
from hsbg_coach.board_opportunity_policy import (
    ALPHA_CANDIDATES,
    METHODOLOGY_VERSION,
    PHASE_2J_CONFIRM_LOBBIES,
    PHASE_2J_CONFIRM_SEED,
    PHASE_2J_FIT_LOBBIES,
    PHASE_2J_FIT_SEED,
    PHASE_2J_REPLICATION_LOBBIES,
    PHASE_2J_REPLICATION_SEED,
    PHASE_2J_SCREEN_LOBBIES,
    PHASE_2J_SCREEN_SEED,
    aggregate_policy_stats,
    policies_for_lobby,
    policy_config_fingerprint,
)
from hsbg_coach.pace import board_stats
from hsbg_coach.persistence_prior import PersistencePrior

from .composition_diagnostic import (METHODOLOGY_VERSION as PHASE_2C_VERSION,
                                     aggregate_diagnostics)
from .composition_trace import RecruitTracer, run_traced_rollouts
from .core_lifecycle_diagnostic import (METHODOLOGY_VERSION as PHASE_2F_VERSION,
                                        analyze_core_lifecycles)
from .fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                               aggregate_turn_curves, real_composition_baseline,
                               run_fidelity_rollouts, summarize_divergence)
from .fidelity_paired import per_lobby_turn_means
from .fidelity_phase_2h import (assert_trace_lobby_integrity,
                                compute_action_deviation_rate)
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .fit_persistence_prior import fit_persistence_prior_from_traces
from .phase_2d_acceptance import (composition_mechanism_summary,
                                  macro_regression_summary)
from .phase_2j_decision import (CONFIRMATION_THRESHOLDS,
                                evaluate_confirmation_acceptance,
                                evaluate_phase_2j_decision,
                                rank_calibration_candidate)

DEFAULT_DIR = "results/sim_fidelity_phase_2j"
PRIOR_FILENAME = "persistence_prior.json"
PHASE = "2J board-relative opportunity-cost policy"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_hash(cfg: Dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _prior_path(out_dir: str) -> str:
    return os.path.join(out_dir, PRIOR_FILENAME)


def load_prior(out_dir: str = DEFAULT_DIR) -> PersistencePrior:
    path = _prior_path(out_dir)
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing persistence prior at {path}; run fit-prior first.")
    return PersistencePrior.load(path)


def run_traced_rollouts_board_opp(
        lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
        scaling_mode: str = "residual") -> Dict:
    all_events: List[Dict] = []
    all_turn_summaries: List[Dict] = []
    all_player_finals: List[Dict] = []
    lobby_meta: List[Dict] = []

    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        policies = policies_for_lobby(alpha, prior, 8)
        env = BGEnv(seed=lobby_seed, scaling_mode=scaling_mode)
        tracer = RecruitTracer(lobby_id=lobby_i, seed=lobby_seed)
        env.play_scripted(list(policies), recruit_tracer=tracer)
        game_length = env.turn
        for pf in tracer.player_finals:
            pf["game_length"] = game_length
        all_events.extend(tracer.events)
        all_turn_summaries.extend(tracer.turn_summaries)
        all_player_finals.extend(tracer.player_finals)
        lobby_meta.append({
            "lobby": lobby_i,
            "seed": lobby_seed,
            "lobby_tribes": list(env.lobby_tribes),
            "game_length": env.turn,
        })
        del env

    traces = {
        "lobbies": lobbies,
        "seed": seed,
        "scaling_mode": scaling_mode,
        "events": all_events,
        "turn_summaries": all_turn_summaries,
        "player_finals": all_player_finals,
        "lobby_meta": lobby_meta,
    }
    assert_trace_lobby_integrity(traces, lobbies)
    return traces


def run_fidelity_rollouts_board_opp(
        lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
        scaling_mode: str = "residual"
        ) -> tuple[List[Dict], List]:
    rows: List[Dict] = []
    all_policies = []
    for i in range(lobbies):
        policies = policies_for_lobby(alpha, prior, 8)
        all_policies.extend(policies)
        env = BGEnv(seed=seed + i, scaling_mode=scaling_mode)
        recs = env.play_scripted(list(policies))
        game_length = max((r["turn"] for r in recs), default=0)
        for r in recs:
            s = r["state"]
            rows.append({
                "lobby": i,
                "seed": seed + i,
                "seat": r["seat"],
                "turn": r["turn"],
                "game_length": game_length,
                "tavern_tier": float(s["tavern_tier"]),
                "gold": float(s["gold"]),
                "board_size": float(len(s["board"])),
                "board_stats": float(board_stats(s)),
                "board": s["board"],
                "players_alive": float(s["players_alive"]),
                "placement": r.get("placement"),
            })
        del env
    return rows, all_policies


def _run_single_policy_arm(lobbies: int, seed: int, policy: Callable,
                           label: str) -> Dict:
    print(f"  [{label}] fidelity rollouts…")
    rows = run_fidelity_rollouts(lobbies, seed=seed, policy=policy,
                                 scaling_mode="residual")
    print(f"  [{label}] composition traces…")
    traces = run_traced_rollouts(lobbies, seed=seed, policy=policy,
                                 scaling_mode="residual")
    assert_trace_lobby_integrity(traces, lobbies)
    diagnostic = aggregate_diagnostics(traces)
    lifecycle = analyze_core_lifecycles(traces)
    turn_curves = aggregate_turn_curves(rows)
    return {
        "label": label,
        "rows": rows,
        "traces": traces,
        "diagnostic": diagnostic,
        "lifecycle": lifecycle,
        "turn_curves": turn_curves,
        "lobby_dynamics": aggregate_lobby_dynamics(rows),
        "composition": aggregate_composition(rows),
        "headline": summarize_divergence(turn_curves),
        "per_lobby_stats": per_lobby_turn_means(rows),
        "mechanism": composition_mechanism_summary(diagnostic),
    }


def _run_board_opp_arm(lobbies: int, seed: int, alpha: float,
                       prior: PersistencePrior, label: str, *,
                       greedy_baseline_traces: Optional[Dict] = None) -> Dict:
    print(f"  [{label}] α={alpha} fidelity rollouts…")
    rows, policies_fidelity = run_fidelity_rollouts_board_opp(
        lobbies, seed, alpha, prior, scaling_mode="residual")
    policy_stats = aggregate_policy_stats(policies_fidelity)

    print(f"  [{label}] α={alpha} composition traces…")
    traces = run_traced_rollouts_board_opp(
        lobbies, seed, alpha, prior, scaling_mode="residual")

    diagnostic = aggregate_diagnostics(traces)
    lifecycle = analyze_core_lifecycles(traces)
    deviation = None
    if greedy_baseline_traces is not None:
        deviation = compute_action_deviation_rate(greedy_baseline_traces, traces)

    turn_curves = aggregate_turn_curves(rows)
    return {
        "label": label,
        "alpha": alpha,
        "methodology_version": METHODOLOGY_VERSION,
        "rows": rows,
        "traces": traces,
        "diagnostic": diagnostic,
        "lifecycle": lifecycle,
        "policy_stats": policy_stats,
        "action_deviation_rate_vs_greedy": deviation,
        "turn_curves": turn_curves,
        "lobby_dynamics": aggregate_lobby_dynamics(rows),
        "composition": aggregate_composition(rows),
        "headline": summarize_divergence(turn_curves),
        "per_lobby_stats": per_lobby_turn_means(rows),
        "mechanism": composition_mechanism_summary(diagnostic),
        "tier_breakdown": tier_band_breakdown_from_traces(traces),
    }


def tier_band_breakdown_from_traces(traces: Dict) -> Dict:
    """Seeded exposures / fulfillment / 2+ progress by tavern-tier band.

    Exposure counts use tavern tier at shop-generation open.
    Progress (reached 2+) uses tavern tier at the turn of first 2-core board.
    Committed states are those that ever meet committed_current_target, attributed
    to the tier at first committed observation.
    """
    from hsbg_coach.build_path import load_archetypes
    from hsbg_coach.persistence_prior import report_tier_band
    from .composition_diagnostic import (
        _archetype_eligible,
        _core_set,
        _is_relevant_at_offer,
        _legally_buyable_cores,
        _lobby_tribes,
        _max_core_count,
        _target_meets_view_threshold,
        _winner_for_lobby,
    )

    bands = {
        "tier_le4": {
            "seeded_exposures": 0, "fulfilled": 0, "rejected": 0,
            "reached_2_core_states": 0, "committed_states": 0,
        },
        "tier_5": {
            "seeded_exposures": 0, "fulfilled": 0, "rejected": 0,
            "reached_2_core_states": 0, "committed_states": 0,
        },
        "tier_6": {
            "seeded_exposures": 0, "fulfilled": 0, "rejected": 0,
            "reached_2_core_states": 0, "committed_states": 0,
        },
    }
    view = "seeded_current_target"
    archetypes = load_archetypes()
    lobbies = traces["lobbies"]

    for lobby in range(lobbies):
        winner = _winner_for_lobby(traces, lobby)
        if winner is None:
            continue
        winner_seat = winner["seat"]
        final_target_key = (winner.get("target") or {}).get("archetype_key")
        lobby_tribes = _lobby_tribes(traces, lobby)

        for arch in archetypes:
            if not _archetype_eligible(arch, lobby_tribes):
                continue
            # seeded_current_target eligibility: arch must appear as seeded target
            seeded_keys = set()
            for ev in traces["events"]:
                if ev["lobby"] != lobby or ev["seat"] != winner_seat:
                    continue
                tb = ev.get("target_before")
                if tb and _target_meets_view_threshold(
                        tb, view, ev.get("tavern_tier")):
                    seeded_keys.add(tb.get("archetype_key"))
            if arch.key not in seeded_keys:
                continue

            core = _core_set(arch)
            # Per-generation exposure latch: name -> tier band at open
            active_gen = None
            active_exposures: Dict[str, str] = {}
            fulfilled_in_gen: set = set()
            prev_gen = None

            def close_gen():
                nonlocal active_gen, active_exposures, fulfilled_in_gen
                for name, band in active_exposures.items():
                    if name not in fulfilled_in_gen:
                        bands[band]["rejected"] += 1
                active_gen = None
                active_exposures = {}
                fulfilled_in_gen = set()

            for ev in traces["events"]:
                if ev["lobby"] != lobby or ev["seat"] != winner_seat:
                    continue
                turn = ev["turn"]
                shop_gen = ev.get("shop_generation", 0)
                gen_key = (turn, shop_gen)
                if prev_gen is not None and gen_key != prev_gen:
                    close_gen()
                prev_gen = gen_key

                pre_shop = ev.get("pre_shop") or []
                legal = ev.get("legal_buy_slots") or []
                buyable = _legally_buyable_cores(pre_shop, legal, core)
                target_before = ev.get("target_before")
                tier = int(ev.get("tavern_tier") or 1)
                if buyable and _is_relevant_at_offer(
                        arch, view, target_before, final_target_key, tier):
                    band = report_tier_band(tier)
                    if active_gen != gen_key:
                        close_gen()
                        active_gen = gen_key
                    for name in buyable:
                        if name not in active_exposures:
                            active_exposures[name] = band
                            bands[band]["seeded_exposures"] += 1

                if ev["action"] == "buy" and ev.get("card"):
                    bought = ev["card"]["name"]
                    if bought in active_exposures and bought not in fulfilled_in_gen:
                        band = active_exposures[bought]
                        bands[band]["fulfilled"] += 1
                        fulfilled_in_gen.add(bought)

                if ev["action"] in ("roll", "end"):
                    close_gen()
                    if ev["action"] == "end":
                        prev_gen = None

            close_gen()

            # Progress: first 2-core and committed attribution by tier
            first_2_band = None
            first_committed_band = None
            for ts in traces["turn_summaries"]:
                if ts["lobby"] != lobby or ts["seat"] != winner_seat:
                    continue
                board = ts.get("board_after_recruit") or []
                count = _max_core_count(board, core)
                tier = int(ts.get("tavern_tier") or 1)
                band = report_tier_band(tier)
                tgt = ts.get("target") or {}
                # Match funnel semantics: 2-core is board count of this arch's
                # cores for a seeded-view lobby-archetype state (no live-target
                # filter at the moment of assembly).
                if first_2_band is None and count >= 2:
                    first_2_band = band
                if (first_committed_band is None
                        and _target_meets_view_threshold(
                            tgt, "committed_current_target", tier)
                        and tgt.get("archetype_key") == arch.key):
                    first_committed_band = band
            if first_2_band is not None:
                bands[first_2_band]["reached_2_core_states"] += 1
            if first_committed_band is not None:
                bands[first_committed_band]["committed_states"] += 1

    return bands


def _macro_ok_vs_greedy(greedy_arm: Dict, candidate_arm: Dict) -> bool:
    macro = macro_regression_summary(
        greedy_arm["turn_curves"], candidate_arm["turn_curves"],
        greedy_arm["lobby_dynamics"], candidate_arm["lobby_dynamics"],
        greedy_arm["headline"], candidate_arm["headline"])
    from .phase_2h_decision import _macro_ok
    return _macro_ok(macro, CONFIRMATION_THRESHOLDS)


def _board_sacrifice_ok(policy_stats: Optional[Dict]) -> bool:
    if not policy_stats:
        return True
    th = CONFIRMATION_THRESHOLDS
    mean_rel = policy_stats.get("mean_relative_tempo_loss")
    p95_rel = policy_stats.get("p95_relative_tempo_loss")
    if mean_rel is not None and mean_rel > th["mean_relative_tempo_loss_max"]:
        return False
    if p95_rel is not None and p95_rel > th["p95_relative_tempo_loss_max"]:
        return False
    return True


def _calibration_row(greedy_arm: Dict, candidate_arm: Dict) -> Dict:
    seeded = candidate_arm["mechanism"].get("seeded_current_target") or {}
    committed = candidate_arm["mechanism"].get("committed_current_target") or {}
    ps = candidate_arm.get("policy_stats") or {}
    return {
        "alpha": candidate_arm.get("alpha"),
        "macro_ok": _macro_ok_vs_greedy(greedy_arm, candidate_arm),
        "board_sacrifice_ok": _board_sacrifice_ok(ps),
        "reached_2_core": seeded.get("reached_2_core", 0),
        "committed_states": committed.get("n_lobby_archetype_states", 0),
        "fulfilled_exposures": seeded.get("fulfilled_exposures", 0),
        "coverage_mean": candidate_arm["mechanism"].get(
            "sim_final_winner_coverage_mean", 0.0),
        "action_deviation_rate": candidate_arm.get(
            "action_deviation_rate_vs_greedy"),
        "played_fulfilled": (candidate_arm["lifecycle"].get("funnel") or {}).get(
            "played", 0),
        "mean_relative_tempo_loss": ps.get("mean_relative_tempo_loss"),
        "p95_relative_tempo_loss": ps.get("p95_relative_tempo_loss"),
        "replacement_transitions": ps.get("replacement_transitions"),
        "mean_persistence_weight": ps.get("mean_persistence_weight"),
        "tier_breakdown": candidate_arm.get("tier_breakdown"),
    }


def run_fit_prior(*, out_dir: str = DEFAULT_DIR,
                  require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2J first.")

    t0 = time.time()
    seed = PHASE_2J_FIT_SEED
    lobbies = PHASE_2J_FIT_LOBBIES
    print(f"Phase 2J fit persistence prior — greedy seeds {seed}–"
          f"{seed + lobbies - 1}")
    traces = run_traced_rollouts(lobbies, seed=seed, policy=greedy_policy,
                                 scaling_mode="residual")
    assert_trace_lobby_integrity(traces, lobbies)
    prior = fit_persistence_prior_from_traces(
        traces, fit_seed_base=seed, fit_lobbies=lobbies)
    path = _prior_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    prior.save(path)
    prior_hash = prior.content_hash_sha256()
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "methodology_version": METHODOLOGY_VERSION,
        "implementation_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "runtime_seconds": round(time.time() - t0, 2),
        "fit_seed_base": seed,
        "fit_lobbies": lobbies,
        "prior_path": path,
        "prior_hash_sha256": prior_hash,
        "prior_summary": {
            "n_cells": len(prior.cells),
            "prior_hash_sha256": prior_hash,
            "global_p_survive_1": prior.global_p_survive_1,
            "global_p_survive_2": prior.global_p_survive_2,
            "collapsed_from": prior.collapsed_from,
            "cells": {k: {
                "n": v.n, "p1": round(v.p_survive_1, 4),
                "p2": round(v.p_survive_2, 4),
                "weight": round(v.persistence_weight, 4),
            } for k, v in sorted(prior.cells.items())},
        },
    }
    _write_json(os.path.join(out_dir, "phase_2j_prior_fit.json"), result)
    return result


def run_calibration(*, out_dir: str = DEFAULT_DIR,
                    require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2J first.")

    prior = load_prior(out_dir)
    t0 = time.time()
    print(f"Phase 2J {METHODOLOGY_VERSION} DEV calibration — screen then replication")

    print(f"Screen seeds {PHASE_2J_SCREEN_SEED}–"
          f"{PHASE_2J_SCREEN_SEED + PHASE_2J_SCREEN_LOBBIES - 1}")
    greedy_screen = _run_single_policy_arm(
        PHASE_2J_SCREEN_LOBBIES, PHASE_2J_SCREEN_SEED, greedy_policy, "greedy")
    screen_rows = []
    for alpha in ALPHA_CANDIDATES:
        arm = _run_board_opp_arm(
            PHASE_2J_SCREEN_LOBBIES, PHASE_2J_SCREEN_SEED, alpha, prior,
            f"board_opp_a{alpha}", greedy_baseline_traces=greedy_screen["traces"])
        row = _calibration_row(greedy_screen, arm)
        row["phase"] = "screen"
        screen_rows.append(row)
    screen_rows.sort(key=rank_calibration_candidate)
    top_two = [r["alpha"] for r in screen_rows[:2]
               if r["macro_ok"] and r.get("board_sacrifice_ok", True)]
    if len(top_two) < 2:
        top_two = [r["alpha"] for r in screen_rows[:2]]

    print(f"Replication seeds {PHASE_2J_REPLICATION_SEED}–"
          f"{PHASE_2J_REPLICATION_SEED + PHASE_2J_REPLICATION_LOBBIES - 1}")
    print(f"  top-two α from screen: {top_two}")
    greedy_rep = _run_single_policy_arm(
        PHASE_2J_REPLICATION_LOBBIES, PHASE_2J_REPLICATION_SEED,
        greedy_policy, "greedy")
    replication_rows = []
    for alpha in top_two:
        arm = _run_board_opp_arm(
            PHASE_2J_REPLICATION_LOBBIES, PHASE_2J_REPLICATION_SEED, alpha,
            prior, f"board_opp_a{alpha}",
            greedy_baseline_traces=greedy_rep["traces"])
        row = _calibration_row(greedy_rep, arm)
        row["phase"] = "replication"
        replication_rows.append(row)
    replication_rows.sort(key=rank_calibration_candidate)
    frozen_alpha = replication_rows[0]["alpha"]

    cfg = policy_config_fingerprint(frozen_alpha, prior)
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "methodology_version": METHODOLOGY_VERSION,
        "implementation_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "runtime_seconds": round(time.time() - t0, 2),
        "screen": {
            "greedy_reached_2_core": (
                (greedy_screen["mechanism"].get("seeded_current_target") or {})
                .get("reached_2_core")),
            "greedy_fulfilled": (
                (greedy_screen["mechanism"].get("seeded_current_target") or {})
                .get("fulfilled_exposures")),
            "candidates": screen_rows,
        },
        "replication": {
            "candidates": replication_rows,
            "frozen_alpha": frozen_alpha,
        },
        "frozen_policy_config": cfg,
        "frozen_policy_config_hash_sha256": _policy_hash(cfg),
        "prior_hash_sha256": prior.content_hash_sha256(),
    }
    _write_json(os.path.join(out_dir, "phase_2j_calibration.json"), result)
    return result


def run_confirmation(*, alpha: float, out_dir: str = DEFAULT_DIR,
                     require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2J first.")

    prior = load_prior(out_dir)
    t0 = time.time()
    seed = PHASE_2J_CONFIRM_SEED
    lobbies = PHASE_2J_CONFIRM_LOBBIES
    print(f"Phase 2J {METHODOLOGY_VERSION} confirmation — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print(f"  frozen α = {alpha}")

    greedy = _run_single_policy_arm(lobbies, seed, greedy_policy, "greedy")
    treatment = _run_board_opp_arm(
        lobbies, seed, alpha, prior, "board_opportunity",
        greedy_baseline_traces=greedy["traces"])
    oracle = _run_single_policy_arm(
        lobbies, seed, seeded_core_deploy_stress_greedy_policy, "oracle")

    macro_delta = macro_regression_summary(
        greedy["turn_curves"], treatment["turn_curves"],
        greedy["lobby_dynamics"], treatment["lobby_dynamics"],
        greedy["headline"], treatment["headline"])
    acceptance = evaluate_confirmation_acceptance(
        greedy["mechanism"], treatment["mechanism"],
        greedy["lifecycle"], treatment["lifecycle"],
        macro_delta, oracle_mechanism=oracle["mechanism"],
        policy_stats=treatment.get("policy_stats"))
    decision = evaluate_phase_2j_decision(
        greedy["mechanism"], treatment["mechanism"],
        greedy["lifecycle"], treatment["lifecycle"],
        macro_delta, acceptance)

    cfg = policy_config_fingerprint(alpha, prior)
    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2j_methodology_version": METHODOLOGY_VERSION,
        "phase_2c_methodology_version": PHASE_2C_VERSION,
        "phase_2f_methodology_version": PHASE_2F_VERSION,
        "frozen_alpha": alpha,
        "policy_config": cfg,
        "policy_config_hash_sha256": _policy_hash(cfg),
        "prior_hash_sha256": prior.content_hash_sha256(),
        "working_tree_clean": tree_clean,
    })

    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "methodology_version": METHODOLOGY_VERSION,
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "implementation_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "frozen_alpha": alpha,
        "frozen_policy_config": cfg,
        "runtime_seconds": round(time.time() - t0, 2),
        "evaluation_seed_base": seed,
        "n_lobbies": lobbies,
        "real_final_winner_coverage_mean": real_composition_baseline()[
            "real_final_winner_coverage_mean"],
        "greedy": _confirm_arm(greedy),
        "treatment": _confirm_arm(treatment),
        "oracle_stress_reference": _confirm_arm(oracle),
        "macro_regression_delta_treatment_minus_greedy": macro_delta,
        "acceptance": acceptance,
        "decision": decision,
        "contract": contract,
        "prior_hash_sha256": prior.content_hash_sha256(),
    }
    _write_json(os.path.join(out_dir, "phase_2j_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def _confirm_arm(arm: Dict) -> Dict:
    lc = arm["lifecycle"]
    seeded = arm["mechanism"].get("seeded_current_target") or {}
    committed = arm["mechanism"].get("committed_current_target") or {}
    out = {
        "label": arm["label"],
        "mechanism": arm["mechanism"],
        "lifecycle": {
            "n_fulfilled_purchases": lc["n_fulfilled_purchases"],
            "funnel": lc["funnel"],
            "board_full_summary": lc.get("board_full_summary"),
            "fate_totals": lc.get("fate_totals"),
        },
        "committed_states": committed.get("n_lobby_archetype_states", 0),
        "seeded_reached_2_core": seeded.get("reached_2_core", 0),
        "seeded_reached_4_core": seeded.get("reached_4_core", 0),
        "headline_divergence": arm["headline"],
        "tier_breakdown": arm.get("tier_breakdown"),
    }
    if arm.get("policy_stats") is not None:
        out["policy_stats"] = arm["policy_stats"]
        out["action_deviation_rate_vs_greedy"] = arm.get(
            "action_deviation_rate_vs_greedy")
    if "alpha" in arm:
        out["alpha"] = arm["alpha"]
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",
                    choices=("fit-prior", "calibrate", "confirm", "full"),
                    default="full", nargs="?")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Frozen α for confirm-only run")
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2J {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")

    try:
        if args.command == "fit-prior":
            run_fit_prior(out_dir=args.out_dir,
                          require_clean_tree=not args.allow_dirty_tree)
        elif args.command == "calibrate":
            run_calibration(out_dir=args.out_dir,
                            require_clean_tree=not args.allow_dirty_tree)
        elif args.command == "confirm":
            alpha = args.alpha
            if alpha is None:
                cal_path = os.path.join(args.out_dir, "phase_2j_calibration.json")
                if os.path.isfile(cal_path):
                    with open(cal_path, encoding="utf-8") as f:
                        alpha = json.load(f)["replication"]["frozen_alpha"]
                else:
                    print("ERROR: --alpha required or run calibrate first",
                          file=sys.stderr)
                    return 1
            result = run_confirmation(
                alpha=alpha, out_dir=args.out_dir,
                require_clean_tree=not args.allow_dirty_tree)
            print(f"\nDecision: {result['decision']['decision_branch']}")
            print(f"  {result['decision']['recommended_next_step']}")
            print(f"  accept={result['acceptance']['flags']['accept_phase_2j_policy']}")
        else:
            run_fit_prior(out_dir=args.out_dir,
                          require_clean_tree=not args.allow_dirty_tree)
            result = run_calibration(
                out_dir=args.out_dir,
                require_clean_tree=not args.allow_dirty_tree)
            alpha = result["replication"]["frozen_alpha"]
            print(f"\nFrozen α = {alpha}")
            print("Commit calibration artifacts, then run:")
            print("  python -m ml.fidelity_phase_2j confirm")
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
