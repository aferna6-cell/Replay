"""Simulator Fidelity Phase 2H — tempo-aware board management (methodology 2h_v3).

    python -m ml.fidelity_phase_2h calibrate
    python -m ml.fidelity_phase_2h confirm --lambda-build 8
    python -m ml.fidelity_phase_2h full   # calibration only; commit before confirm

DEV calibration on seeds 3000–3499; frozen confirmation on 6000–6199.
Invalidated: 4000–4199 (v1), 5000–5199 (v2) under invalidated_v1/ and invalidated_v2/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

from hsbg_coach.bg_env import BGEnv, greedy_policy, seeded_core_deploy_stress_greedy_policy
from hsbg_coach.pace import board_stats
from hsbg_coach.tempo_board_policy import (
    LAMBDA_BUILD_CANDIDATES,
    METHODOLOGY_VERSION,
    PHASE_2H_CONFIRM_LOBBIES,
    PHASE_2H_CONFIRM_SEED,
    PHASE_2H_INVALIDATED_V1_CONFIRM_LOBBIES,
    PHASE_2H_INVALIDATED_V1_CONFIRM_SEED,
    PHASE_2H_INVALIDATED_V2_CONFIRM_LOBBIES,
    PHASE_2H_INVALIDATED_V2_CONFIRM_SEED,
    PHASE_2H_REPLICATION_LOBBIES,
    PHASE_2H_REPLICATION_SEED,
    PHASE_2H_SCREEN_LOBBIES,
    PHASE_2H_SCREEN_SEED,
    aggregate_policy_stats,
    policies_for_lobby,
    policy_config_fingerprint,
)

from .composition_diagnostic import (METHODOLOGY_VERSION as PHASE_2C_VERSION,
                                     aggregate_diagnostics)
from .composition_trace import RecruitTracer, run_traced_rollouts
from .core_lifecycle_diagnostic import (METHODOLOGY_VERSION as PHASE_2F_VERSION,
                                        analyze_core_lifecycles)
from .fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                               aggregate_turn_curves, real_composition_baseline,
                               run_fidelity_rollouts, summarize_divergence)
from .fidelity_paired import per_lobby_turn_means
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .phase_2d_acceptance import (composition_mechanism_summary,
                                  macro_regression_summary)
from .phase_2h_decision import (evaluate_confirmation_acceptance,
                                evaluate_phase_2h_decision,
                                rank_calibration_candidate)

DEFAULT_DIR = "results/sim_fidelity_phase_2h"
INVALIDATED_V1_DIR = "results/sim_fidelity_phase_2h/invalidated_v1"
INVALIDATED_V2_DIR = "results/sim_fidelity_phase_2h/invalidated_v2"
PHASE = "2H tempo-aware board management"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_hash(cfg: Dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def assert_trace_lobby_integrity(traces: Dict, lobbies: int) -> None:
    """Ensure every lobby 0..lobbies-1 appears in trace aggregates."""
    expected = set(range(lobbies))
    event_lobbies = {e["lobby"] for e in traces.get("events", [])}
    final_lobbies = {p["lobby"] for p in traces.get("player_finals", [])}
    turn_lobbies = {t["lobby"] for t in traces.get("turn_summaries", [])}
    meta_lobbies = {m["lobby"] for m in traces.get("lobby_meta", [])}
    assert event_lobbies == expected, (
        f"event lobby IDs {sorted(event_lobbies)} != {sorted(expected)}")
    assert final_lobbies == expected, (
        f"player_final lobby IDs {sorted(final_lobbies)} != {sorted(expected)}")
    assert turn_lobbies == expected, (
        f"turn_summary lobby IDs {sorted(turn_lobbies)} != {sorted(expected)}")
    assert meta_lobbies == expected, (
        f"lobby_meta IDs {sorted(meta_lobbies)} != {sorted(expected)}")
    assert traces.get("lobbies") == lobbies


def compute_action_deviation_rate(base_traces: Dict, alt_traces: Dict) -> float:
    """Fraction of recruit actions (all seats) that differ vs baseline."""
    lobbies = base_traces["lobbies"]
    total = 0
    diff = 0
    for lobby in range(lobbies):
        for seat in range(8):
            base_ev = [e for e in base_traces["events"]
                       if e["lobby"] == lobby and e["seat"] == seat]
            alt_ev = [e for e in alt_traces["events"]
                      if e["lobby"] == lobby and e["seat"] == seat]
            n = min(len(base_ev), len(alt_ev))
            for i in range(n):
                total += 1
                if base_ev[i]["action"] != alt_ev[i]["action"]:
                    diff += 1
    return diff / total if total else 0.0


def run_traced_rollouts_policy_list(
        lobbies: int, seed: int, policies: Sequence[Callable],
        scaling_mode: str = "residual") -> Dict:
    """Traced rollouts with per-seat policy list (one shared list — legacy)."""
    return run_traced_rollouts_tempo(
        lobbies, seed, policies[0].lambda_build, scaling_mode=scaling_mode)


def run_traced_rollouts_tempo(
        lobbies: int, seed: int, lambda_build: float,
        scaling_mode: str = "residual") -> Dict:
    """Traced rollouts with fresh policy instances per lobby."""
    all_events: List[Dict] = []
    all_turn_summaries: List[Dict] = []
    all_player_finals: List[Dict] = []
    lobby_meta: List[Dict] = []

    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        policies = policies_for_lobby(lambda_build, 8)
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


def run_fidelity_rollouts_policy_list(
        lobbies: int, seed: int, policies: Sequence[Callable],
        scaling_mode: str = "residual") -> List[Dict]:
    """Fidelity rollouts with one shared policy list (legacy)."""
    return run_fidelity_rollouts_tempo(
        lobbies, seed, policies[0].lambda_build,
        scaling_mode=scaling_mode)[0]


def run_fidelity_rollouts_tempo(
        lobbies: int, seed: int, lambda_build: float,
        scaling_mode: str = "residual"
        ) -> tuple[List[Dict], List]:
    """Fidelity rollouts with fresh policy instances per lobby."""
    rows: List[Dict] = []
    all_policies = []
    for i in range(lobbies):
        policies = policies_for_lobby(lambda_build, 8)
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


def _run_tempo_arm(lobbies: int, seed: int, lambda_build: float,
                   label: str, *, greedy_baseline_traces: Optional[Dict] = None,
                   collect_policy_stats: bool = True) -> Dict:
    print(f"  [{label}] λ={lambda_build} fidelity rollouts…")
    rows, policies_fidelity = run_fidelity_rollouts_tempo(
        lobbies, seed, lambda_build, scaling_mode="residual")
    policy_stats = (aggregate_policy_stats(policies_fidelity)
                    if collect_policy_stats else None)

    print(f"  [{label}] λ={lambda_build} composition traces…")
    traces = run_traced_rollouts_tempo(
        lobbies, seed, lambda_build, scaling_mode="residual")

    diagnostic = aggregate_diagnostics(traces)
    lifecycle = analyze_core_lifecycles(traces)
    deviation = None
    if greedy_baseline_traces is not None:
        deviation = compute_action_deviation_rate(greedy_baseline_traces, traces)

    turn_curves = aggregate_turn_curves(rows)
    return {
        "label": label,
        "lambda_build": lambda_build,
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
    }


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


def _macro_ok_vs_greedy(greedy_arm: Dict, candidate_arm: Dict) -> bool:
    macro = macro_regression_summary(
        greedy_arm["turn_curves"], candidate_arm["turn_curves"],
        greedy_arm["lobby_dynamics"], candidate_arm["lobby_dynamics"],
        greedy_arm["headline"], candidate_arm["headline"])
    from .phase_2h_decision import _macro_ok, CONFIRMATION_THRESHOLDS
    return _macro_ok(macro, CONFIRMATION_THRESHOLDS)


def _calibration_row(greedy_arm: Dict, candidate_arm: Dict) -> Dict:
    seeded = candidate_arm["mechanism"].get("seeded_current_target") or {}
    committed = candidate_arm["mechanism"].get("committed_current_target") or {}
    return {
        "lambda_build": candidate_arm.get("lambda_build"),
        "macro_ok": _macro_ok_vs_greedy(greedy_arm, candidate_arm),
        "reached_2_core": seeded.get("reached_2_core", 0),
        "committed_states": committed.get("n_lobby_archetype_states", 0),
        "coverage_mean": candidate_arm["mechanism"].get(
            "sim_final_winner_coverage_mean", 0.0),
        "action_deviation_rate": candidate_arm.get(
            "action_deviation_rate_vs_greedy"),
        "fulfilled_exposures": seeded.get("fulfilled_exposures", 0),
        "played_fulfilled": (candidate_arm["lifecycle"].get("funnel") or {}).get(
            "played", 0),
    }


def _preserve_invalidated_v2_artifacts(out_dir: str) -> None:
    """Archive invalidated v2 confirmation before overwriting."""
    inv_dir = INVALIDATED_V2_DIR
    os.makedirs(inv_dir, exist_ok=True)
    note = {
        "status": "invalidated",
        "methodology_version": "2h_v2",
        "reason": (
            "Incorrect shop full-board compound transition (buy before sell); "
            "cross-lobby policy state reuse; telemetry mislabeled core actions; "
            "confirmation run with working_tree_clean=false."),
        "confirmation_seeds": (
            f"{PHASE_2H_INVALIDATED_V2_CONFIRM_SEED}–"
            f"{PHASE_2H_INVALIDATED_V2_CONFIRM_SEED + PHASE_2H_INVALIDATED_V2_CONFIRM_LOBBIES - 1}"),
        "do_not_use_for_decisions": True,
    }
    _write_json(os.path.join(inv_dir, "invalidated_note.json"), note)
    for name in ("phase_2h_report.json", "contract.json"):
        src = os.path.join(out_dir, name)
        if os.path.isfile(src):
            dst = os.path.join(inv_dir, name)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)


def _preserve_invalidated_v1_artifacts(out_dir: str) -> None:
    """Move pre-v2 confirmation artifacts without deleting history."""
    inv_dir = INVALIDATED_V1_DIR
    os.makedirs(inv_dir, exist_ok=True)
    note = {
        "status": "invalidated",
        "methodology_version": "2h_v1",
        "reason": (
            "Treatment trace lobby-ID collapse (all lobbies recorded as lobby=0) "
            "and incomplete compound sell→play/buy transition semantics."),
        "confirmation_seeds": (
            f"{PHASE_2H_INVALIDATED_V1_CONFIRM_SEED}–"
            f"{PHASE_2H_INVALIDATED_V1_CONFIRM_SEED + PHASE_2H_INVALIDATED_V1_CONFIRM_LOBBIES - 1}"),
        "do_not_use_for_decisions": True,
    }
    _write_json(os.path.join(inv_dir, "invalidated_note.json"), note)
    for name in ("phase_2h_report.json", "contract.json", "phase_2h_calibration.json"):
        src = os.path.join(out_dir, name)
        if os.path.isfile(src):
            dst = os.path.join(inv_dir, name)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)


def run_calibration(*, out_dir: str = DEFAULT_DIR,
                    require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2H first.")

    _preserve_invalidated_v1_artifacts(out_dir)
    _preserve_invalidated_v2_artifacts(out_dir)
    t0 = time.time()
    print(f"Phase 2H {METHODOLOGY_VERSION} DEV calibration — screen then replication")

    print(f"Screen seeds {PHASE_2H_SCREEN_SEED}–"
          f"{PHASE_2H_SCREEN_SEED + PHASE_2H_SCREEN_LOBBIES - 1}")
    greedy_screen = _run_single_policy_arm(
        PHASE_2H_SCREEN_LOBBIES, PHASE_2H_SCREEN_SEED, greedy_policy, "greedy")
    screen_rows = []
    for lb in LAMBDA_BUILD_CANDIDATES:
        arm = _run_tempo_arm(
            PHASE_2H_SCREEN_LOBBIES, PHASE_2H_SCREEN_SEED, lb,
            f"tempo_lb{lb}", greedy_baseline_traces=greedy_screen["traces"])
        row = _calibration_row(greedy_screen, arm)
        row["phase"] = "screen"
        screen_rows.append(row)
    screen_rows.sort(key=rank_calibration_candidate)
    top_two = [r["lambda_build"] for r in screen_rows[:2] if r["macro_ok"]]
    if len(top_two) < 2:
        top_two = [r["lambda_build"] for r in screen_rows[:2]]

    print(f"Replication seeds {PHASE_2H_REPLICATION_SEED}–"
          f"{PHASE_2H_REPLICATION_SEED + PHASE_2H_REPLICATION_LOBBIES - 1}")
    print(f"  top-two λ from screen: {top_two}")
    greedy_rep = _run_single_policy_arm(
        PHASE_2H_REPLICATION_LOBBIES, PHASE_2H_REPLICATION_SEED,
        greedy_policy, "greedy")
    replication_rows = []
    for lb in top_two:
        arm = _run_tempo_arm(
            PHASE_2H_REPLICATION_LOBBIES, PHASE_2H_REPLICATION_SEED, lb,
            f"tempo_lb{lb}", greedy_baseline_traces=greedy_rep["traces"])
        row = _calibration_row(greedy_rep, arm)
        row["phase"] = "replication"
        replication_rows.append(row)
    replication_rows.sort(key=rank_calibration_candidate)
    frozen_lambda = replication_rows[0]["lambda_build"]

    cfg = policy_config_fingerprint(frozen_lambda)
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "methodology_version": METHODOLOGY_VERSION,
        "implementation_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "runtime_seconds": round(time.time() - t0, 2),
        "screen": {"candidates": screen_rows},
        "replication": {
            "candidates": replication_rows,
            "frozen_lambda_build": frozen_lambda,
        },
        "frozen_policy_config": cfg,
        "frozen_policy_config_hash_sha256": _policy_hash(cfg),
        "invalidated_v1_note": (
            "Prior confirmation on seeds 4000–4199 invalidated; see invalidated_v1/"),
        "invalidated_v2_note": (
            "Prior confirmation on seeds 5000–5199 invalidated; see invalidated_v2/"),
    }
    _write_json(os.path.join(out_dir, "phase_2h_calibration.json"), result)
    return result


def run_confirmation(*, lambda_build: float,
                     out_dir: str = DEFAULT_DIR,
                     require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2H first.")

    _preserve_invalidated_v1_artifacts(out_dir)
    _preserve_invalidated_v2_artifacts(out_dir)
    t0 = time.time()
    seed = PHASE_2H_CONFIRM_SEED
    lobbies = PHASE_2H_CONFIRM_LOBBIES
    print(f"Phase 2H {METHODOLOGY_VERSION} confirmation — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print(f"  frozen λ_build = {lambda_build}")

    greedy = _run_single_policy_arm(lobbies, seed, greedy_policy, "greedy")
    treatment = _run_tempo_arm(
        lobbies, seed, lambda_build, "tempo_board",
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
        macro_delta, oracle_mechanism=oracle["mechanism"])
    decision = evaluate_phase_2h_decision(
        greedy["mechanism"], treatment["mechanism"],
        greedy["lifecycle"], treatment["lifecycle"],
        macro_delta, acceptance)

    cfg = policy_config_fingerprint(lambda_build)
    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2h_methodology_version": METHODOLOGY_VERSION,
        "phase_2c_methodology_version": PHASE_2C_VERSION,
        "phase_2f_methodology_version": PHASE_2F_VERSION,
        "frozen_lambda_build": lambda_build,
        "policy_config": cfg,
        "policy_config_hash_sha256": _policy_hash(cfg),
        "working_tree_clean": tree_clean,
    })

    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "methodology_version": METHODOLOGY_VERSION,
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "implementation_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "frozen_lambda_build": lambda_build,
        "frozen_policy_config": cfg,
        "runtime_seconds": round(time.time() - t0, 2),
        "evaluation_seed_base": seed,
        "n_lobbies": lobbies,
        "real_final_winner_coverage_mean": real_composition_baseline()[
            "real_final_winner_coverage_mean"],
        "greedy": _confirm_arm(greedy),
        "treatment": _confirm_arm(treatment),
        "oracle_upper_bound": _confirm_arm(oracle),
        "macro_regression_delta_treatment_minus_greedy": macro_delta,
        "acceptance": acceptance,
        "decision": decision,
        "contract": contract,
        "invalidated_v1_confirmation": (
            f"seeds {PHASE_2H_INVALIDATED_V1_CONFIRM_SEED}–"
            f"{PHASE_2H_INVALIDATED_V1_CONFIRM_SEED + PHASE_2H_INVALIDATED_V1_CONFIRM_LOBBIES - 1} "
            "preserved under invalidated_v1/ — do not use"),
        "invalidated_v2_confirmation": (
            f"seeds {PHASE_2H_INVALIDATED_V2_CONFIRM_SEED}–"
            f"{PHASE_2H_INVALIDATED_V2_CONFIRM_SEED + PHASE_2H_INVALIDATED_V2_CONFIRM_LOBBIES - 1} "
            "preserved under invalidated_v2/ — do not use"),
    }
    _write_json(os.path.join(out_dir, "phase_2h_report.json"), result)
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
    }
    if arm.get("policy_stats") is not None:
        out["policy_stats"] = arm["policy_stats"]
        out["action_deviation_rate_vs_greedy"] = arm.get(
            "action_deviation_rate_vs_greedy")
    if "lambda_build" in arm:
        out["lambda_build"] = arm["lambda_build"]
    return out


def run_phase_2h_full(*, out_dir: str = DEFAULT_DIR,
                      require_clean_tree: bool = True) -> Dict:
    """Run DEV calibration. Commit artifacts before ``confirm``."""
    return {"calibration": run_calibration(
        out_dir=out_dir, require_clean_tree=require_clean_tree)}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("calibrate", "confirm", "full"),
                    default="full", nargs="?")
    ap.add_argument("--lambda-build", type=float, default=None,
                    help="Frozen λ for confirm-only run")
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2H {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")

    try:
        if args.command == "calibrate":
            run_calibration(out_dir=args.out_dir,
                            require_clean_tree=not args.allow_dirty_tree)
        elif args.command == "confirm":
            lb = args.lambda_build
            if lb is None:
                cal_path = os.path.join(args.out_dir, "phase_2h_calibration.json")
                if os.path.isfile(cal_path):
                    with open(cal_path, encoding="utf-8") as f:
                        lb = json.load(f)["replication"]["frozen_lambda_build"]
                else:
                    print("ERROR: --lambda-build required or run calibrate first",
                          file=sys.stderr)
                    return 1
            result = run_confirmation(
                lambda_build=lb, out_dir=args.out_dir,
                require_clean_tree=not args.allow_dirty_tree)
            print(f"\nDecision: {result['decision']['decision_branch']}")
            print(f"  {result['decision']['recommended_next_step']}")
            print(f"  accept={result['acceptance']['flags']['accept_phase_2h_policy']}")
        else:
            result = run_phase_2h_full(
                out_dir=args.out_dir,
                require_clean_tree=not args.allow_dirty_tree)
            lb = result["calibration"]["replication"]["frozen_lambda_build"]
            print(f"\nFrozen λ_build = {lb}")
            print("Commit calibration artifacts, then run:")
            print("  python -m ml.fidelity_phase_2h confirm")
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
