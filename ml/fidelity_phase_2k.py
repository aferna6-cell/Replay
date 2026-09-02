"""Simulator Fidelity Phase 2K — post-assembly residual composition-gap diagnostic.

    python -m ml.fidelity_phase_2k

Frozen Phase 2J policy (α=0.5, prior_hash=9b31c93a…). DEV seeds 9000–9499.
Measurement-only. Does not reuse 8000–8199.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from hsbg_coach.board_opportunity_policy import (
    METHODOLOGY_VERSION as PHASE_2J_VERSION,
    policies_for_lobby,
)
from hsbg_coach.bg_env import BGEnv
from hsbg_coach.persistence_prior import PersistencePrior

from .composition_trace import RecruitTracer, board_fingerprint
from .fidelity_phase_2h import assert_trace_lobby_integrity
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .phase_2k_decision import evaluate_phase_2k_decision
from .post_assembly_gap_diagnostic import (
    FORBIDDEN_CONFIRM_LOBBIES,
    FORBIDDEN_CONFIRM_SEED,
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    METHODOLOGY_VERSION,
    PHASE_2J_PRIOR_PATH,
    PHASE_2K_EXPAND_THROUGH,
    PHASE_2K_LOBBIES,
    PHASE_2K_MIN_STATES,
    PHASE_2K_SEED,
    analyze_post_assembly_gap,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2k"
PHASE = "2K post-assembly residual composition-gap diagnostic"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_frozen_prior(path: str = PHASE_2J_PRIOR_PATH) -> PersistencePrior:
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing Phase 2J prior at {path}")
    prior = PersistencePrior.load(path)
    h = prior.content_hash_sha256()
    if h != FROZEN_PRIOR_HASH:
        raise RuntimeError(
            f"Prior hash mismatch: got {h}, expected {FROZEN_PRIOR_HASH}")
    return prior


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """Reject Phase 2J confirmation seeds by default."""
    forbidden_lo = FORBIDDEN_CONFIRM_SEED
    forbidden_hi = FORBIDDEN_CONFIRM_SEED + FORBIDDEN_CONFIRM_LOBBIES - 1
    run_lo, run_hi = seed, seed + lobbies - 1
    if not (run_hi < forbidden_lo or run_lo > forbidden_hi):
        raise RuntimeError(
            f"Phase 2K default runner rejects seeds overlapping "
            f"{forbidden_lo}–{forbidden_hi} (Phase 2J confirmation). "
            f"Requested {run_lo}–{run_hi}.")


def run_board_opp_traced(lobbies: int, seed: int, prior: PersistencePrior,
                         alpha: float = FROZEN_ALPHA) -> Dict:
    all_events: List[Dict] = []
    all_turn_summaries: List[Dict] = []
    all_player_finals: List[Dict] = []
    lobby_meta: List[Dict] = []

    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        policies = policies_for_lobby(alpha, prior, 8)
        env = BGEnv(seed=lobby_seed, scaling_mode="residual")
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
        "scaling_mode": "residual",
        "events": all_events,
        "turn_summaries": all_turn_summaries,
        "player_finals": all_player_finals,
        "lobby_meta": lobby_meta,
    }
    assert_trace_lobby_integrity(traces, lobbies)
    return traces


def finals_fingerprint(lobbies: int, seed: int, prior: PersistencePrior, *,
                       with_tracer: bool) -> list:
    rows = []
    if with_tracer:
        traces = run_board_opp_traced(lobbies, seed, prior)
        for pf in traces["player_finals"]:
            rows.append((
                pf["lobby"], pf["seat"], pf.get("placement"),
                board_fingerprint(pf.get("final_board") or []),
            ))
    else:
        for i in range(lobbies):
            policies = policies_for_lobby(FROZEN_ALPHA, prior, 8)
            env = BGEnv(seed=seed + i, scaling_mode="residual")
            env.play_scripted(list(policies))
            for seat, p in enumerate(env.players):
                board = [{"name": m.name, "attack": m.attack,
                          "health": m.health, "golden": m.golden}
                         for m in p.board]
                rows.append((i, seat, p.placement, board_fingerprint(board)))
            del env
    return sorted(rows)


def run_phase_2k(*, seed: int = PHASE_2K_SEED,
                 lobbies: int = PHASE_2K_LOBBIES,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True,
                 allow_expand: bool = True) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2K first.")

    prior = load_frozen_prior()
    t0 = time.time()
    print(f"Phase 2K {METHODOLOGY_VERSION} — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print(f"  frozen α={FROZEN_ALPHA}, prior_hash={FROZEN_PRIOR_HASH[:12]}…")

    traces = run_board_opp_traced(lobbies, seed, prior)
    analysis = analyze_post_assembly_gap(traces)
    expanded = False
    if allow_expand and analysis["n_states"] < PHASE_2K_MIN_STATES:
        # Extend through PHASE_2K_EXPAND_THROUGH
        extra_hi = PHASE_2K_EXPAND_THROUGH
        extra_lo = seed + lobbies
        if extra_lo <= extra_hi:
            extra_n = extra_hi - extra_lo + 1
            print(f"  adaptive expand: {analysis['n_states']} < {PHASE_2K_MIN_STATES}; "
                  f"adding seeds {extra_lo}–{extra_hi}")
            assert_seed_range_allowed(extra_lo, extra_n)
            traces2 = run_board_opp_traced(extra_n, extra_lo, prior)
            # Merge traces with remapped lobby ids
            offset = lobbies
            for ev in traces2["events"]:
                ev["lobby"] += offset
            for ts in traces2["turn_summaries"]:
                ts["lobby"] += offset
            for pf in traces2["player_finals"]:
                pf["lobby"] += offset
            for m in traces2["lobby_meta"]:
                m["lobby"] += offset
            traces["events"].extend(traces2["events"])
            traces["turn_summaries"].extend(traces2["turn_summaries"])
            traces["player_finals"].extend(traces2["player_finals"])
            traces["lobby_meta"].extend(traces2["lobby_meta"])
            traces["lobbies"] = lobbies + extra_n
            analysis = analyze_post_assembly_gap(traces)
            expanded = True
            lobbies = traces["lobbies"]

    decision = evaluate_phase_2k_decision(analysis)
    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2k_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "measurement_only": True,
        "forbidden_seed_range": (
            f"{FORBIDDEN_CONFIRM_SEED}–"
            f"{FORBIDDEN_CONFIRM_SEED + FORBIDDEN_CONFIRM_LOBBIES - 1}"),
        "note": "Observational post-assembly gap accounting; no policy changes.",
    })

    # Drop bulky per-card detail from top-level report copy if huge — keep records
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "methodology_version": METHODOLOGY_VERSION,
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "implementation_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "runtime_seconds": round(time.time() - t0, 2),
        "evaluation_seed_base": seed,
        "n_lobbies": lobbies,
        "adaptive_expanded": expanded,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "analysis": {
            "n_states": analysis["n_states"],
            "funnel": analysis["funnel"],
            "distribution_summary": analysis["distribution_summary"],
            "weighted_funnel": analysis["weighted_funnel"],
            "missing_coverage_mass_by_cause": (
                analysis["missing_coverage_mass_by_cause"]),
            "missing_coverage_mass_share_by_cause": (
                analysis["missing_coverage_mass_share_by_cause"]),
            "missing_coverage_card_counts_by_cause": (
                analysis["missing_coverage_card_counts_by_cause"]),
            "mass_reconciliation": analysis["mass_reconciliation"],
            "dominant_cause": analysis["dominant_cause"],
            "dominant_share": analysis["dominant_share"],
        },
        "decision": decision,
        "contract": contract,
        "state_records": analysis["state_records"],
    }
    _write_json(os.path.join(out_dir, "phase_2k_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=PHASE_2K_SEED)
    ap.add_argument("--lobbies", type=int, default=PHASE_2K_LOBBIES)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    ap.add_argument("--no-expand", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2K {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")

    try:
        result = run_phase_2k(
            seed=args.seed, lobbies=args.lobbies, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree,
            allow_expand=not args.no_expand)
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    a = result["analysis"]
    d = result["decision"]
    print(f"\nPost-assembly states: {a['n_states']}")
    print(f"Dominant missing-mass cause: {a.get('dominant_cause')} "
          f"({a.get('dominant_share')})")
    print(f"Decision: {d['decision_branch']}")
    print(f"  {d['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
