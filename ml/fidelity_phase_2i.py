"""Simulator Fidelity Phase 2I — seeded opportunity decision-margin diagnostic.

    python -m ml.fidelity_phase_2i

Measurement-only on DEV seeds 3000–3499, frozen Phase 2H v3 policy λ_build=12.
No policy behavior changes; observational audit only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from hsbg_coach.bg_env import BGEnv
from hsbg_coach.tempo_board_policy import (
    PHASE_2H_REPLICATION_LOBBIES,
    PHASE_2H_REPLICATION_SEED,
    PHASE_2H_SCREEN_LOBBIES,
    PHASE_2H_SCREEN_SEED,
    METHODOLOGY_VERSION as PHASE_2H_VERSION,
    policies_for_lobby,
    policy_config_fingerprint,
)
from hsbg_coach.tempo_margin_audit import TempoMarginAuditCollector

from .composition_trace import RecruitTracer, board_fingerprint
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .phase_2i_decision import evaluate_phase_2i_decision
from .seeded_margin_diagnostic import (METHODOLOGY_VERSION,
                                       analyze_margin_exposures)

DEFAULT_DIR = "results/sim_fidelity_phase_2i"
PHASE = "2I seeded opportunity decision-margin diagnostic"
PHASE_2I_SEED = PHASE_2H_SCREEN_SEED
PHASE_2I_LOBBIES = PHASE_2H_SCREEN_LOBBIES + PHASE_2H_REPLICATION_LOBBIES
FROZEN_LAMBDA = 12.0


class MarginAuditTracer(RecruitTracer):
    """RecruitTracer that injects audit context and links events to audit indices."""

    def __init__(self, lobby_id: int, seed: int, policies: List):
        super().__init__(lobby_id, seed)
        self.policies = policies
        self.audit_event_links: List[Optional[int]] = []

    def before_action(self, seat: int, turn: int, shop_generation: int,
                      obs: Dict, mask: List[bool]) -> None:
        super().before_action(seat, turn, shop_generation, obs, mask)
        if seat < len(self.policies):
            p = self.policies[seat]
            if hasattr(p, "set_audit_context"):
                p.set_audit_context(
                    lobby=self.lobby_id, seat=seat, turn=turn,
                    shop_generation=shop_generation)

    def after_action(self, seat: int, turn: int, shop_generation: int,
                     action: int, ended: bool, player=None) -> None:
        audit_idx = None
        if seat < len(self.policies):
            audit_idx = getattr(self.policies[seat], "last_audit_index", None)
        self.audit_event_links.append(audit_idx)
        super().after_action(seat, turn, shop_generation, action, ended, player)


def run_margin_audit_rollouts(lobbies: int, seed: int, lambda_build: float,
                              scaling_mode: str = "residual") -> Dict:
    """Traced rollouts with per-lobby fresh policies and margin audit."""
    audit = TempoMarginAuditCollector()
    all_events: List[Dict] = []
    all_turn_summaries: List[Dict] = []
    all_player_finals: List[Dict] = []
    lobby_meta: List[Dict] = []
    all_links: List[Optional[int]] = []

    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        policies = policies_for_lobby(lambda_build, 8, audit=audit)
        env = BGEnv(seed=lobby_seed, scaling_mode=scaling_mode)
        tracer = MarginAuditTracer(lobby_i, lobby_seed, list(policies))
        env.play_scripted(list(policies), recruit_tracer=tracer)
        game_length = env.turn
        for pf in tracer.player_finals:
            pf["game_length"] = game_length
        all_events.extend(tracer.events)
        all_turn_summaries.extend(tracer.turn_summaries)
        all_player_finals.extend(tracer.player_finals)
        all_links.extend(tracer.audit_event_links)
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
    return {"traces": traces, "audit": audit, "audit_event_links": all_links}


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_phase_2i(*, lobbies: int = PHASE_2I_LOBBIES,
                 seed: int = PHASE_2I_SEED,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2I first.")

    t0 = time.time()
    print(f"Phase 2I {METHODOLOGY_VERSION} — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}, λ={FROZEN_LAMBDA}")

    rollout = run_margin_audit_rollouts(lobbies, seed, FROZEN_LAMBDA)
    analysis = analyze_margin_exposures(
        rollout["traces"], rollout["audit"], rollout["audit_event_links"])
    decision = evaluate_phase_2i_decision(analysis)

    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2i_methodology_version": METHODOLOGY_VERSION,
        "phase_2h_methodology_version": PHASE_2H_VERSION,
        "frozen_lambda_build": FROZEN_LAMBDA,
        "policy_config": policy_config_fingerprint(FROZEN_LAMBDA),
        "measurement_only": True,
        "note": "Observational audit; no policy behavior changes.",
    })

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
        "frozen_lambda_build": FROZEN_LAMBDA,
        "analysis": {
            "funnel": analysis["funnel"],
            "reconciliation_2c_v3": analysis["reconciliation_2c_v3"],
            "headline_metrics": analysis["headline_metrics"],
            "break_even_lambda": analysis["break_even_lambda"],
            "breakdown_by_tier": analysis["breakdown_by_tier"],
            "breakdown_by_core_have": analysis["breakdown_by_core_have"],
            "breakdown_by_board_full": analysis["breakdown_by_board_full"],
            "breakdown_by_archetype": analysis["breakdown_by_archetype"],
        },
        "decision": decision,
        "contract": contract,
        "rejected_exposure_records": analysis["rejected_exposure_records"],
    }

    _write_json(os.path.join(out_dir, "phase_2i_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=PHASE_2I_LOBBIES)
    ap.add_argument("--seed", type=int, default=PHASE_2I_SEED)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2I {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")

    try:
        result = run_phase_2i(
            lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    dec = result["decision"]
    funnel = result["analysis"]["funnel"]
    print(f"\nExposures: {funnel['seeded_legally_buyable_exposures']} "
          f"(fulfilled={funnel['fulfilled']}, rejected={funnel['rejected']})")
    print(f"2c reconciliation: {result['analysis']['reconciliation_2c_v3']['counts_match']}")
    print(f"Decision: {dec['decision_branch']}")
    print(f"  {dec['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
