"""Replay Simulator Fidelity Benchmark v1 — measure sim vs real divergence.

    python -m ml.fidelity_benchmark
    python -m ml.fidelity_benchmark --lobbies 500 --json-out results/sim_fidelity_v1/baseline.json

This is NOT Replay Benchmark v1 (agent strength). It quantifies where
Simulator v1 diverges from Firestone reference curves before any env changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from .fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                               aggregate_turn_curves, run_fidelity_rollouts,
                               summarize_divergence)
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 build_simulator_v1_contract)

DEFAULT_OUT = "results/sim_fidelity_v1/baseline.json"
DEFAULT_CONTRACT = "results/sim_fidelity_v1/contract.json"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_benchmark(*, lobbies: int = 200, seed: int = 0) -> Dict:
    t0 = time.time()
    contract = build_simulator_v1_contract(evaluation_seed=seed, lobbies=lobbies)
    rows = run_fidelity_rollouts(lobbies, seed=seed)
    turn_curves = aggregate_turn_curves(rows)
    composition = aggregate_composition(rows)
    lobby = aggregate_lobby_dynamics(rows)
    headline = summarize_divergence(turn_curves)

    # Strip board payloads from stored rows count only
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "question": "Where does Simulator v1 diverge from real Battlegrounds?",
        "contract": contract,
        "runtime_seconds": round(time.time() - t0, 2),
        "headline_divergence": headline,
        "turn_curves": turn_curves,
        "composition": composition,
        "lobby_dynamics": lobby,
        "combat": {
            "measured_in_v1": False,
            "prior_spot_check_note": (
                "Historical work: built-in combat matched Firestone in ~29/30 "
                "sampled fights. Re-benchmark separately if combat code changes."
            ),
        },
        "shop_recruit": {
            "measured_in_v1": False,
            "note": "Shop offer quality and recruit mix require trajectory logs; deferred.",
        },
    }
    return result


def print_table(turn_curves: Dict[str, Dict]) -> None:
    print(f"\n{'Turn':>4} {'Real stats':>11} {'Sim stats':>11} {'Ratio':>8} "
          f"{'Rel err':>9} {'Real tier':>10} {'Sim tier':>9}")
    for t in sorted(int(k) for k in turn_curves):
        row = turn_curves[str(t)]
        rs = row.get("real_board_stats")
        ss = row.get("sim_board_stats")
        ratio = row.get("stats_ratio_sim_over_real")
        rel = row.get("stats_relative_error")
        rt = row.get("real_tavern_tier")
        stier = row.get("sim_tavern_tier")
        rs_s = f"{rs:.0f}" if rs is not None else "—"
        ss_s = f"{ss:.0f}" if ss is not None else "—"
        ratio_s = f"{ratio:.2f}×" if ratio is not None else "—"
        rel_s = f"{100 * rel:+.0f}%" if rel is not None else "—"
        rt_s = f"{rt:.2f}" if rt is not None else "—"
        st_s = f"{stier:.2f}" if stier is not None else "—"
        print(f"{t:>4} {rs_s:>11} {ss_s:>11} {ratio_s:>8} {rel_s:>9} {rt_s:>10} {st_s:>9}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=200,
                    help="greedy lobbies to simulate (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=DEFAULT_OUT)
    ap.add_argument("--contract-out", default=DEFAULT_CONTRACT)
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION}")
    print(f"Simulator v1 · {args.lobbies} greedy lobbies · seed base {args.seed}")
    result = run_benchmark(lobbies=args.lobbies, seed=args.seed)
    _write_json(args.json_out, result)
    _write_json(args.contract_out, result["contract"])
    print_table(result["turn_curves"])

    comp = result["composition"]
    print(f"\nComposition (turns 8–14, build_path coverage):")
    print(f"  Sim mean:  {comp['sim_coverage_mean_turns_8_14']:.3f}")
    print(f"  Real mean: {comp['real_coverage_mean']:.3f}  "
          f"(n={comp['n_boards']} example boards)")

    ld = result["lobby_dynamics"]
    print(f"\nLobby dynamics:")
    print(f"  Avg game length: {ld['avg_game_length']:.1f} turns")

    h = result["headline_divergence"]
    if h.get("stats_ratio_turn_14") is not None:
        print(f"\nHeadline: turn-14 stats ratio sim/real = "
              f"{h['stats_ratio_turn_14']:.2f}×")

    print(f"\nSaved -> {args.json_out}")
    print(f"Contract -> {args.contract_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
