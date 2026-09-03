"""Simulator Fidelity Phase 2C — composition assembly diagnostic (measurement only).

    python -m ml.fidelity_phase_2c
    python -m ml.fidelity_phase_2c --lobbies 200 --seed 0

Records event-level recruit traces and aggregates availability → purchase →
retention → assembly funnels per Firestone archetype. Does not change sim mechanics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from .composition_diagnostic import aggregate_diagnostics
from .composition_trace import run_traced_rollouts
from .fidelity_metrics import real_composition_baseline
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)

DEFAULT_DIR = "results/sim_fidelity_phase_2c"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _print_funnel(label: str, funnel: Dict) -> None:
    if not funnel:
        return
    legally = funnel.get("legally_buyable_exposures", 0)
    fulfilled = funnel.get("fulfilled_exposures", 0)
    rejected = funnel.get("rejected_exposures", 0)
    rate = funnel.get("rejection_rate", 0.0)
    valid = funnel.get("exposure_accounting_valid", False)
    print(f"\n{label}:")
    print(f"  legally buyable exposures: {legally}")
    print(f"  fulfilled exposures:         {fulfilled}")
    print(f"  rejected exposures:          {rejected}")
    print(f"  rejection rate:              {rate:.1%}")
    print(f"  accounting valid:            {valid}")


def run_phase_2c(*, lobbies: int = 200, seed: int = 0,
                 out_dir: str = DEFAULT_DIR,
                 save_events: bool = False,
                 require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError(
            "Working tree is not clean. Commit Phase 2C implementation first, "
            "then rerun with a clean tree (Phase 2B reproducibility standard).")

    t0 = time.time()
    print(f"Tracing {lobbies} lobbies (greedy, Simulator v1.1 residual scaling)…")
    traces = run_traced_rollouts(lobbies, seed=seed, scaling_mode="residual")
    print("Aggregating composition funnels by archetype…")
    diagnostic = aggregate_diagnostics(traces)
    real = real_composition_baseline()
    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["working_tree_clean"] = tree_clean

    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": "2C composition assembly diagnostic",
        "methodology_version": diagnostic["methodology_version"],
        "implementation_commit": impl_commit,
        "code_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "measurement_only": True,
        "invalidated_prior_results": diagnostic["invalidated_prior_results"],
        "invalidated_v2_broad_current_target": diagnostic.get(
            "invalidated_v2_broad_current_target"),
        "contract": contract,
        "runtime_seconds": round(time.time() - t0, 2),
        "n_lobbies": lobbies,
        "n_events": len(traces["events"]),
        "real_final_winner_coverage_mean": real["real_final_winner_coverage_mean"],
        "sim_final_winner_coverage_mean": diagnostic["sim_final_winner_coverage_mean"],
        "diagnostic": diagnostic,
        "recommended_phase_2d_intervention": diagnostic["recommended_phase_2d_intervention"],
    }

    _write_json(os.path.join(out_dir, "phase_2c_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    if save_events:
        slim = {k: v for k, v in traces.items() if k != "events"}
        slim["events_sample"] = traces["events"][:500]
        _write_json(os.path.join(out_dir, "traces_summary.json"), slim)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--save-events", action="store_true",
                    help="Write traces_summary.json with event sample")
    ap.add_argument("--allow-dirty-tree", action="store_true",
                    help="Skip clean working-tree check (not for canonical runs)")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2C")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")

    try:
        result = run_phase_2c(
            lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
            save_events=args.save_events,
            require_clean_tree=not args.allow_dirty_tree)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    rec = result["recommended_phase_2d_intervention"]
    print(f"\nMethodology: {result['methodology_version']}")
    print(f"Implementation commit: {result['implementation_commit']}")
    print(f"Working tree clean: {result['working_tree_clean']}")
    print(f"Sim final winner coverage mean: {result['sim_final_winner_coverage_mean']:.3f}")
    print(f"Real final winner coverage mean: {result['real_final_winner_coverage_mean']:.3f}")
    print(f"\nRecommended Phase 2D: {rec['phase_2d_title']}")
    print(f"  {rec['rationale']}")

    _print_funnel("Seeded current-target (primary)",
                  rec.get("funnel_seeded_current_target"))
    _print_funnel("Committed current-target (confirmation)",
                  rec.get("funnel_committed_current_target"))
    _print_funnel("Final-target hindsight",
                  rec.get("funnel_final_target_hindsight"))
    _print_funnel("Broad current-target (exploratory)",
                  rec.get("funnel_broad_current_target_exploratory"))

    print(f"\nSaved -> {args.out_dir}/phase_2c_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
