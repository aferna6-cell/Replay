"""Simulator Fidelity Phase 2M — shop/pool rules audit.

    python -m ml.fidelity_phase_2m

Measurement-only. Frozen Phase 2J policy. Reuses Phase 2L DEV seeds
10200–10699. Reserves 11000–11499 / 11500–11699 for Phase 2N — does not
consume them. Documents rule mismatches without patching the simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from hsbg_coach.board_opportunity_policy import (
    METHODOLOGY_VERSION as PHASE_2J_VERSION,
)

from .availability_decomposition import FROZEN_PRIOR_HASH, PHASE_2J_PRIOR_PATH
from .fidelity_phase_2k import load_frozen_prior
from .fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from .phase_2m_decision import evaluate_phase_2m_decision
from .shop_pool_audit import (
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    METHODOLOGY_VERSION,
    PHASE_2M_LOBBIES,
    PHASE_2M_SEED,
    analyze_shop_pool_audit,
    run_board_opp_with_pool_audit,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2m"
PHASE = "2M shop/pool rules audit"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    run_lo, run_hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        # Diagnostic DEV 10200–10699 is allowed (not in forbidden list).
        if not (run_hi < flo or run_lo > fhi):
            raise RuntimeError(
                f"Phase 2M rejects seeds overlapping {flo}–{fhi}. "
                f"Requested {run_lo}–{run_hi}.")


def run_phase_2m(*, seed: int = PHASE_2M_SEED,
                 lobbies: int = PHASE_2M_LOBBIES,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2M first.")

    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    assert prior.content_hash_sha256() == FROZEN_PRIOR_HASH

    t0 = time.time()
    print(f"Phase 2M {METHODOLOGY_VERSION} — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print(f"  frozen α={FROZEN_ALPHA}, prior_hash={FROZEN_PRIOR_HASH[:12]}…")
    print("  measurement-only: catalogue sync + rule doc + live pool calib")

    traces = run_board_opp_with_pool_audit(lobbies, seed, prior)
    analysis = analyze_shop_pool_audit(traces)
    decision = evaluate_phase_2m_decision(analysis)

    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2m_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "measurement_only": True,
        "no_simulator_patches": True,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "reserved_seeds": analysis["reserved_seeds"],
        "note": (
            "Shop/pool rules audit; documents mismatches without changing "
            "POOL_COPIES / draw / lifecycle behavior."),
    })

    # Slim analysis for top-level report (drop bulky per-core rows to sidecar)
    analysis_slim = {
        k: v for k, v in analysis.items() if k != "catalogue_rows"
    }
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
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "analysis": analysis_slim,
        "decision": decision,
        "contract": contract,
    }
    _write_json(os.path.join(out_dir, "phase_2m_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "catalogue_sync.json"), {
        "catalogue_synchronization": analysis["catalogue_synchronization"],
        "rows": analysis["catalogue_rows"],
    })
    _write_json(os.path.join(out_dir, "rule_mismatches.json"),
                analysis["rule_mismatches"])
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=PHASE_2M_SEED)
    ap.add_argument("--lobbies", type=int, default=PHASE_2M_LOBBIES)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2M {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")

    try:
        result = run_phase_2m(
            seed=args.seed, lobbies=args.lobbies, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    a = result["analysis"]
    d = result["decision"]
    h = a.get("headlines") or {}
    live = a.get("live_calibration") or {}
    print(f"\nPost-assembly states (2L lens): {a.get('n_states_phase_2l')}")
    print(f"Cores missing from KB: {h.get('pct_cores_missing_from_kb')}")
    print(f"Demonstrated rule mismatches: "
          f"{h.get('n_demonstrated_rule_mismatches')}")
    print(f"Live calib windows: {live.get('n_card_windows')}")
    print(f"Live zero-offer observed={h.get('live_observed_zero_offer_rate')} "
          f"vs expected={h.get('live_expected_zero_offer_rate')}")
    print(f"Live raw expected={h.get('live_sum_expected_raw')} "
          f"observed={h.get('live_sum_observed_raw')}")
    print(f"Decision: {d['decision_branch']}")
    print(f"  {d['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
