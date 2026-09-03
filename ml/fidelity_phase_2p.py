"""Simulator Fidelity Phase 2P — replacement-value contamination diagnostic.

Measurement only. Fresh DEV seeds 12700–13199 (500 lobbies).
Arms: raw greedy + frozen Phase 2J BoardOpp α=0.5.

Quantifies full-board recruit states where abstract scaling alone flips the
raw-stat greedy replacement rule from replace -> don't replace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

from hsbg_coach.board_opportunity_policy import (
    METHODOLOGY_VERSION as PHASE_2J_VERSION,
)
from ml.availability_decomposition import (
    FROZEN_ALPHA,
    PHASE_2J_PRIOR_PATH,
)
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from ml.replacement_value_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2P_LOBBIES,
    PHASE_2P_SEED,
    assert_seed_range_allowed,
    diagnose_contamination,
    run_greedy_arm,
    run_phase_2j_arm,
    summarize_arm,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2p"
PHASE = "2P replacement-value / scaling-contamination diagnostic"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_phase_2p(
    *,
    lobbies: int = PHASE_2P_LOBBIES,
    seed: int = PHASE_2P_SEED,
    out_dir: str = DEFAULT_DIR,
    alpha: float = FROZEN_ALPHA,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    if abs(alpha - FROZEN_ALPHA) > 1e-12:
        raise ValueError(f"Phase 2P must use frozen α={FROZEN_ALPHA}, got {alpha}")

    t0 = time.time()
    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    prior_hash = prior.content_hash_sha256()

    print(f"[2P] greedy arm — {lobbies} lobbies, seeds {seed}–{seed + lobbies - 1}")
    greedy_raw = run_greedy_arm(lobbies, seed)
    print(f"[2P] Phase 2J arm α={alpha} — same seeds")
    phase_2j_raw = run_phase_2j_arm(lobbies, seed, alpha, prior)

    greedy = summarize_arm(greedy_raw)
    phase_2j = summarize_arm(phase_2j_raw)
    decision = diagnose_contamination(greedy, phase_2j)

    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["evaluation"] = {
        "policy": "greedy + frozen BoardOpp α=0.5",
        "lobbies": lobbies,
        "base_seed": seed,
        "note": "Phase 2P contamination diagnostic; not Benchmark v1.",
    }
    contract.update({
        "phase": PHASE,
        "phase_2p_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version_label": SIMULATOR_V1_1_VERSION,
        "frozen_alpha": alpha,
        "prior_hash_sha256": prior_hash,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "instrument_turns": list(range(7, 15)),
        "measurement_only": True,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "decision": {
            **decision,
            "measurement_only": True,
            "keep_pr_29_hold": True,
            "keep_phase_2j_alpha": FROZEN_ALPHA,
            "confirm_seeds_reserved": "11500–11699",
        },
        "contract": contract,
        "greedy": greedy,
        "phase_2j": phase_2j,
    }

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_2p_report.json"), report)
    _write_json(os.path.join(out_dir, "decision.json"), report["decision"])
    _write_json(os.path.join(out_dir, "state_rows_greedy.json"), {
        "rows": greedy_raw["state_rows"]
    })
    _write_json(os.path.join(out_dir, "state_rows_phase_2j.json"), {
        "rows": phase_2j_raw["state_rows"]
    })
    _write_json(os.path.join(out_dir, "candidate_rows_greedy.json"), {
        "rows": greedy_raw["candidate_rows"]
    })
    _write_json(os.path.join(out_dir, "candidate_rows_phase_2j.json"), {
        "rows": phase_2j_raw["candidate_rows"]
    })

    print(f"[2P] primary_finding={report['decision']['primary_finding']}")
    print(f"[2P] next={report['decision']['recommended_next_step']}")
    print(f"[2P] wrote {out_dir}/ ({contract['runtime_sec']}s)")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=PHASE_2P_LOBBIES)
    ap.add_argument("--seed", type=int, default=PHASE_2P_SEED)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--alpha", type=float, default=FROZEN_ALPHA)
    args = ap.parse_args(argv)
    try:
        run_phase_2p(
            lobbies=args.lobbies,
            seed=args.seed,
            out_dir=args.out_dir,
            alpha=args.alpha,
        )
    except (RuntimeError, AssertionError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
