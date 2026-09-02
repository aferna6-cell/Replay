"""Simulator Fidelity Phase 2L — post-assembly availability decomposition.

    python -m ml.fidelity_phase_2l

Frozen Phase 2J policy. DEV seeds 10200–10699 (expand through 10999).
Decomposes 2K's never-legally-buyable missing mass into tier / raw / legal /
pool constraints. Measurement-only.
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
)

from .availability_decomposition import (
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    METHODOLOGY_VERSION,
    PHASE_2J_PRIOR_PATH,
    PHASE_2L_EXPAND_THROUGH,
    PHASE_2L_LOBBIES,
    PHASE_2L_MIN_STATES,
    PHASE_2L_SEED,
    analyze_availability_decomposition,
)
from .fidelity_phase_2k import load_frozen_prior, run_board_opp_traced
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .phase_2l_decision import evaluate_phase_2l_decision

DEFAULT_DIR = "results/sim_fidelity_phase_2l"
PHASE = "2L post-assembly availability decomposition"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    run_lo, run_hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if not (run_hi < flo or run_lo > fhi):
            raise RuntimeError(
                f"Phase 2L rejects seeds overlapping {flo}–{fhi}. "
                f"Requested {run_lo}–{run_hi}.")


def run_phase_2l(*, seed: int = PHASE_2L_SEED,
                 lobbies: int = PHASE_2L_LOBBIES,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True,
                 allow_expand: bool = True) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2L first.")

    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    assert prior.content_hash_sha256() == FROZEN_PRIOR_HASH

    t0 = time.time()
    print(f"Phase 2L {METHODOLOGY_VERSION} — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print(f"  frozen α={FROZEN_ALPHA}, prior_hash={FROZEN_PRIOR_HASH[:12]}…")

    traces = run_board_opp_traced(lobbies, seed, prior)
    analysis = analyze_availability_decomposition(traces)
    expanded = False
    if allow_expand and analysis["n_states"] < PHASE_2L_MIN_STATES:
        extra_lo = seed + lobbies
        extra_hi = PHASE_2L_EXPAND_THROUGH
        if extra_lo <= extra_hi:
            extra_n = extra_hi - extra_lo + 1
            print(f"  adaptive expand: {analysis['n_states']} < {PHASE_2L_MIN_STATES}; "
                  f"adding seeds {extra_lo}–{extra_hi}")
            assert_seed_range_allowed(extra_lo, extra_n)
            traces2 = run_board_opp_traced(extra_n, extra_lo, prior)
            offset = lobbies
            for collection in ("events", "turn_summaries", "player_finals",
                               "lobby_meta"):
                for row in traces2[collection]:
                    row["lobby"] += offset
                traces[collection].extend(traces2[collection])
            traces["lobbies"] = lobbies + extra_n
            analysis = analyze_availability_decomposition(traces)
            expanded = True
            lobbies = traces["lobbies"]

    decision = evaluate_phase_2l_decision(analysis)
    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2l_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "measurement_only": True,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "note": (
            "Decomposes never-legally-buyable missing mass after first-2; "
            "no policy changes."),
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
        "adaptive_expanded": expanded,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "analysis": {
            "n_states": analysis["n_states"],
            "never_legal_missing_mass": analysis["never_legal_missing_mass"],
            "total_missing_final_coverage_mass": (
                analysis["total_missing_final_coverage_mass"]),
            "never_legal_share_of_total_missing": (
                analysis["never_legal_share_of_total_missing"]),
            "subfate_mass": analysis["subfate_mass"],
            "subfate_share_of_never_legal": (
                analysis["subfate_share_of_never_legal"]),
            "subfate_card_counts": analysis["subfate_card_counts"],
            "headlines": analysis["headlines"],
            "sampler_diagnostic": analysis["sampler_diagnostic"],
            "dominant_subfate": analysis["dominant_subfate"],
            "dominant_share": analysis["dominant_share"],
        },
        "decision": decision,
        "contract": contract,
        "state_records": analysis["state_records"],
    }
    _write_json(os.path.join(out_dir, "phase_2l_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=PHASE_2L_SEED)
    ap.add_argument("--lobbies", type=int, default=PHASE_2L_LOBBIES)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    ap.add_argument("--no-expand", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2L {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")

    try:
        result = run_phase_2l(
            seed=args.seed, lobbies=args.lobbies, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree,
            allow_expand=not args.no_expand)
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    a = result["analysis"]
    d = result["decision"]
    h = a.get("headlines") or {}
    print(f"\nPost-assembly states: {a['n_states']}")
    print(f"Never-legal missing mass share of total missing: "
          f"{a.get('never_legal_share_of_total_missing')}")
    print(f"Headline zero-raw (tier-eligible): "
          f"{h.get('pct_never_legal_mass_tier_eligible_zero_raw')}")
    print(f"Headline raw-but-illegal: "
          f"{h.get('pct_never_legal_mass_raw_but_zero_legal')}")
    print(f"Dominant subfate: {a.get('dominant_subfate')} "
          f"({a.get('dominant_share')})")
    print(f"Decision: {d['decision_branch']}")
    print(f"  {d['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
