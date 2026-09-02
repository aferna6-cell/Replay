"""Simulator Fidelity Phase 2N — shop/pool fidelity interventions + measure.

    python -m ml.fidelity_phase_2n

Implements / measures the three Phase 2M actionable fixes:
  2N-A catalogue/KB sync (data refresh + T7 core hygiene)
  2N-B death return + freeze top-up
  2N-C T6 copies 6→7

Measurement consumes reserved intervention seeds **11000–11499** once on the
completed 2N simulator. Confirmation **11500–11699** reserved for freeze.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from hsbg_coach.bg_env import (
    PHASE_2N_DEATH_RETURN,
    PHASE_2N_FREEZE_TOPUP,
    POOL_COPIES,
)
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
from .shop_pool_audit import (
    FROZEN_ALPHA,
    analyze_shop_pool_audit,
    audit_catalogue_synchronization,
    audit_rule_mismatches,
    run_board_opp_with_pool_audit,
)

METHODOLOGY_VERSION = "2n_v1"
PHASE_2N_SEED = 11000
PHASE_2N_LOBBIES = 500
PHASE_2N_CONFIRM_SEED = 11500
PHASE_2N_CONFIRM_LOBBIES = 200

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),  # Phase 2L/2M diagnostic DEV
    (11500, 11699),  # confirmation — not for intervention measure
)

DEFAULT_DIR = "results/sim_fidelity_phase_2n"
PHASE = "2N shop/pool fidelity interventions"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    run_lo, run_hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if not (run_hi < flo or run_lo > fhi):
            raise RuntimeError(
                f"Phase 2N rejects seeds overlapping {flo}–{fhi}. "
                f"Requested {run_lo}–{run_hi}.")


def evaluate_phase_2n_decision(analysis: Dict) -> Dict:
    cat = analysis.get("catalogue_synchronization") or {}
    rules = analysis.get("rule_mismatches") or {}
    live = analysis.get("live_calibration") or {}
    primary = live.get("primary_deal_level") or {}
    headlines = analysis.get("headlines") or {}

    missing = int(cat.get("n_missing_from_kb") or 0)
    invalid = int((cat.get("status_counts") or {}).get(
        "MISSING_OR_INVALID_TIER", 0))
    actionable = list(rules.get("phase_2n_actionable_ids") or [])
    raw_ratio = primary.get("raw_ratio_obs_over_exp")
    clustered = (primary.get("lobby_clustered") or {}).get(
        "raw_obs_minus_exp") or {}

    catalogue_clean = missing == 0 and invalid == 0
    lifecycle_copy_clean = len(actionable) == 0
    draw_ok = (
        raw_ratio is not None
        and 0.70 <= float(raw_ratio) <= 1.30
    )

    if catalogue_clean and lifecycle_copy_clean and draw_ok:
        branch = "accept_simulator_v1_x_candidate"
        next_step = (
            "Freeze Simulator v1.x candidate; run confirmation on "
            f"{PHASE_2N_CONFIRM_SEED}–"
            f"{PHASE_2N_CONFIRM_SEED + PHASE_2N_CONFIRM_LOBBIES - 1}.")
    elif catalogue_clean and lifecycle_copy_clean:
        branch = "interventions_applied_draw_residual"
        next_step = (
            "Catalogue/lifecycle/T6 applied; deal-level calib still off — "
            "inspect residual before confirmation.")
    else:
        branch = "interventions_incomplete"
        next_step = (
            f"Remaining actionable mismatches: {actionable}; "
            f"missing_kb={missing}, invalid_tier={invalid}.")

    return {
        "decision_branch": branch,
        "recommended_next_step": next_step,
        "catalogue_clean": catalogue_clean,
        "lifecycle_copy_clean": lifecycle_copy_clean,
        "deal_level_raw_ratio": raw_ratio,
        "lobby_raw_ci95": clustered.get("ci95"),
        "phase_2n_flags": {
            "death_return": PHASE_2N_DEATH_RETURN,
            "freeze_topup": PHASE_2N_FREEZE_TOPUP,
            "pool_copies_t6": POOL_COPIES[6],
        },
        "headlines": headlines,
    }


def run_phase_2n(*, seed: int = PHASE_2N_SEED,
                 lobbies: int = PHASE_2N_LOBBIES,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError("Working tree is not clean. Commit Phase 2N first.")

    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    assert prior.content_hash_sha256() == FROZEN_PRIOR_HASH

    t0 = time.time()
    print(f"Phase 2N {METHODOLOGY_VERSION} — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print(f"  flags: death_return={PHASE_2N_DEATH_RETURN} "
          f"freeze_topup={PHASE_2N_FREEZE_TOPUP} T6_copies={POOL_COPIES[6]}")

    # Static audits first
    catalogue = audit_catalogue_synchronization()
    rules = audit_rule_mismatches()

    traces = run_board_opp_with_pool_audit(lobbies, seed, prior)
    analysis = analyze_shop_pool_audit(traces)
    decision = evaluate_phase_2n_decision(analysis)

    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract.update({
        "phase": PHASE,
        "phase_2n_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "phase_2n_death_return": PHASE_2N_DEATH_RETURN,
        "phase_2n_freeze_topup": PHASE_2N_FREEZE_TOPUP,
        "pool_copies": dict(POOL_COPIES),
        "note": (
            "Combined measurement after 2N-A/B/C. No _draw rewrite; "
            "no buy/economy; no card effects; no BC/DAgger/PPO."),
    })

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
        "static_catalogue": {
            k: v for k, v in catalogue.items() if k != "rows"
        },
        "static_rule_mismatches": rules,
        "analysis": analysis_slim,
        "decision": decision,
        "contract": contract,
        "phase_2m_baseline_deal_level": {
            "note": "2m_v2 DEV 10200–10699 primary deal-level",
            "sum_expected_raw": 74.86883120790482,
            "sum_observed_raw": 60.0,
            "raw_ratio_obs_over_exp": 0.801401585038569,
        },
    }
    _write_json(os.path.join(out_dir, "phase_2n_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "rule_mismatches.json"), rules)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=PHASE_2N_SEED)
    ap.add_argument("--lobbies", type=int, default=PHASE_2N_LOBBIES)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2N {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")
    try:
        result = run_phase_2n(
            seed=args.seed, lobbies=args.lobbies, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    d = result["decision"]
    live = (result["analysis"].get("live_calibration") or {}).get(
        "primary_deal_level") or {}
    print(f"\nCatalogue missing_kb: "
          f"{result['static_catalogue'].get('n_missing_from_kb')}")
    print(f"Actionable mismatches: "
          f"{result['static_rule_mismatches'].get('phase_2n_actionable_ids')}")
    print(f"Deal-level raw expected={live.get('sum_expected_raw')} "
          f"observed={live.get('sum_observed_raw')} "
          f"ratio={live.get('raw_ratio_obs_over_exp')}")
    print(f"Decision: {d['decision_branch']}")
    print(f"  {d['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
