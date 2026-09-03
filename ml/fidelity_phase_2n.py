"""Simulator Fidelity Phase 2N — shop/pool fidelity interventions + measure.

    python -m ml.fidelity_phase_2n

Implements / measures Phase 2M actionable fixes plus 2N-D, then validates
the candidate with a paired behavioral/macro regression panel (`2n_v3`):

  2N-A catalogue/KB sync (data refresh + T7 core hygiene)
  2N-B death return + freeze top-up
  2N-C T6 copies 6→7
  2N-D current active Tavern-pool manifest ∩ build_pool (precision+recall)
  2n_v3 validation-only paired panel (greedy vs frozen BoardOpp α=0.5)
        + macro fidelity vs Firestone reference curves

No simulator changes in 2n_v3. DEV seeds **11700–12199** reused.
Confirmation **11500–11699** reserved until the full candidate gate passes
and an explicit Simulator v1.x freeze fingerprints the stack.
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
    audit_active_pool_precision_recall,
    audit_catalogue_synchronization,
    audit_rule_mismatches,
    run_board_opp_with_pool_audit,
)
from .phase_2n_regression import (
    build_simulator_v1_x_fingerprint,
    run_paired_regression_panel,
)

METHODOLOGY_VERSION = "2n_v3"
# Fresh DEV after 2N-D active-pool intervention (11000–11499 already informed 2N-D).
PHASE_2N_SEED = 11700
PHASE_2N_LOBBIES = 500  # 11700–12199
PHASE_2N_CONFIRM_SEED = 11500
PHASE_2N_CONFIRM_LOBBIES = 200  # reserved — do not consume until v1.x freeze

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),  # Phase 2L/2M diagnostic DEV
    (11000, 11499),  # Phase 2N 2n_v1 combined measure (consumed)
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
    active = analysis.get("active_pool_precision_recall") or {}
    rules = analysis.get("rule_mismatches") or {}
    live = analysis.get("live_calibration") or {}
    primary = live.get("primary_deal_level") or {}
    cons = analysis.get("pool_conservation") or {}
    panel = analysis.get("paired_regression_panel") or {}
    headlines = analysis.get("headlines") or {}

    missing = int(cat.get("n_missing_from_kb") or 0)
    invalid = int((cat.get("status_counts") or {}).get(
        "MISSING_OR_INVALID_TIER", 0))
    actionable = list(rules.get("phase_2n_actionable_ids") or [])
    raw_ratio = primary.get("raw_ratio_obs_over_exp")
    clustered = (primary.get("lobby_clustered") or {}).get(
        "raw_obs_minus_exp") or {}

    active_recall = active.get("active_pool_recall")
    active_precision = active.get("active_pool_precision")
    active_gates = bool(active.get("gates_pass"))
    catalogue_clean = (
        missing == 0 and invalid == 0
        and active_gates
        and active_recall == 1.0
        and active_precision == 1.0
    )
    lifecycle_copy_clean = len(actionable) == 0
    conservation_ok = bool(cons.get("conservation_ok", True))
    draw_ok = (
        raw_ratio is not None
        and 0.70 <= float(raw_ratio) <= 1.30
    )
    mechanism_ok = bool(panel.get("phase_2j_mechanism_regression_pass"))
    macro_fid_ok = bool(panel.get("macro_fidelity_pass"))
    panel_ok = bool(panel.get("panel_pass"))

    if (catalogue_clean and lifecycle_copy_clean and conservation_ok
            and draw_ok and mechanism_ok and macro_fid_ok):
        branch = "accept_simulator_v1_x_candidate"
        next_step = (
            "Candidate gate passed (active-pool + conservation + draw + "
            "Phase 2J mechanism + macro fidelity). Explicitly freeze "
            "Simulator v1.x fingerprints, then run confirmation ONCE on "
            f"{PHASE_2N_CONFIRM_SEED}–"
            f"{PHASE_2N_CONFIRM_SEED + PHASE_2N_CONFIRM_LOBBIES - 1}."
        )
    elif not catalogue_clean:
        branch = "active_pool_precision_incomplete"
        next_step = (
            "Active Tavern-pool precision/recall gates failed — "
            f"recall={active_recall}, precision={active_precision}, "
            f"gates_pass={active_gates}, missing_kb={missing}, "
            f"invalid_tier={invalid}."
        )
    elif not conservation_ok:
        branch = "pool_conservation_broken"
        next_step = (
            "Pool conservation invariant failed after death-return; "
            "inspect pool_conservation before acceptance."
        )
    elif not draw_ok:
        branch = "interventions_applied_draw_residual"
        next_step = (
            "Active-pool/lifecycle/T6 applied; deal-level calib still off — "
            "inspect residual before confirmation."
        )
    elif not mechanism_ok:
        branch = "phase_2j_mechanism_regression"
        next_step = (
            "Paired BoardOpp vs greedy mechanism regression failed — "
            "inspect phase_2j_acceptance before freeze/confirmation."
        )
    elif not macro_fid_ok:
        branch = "macro_fidelity_regression"
        next_step = (
            "Macro fidelity vs Firestone reference envelope failed — "
            "report only; do not retune. Inspect "
            "macro_fidelity_vs_reference gates."
        )
    else:
        branch = "interventions_incomplete"
        next_step = (
            f"Remaining actionable mismatches: {actionable}; "
            f"missing_kb={missing}, invalid_tier={invalid}."
        )

    return {
        "decision_branch": branch,
        "recommended_next_step": next_step,
        "catalogue_clean": catalogue_clean,
        "active_pool_recall": active_recall,
        "active_pool_precision": active_precision,
        "active_pool_gates_pass": active_gates,
        "lifecycle_copy_clean": lifecycle_copy_clean,
        "pool_conservation_ok": conservation_ok,
        "deal_level_raw_ratio": raw_ratio,
        "lobby_raw_ci95": clustered.get("ci95"),
        "phase_2j_mechanism_regression_pass": mechanism_ok,
        "macro_fidelity_pass": macro_fid_ok,
        "paired_regression_panel_pass": panel_ok,
        "phase_2n_flags": {
            "death_return": PHASE_2N_DEATH_RETURN,
            "freeze_topup": PHASE_2N_FREEZE_TOPUP,
            "pool_copies_t6": POOL_COPIES[6],
            "active_tavern_pool_filter": True,
            "frozen_alpha": FROZEN_ALPHA,
        },
        "headlines": headlines,
    }


def run_phase_2n(*, seed: int = PHASE_2N_SEED,
                 lobbies: int = PHASE_2N_LOBBIES,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True,
                 reuse_pool_audit_report: str | None = None) -> Dict:
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
    active_pool = audit_active_pool_precision_recall()
    rules = audit_rule_mismatches()

    if reuse_pool_audit_report:
        print(f"  reusing pool-audit analysis from {reuse_pool_audit_report}")
        with open(reuse_pool_audit_report, encoding="utf-8") as fh:
            prior_report = json.load(fh)
        if prior_report.get("evaluation_seed_base") != seed:
            raise RuntimeError(
                "reuse_pool_audit_report seed mismatch: "
                f"{prior_report.get('evaluation_seed_base')} != {seed}")
        if prior_report.get("n_lobbies") != lobbies:
            raise RuntimeError(
                "reuse_pool_audit_report lobbies mismatch: "
                f"{prior_report.get('n_lobbies')} != {lobbies}")
        analysis = dict(prior_report.get("analysis") or {})
        # Refresh static active-pool / catalogue / rules onto reused analysis.
        analysis["catalogue_synchronization"] = {
            k: v for k, v in catalogue.items() if k != "rows"
        }
        analysis["active_pool_precision_recall"] = active_pool
        analysis["rule_mismatches"] = rules
    else:
        traces = run_board_opp_with_pool_audit(lobbies, seed, prior)
        analysis = analyze_shop_pool_audit(traces)
        analysis.setdefault("active_pool_precision_recall", active_pool)

    panel = run_paired_regression_panel(
        seed=seed, lobbies=lobbies, alpha=FROZEN_ALPHA,
        prior_path=PHASE_2J_PRIOR_PATH, prior_hash=FROZEN_PRIOR_HASH)
    # Slim flags for decision / analysis; full panel written separately.
    analysis["paired_regression_panel"] = {
        "phase_2j_mechanism_regression_pass": panel[
            "phase_2j_mechanism_regression_pass"],
        "macro_fidelity_pass": panel["macro_fidelity_pass"],
        "panel_pass": panel["panel_pass"],
        "phase_2j_acceptance_flags": (
            (panel.get("phase_2j_acceptance") or {}).get("flags")),
        "macro_fidelity_gates": (
            (panel.get("macro_fidelity_vs_reference") or {}).get("gates")),
    }
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
            "2n_v3 validation after 2N-A/B/C/D: paired greedy vs BoardOpp "
            "α=0.5 + macro fidelity vs Firestone reference. No simulator "
            "changes in 2n_v3. Confirm seeds 11500–11699 reserved."),
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
        "static_active_pool": active_pool,
        "static_rule_mismatches": rules,
        "analysis": analysis_slim,
        "paired_regression_panel": panel,
        "decision": decision,
        "simulator_v1_x_fingerprint": build_simulator_v1_x_fingerprint(
            implementation_commit=impl_commit),
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
    _write_json(os.path.join(out_dir, "paired_regression_panel.json"), panel)
    _write_json(
        os.path.join(out_dir, "simulator_v1_x_fingerprint.json"),
        result["simulator_v1_x_fingerprint"])
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=PHASE_2N_SEED)
    ap.add_argument("--lobbies", type=int, default=PHASE_2N_LOBBIES)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    ap.add_argument(
        "--reuse-pool-audit-report", default=None,
        help="Reuse prior 2n_v2 pool-audit analysis JSON (same seeds; "
             "no simulator changes). Still runs the 2n_v3 regression panel.")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2N {METHODOLOGY_VERSION}")
    print(f"Implementation commit: {git_commit()}")
    try:
        result = run_phase_2n(
            seed=args.seed, lobbies=args.lobbies, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree,
            reuse_pool_audit_report=args.reuse_pool_audit_report)
    except (RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    d = result["decision"]
    live = (result["analysis"].get("live_calibration") or {}).get(
        "primary_deal_level") or {}
    ap = result.get("static_active_pool") or {}
    print(f"\nActive-pool recall={ap.get('active_pool_recall')} "
          f"precision={ap.get('active_pool_precision')} "
          f"gates={ap.get('gates_pass')} "
          f"n_build_pool={ap.get('n_build_pool')}")
    print(f"Catalogue missing_kb: "
          f"{result['static_catalogue'].get('n_missing_from_kb')}")
    panel = result.get("paired_regression_panel") or {}
    acc = (panel.get("phase_2j_acceptance") or {}).get("flags") or {}
    print(f"Phase 2J mechanism pass={panel.get('phase_2j_mechanism_regression_pass')} "
          f"(macro_reg={acc.get('macro_regression_ok')} "
          f"mech={acc.get('mechanism_up')} cov={acc.get('coverage_up')} "
          f"sacrifice={acc.get('board_sacrifice_ok')})")
    print(f"Macro fidelity pass={panel.get('macro_fidelity_pass')}")
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
