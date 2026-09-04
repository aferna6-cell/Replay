"""Simulator Fidelity Phase 2S — board-level abstract scaling.

Implementation harness. Default-OFF ``PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING``.
Treatment = 2Q recruit-value selection ON **and** 2S board-level pool ON.
Control = both OFF. Residual/ratio budget math and α=0.5 unchanged.

Full evaluative DEV: 500 lobbies, seeds 14200–14699 — **not this hour**.
Tiny smoke (``--non-evaluative``) is runtime/accounting only and must not
route the 500-lobby decision.

    python -m ml.fidelity_phase_2s --non-evaluative --skip-phase-2j
    python -m ml.fidelity_phase_2s          # 500-lobby DEV — later hour
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
)
from ml.availability_decomposition import FROZEN_ALPHA
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from ml.phase_2s_prereg import (
    FEATURE_TOGGLE,
    FORBIDDEN_RANGES,
    HOLD_PRS,
    METHODOLOGY_VERSION,
    PHASE_2S_LOBBIES,
    PHASE_2S_SEED,
    SMOKE_LOBBIES,
    SMOKE_SEED,
    assert_seed_range_allowed,
    diagnose_phase_2s,
)
from ml.replacement_churn_diagnostic import (
    compare_control_treatment,
    run_greedy_2s_treatment,
    run_greedy_control,
    summarize_churn_arm,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2s"
SMOKE_DIR = "results/sim_fidelity_phase_2s_smoke"
PHASE = "2S board-level abstract scaling"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _slim_arm(summary: Dict) -> Dict:
    out = {k: v for k, v in summary.items() if k != "example_replacement_events"}
    return out


def run_phase_2s(
    *,
    lobbies: int = PHASE_2S_LOBBIES,
    seed: int = PHASE_2S_SEED,
    out_dir: str = DEFAULT_DIR,
    non_evaluative: bool = False,
    skip_phase_2j: bool = True,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    if PHASE_2Q_RECRUIT_VALUE_STATS or PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING:
        raise RuntimeError(
            "2Q/2S toggles must default OFF outside arm context "
            f"(2Q={PHASE_2Q_RECRUIT_VALUE_STATS}, "
            f"2S={PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING})"
        )
    if abs(FROZEN_ALPHA - 0.5) > 1e-12:
        raise ValueError(f"Phase 2S must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[2S] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    greedy_c_raw = run_greedy_control(lobbies, seed)
    greedy_c = summarize_churn_arm(greedy_c_raw)
    del greedy_c_raw
    print("[2S] greedy TREATMENT (2Q recruit-value + 2S pool) — same seeds",
          flush=True)
    greedy_t_raw = run_greedy_2s_treatment(lobbies, seed)
    accounting = list(greedy_t_raw.get("pool_accounting") or [])
    greedy_t = summarize_churn_arm(greedy_t_raw)
    del greedy_t_raw
    greedy_cmp = compare_control_treatment(greedy_c, greedy_t)
    if non_evaluative:
        greedy_cmp["non_evaluative"] = True
        greedy_cmp["evaluative"] = False
        greedy_cmp["note"] = (
            "tiny smoke for runtime/accounting only; not the 500-lobby DEV"
        )

    worst_drift = max(
        (float(s.get("worst_abs_drift") or 0) for s in accounting),
        default=0.0,
    )
    decision = diagnose_phase_2s(greedy_cmp, non_evaluative=non_evaluative)
    decision["pool_accounting_worst_abs_drift"] = worst_drift
    decision["pool_accounting_lobbies_checked"] = len(accounting)
    decision["skip_phase_2j"] = bool(skip_phase_2j)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(os.path.join(out_dir, "greedy_comparison.json"), greedy_cmp)
    _write_json(
        os.path.join(out_dir, "pool_accounting.json"),
        {
            "non_evaluative": non_evaluative,
            "worst_abs_drift": worst_drift,
            "lobbies": accounting,
        },
    )
    _write_json(
        os.path.join(out_dir, "per_turn_decomposition_greedy.json"),
        {
            "control": greedy_c.get("per_turn_decomposition"),
            "treatment": greedy_t.get("per_turn_decomposition"),
            "delta": greedy_cmp.get("per_turn_decomposition_delta"),
        },
    )

    try:
        contract = build_simulator_v1_1_contract(
            evaluation_seed=seed, lobbies=lobbies
        )
    except ModuleNotFoundError as exc:
        # Torch is optional for this accounting smoke (same 2R persist-first
        # reason). Keep a usable contract without the runtime fingerprint.
        contract = {
            "runtime_fingerprint_error": str(exc),
            "evaluation_seed": seed,
            "lobbies": lobbies,
        }
    contract["evaluation"] = {
        "policy": (
            "greedy control (2Q/2S OFF) vs greedy treatment "
            "(2Q recruit-value + 2S board-level pool)"
        ),
        "lobbies": lobbies,
        "base_seed": seed,
        "note": (
            "Phase 2S non-evaluative smoke; not the 500-lobby DEV."
            if non_evaluative
            else "Phase 2S evaluative DEV 14200–14699."
        ),
        "non_evaluative": non_evaluative,
        "skip_phase_2j": bool(skip_phase_2j),
    }
    contract.update({
        "phase": PHASE,
        "phase_2s_methodology_version": METHODOLOGY_VERSION,
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version_label": SIMULATOR_V1_1_VERSION,
        "frozen_alpha": FROZEN_ALPHA,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default": False,
        "phase_2q_toggle_default": False,
        "hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "non_evaluative": non_evaluative,
        "decision": decision,
        "contract": contract,
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "greedy_comparison": greedy_cmp,
        "pool_accounting_worst_abs_drift": worst_drift,
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_2s_report.json"), report)

    print(
        f"[2S] primary_finding={decision['primary_finding']} "
        f"pool_drift={worst_drift} evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[2S] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--non-evaluative",
        action="store_true",
        help="tiny smoke only; do not route the 500-lobby DEV",
    )
    p.add_argument("--skip-phase-2j", action="store_true", default=True)
    args = p.parse_args(argv)
    non_eval = bool(args.non_evaluative)
    lobbies = args.lobbies if args.lobbies is not None else (
        SMOKE_LOBBIES if non_eval else PHASE_2S_LOBBIES
    )
    seed = args.seed if args.seed is not None else (
        SMOKE_SEED if non_eval else PHASE_2S_SEED
    )
    out_dir = args.out_dir or (SMOKE_DIR if non_eval else DEFAULT_DIR)
    run_phase_2s(
        lobbies=lobbies,
        seed=seed,
        out_dir=out_dir,
        non_evaluative=non_eval,
        skip_phase_2j=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
