"""Simulator Fidelity Phase 2R — replacement-collapse mechanism diagnostic.

Measurement only. Fresh DEV seeds 13700–14199.
Primary: greedy control vs Phase 2Q recruit-value treatment.
Secondary: Phase 2J α=0.5, report-only (no retune).

Does not alter residual scaling, α, pool, economy, card effects, combat, or
the Phase 2Q toggle default (OFF). Confirm 11500–11699 reserved.

    python -m ml.fidelity_phase_2r
    python -m ml.fidelity_phase_2r --lobbies 2 --seed 13700
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
from ml.collapse_mechanism_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2R_LOBBIES,
    PHASE_2R_SEED,
    assert_seed_range_allowed,
    compare_control_treatment,
    diagnose_phase_2r,
    run_greedy_control,
    run_greedy_treatment,
    run_phase_2j_control,
    run_phase_2j_treatment,
    summarize_collapse_arm,
)
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2r"
PHASE = "2R replacement-collapse mechanism diagnostic"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_phase_2r(
    *,
    lobbies: int = PHASE_2R_LOBBIES,
    seed: int = PHASE_2R_SEED,
    out_dir: str = DEFAULT_DIR,
    alpha: float = FROZEN_ALPHA,
    skip_phase_2j: bool = False,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    if abs(alpha - FROZEN_ALPHA) > 1e-12:
        raise ValueError(f"Phase 2R must use frozen α={FROZEN_ALPHA}, got {alpha}")

    t0 = time.time()
    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    prior_hash = prior.content_hash_sha256()

    print(f"[2R] greedy CONTROL — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}", flush=True)
    greedy_c = summarize_collapse_arm(run_greedy_control(lobbies, seed))
    print(f"[2R] greedy TREATMENT (recruit-value) — same seeds", flush=True)
    greedy_t = summarize_collapse_arm(run_greedy_treatment(lobbies, seed))
    greedy_cmp = compare_control_treatment(greedy_c, greedy_t)
    print(
        f"[2R] greedy T10 post/FS "
        f"{(greedy_cmp.get('control') or {}).get('post_scale_over_firestone_t10')} → "
        f"{(greedy_cmp.get('treatment') or {}).get('post_scale_over_firestone_t10')}",
        flush=True,
    )

    phase_2j_c = phase_2j_t = phase_2j_cmp = None
    if not skip_phase_2j:
        print(f"[2R] Phase 2J CONTROL α={alpha} — same seeds (report-only)",
              flush=True)
        phase_2j_c = summarize_collapse_arm(
            run_phase_2j_control(lobbies, seed, alpha, prior)
        )
        print(f"[2R] Phase 2J TREATMENT α={alpha} — same seeds (report-only)",
              flush=True)
        phase_2j_t = summarize_collapse_arm(
            run_phase_2j_treatment(lobbies, seed, alpha, prior)
        )
        phase_2j_cmp = compare_control_treatment(phase_2j_c, phase_2j_t)

    decision = diagnose_phase_2r(greedy_cmp, phase_2j_cmp)

    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["evaluation"] = {
        "policy": "greedy ± recruit-value; BoardOpp α=0.5 ± recruit-value (secondary)",
        "lobbies": lobbies,
        "base_seed": seed,
        "note": "Phase 2R measurement-only collapse mechanism; not Benchmark v1.",
    }
    contract.update({
        "phase": PHASE,
        "phase_2r_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version_label": SIMULATOR_V1_1_VERSION,
        "frozen_alpha": alpha,
        "prior_hash_sha256": prior_hash,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "feature_toggle": "PHASE_2Q_RECRUIT_VALUE_STATS (default OFF)",
        "measurement_only": True,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "decision": decision,
        "contract": contract,
        "greedy_control": greedy_c,
        "greedy_treatment": greedy_t,
        "greedy_comparison": greedy_cmp,
        "phase_2j_control": phase_2j_c,
        "phase_2j_treatment": phase_2j_t,
        "phase_2j_comparison": phase_2j_cmp,
    }

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(os.path.join(out_dir, "phase_2r_report.json"), report)
    _write_json(os.path.join(out_dir, "greedy_comparison.json"), greedy_cmp)
    _write_json(
        os.path.join(out_dir, "greedy_per_turn_decomposition.json"),
        {
            "control": greedy_c.get("per_turn_decomposition"),
            "treatment": greedy_t.get("per_turn_decomposition"),
            "gap": greedy_cmp.get("gap_decomposition_by_turn"),
        },
    )
    _write_json(
        os.path.join(out_dir, "greedy_replacement_loss_distribution.json"),
        {
            "control": greedy_c.get("replacement_loss_distribution"),
            "treatment": greedy_t.get("replacement_loss_distribution"),
            "by_turn_control": greedy_c.get("replacement_loss_by_turn"),
            "by_turn_treatment": greedy_t.get("replacement_loss_by_turn"),
        },
    )
    _write_json(
        os.path.join(out_dir, "paired_post_scale_and_alive.json"),
        {
            "post_scale_firestone": greedy_cmp.get("paired_post_scale_firestone"),
            "alive_curve": greedy_cmp.get("paired_alive_curve"),
            "phase_2j_post_scale_firestone": (
                (phase_2j_cmp or {}).get("paired_post_scale_firestone")
            ),
            "phase_2j_alive_curve": (phase_2j_cmp or {}).get("paired_alive_curve"),
        },
    )
    if phase_2j_cmp is not None:
        _write_json(os.path.join(out_dir, "phase_2j_comparison.json"), phase_2j_cmp)

    print(f"[2R] primary_finding={decision['primary_finding']}", flush=True)
    print(f"[2R] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=PHASE_2R_LOBBIES)
    p.add_argument("--seed", type=int, default=PHASE_2R_SEED)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--skip-phase-2j", action="store_true")
    args = p.parse_args(argv)
    run_phase_2r(
        lobbies=args.lobbies,
        seed=args.seed,
        out_dir=args.out_dir,
        skip_phase_2j=args.skip_phase_2j,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
