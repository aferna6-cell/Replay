"""Simulator Fidelity Phase 2W — Firestone final-board vs 2Q selection.

Measurement only. Reuses consumed Phase 2S–2V DEV 14200–14699. No new seeds.
Does not change α, scaling math, `_hero_damage`, gates, or defaults.

    python -m ml.fidelity_phase_2w
    python -m ml.fidelity_phase_2w --lobbies 8 --seed 14200 --non-evaluative
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
from ml.firestone_composition_reference import build_firestone_reference
from ml.phase_2w_diagnostic import (
    compare_arms,
    run_greedy_2s_treatment,
    run_greedy_control,
    summarize_arm,
)
from ml.phase_2w_prereg import (
    FEATURE_TOGGLE,
    FORBIDDEN_RANGES,
    HOLD_PRS,
    METHODOLOGY_VERSION,
    PHASE_2W_LOBBIES,
    PHASE_2W_SEED,
    assert_seed_range_allowed,
    diagnose_phase_2w,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2w"
PHASE = "2W Firestone final-board vs 2Q selection"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _slim_arm(summary: Dict) -> Dict:
    skip = {"late"}
    return {k: v for k, v in summary.items() if k not in skip}


def run_phase_2w(
    *,
    lobbies: int = PHASE_2W_LOBBIES,
    seed: int = PHASE_2W_SEED,
    out_dir: str = DEFAULT_DIR,
    non_evaluative: bool = False,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    if PHASE_2Q_RECRUIT_VALUE_STATS or PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING:
        raise RuntimeError(
            "2Q/2S toggles must default OFF outside arm context "
            f"(2Q={PHASE_2Q_RECRUIT_VALUE_STATS}, "
            f"2S={PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING})"
        )
    if abs(FROZEN_ALPHA - 0.5) > 1e-12:
        raise ValueError(f"Phase 2W must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    firestone = build_firestone_reference()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[2W] {tag} Firestone reference — "
        f"{firestone['coverage']['n_example_boards']} boards, "
        f"join={firestone['coverage']['join_rate']}",
        flush=True,
    )
    print(
        f"[2W] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    greedy_c = summarize_arm(run_greedy_control(lobbies, seed), firestone)
    print("[2W] greedy TREATMENT (2Q + 2S) — same seeds", flush=True)
    greedy_t = summarize_arm(run_greedy_2s_treatment(lobbies, seed), firestone)
    greedy_cmp = compare_arms(greedy_c, greedy_t, firestone)
    if non_evaluative:
        greedy_cmp["non_evaluative"] = True

    decision = diagnose_phase_2w(greedy_cmp, non_evaluative=non_evaluative)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(
        os.path.join(out_dir, "firestone_reference.json"),
        {
            "meta": firestone.get("meta"),
            "coverage": firestone.get("coverage"),
            "weighted": firestone.get("weighted"),
            "unweighted": firestone.get("unweighted"),
            "reconciliation": firestone.get("reconciliation"),
        },
    )
    _write_json(
        os.path.join(out_dir, "last_boards.json"),
        {
            "control": greedy_c.get("last_alive"),
            "treatment": greedy_t.get("last_alive"),
            "control_all_players": greedy_c.get("last_alive_all"),
            "treatment_all_players": greedy_t.get("last_alive_all"),
            "deltas": greedy_cmp.get("last_alive"),
        },
    )
    _write_json(
        os.path.join(out_dir, "t12_t14.json"),
        {
            "control": greedy_c.get("late"),
            "treatment": greedy_t.get("late"),
            "delta": greedy_cmp.get("late"),
        },
    )
    _write_json(
        os.path.join(out_dir, "replacements.json"),
        {
            "control": greedy_c.get("replacements"),
            "treatment": greedy_t.get("replacements"),
            "replace_rate": greedy_cmp.get("replace_rate"),
        },
    )
    _write_json(
        os.path.join(out_dir, "comparison.json"),
        {
            "last_alive": greedy_cmp.get("last_alive"),
            "late": greedy_cmp.get("late"),
            "overlap": greedy_cmp.get("overlap"),
            "replace_rate": greedy_cmp.get("replace_rate"),
        },
    )
    _write_json(
        os.path.join(out_dir, "reconciliation.json"),
        greedy_cmp.get("reconciliation") or {},
    )

    try:
        contract = build_simulator_v1_1_contract(
            evaluation_seed=seed, lobbies=lobbies
        )
    except ModuleNotFoundError as exc:
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
            "Phase 2W reuses consumed 2S–2V DEV 14200–14699. "
            "Measurement only; no new seeds. Firestone is final-board data."
        ),
        "non_evaluative": non_evaluative,
        "reused_phase_2s_dev": True,
        "stacked_on_phase_2v": True,
    }
    contract.update({
        "phase": PHASE,
        "phase_2w_methodology_version": METHODOLOGY_VERSION,
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version_label": SIMULATOR_V1_1_VERSION,
        "frozen_alpha": FROZEN_ALPHA,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default": False,
        "phase_2q_toggle_default": False,
        "hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "no_hero_damage_change": True,
        "no_gate_change": True,
        "no_behavior_change": True,
        "firestone_is_final_board_data": True,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "non_evaluative": non_evaluative,
        "decision": decision,
        "contract": contract,
        "firestone": {
            "coverage": firestone.get("coverage"),
            "weighted": firestone.get("weighted"),
        },
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "comparison": {
            "last_alive": greedy_cmp.get("last_alive"),
            "overlap": greedy_cmp.get("overlap"),
            "replace_rate": greedy_cmp.get("replace_rate"),
        },
        "reconciliation": greedy_cmp.get("reconciliation"),
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_2w_report.json"), report)

    last = greedy_cmp.get("last_alive") or {}
    print(
        f"[2W] primary_finding={decision['primary_finding']} "
        f"T4+ t-c={last.get('t4_share_treatment_minus_control')} "
        f"T4+ t-fs={last.get('t4_share_treatment_minus_firestone')} "
        f"tier t-fs={last.get('mean_printed_tier_treatment_minus_firestone')} "
        f"evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[2W] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--non-evaluative", action="store_true")
    args = p.parse_args(argv)
    run_phase_2w(
        lobbies=args.lobbies if args.lobbies is not None else PHASE_2W_LOBBIES,
        seed=args.seed if args.seed is not None else PHASE_2W_SEED,
        out_dir=args.out_dir,
        non_evaluative=bool(args.non_evaluative),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
