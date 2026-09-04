"""Simulator Fidelity Phase 3E — board-pool lifecycle attribution.

Measurement only. Reuses consumed Phase 2S–3D DEV 14200–14699. No new seeds.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

    python -m ml.fidelity_phase_3e
    python -m ml.fidelity_phase_3e --lobbies 8 --seed 14200 --non-evaluative
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
from ml.phase_3e_prereg import (
    FEATURE_TOGGLE,
    FORBIDDEN_RANGES,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    METHODOLOGY_VERSION,
    PHASE_3E_LOBBIES,
    PHASE_3E_SEED,
    POOL_FLOW_IDENTITY,
    assert_seed_range_allowed,
    diagnose_phase_3e,
)
from ml.pool_lifecycle_diagnostic import (
    compare_lifecycle,
    run_greedy_2s_treatment_lifecycle,
    run_greedy_control_lifecycle,
    summarize_lifecycle_arm,
)

DEFAULT_DIR = "results/sim_fidelity_phase_3e"
PHASE = "3E board-pool lifecycle attribution"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _slim_arm(summary: Dict) -> Dict:
    skip = {
        "example_fights", "per_turn", "_rows", "example_minions",
        "example_turns", "example_replacements",
    }
    return {k: v for k, v in summary.items() if k not in skip}


def run_phase_3e(
    *,
    lobbies: int = PHASE_3E_LOBBIES,
    seed: int = PHASE_3E_SEED,
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
        raise ValueError(f"Phase 3E must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[3E] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    greedy_c = summarize_lifecycle_arm(run_greedy_control_lifecycle(lobbies, seed))
    print("[3E] greedy TREATMENT (2Q + 2S) — same seeds", flush=True)
    greedy_t = summarize_lifecycle_arm(
        run_greedy_2s_treatment_lifecycle(lobbies, seed)
    )
    greedy_cmp = compare_lifecycle(greedy_c, greedy_t)
    if non_evaluative:
        greedy_cmp["non_evaluative"] = True

    decision = diagnose_phase_3e(greedy_cmp, non_evaluative=non_evaluative)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(
        os.path.join(out_dir, "reweighting.json"),
        greedy_cmp.get("reweighting") or {},
    )
    _write_json(
        os.path.join(out_dir, "reweighting_3d.json"),
        greedy_cmp.get("reweighting_3d") or {},
    )
    _write_json(
        os.path.join(out_dir, "by_tier_lifecycle.json"),
        greedy_cmp.get("by_tier_lifecycle") or {},
    )
    _write_json(
        os.path.join(out_dir, "reconciliation.json"),
        greedy_cmp.get("reconciliation") or {},
    )
    _write_json(
        os.path.join(out_dir, "additive_flow.json"),
        greedy_cmp.get("additive_flow") or {},
    )
    _write_json(
        os.path.join(out_dir, "example_turns.json"),
        greedy_cmp.get("example_turns") or {},
    )
    _write_json(
        os.path.join(out_dir, "example_replacements.json"),
        greedy_cmp.get("example_replacements") or {},
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
            "Phase 3E reuses consumed 2S–3D DEV 14200–14699. "
            "Measurement only; no new seeds. Do not rewrite 2Q."
        ),
        "non_evaluative": non_evaluative,
        "reused_phase_2s_dev": True,
        "stacked_on_phase_3d": True,
    }
    contract.update({
        "phase": PHASE,
        "phase_3e_methodology_version": METHODOLOGY_VERSION,
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
        "no_2q_rewrite": True,
        "no_scaling_constant_change": True,
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
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
        "reweighting": greedy_cmp.get("reweighting"),
        "reweighting_3d": greedy_cmp.get("reweighting_3d"),
        "reweighting_3c": greedy_cmp.get("reweighting_3c"),
        "by_tier_lifecycle": greedy_cmp.get("by_tier_lifecycle"),
        "by_tier_source": greedy_cmp.get("by_tier_source"),
        "reconciliation": greedy_cmp.get("reconciliation"),
        "additive_flow": greedy_cmp.get("additive_flow"),
        "deltas": greedy_cmp.get("deltas"),
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_3e_report.json"), report)

    rw = greedy_cmp.get("reweighting") or {}
    rec = greedy_cmp.get("reconciliation") or {}
    print(
        f"[3E] primary_finding={decision['primary_finding']} "
        f"a1_3d={rec.get('reproduced_3d_board_pool_magnitude')} "
        f"carry={rw.get('share_of_a1_inherited_carry_pool')} "
        f"scale={rw.get('share_of_a1_current_turn_scaling_add')} "
        f"replace={rw.get('share_of_a1_replacement_churn')} "
        f"select={rw.get('share_of_a1_lifecycle_selection_plus_leftover')} "
        f"evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[3E] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--non-evaluative", action="store_true")
    args = p.parse_args(argv)
    run_phase_3e(
        lobbies=args.lobbies if args.lobbies is not None else PHASE_3E_LOBBIES,
        seed=args.seed if args.seed is not None else PHASE_3E_SEED,
        out_dir=args.out_dir,
        non_evaluative=bool(args.non_evaluative),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
