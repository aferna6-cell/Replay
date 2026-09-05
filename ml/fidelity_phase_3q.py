"""Simulator Fidelity Phase 3Q — play-lifecycle sticky-vs-repaint causal audit.

Measurement only. Reuses consumed Phase 2S–3P DEV 14200–14699. No new seeds.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

    python -m ml.fidelity_phase_3q
    python -m ml.fidelity_phase_3q --lobbies 8 --seed 14200 --non-evaluative
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
from ml.carry_divergence_diagnostic import compare_divergence
from ml.elimination_chain_diagnostic import compare_chain
from ml.elimination_timing_diagnostic import compare_elimination
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from ml.hp_divergence_diagnostic import compare_first_divergence
from ml.matched_state_damage_diagnostic import compare_matched_state_damage
from ml.pairing_who_wins_diagnostic import compare_pairing
from ml.phase_3q_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    FEATURE_TOGGLE,
    FORBIDDEN_RANGES,
    HOLD_PRS,
    HISTORY_LINK_IDENTITY,
    HP_FLOW_IDENTITY,
    HP_GAP_RECONCILE_IDENTITY,
    IMPACT_ATTACK_IDENTITY,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    MATCHMAKING_RECONCILE_IDENTITY,
    METHODOLOGY_VERSION,
    NESTED_ALLOCATION_IDENTITY,
    NESTED_LIFECYCLE_IDENTITY,
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PAINT_EQUATION_IDENTITY,
    PAINT_RECONCILE_IDENTITY,
    PHASE_3Q_LOBBIES,
    PHASE_3Q_SEED,
    PLAY_POOL_RECONCILE_IDENTITY,
    POOL_FLOW_IDENTITY,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SAME_STATE_IDENTITY,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    diagnose_phase_3q,
)
from ml.play_lifecycle_diagnostic import (
    compare_play_lifecycle,
    run_greedy_2s_treatment_lifecycle,
    run_greedy_control_lifecycle,
)
from ml.pool_lifecycle_diagnostic import compare_lifecycle, summarize_lifecycle_arm
from ml.punch_selection_diagnostic import compare_selection
from ml.allocation_input_diagnostic import compare_allocation_inputs
from ml.survivor_mechanic_diagnostic import compare_survivor_mechanics

DEFAULT_DIR = "results/sim_fidelity_phase_3q"
PHASE = "3Q play-lifecycle sticky-vs-repaint causal audit"


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


def _slim_attr(attr: Optional[Dict]) -> Dict:
    if not attr:
        return {}
    skip = {"examples", "hp_examples", "_chain_cmp", "control_bodies",
            "treatment_bodies"}
    return {k: v for k, v in attr.items() if k not in skip}


def run_phase_3q(
    *,
    lobbies: int = PHASE_3Q_LOBBIES,
    seed: int = PHASE_3Q_SEED,
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
        raise ValueError(f"Phase 3Q must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[3Q] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    control_raw = run_greedy_control_lifecycle(lobbies, seed)
    greedy_c = summarize_lifecycle_arm(control_raw)
    print("[3Q] greedy TREATMENT (2Q + 2S) — same seeds", flush=True)
    treatment_raw = run_greedy_2s_treatment_lifecycle(lobbies, seed)
    greedy_t = summarize_lifecycle_arm(treatment_raw)
    greedy_cmp = compare_lifecycle(greedy_c, greedy_t)
    print("[3Q] pairing carry trajectories (3F lock)", flush=True)
    divergence = compare_divergence(
        control_raw, treatment_raw, lifecycle_cmp=greedy_cmp,
    )
    print("[3Q] reproducing 3G punch-sample mixture", flush=True)
    selection = compare_selection(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
    )
    print("[3Q] reproducing 3I pairing-schedule leftover", flush=True)
    pairing = compare_pairing(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence, selection=selection,
    )
    print("[3Q] reproducing 3K elimination-timing leftover", flush=True)
    timing = compare_elimination(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing,
    )
    print("[3Q] reproducing 3L third-party elimination chain", flush=True)
    chain = compare_chain(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing,
    )
    print("[3Q] reproducing 3M first-split classes", flush=True)
    first = compare_first_divergence(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
    )
    print("[3Q] reproducing 3N matched-state damage", flush=True)
    matched = compare_matched_state_damage(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first,
    )
    print("[3Q] reproducing 3O T5/T6 survivor-mechanic lock", flush=True)
    mechanics = compare_survivor_mechanics(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched,
    )
    print("[3Q] reproducing 3P allocation-input lock", flush=True)
    allocation = compare_allocation_inputs(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched, mechanics=mechanics,
    )
    print("[3Q] T5/T6 class-(3) play-lifecycle causal audit", flush=True)
    lifecycle = compare_play_lifecycle(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched, mechanics=mechanics,
        allocation=allocation,
    )
    if non_evaluative:
        greedy_cmp["non_evaluative"] = True
        divergence["non_evaluative"] = True
        selection["non_evaluative"] = True
        pairing["non_evaluative"] = True
        timing["non_evaluative"] = True
        chain["non_evaluative"] = True
        first["non_evaluative"] = True
        matched["non_evaluative"] = True
        mechanics["non_evaluative"] = True
        allocation["non_evaluative"] = True
        lifecycle["non_evaluative"] = True

    decision = diagnose_phase_3q(lifecycle, non_evaluative=non_evaluative)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(
        os.path.join(out_dir, "attribution.json"),
        _slim_attr(lifecycle.get("attribution")),
    )
    _write_json(
        os.path.join(out_dir, "primary.json"),
        lifecycle.get("primary") or {},
    )
    _write_json(
        os.path.join(out_dir, "per_tier.json"),
        lifecycle.get("per_tier") or {},
    )
    _write_json(
        os.path.join(out_dir, "plays.json"),
        {
            "control": lifecycle.get("plays_control") or {},
            "treatment": lifecycle.get("plays_treatment") or {},
        },
    )
    _write_json(
        os.path.join(out_dir, "source.json"),
        lifecycle.get("source") or {},
    )
    _write_json(
        os.path.join(out_dir, "matched_state_3n.json"),
        _slim_attr(lifecycle.get("matched_state")),
    )
    _write_json(
        os.path.join(out_dir, "mechanics_3o.json"),
        lifecycle.get("mechanics_3o") or {},
    )
    _write_json(
        os.path.join(out_dir, "allocation_3p.json"),
        lifecycle.get("allocation_3p") or {},
    )
    _write_json(
        os.path.join(out_dir, "examples.json"),
        {
            "t5_t6_class3": (
                (lifecycle.get("attribution") or {}).get("examples") or []
            ),
        },
    )
    _write_json(
        os.path.join(out_dir, "reconciliation.json"),
        lifecycle.get("reconciliation") or {},
    )
    _write_json(
        os.path.join(out_dir, "decomposition_3g.json"),
        lifecycle.get("decomposition_3g") or {},
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
            "Phase 3Q reuses consumed 2S–3P DEV 14200–14699. "
            "Measurement only; no new seeds. Do not rewrite 2Q. "
            "Do not change `_hero_damage`."
        ),
        "non_evaluative": non_evaluative,
        "reused_phase_2s_dev": True,
        "stacked_on_phase_3p": True,
    }
    contract.update({
        "phase": PHASE,
        "phase_3q_methodology_version": METHODOLOGY_VERSION,
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
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "lineage_identity": LINEAGE_IDENTITY,
        "paired_seat_identity": PAIRED_SEAT_IDENTITY,
        "pairing_identity": PAIRING_IDENTITY,
        "leftover_reconcile_identity": LEFTOVER_RECONCILE_IDENTITY,
        "candidate_choice_identity": CANDIDATE_CHOICE_IDENTITY,
        "matchmaking_reconcile_identity": MATCHMAKING_RECONCILE_IDENTITY,
        "hp_flow_identity": HP_FLOW_IDENTITY,
        "elimination_identity": ELIMINATION_IDENTITY,
        "eligibility_timing_identity": ELIGIBILITY_TIMING_IDENTITY,
        "hp_gap_reconcile_identity": HP_GAP_RECONCILE_IDENTITY,
        "chain_reconcile_identity": CHAIN_RECONCILE_IDENTITY,
        "chain_hp_reconcile_identity": CHAIN_HP_RECONCILE_IDENTITY,
        "row_elim_hp_identity": ROW_ELIM_HP_IDENTITY,
        "row_history_divergence_identity": ROW_HISTORY_DIVERGENCE_IDENTITY,
        "row_damage_reconcile_identity": ROW_DAMAGE_RECONCILE_IDENTITY,
        "paint_equation_identity": PAINT_EQUATION_IDENTITY,
        "paint_reconcile_identity": PAINT_RECONCILE_IDENTITY,
        "nested_allocation_identity": NESTED_ALLOCATION_IDENTITY,
        "same_state_identity": SAME_STATE_IDENTITY,
        "nested_lifecycle_identity": NESTED_LIFECYCLE_IDENTITY,
        "play_pool_reconcile_identity": PLAY_POOL_RECONCILE_IDENTITY,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "non_evaluative": non_evaluative,
        "decision": decision,
        "contract": contract,
        "attribution": _slim_attr(lifecycle.get("attribution")),
        "primary": lifecycle.get("primary"),
        "per_tier": lifecycle.get("per_tier"),
        "plays_control": lifecycle.get("plays_control"),
        "plays_treatment": lifecycle.get("plays_treatment"),
        "source": lifecycle.get("source"),
        "matched_state_3n": _slim_attr(lifecycle.get("matched_state")),
        "mechanics_3o": lifecycle.get("mechanics_3o"),
        "allocation_3p": lifecycle.get("allocation_3p"),
        "reconciliation": lifecycle.get("reconciliation"),
        "decomposition_3g": lifecycle.get("decomposition_3g"),
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "reweighting_3e": greedy_cmp.get("reweighting"),
        "additive_flow": greedy_cmp.get("additive_flow"),
        "deltas": greedy_cmp.get("deltas"),
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_3q_report.json"), report)

    attr = lifecycle.get("attribution") or {}
    primary = lifecycle.get("primary") or {}
    rec = lifecycle.get("reconciliation") or {}
    print(
        f"[3Q] primary_finding={decision['primary_finding']} "
        f"same_state={primary.get('share_of_delta_same_state_repaint')} "
        f"lifecycle={primary.get('share_of_delta_replacement_lifecycle')} "
        f"scaling={primary.get('share_of_delta_subsequent_scaling')} "
        f"residual={primary.get('share_of_delta_residual')} "
        f"delta={primary.get('pooled_signed_delta_synth')} "
        f"B3n_ok={attr.get('phase_3n_B_reproduced')} "
        f"B3o_ok={attr.get('phase_3o_B_reproduced')} "
        f"t1_ok={attr.get('t1_synth_reproduced')} "
        f"t3_ok={attr.get('t3_synth_reproduced')} "
        f"paint_ok={rec.get('paint_ok')} "
        f"nested_ok={rec.get('primary_nested_ok')} "
        f"evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[3Q] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--non-evaluative", action="store_true")
    args = p.parse_args(argv)
    run_phase_3q(
        lobbies=args.lobbies if args.lobbies is not None else PHASE_3Q_LOBBIES,
        seed=args.seed if args.seed is not None else PHASE_3Q_SEED,
        out_dir=args.out_dir,
        non_evaluative=bool(args.non_evaluative),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
