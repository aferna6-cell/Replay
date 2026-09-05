"""Simulator Fidelity Phase 3S — open-slot board-formation attribution.

Measurement only. Reuses consumed Phase 2S–3R DEV 14200–14699. No new seeds.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

    python -m ml.fidelity_phase_3s
    python -m ml.fidelity_phase_3s --lobbies 8 --seed 14200 --non-evaluative
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
from ml.phase_3s_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    EXCLUSIVE_FIRST_DIFF_IDENTITY,
    FEATURE_TOGGLE,
    FORBIDDEN_RANGES,
    FORMATION_FLOW_RECONCILE_IDENTITY,
    HOLD_PRS,
    HISTORY_LINK_IDENTITY,
    HP_FLOW_IDENTITY,
    HP_GAP_RECONCILE_IDENTITY,
    IMPACT_ATTACK_IDENTITY,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    MATCHMAKING_RECONCILE_IDENTITY,
    MEMBERSHIP_PROPAGATION_IDENTITY,
    METHODOLOGY_VERSION,
    NESTED_ALLOCATION_IDENTITY,
    NESTED_FORMATION_IDENTITY,
    NESTED_LIFECYCLE_IDENTITY,
    NESTED_SCALE_SYNC_IDENTITY,
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PAINT_EQUATION_IDENTITY,
    PAINT_RECONCILE_IDENTITY,
    PHASE_3S_LOBBIES,
    PHASE_3S_SEED,
    PLAY_POOL_RECONCILE_IDENTITY,
    POOL_FLOW_IDENTITY,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SAME_STATE_IDENTITY,
    SAME_TURN_SYNC_IDENTITY,
    SCALE_FLOW_RECONCILE_IDENTITY,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    diagnose_phase_3s,
)
from ml.play_lifecycle_diagnostic import compare_play_lifecycle
from ml.pool_lifecycle_diagnostic import compare_lifecycle, summarize_lifecycle_arm
from ml.punch_selection_diagnostic import compare_selection
from ml.allocation_input_diagnostic import compare_allocation_inputs
from ml.open_slot_formation_diagnostic import (
    compare_open_slot_formation,
    run_greedy_2s_treatment_open_slot,
    run_greedy_control_open_slot,
)
from ml.scale_sync_diagnostic import compare_scale_sync
from ml.survivor_mechanic_diagnostic import compare_survivor_mechanics

DEFAULT_DIR = "results/sim_fidelity_phase_3s"
PHASE = "3S open-slot board-formation attribution"


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


def run_phase_3s(
    *,
    lobbies: int = PHASE_3S_LOBBIES,
    seed: int = PHASE_3S_SEED,
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
        raise ValueError(f"Phase 3S must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[3S] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    control_raw = run_greedy_control_open_slot(lobbies, seed)
    greedy_c = summarize_lifecycle_arm(control_raw)
    print("[3S] greedy TREATMENT (2Q + 2S) — same seeds", flush=True)
    treatment_raw = run_greedy_2s_treatment_open_slot(lobbies, seed)
    greedy_t = summarize_lifecycle_arm(treatment_raw)
    greedy_cmp = compare_lifecycle(greedy_c, greedy_t)
    print("[3S] pairing carry trajectories (3F lock)", flush=True)
    divergence = compare_divergence(
        control_raw, treatment_raw, lifecycle_cmp=greedy_cmp,
    )
    print("[3S] reproducing 3G punch-sample mixture", flush=True)
    selection = compare_selection(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
    )
    print("[3S] reproducing 3I pairing-schedule leftover", flush=True)
    pairing = compare_pairing(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence, selection=selection,
    )
    print("[3S] reproducing 3K elimination-timing leftover", flush=True)
    timing = compare_elimination(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing,
    )
    print("[3S] reproducing 3L third-party elimination chain", flush=True)
    chain = compare_chain(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing,
    )
    print("[3S] reproducing 3M first-split classes", flush=True)
    first = compare_first_divergence(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
    )
    print("[3S] reproducing 3N matched-state damage", flush=True)
    matched = compare_matched_state_damage(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first,
    )
    print("[3S] reproducing 3O T5/T6 survivor-mechanic lock", flush=True)
    mechanics = compare_survivor_mechanics(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched,
    )
    print("[3S] reproducing 3P allocation-input lock", flush=True)
    allocation = compare_allocation_inputs(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched, mechanics=mechanics,
    )
    print("[3S] reproducing 3Q play-lifecycle lock", flush=True)
    lifecycle = compare_play_lifecycle(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched, mechanics=mechanics,
        allocation=allocation,
    )
    print("[3S] reproducing 3R scale-sync lock", flush=True)
    scale = compare_scale_sync(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched, mechanics=mechanics,
        allocation=allocation, lifecycle=lifecycle,
    )
    print("[3S] T5/T6 class-(3) open-slot formation attribution", flush=True)
    formation = compare_open_slot_formation(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first, matched=matched, mechanics=mechanics,
        allocation=allocation, lifecycle=lifecycle, scale=scale,
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
        scale["non_evaluative"] = True
        formation["non_evaluative"] = True

    decision = diagnose_phase_3s(formation, non_evaluative=non_evaluative)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(
        os.path.join(out_dir, "attribution.json"),
        _slim_attr(formation.get("attribution")),
    )
    _write_json(
        os.path.join(out_dir, "primary.json"),
        formation.get("primary") or {},
    )
    _write_json(
        os.path.join(out_dir, "per_tier.json"),
        formation.get("per_tier") or {},
    )
    _write_json(
        os.path.join(out_dir, "membership_propagation.json"),
        formation.get("membership_propagation") or {},
    )
    _write_json(
        os.path.join(out_dir, "opening_causes.json"),
        formation.get("opening_causes") or {},
    )
    _write_json(
        os.path.join(out_dir, "source.json"),
        formation.get("source") or {},
    )
    _write_json(
        os.path.join(out_dir, "matched_state_3n.json"),
        _slim_attr(formation.get("matched_state")),
    )
    _write_json(
        os.path.join(out_dir, "mechanics_3o.json"),
        formation.get("mechanics_3o") or {},
    )
    _write_json(
        os.path.join(out_dir, "allocation_3p.json"),
        formation.get("allocation_3p") or {},
    )
    _write_json(
        os.path.join(out_dir, "lifecycle_3q.json"),
        formation.get("lifecycle_3q") or {},
    )
    _write_json(
        os.path.join(out_dir, "scale_3r.json"),
        formation.get("scale_3r") or {},
    )
    _write_json(
        os.path.join(out_dir, "examples.json"),
        {
            "t5_t6_class3": (
                (formation.get("attribution") or {}).get("examples") or []
            ),
        },
    )
    _write_json(
        os.path.join(out_dir, "reconciliation.json"),
        formation.get("reconciliation") or {},
    )
    _write_json(
        os.path.join(out_dir, "decomposition_3g.json"),
        formation.get("decomposition_3g") or {},
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
            "Phase 3S reuses consumed 2S–3R DEV 14200–14699. "
            "Measurement only; no new seeds. Do not rewrite 2Q. "
            "Do not change `_hero_damage`."
        ),
        "non_evaluative": non_evaluative,
        "reused_phase_2s_dev": True,
        "stacked_on_phase_3r": True,
    }
    contract.update({
        "phase": PHASE,
        "phase_3s_methodology_version": METHODOLOGY_VERSION,
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
        "nested_scale_sync_identity": NESTED_SCALE_SYNC_IDENTITY,
        "scale_flow_reconcile_identity": SCALE_FLOW_RECONCILE_IDENTITY,
        "same_turn_sync_identity": SAME_TURN_SYNC_IDENTITY,
        "nested_formation_identity": NESTED_FORMATION_IDENTITY,
        "formation_flow_reconcile_identity": FORMATION_FLOW_RECONCILE_IDENTITY,
        "exclusive_first_diff_identity": EXCLUSIVE_FIRST_DIFF_IDENTITY,
        "membership_propagation_identity": MEMBERSHIP_PROPAGATION_IDENTITY,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "non_evaluative": non_evaluative,
        "decision": decision,
        "contract": contract,
        "attribution": _slim_attr(formation.get("attribution")),
        "primary": formation.get("primary"),
        "per_tier": formation.get("per_tier"),
        "membership_propagation": formation.get("membership_propagation"),
        "opening_causes": formation.get("opening_causes"),
        "lifecycle_3q": formation.get("lifecycle_3q"),
        "scale_3r": formation.get("scale_3r"),
        "source": formation.get("source"),
        "matched_state_3n": _slim_attr(formation.get("matched_state")),
        "mechanics_3o": formation.get("mechanics_3o"),
        "allocation_3p": formation.get("allocation_3p"),
        "reconciliation": formation.get("reconciliation"),
        "decomposition_3g": formation.get("decomposition_3g"),
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "reweighting_3e": greedy_cmp.get("reweighting"),
        "additive_flow": greedy_cmp.get("additive_flow"),
        "deltas": greedy_cmp.get("deltas"),
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_3s_report.json"), report)

    attr = formation.get("attribution") or {}
    primary = formation.get("primary") or {}
    rec = formation.get("reconciliation") or {}
    prop = formation.get("membership_propagation") or {}
    print(
        f"[3S] primary_finding={decision['primary_finding']} "
        f"membership={primary.get('share_of_delta_pre_play_membership')} "
        f"incoming={primary.get('share_of_delta_incoming_identity')} "
        f"opening={primary.get('share_of_delta_slot_opening_cause')} "
        f"order={primary.get('share_of_delta_buy_play_order')} "
        f"residual={primary.get('share_of_delta_residual')} "
        f"lifecycle3q={attr.get('phase_3q_share_lifecycle')} "
        f"lifecycle_ok={attr.get('phase_3q_lifecycle_reproduced')} "
        f"memb3r={attr.get('phase_3r_share_membership')} "
        f"memb_ok={attr.get('phase_3r_membership_reproduced')} "
        f"B3n_ok={attr.get('phase_3n_B_reproduced')} "
        f"B3o_ok={attr.get('phase_3o_B_reproduced')} "
        f"t1_ok={attr.get('t1_synth_reproduced')} "
        f"t3_ok={attr.get('t3_synth_reproduced')} "
        f"nested_ok={rec.get('formation_nested_ok')} "
        f"flow_ok={rec.get('event_board_flow_ok')} "
        f"prop_membership={prop.get('share_of_membership_pre_play_membership')} "
        f"earliest={attr.get('modal_earliest_membership_diverge_turn')} "
        f"evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[3S] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--non-evaluative", action="store_true")
    args = p.parse_args(argv)
    run_phase_3s(
        lobbies=args.lobbies if args.lobbies is not None else PHASE_3S_LOBBIES,
        seed=args.seed if args.seed is not None else PHASE_3S_SEED,
        out_dir=args.out_dir,
        non_evaluative=bool(args.non_evaluative),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
