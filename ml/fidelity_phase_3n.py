"""Simulator Fidelity Phase 3N — first-split matched-state damage attribution.

Measurement only. Reuses consumed Phase 2S–3M DEV 14200–14699. No new seeds.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

    python -m ml.fidelity_phase_3n
    python -m ml.fidelity_phase_3n --lobbies 8 --seed 14200 --non-evaluative
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
from ml.matched_state_damage_diagnostic import (
    compare_matched_state_damage,
    run_greedy_2s_treatment_matched,
    run_greedy_control_matched,
)
from ml.pairing_who_wins_diagnostic import compare_pairing
from ml.phase_3n_prereg import (
    APPLIED_RECONCILE_IDENTITY,
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    COUNTERFACTUAL_IDENTITY,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    FEATURE_TOGGLE,
    FIELD_VS_SURVIVAL_IDENTITY,
    FIRST_DIVERGENCE_RECONCILE_IDENTITY,
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
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PHASE_3N_LOBBIES,
    PHASE_3N_SEED,
    POOL_FLOW_IDENTITY,
    PROXY_ERROR_IDENTITY,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    diagnose_phase_3n,
)
from ml.pool_lifecycle_diagnostic import compare_lifecycle, summarize_lifecycle_arm
from ml.punch_selection_diagnostic import compare_selection

DEFAULT_DIR = "results/sim_fidelity_phase_3n"
PHASE = "3N first-split matched-state damage attribution"


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
    skip = {"examples", "hp_examples", "_chain_cmp"}
    return {k: v for k, v in attr.items() if k not in skip}


def _slim_example(rec: Dict) -> Dict:
    skip_state = {"board_a", "board_b", "winner_board", "loser_board", "survivors"}
    out = {
        k: v for k, v in rec.items()
        if k not in (
            "control_state", "treatment_state", "control", "treatment",
            "control_decisive", "treatment_decisive",
        )
    }
    for key in ("control_state", "treatment_state"):
        state = rec.get(key) or {}
        out[key] = {k: v for k, v in state.items() if k not in skip_state}
        out[key]["survivor_identities"] = state.get("survivor_identities")
        out[key]["winner_start_n"] = state.get("winner_start_n")
        out[key]["winner_start_tier_sum"] = state.get("winner_start_tier_sum")
    return out


def run_phase_3n(
    *,
    lobbies: int = PHASE_3N_LOBBIES,
    seed: int = PHASE_3N_SEED,
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
        raise ValueError(f"Phase 3N must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[3N] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    control_raw = run_greedy_control_matched(lobbies, seed)
    greedy_c = summarize_lifecycle_arm(control_raw)
    print("[3N] greedy TREATMENT (2Q + 2S) — same seeds", flush=True)
    treatment_raw = run_greedy_2s_treatment_matched(lobbies, seed)
    greedy_t = summarize_lifecycle_arm(treatment_raw)
    greedy_cmp = compare_lifecycle(greedy_c, greedy_t)
    print("[3N] pairing carry trajectories (3F lock)", flush=True)
    divergence = compare_divergence(
        control_raw, treatment_raw, lifecycle_cmp=greedy_cmp,
    )
    print("[3N] reproducing 3G punch-sample mixture", flush=True)
    selection = compare_selection(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
    )
    print("[3N] reproducing 3I pairing-schedule leftover", flush=True)
    pairing = compare_pairing(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence, selection=selection,
    )
    print("[3N] reproducing 3K elimination-timing leftover", flush=True)
    timing = compare_elimination(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing,
    )
    print("[3N] reproducing 3L third-party elimination chain", flush=True)
    chain = compare_chain(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing,
    )
    print("[3N] reproducing 3M first-split classes", flush=True)
    first = compare_first_divergence(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
    )
    print("[3N] class-(3) matched-state damage attribution", flush=True)
    matched = compare_matched_state_damage(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing, timing=timing, chain=chain,
        first=first,
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

    decision = diagnose_phase_3n(matched, non_evaluative=non_evaluative)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(
        os.path.join(out_dir, "attribution.json"),
        _slim_attr(matched.get("attribution")),
    )
    _write_json(
        os.path.join(out_dir, "source.json"),
        matched.get("source") or {},
    )
    _write_json(
        os.path.join(out_dir, "very_late_attribution.json"),
        _slim_attr(matched.get("very_late_attribution")),
    )
    _write_json(
        os.path.join(out_dir, "first_divergence_3m.json"),
        _slim_attr(matched.get("first_divergence_3m")),
    )
    _write_json(
        os.path.join(out_dir, "chain_3l.json"),
        _slim_attr(matched.get("chain_3l")),
    )
    _write_json(
        os.path.join(out_dir, "timing_3k.json"),
        _slim_attr(matched.get("timing_3k")),
    )
    _write_json(
        os.path.join(out_dir, "matchmaking_3j.json"),
        _slim_attr(matched.get("matchmaking_3j")),
    )
    _write_json(
        os.path.join(out_dir, "pairing_3i.json"),
        _slim_attr(matched.get("pairing_3i")),
    )
    _write_json(
        os.path.join(out_dir, "leftover_3h.json"),
        matched.get("leftover_3h") or {},
    )
    _write_json(
        os.path.join(out_dir, "examples.json"),
        {
            "class3_matched_state": [
                _slim_example(r)
                for r in (matched.get("attribution") or {}).get("examples") or []
            ],
        },
    )
    _write_json(
        os.path.join(out_dir, "reconciliation.json"),
        matched.get("reconciliation") or {},
    )
    _write_json(
        os.path.join(out_dir, "decomposition_3g.json"),
        matched.get("decomposition_3g") or {},
    )
    _write_json(
        os.path.join(out_dir, "timing_3f.json"),
        divergence.get("timing") or {},
    )
    _write_json(
        os.path.join(out_dir, "reweighting_3e.json"),
        greedy_cmp.get("reweighting") or {},
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
            "Phase 3N reuses consumed 2S–3M DEV 14200–14699. "
            "Measurement only; no new seeds. Do not rewrite 2Q. "
            "Do not change `_hero_damage`."
        ),
        "non_evaluative": non_evaluative,
        "reused_phase_2s_dev": True,
        "stacked_on_phase_3m": True,
    }
    contract.update({
        "phase": PHASE,
        "phase_3n_methodology_version": METHODOLOGY_VERSION,
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
        "first_divergence_reconcile_identity": FIRST_DIVERGENCE_RECONCILE_IDENTITY,
        "row_history_divergence_identity": ROW_HISTORY_DIVERGENCE_IDENTITY,
        "applied_reconcile_identity": APPLIED_RECONCILE_IDENTITY,
        "counterfactual_identity": COUNTERFACTUAL_IDENTITY,
        "proxy_error_identity": PROXY_ERROR_IDENTITY,
        "field_vs_survival_identity": FIELD_VS_SURVIVAL_IDENTITY,
        "row_damage_reconcile_identity": ROW_DAMAGE_RECONCILE_IDENTITY,
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "non_evaluative": non_evaluative,
        "decision": decision,
        "contract": contract,
        "attribution": _slim_attr(matched.get("attribution")),
        "source": matched.get("source"),
        "very_late_attribution": _slim_attr(matched.get("very_late_attribution")),
        "first_divergence_3m": _slim_attr(matched.get("first_divergence_3m")),
        "chain_3l": _slim_attr(matched.get("chain_3l")),
        "timing_3k": _slim_attr(matched.get("timing_3k")),
        "matchmaking_3j": _slim_attr(matched.get("matchmaking_3j")),
        "pairing_3i": _slim_attr(matched.get("pairing_3i")),
        "leftover_3h": matched.get("leftover_3h"),
        "reconciliation": matched.get("reconciliation"),
        "decomposition_3g": matched.get("decomposition_3g"),
        "timing_3f": divergence.get("timing"),
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "reweighting_3e": greedy_cmp.get("reweighting"),
        "additive_flow": greedy_cmp.get("additive_flow"),
        "deltas": greedy_cmp.get("deltas"),
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_3n_report.json"), report)

    attr = matched.get("attribution") or {}
    source = matched.get("source") or {}
    rec = matched.get("reconciliation") or {}
    lock = matched.get("leftover_3h") or {}
    decomp = matched.get("decomposition_3g") or {}
    first_3m = matched.get("first_divergence_3m") or {}
    print(
        f"[3N] primary_finding={decision['primary_finding']} "
        f"tavern={attr.get('share_winner_tavern_tier')} "
        f"count={attr.get('share_survivor_count')} "
        f"composition={attr.get('share_survivor_composition')} "
        f"proxy={attr.get('share_proxy_formula_error')} "
        f"pre_fight={source.get('share_pre_fight_board')} "
        f"within={source.get('share_within_fight_survival')} "
        f"class3_n={attr.get('n_same_outcome_damage')} "
        f"class1_n={first_3m.get('n_same_seat_earlier')} "
        f"leftover_3h={lock.get('leftover')} "
        f"mix3g={decomp.get('share_mixture_turn_winner_tier')} "
        f"hp_ok={rec.get('hp_flow_ok')} "
        f"row_dmg_ok={rec.get('row_damage_ok')} "
        f"recon_ok={attr.get('reconciliation_ok')} "
        f"evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[3N] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--non-evaluative", action="store_true")
    args = p.parse_args(argv)
    run_phase_3n(
        lobbies=args.lobbies if args.lobbies is not None else PHASE_3N_LOBBIES,
        seed=args.seed if args.seed is not None else PHASE_3N_SEED,
        out_dir=args.out_dir,
        non_evaluative=bool(args.non_evaluative),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
