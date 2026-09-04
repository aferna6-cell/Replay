"""Simulator Fidelity Phase 3K — elimination-timing attribution.

Measurement only. Reuses consumed Phase 2S–3J DEV 14200–14699. No new seeds.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

    python -m ml.fidelity_phase_3k
    python -m ml.fidelity_phase_3k --lobbies 8 --seed 14200 --non-evaluative
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
from ml.elimination_timing_diagnostic import (
    compare_elimination,
    run_greedy_2s_treatment_elimination,
    run_greedy_control_elimination,
)
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from ml.pairing_who_wins_diagnostic import compare_pairing
from ml.phase_3k_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
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
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PHASE_3K_LOBBIES,
    PHASE_3K_SEED,
    POOL_FLOW_IDENTITY,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    diagnose_phase_3k,
)
from ml.pool_lifecycle_diagnostic import compare_lifecycle, summarize_lifecycle_arm
from ml.punch_selection_diagnostic import compare_selection

DEFAULT_DIR = "results/sim_fidelity_phase_3k"
PHASE = "3K elimination-timing attribution"


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
    skip = {"examples", "hp_examples"}
    return {k: v for k, v in attr.items() if k not in skip}


def run_phase_3k(
    *,
    lobbies: int = PHASE_3K_LOBBIES,
    seed: int = PHASE_3K_SEED,
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
        raise ValueError(f"Phase 3K must keep frozen α=0.5, got {FROZEN_ALPHA}")

    t0 = time.time()
    tag = "SMOKE (non-evaluative)" if non_evaluative else "DEV"
    print(
        f"[3K] {tag} greedy CONTROL — {lobbies} lobbies, "
        f"seeds {seed}–{seed + lobbies - 1}",
        flush=True,
    )
    control_raw = run_greedy_control_elimination(lobbies, seed)
    greedy_c = summarize_lifecycle_arm(control_raw)
    print("[3K] greedy TREATMENT (2Q + 2S) — same seeds", flush=True)
    treatment_raw = run_greedy_2s_treatment_elimination(lobbies, seed)
    greedy_t = summarize_lifecycle_arm(treatment_raw)
    greedy_cmp = compare_lifecycle(greedy_c, greedy_t)
    print("[3K] pairing carry trajectories (3F lock)", flush=True)
    divergence = compare_divergence(
        control_raw, treatment_raw, lifecycle_cmp=greedy_cmp,
    )
    print("[3K] reproducing 3G punch-sample mixture", flush=True)
    selection = compare_selection(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
    )
    print("[3K] reproducing 3I pairing-schedule leftover", flush=True)
    pairing = compare_pairing(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence, selection=selection,
    )
    print("[3K] elimination-timing attribution", flush=True)
    timing = compare_elimination(
        control_raw, treatment_raw,
        lifecycle_cmp=greedy_cmp, divergence=divergence,
        selection=selection, pairing=pairing,
    )
    if non_evaluative:
        greedy_cmp["non_evaluative"] = True
        divergence["non_evaluative"] = True
        selection["non_evaluative"] = True
        pairing["non_evaluative"] = True
        timing["non_evaluative"] = True

    decision = diagnose_phase_3k(timing, non_evaluative=non_evaluative)

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(
        os.path.join(out_dir, "attribution.json"),
        _slim_attr(timing.get("attribution")),
    )
    _write_json(
        os.path.join(out_dir, "very_late_attribution.json"),
        _slim_attr(timing.get("very_late_attribution")),
    )
    _write_json(
        os.path.join(out_dir, "matchmaking_3j.json"),
        _slim_attr(timing.get("matchmaking_3j")),
    )
    _write_json(
        os.path.join(out_dir, "pairing_3i.json"),
        _slim_attr(timing.get("pairing_3i")),
    )
    _write_json(
        os.path.join(out_dir, "leftover_3h.json"),
        timing.get("leftover_3h") or {},
    )
    _write_json(
        os.path.join(out_dir, "examples.json"),
        {
            "timing": (timing.get("attribution") or {}).get("examples") or {},
            "hp_gap": (timing.get("attribution") or {}).get("hp_examples") or {},
        },
    )
    _write_json(
        os.path.join(out_dir, "reconciliation.json"),
        timing.get("reconciliation") or {},
    )
    _write_json(
        os.path.join(out_dir, "decomposition_3g.json"),
        timing.get("decomposition_3g") or {},
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
            "Phase 3K reuses consumed 2S–3J DEV 14200–14699. "
            "Measurement only; no new seeds. Do not rewrite 2Q."
        ),
        "non_evaluative": non_evaluative,
        "reused_phase_2s_dev": True,
        "stacked_on_phase_3j": True,
    }
    contract.update({
        "phase": PHASE,
        "phase_3k_methodology_version": METHODOLOGY_VERSION,
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
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "non_evaluative": non_evaluative,
        "decision": decision,
        "contract": contract,
        "attribution": _slim_attr(timing.get("attribution")),
        "very_late_attribution": _slim_attr(timing.get("very_late_attribution")),
        "matchmaking_3j": _slim_attr(timing.get("matchmaking_3j")),
        "pairing_3i": _slim_attr(timing.get("pairing_3i")),
        "leftover_3h": timing.get("leftover_3h"),
        "reconciliation": timing.get("reconciliation"),
        "decomposition_3g": timing.get("decomposition_3g"),
        "timing_3f": divergence.get("timing"),
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "reweighting_3e": greedy_cmp.get("reweighting"),
        "additive_flow": greedy_cmp.get("additive_flow"),
        "deltas": greedy_cmp.get("deltas"),
    }
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_3k_report.json"), report)

    attr = timing.get("attribution") or {}
    rec = timing.get("reconciliation") or {}
    lock = timing.get("leftover_3h") or {}
    decomp = timing.get("decomposition_3g") or {}
    mm = timing.get("matchmaking_3j") or {}
    print(
        f"[3K] primary_finding={decision['primary_finding']} "
        f"treat={attr.get('share_treatment_eliminated_earlier')} "
        f"ctrl={attr.get('share_control_opponent_eliminated_earlier')} "
        f"third={attr.get('share_ghost_bye_third_party')} "
        f"unrec={attr.get('share_unreconciled')} "
        f"prior={attr.get('share_accumulated_prior_hp')} "
        f"hit={attr.get('share_current_fight_hit')} "
        f"mag={attr.get('share_current_fight_damage_magnitude')} "
        f"elig_n={attr.get('n_eligibility')} "
        f"elig3j={mm.get('attributed', mm).get('eligibility') if isinstance(mm.get('attributed', mm), dict) else mm.get('eligibility')} "
        f"leftover_3h={lock.get('leftover')} "
        f"mix3g={decomp.get('share_mixture_turn_winner_tier')} "
        f"hp_ok={rec.get('hp_flow_ok')} "
        f"elim_ok={rec.get('elimination_ok')} "
        f"recon_ok={attr.get('reconciliation_ok')} "
        f"evaluative={decision.get('evaluative')}",
        flush=True,
    )
    print(f"[3K] wrote {out_dir}/ ({contract['runtime_sec']}s)", flush=True)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--non-evaluative", action="store_true")
    args = p.parse_args(argv)
    run_phase_3k(
        lobbies=args.lobbies if args.lobbies is not None else PHASE_3K_LOBBIES,
        seed=args.seed if args.seed is not None else PHASE_3K_SEED,
        out_dir=args.out_dir,
        non_evaluative=bool(args.non_evaluative),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
