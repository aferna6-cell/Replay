"""Simulator Fidelity Phase 2O — midgame scaling-budget diagnostic.

Measurement only. Fresh DEV seeds 12200–12699 (500 lobbies).
Arms: raw greedy + frozen Phase 2J BoardOpp α=0.5.

Does not alter α, active pool, shop generation, economy, card effects,
combat, PPO, or residual scaling math. Confirm seeds 11500–11699 reserved.

    python -m ml.fidelity_phase_2o
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
    FROZEN_PRIOR_HASH,
    PHASE_2J_PRIOR_PATH,
)
from ml.fidelity_metrics import (
    aggregate_lobby_dynamics,
    aggregate_turn_curves,
    summarize_divergence,
)
from ml.fidelity_phase_2b import THRESHOLDS_PATH, load_frozen_thresholds
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from ml.scaling_budget_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2O_LOBBIES,
    PHASE_2O_SEED,
    PROSPECTIVE_MACRO_POLICY_HARM,
    ROUTING_TABLE,
    aggregate_scaling_budget,
    assert_seed_range_allowed,
    directional_macro_policy_harm,
    route_phase_2o_finding,
    run_greedy_arm,
    run_phase_2j_arm,
    symmetric_absolute_fidelity,
    t10_headline_decomposition,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2o"
PHASE = "2O midgame residual scaling-budget diagnostic"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _arm_summary(raw: Dict) -> Dict:
    records = raw["records"]
    rows = raw["rows"]
    agg = aggregate_scaling_budget(records)
    fid = symmetric_absolute_fidelity(records)
    turn_curves = aggregate_turn_curves(rows)
    lobby = aggregate_lobby_dynamics(rows)
    return {
        "arm": raw["arm"],
        "n_lobbies": raw["n_lobbies"],
        "seed_base": raw["seed_base"],
        "n_records": len(records),
        "aggregation": agg,
        "symmetric_absolute_fidelity_turns_8_14": fid,
        "t10_headline_decomposition": t10_headline_decomposition(agg),
        "turn_curves_end_recruit": turn_curves,
        "lobby_dynamics": lobby,
        "headline_end_recruit": summarize_divergence(turn_curves),
    }


def run_phase_2o(*, lobbies: int = PHASE_2O_LOBBIES,
                 seed: int = PHASE_2O_SEED,
                 out_dir: str = DEFAULT_DIR,
                 alpha: float = FROZEN_ALPHA) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    if abs(alpha - FROZEN_ALPHA) > 1e-12:
        raise ValueError(
            f"Phase 2O must use frozen α={FROZEN_ALPHA}, got {alpha}")

    t0 = time.time()
    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    prior_hash = prior.content_hash_sha256()

    print(f"[2O] greedy arm — {lobbies} lobbies, seeds {seed}–{seed + lobbies - 1}")
    greedy_raw = run_greedy_arm(lobbies, seed)
    print(f"[2O] Phase 2J arm α={alpha} — same seeds")
    treatment_raw = run_phase_2j_arm(lobbies, seed, alpha, prior)

    greedy = _arm_summary(greedy_raw)
    treatment = _arm_summary(treatment_raw)

    routing = route_phase_2o_finding(
        greedy["aggregation"], treatment["aggregation"],
        greedy["symmetric_absolute_fidelity_turns_8_14"],
        treatment["symmetric_absolute_fidelity_turns_8_14"])

    policy_harm = directional_macro_policy_harm(
        greedy["symmetric_absolute_fidelity_turns_8_14"],
        treatment["symmetric_absolute_fidelity_turns_8_14"])

    thresholds, thresholds_sha = load_frozen_thresholds(THRESHOLDS_PATH)
    # Historical Phase 2B envelope reported only (upper-bound era); not retuned.
    hist_envelope = {
        "thresholds_sha256": thresholds_sha,
        "gates": thresholds.get("gates"),
        "note": (
            "Phase 2B envelope was designed around overscaling (T12/T14 upper "
            "bounds only). Preserved unchanged. Phase 2O symmetric absolute "
            "fidelity reports undershoot that those gates cannot catch."
        ),
    }

    decision = {
        "phase": "2O",
        "methodology_version": METHODOLOGY_VERSION,
        "measurement_only": True,
        "keep_2n_catalogue": True,
        "keep_phase_2j_alpha": FROZEN_ALPHA,
        "do_not_merge_29_for_freeze": True,
        "confirm_seeds_reserved": "11500–11699",
        "primary_finding": routing["primary_finding"],
        "recommended_next_step": routing["recommended_next_step"],
        "routing": routing,
        "prospective_macro_policy_harm": policy_harm,
        "note_2n_v3": (
            "2n_v3 remains FAIL as run. T14 treatment−greedy −0.314 is a "
            "semantic failure of a symmetric control-difference guard; do not "
            "retune α or weaken Phase 2J. Prospective directional harm metric "
            "applies to the next fresh evaluation only."
        ),
    }

    contract = build_simulator_v1_1_contract(
        evaluation_seed=seed,
        lobbies=lobbies,
        success_thresholds_sha256=thresholds_sha,
        success_thresholds_path=THRESHOLDS_PATH,
    )
    contract["evaluation"] = {
        "policy": "greedy + frozen BoardOpp α=0.5",
        "lobbies": lobbies,
        "base_seed": seed,
        "note": "Phase 2O scaling-budget diagnostic; not Benchmark v1.",
    }
    contract.update({
        "phase": PHASE,
        "phase_2o_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version_label": SIMULATOR_V1_1_VERSION,
        "frozen_alpha": alpha,
        "prior_hash_sha256": prior_hash,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "instrument_turns": list(range(7, 15)),
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "decision": decision,
        "contract": contract,
        "historical_phase_2b_envelope": hist_envelope,
        "prospective_macro_policy_harm_definition": PROSPECTIVE_MACRO_POLICY_HARM,
        "routing_table": [
            {"finding": f, "next_step": s} for f, s in ROUTING_TABLE],
        "greedy": {
            k: v for k, v in greedy.items()
            if k not in ("turn_curves_end_recruit",)
        },
        "phase_2j": {
            k: v for k, v in treatment.items()
            if k not in ("turn_curves_end_recruit",)
        },
        # Keep full turn curves in separate artifact for size.
        "greedy_turn_curves_end_recruit": greedy["turn_curves_end_recruit"],
        "phase_2j_turn_curves_end_recruit": treatment["turn_curves_end_recruit"],
    }

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "phase_2o_report.json"), report)
    _write_json(os.path.join(out_dir, "t10_decomposition.json"), {
        "greedy": greedy["t10_headline_decomposition"],
        "phase_2j": treatment["t10_headline_decomposition"],
    })
    _write_json(os.path.join(out_dir, "symmetric_fidelity.json"), {
        "greedy": greedy["symmetric_absolute_fidelity_turns_8_14"],
        "phase_2j": treatment["symmetric_absolute_fidelity_turns_8_14"],
        "directional_macro_policy_harm": policy_harm,
    })
    _write_json(os.path.join(out_dir, "routing.json"), routing)

    # Compact per-turn series for the experiment doc.
    _write_json(os.path.join(out_dir, "midgame_curves.json"), {
        "greedy_post_over_firestone": {
            t: (greedy["symmetric_absolute_fidelity_turns_8_14"].get(t) or {}
                ).get("mean_post_scale_over_firestone")
            for t in map(str, range(8, 15))
        },
        "phase_2j_post_over_firestone": {
            t: (treatment["symmetric_absolute_fidelity_turns_8_14"].get(t) or {}
                ).get("mean_post_scale_over_firestone")
            for t in map(str, range(8, 15))
        },
        "greedy_t10": greedy["t10_headline_decomposition"],
        "phase_2j_t10": treatment["t10_headline_decomposition"],
    })

    print(f"[2O] primary_finding={routing['primary_finding']}")
    print(f"[2O] next={routing['recommended_next_step']}")
    print(f"[2O] wrote {out_dir}/ ({contract['runtime_sec']}s)")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=PHASE_2O_LOBBIES)
    p.add_argument("--seed", type=int, default=PHASE_2O_SEED)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--alpha", type=float, default=FROZEN_ALPHA)
    args = p.parse_args(argv)
    run_phase_2o(
        lobbies=args.lobbies, seed=args.seed,
        out_dir=args.out_dir, alpha=args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
