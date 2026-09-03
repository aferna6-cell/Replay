"""Simulator Fidelity Phase 2Q — recruit vs combat representation split.

Feature toggle ``PHASE_2Q_RECRUIT_VALUE_STATS``:
  Control  False — replacement uses live combat stats (contaminated)
  Treatment True  — replacement uses recruit-value stats (excl. synthetic scaling)

Combat path unchanged. Residual scaling magnitude unchanged. α unchanged.

Fresh DEV seeds 13200–13699. Confirm 11500–11699 reserved.

    python -m ml.fidelity_phase_2q
    python -m ml.fidelity_phase_2q --lobbies 2 --seed 13200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from hsbg_coach.board_opportunity_policy import (
    METHODOLOGY_VERSION as PHASE_2J_VERSION,
)
from ml.availability_decomposition import (
    FROZEN_ALPHA,
    PHASE_2J_PRIOR_PATH,
)
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_reference import (
    FIDELITY_BENCHMARK_VERSION,
    SIMULATOR_V1_1_VERSION,
    build_simulator_v1_1_contract,
    git_commit,
    git_working_tree_clean,
)
from ml.recruit_combat_split_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2Q_LOBBIES,
    PHASE_2Q_SEED,
    assert_seed_range_allowed,
    compare_control_treatment,
    diagnose_phase_2q,
    run_greedy_control,
    run_greedy_treatment,
    run_phase_2j_control,
    run_phase_2j_treatment,
    summarize_split_arm,
)

DEFAULT_DIR = "results/sim_fidelity_phase_2q"
PHASE = "2Q recruit-value vs combat representation split"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _slim_arm(summary: Dict) -> Dict:
    """Drop heavy turn_curves from the main report (kept separately if needed)."""
    out = {k: v for k, v in summary.items() if k != "turn_curves_end_recruit"}
    return out


def _mechanism_snapshot(summary: Dict) -> Dict:
    mech = summary.get("mechanism") or {}
    seeded = mech.get("seeded_current_target") or {}
    committed = mech.get("committed_current_target") or {}
    ps = summary.get("policy_stats") or {}
    funnel = summary.get("lifecycle_funnel") or {}
    played = funnel.get("played")
    fulfilled = seeded.get("fulfilled_exposures")
    played_rate = None
    if played is not None and fulfilled:
        played_rate = played / fulfilled if fulfilled else None
    return {
        "persistent_2_plus": seeded.get("reached_2_core"),
        "committed_states": committed.get("n_lobby_archetype_states"),
        "fulfilled_exposures": fulfilled,
        "played": played,
        "played_rate": played_rate,
        "coverage_mean": mech.get("sim_final_winner_coverage_mean"),
        "replacement_transitions": ps.get("replacement_transitions"),
        "mean_relative_tempo_loss": ps.get("mean_relative_tempo_loss"),
        "p95_relative_tempo_loss": ps.get("p95_relative_tempo_loss"),
    }


def run_phase_2q(
    *,
    lobbies: int = PHASE_2Q_LOBBIES,
    seed: int = PHASE_2Q_SEED,
    out_dir: str = DEFAULT_DIR,
    alpha: float = FROZEN_ALPHA,
    skip_phase_2j: bool = False,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    if abs(alpha - FROZEN_ALPHA) > 1e-12:
        raise ValueError(f"Phase 2Q must use frozen α={FROZEN_ALPHA}, got {alpha}")

    t0 = time.time()
    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    prior_hash = prior.content_hash_sha256()

    print(f"[2Q] greedy CONTROL (scaled valuation) — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}", flush=True)
    greedy_c_raw = run_greedy_control(lobbies, seed)
    greedy_c = summarize_split_arm(greedy_c_raw)
    del greedy_c_raw
    print(f"[2Q] greedy TREATMENT (recruit-value valuation) — same seeds",
          flush=True)
    greedy_t_raw = run_greedy_treatment(lobbies, seed)
    greedy_t = summarize_split_arm(greedy_t_raw)
    del greedy_t_raw
    greedy_cmp = compare_control_treatment(greedy_c, greedy_t)
    print(f"[2Q] greedy gates_passed={greedy_cmp['gates_passed']}/"
          f"{greedy_cmp['gates_total']} {greedy_cmp['gates']}", flush=True)

    phase_2j_c = phase_2j_t = phase_2j_cmp = None
    phase_2j_mechanism = None
    if not skip_phase_2j:
        print(f"[2Q] Phase 2J CONTROL α={alpha} — same seeds", flush=True)
        j_c_raw = run_phase_2j_control(
            lobbies, seed, alpha, prior, include_composition=False
        )
        phase_2j_c = summarize_split_arm(j_c_raw)
        del j_c_raw
        print(f"[2Q] Phase 2J TREATMENT α={alpha} — same seeds", flush=True)
        j_t_raw = run_phase_2j_treatment(
            lobbies, seed, alpha, prior, include_composition=False
        )
        phase_2j_t = summarize_split_arm(j_t_raw)
        del j_t_raw
        phase_2j_cmp = compare_control_treatment(phase_2j_c, phase_2j_t)
        phase_2j_mechanism = {
            "control": _mechanism_snapshot(phase_2j_c),
            "treatment": _mechanism_snapshot(phase_2j_t),
            "deltas": {
                "persistent_2_plus": _delta_safe(
                    _mechanism_snapshot(phase_2j_c).get("persistent_2_plus"),
                    _mechanism_snapshot(phase_2j_t).get("persistent_2_plus"),
                ),
                "committed_states": _delta_safe(
                    _mechanism_snapshot(phase_2j_c).get("committed_states"),
                    _mechanism_snapshot(phase_2j_t).get("committed_states"),
                ),
                "played_rate": _delta_safe(
                    _mechanism_snapshot(phase_2j_c).get("played_rate"),
                    _mechanism_snapshot(phase_2j_t).get("played_rate"),
                ),
                "coverage_mean": _delta_safe(
                    _mechanism_snapshot(phase_2j_c).get("coverage_mean"),
                    _mechanism_snapshot(phase_2j_t).get("coverage_mean"),
                ),
                "replacement_transitions": _delta_safe(
                    _mechanism_snapshot(phase_2j_c).get("replacement_transitions"),
                    _mechanism_snapshot(phase_2j_t).get("replacement_transitions"),
                ),
                "mean_relative_tempo_loss": _delta_safe(
                    _mechanism_snapshot(phase_2j_c).get("mean_relative_tempo_loss"),
                    _mechanism_snapshot(phase_2j_t).get("mean_relative_tempo_loss"),
                ),
            },
            "directional_macro_policy_harm": phase_2j_cmp.get(
                "directional_macro_policy_harm"
            ),
            "note": (
                "α=0.5 frozen without retune; input value space changed so "
                "numerical optimality is not required — mechanism survival only. "
                "Composition-trace persistent-2+/coverage omitted from this DEV "
                "pass for memory; policy_stats replacement/tempo reported."
            ),
        }
        print(f"[2Q] phase_2j gates_passed={phase_2j_cmp['gates_passed']}/"
              f"{phase_2j_cmp['gates_total']}", flush=True)

    decision = diagnose_phase_2q(
        greedy_cmp, phase_2j_cmp, phase_2j_mechanism=phase_2j_mechanism
    )

    contract = build_simulator_v1_1_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["evaluation"] = {
        "policy": "greedy ± recruit-value; BoardOpp α=0.5 ± recruit-value",
        "lobbies": lobbies,
        "base_seed": seed,
        "note": "Phase 2Q representation-split A/B; not Benchmark v1.",
    }
    contract.update({
        "phase": PHASE,
        "phase_2q_methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "fidelity_benchmark_version": FIDELITY_BENCHMARK_VERSION,
        "simulator_version_label": SIMULATOR_V1_1_VERSION,
        "frozen_alpha": alpha,
        "prior_hash_sha256": prior_hash,
        "forbidden_seed_ranges": [f"{a}–{b}" for a, b in FORBIDDEN_RANGES],
        "feature_toggle": "PHASE_2Q_RECRUIT_VALUE_STATS",
        "code_commit": git_commit(),
        "working_tree_clean": git_working_tree_clean(),
        "runtime_sec": round(time.time() - t0, 2),
    })

    report = {
        "methodology_version": METHODOLOGY_VERSION,
        "decision": decision,
        "contract": contract,
        "greedy_control": _slim_arm(greedy_c),
        "greedy_treatment": _slim_arm(greedy_t),
        "greedy_comparison": greedy_cmp,
        "phase_2j_control": _slim_arm(phase_2j_c) if phase_2j_c else None,
        "phase_2j_treatment": _slim_arm(phase_2j_t) if phase_2j_t else None,
        "phase_2j_comparison": phase_2j_cmp,
        "phase_2j_mechanism": phase_2j_mechanism,
    }

    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    _write_json(os.path.join(out_dir, "decision.json"), decision)
    _write_json(os.path.join(out_dir, "phase_2q_report.json"), report)
    _write_json(os.path.join(out_dir, "greedy_comparison.json"), greedy_cmp)
    if phase_2j_cmp is not None:
        _write_json(os.path.join(out_dir, "phase_2j_comparison.json"), phase_2j_cmp)
    if phase_2j_mechanism is not None:
        _write_json(
            os.path.join(out_dir, "phase_2j_mechanism.json"), phase_2j_mechanism
        )

    print(f"[2Q] primary_finding={decision['primary_finding']}")
    print(f"[2Q] wrote {out_dir}/ ({contract['runtime_sec']}s)")
    return report


def _delta_safe(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lobbies", type=int, default=PHASE_2Q_LOBBIES)
    p.add_argument("--seed", type=int, default=PHASE_2Q_SEED)
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    p.add_argument("--skip-phase-2j", action="store_true")
    args = p.parse_args(argv)
    run_phase_2q(
        lobbies=args.lobbies,
        seed=args.seed,
        out_dir=args.out_dir,
        skip_phase_2j=args.skip_phase_2j,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
