"""Phase 2B: paired Simulator v1 vs v1.1 fidelity experiment.

    python -m ml.fidelity_phase_2b
    python -m ml.fidelity_phase_2b --lobbies 200 --seed 0

Steps:
1. Re-run Simulator v1 (ratio scaling) on the evaluation seeds (paired baseline).
2. Load the frozen bootstrap success thresholds (do not overwrite by default).
3. Run Simulator v1.1 (residual scaling) on the same seeds.
4. Report paired per-lobby comparison and accept/reject gates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

from .fidelity_benchmark import print_table
from .fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                               aggregate_turn_curves, run_fidelity_rollouts,
                               summarize_divergence)
from .fidelity_paired import (evaluate_gates, freeze_success_thresholds,
                              paired_turn_comparison, per_lobby_turn_means)
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION, SIMULATOR_VERSION,
                                 build_simulator_v1_1_contract, file_sha256,
                                 git_commit, git_working_tree_clean)

DEFAULT_DIR = "results/sim_fidelity_v1_1"
THRESHOLDS_PATH = "results/sim_fidelity_v1/success_thresholds.json"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_frozen_thresholds(path: str) -> Tuple[Dict, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Frozen success thresholds not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, file_sha256(path)


def run_phase_2b(*, lobbies: int = 200, seed: int = 0,
                 out_dir: str = DEFAULT_DIR,
                 thresholds_path: str = THRESHOLDS_PATH,
                 freeze_thresholds: bool = False) -> Dict:
    t0 = time.time()

    print(f"Step 1/3: Simulator v1 rollouts ({lobbies} lobbies, ratio scaling)…")
    rows_v1 = run_fidelity_rollouts(lobbies, seed=seed, scaling_mode="ratio")
    per_v1 = per_lobby_turn_means(rows_v1)

    if freeze_thresholds:
        print("Step 2/3: Freeze bootstrap success thresholds from v1 variability…")
        thresholds = freeze_success_thresholds(per_v1, seed=seed)
        _write_json(thresholds_path, thresholds)
        thresholds_sha = file_sha256(thresholds_path)
    else:
        print(f"Step 2/3: Load frozen success thresholds ({thresholds_path})…")
        thresholds, thresholds_sha = load_frozen_thresholds(thresholds_path)

    print(f"Step 3/3: Simulator v1.1 rollouts ({lobbies} lobbies, residual scaling)…")
    t1 = time.time()
    rows_v11 = run_fidelity_rollouts(lobbies, seed=seed, scaling_mode="residual")
    runtime_v11 = round(time.time() - t1, 2)

    per_v11 = per_lobby_turn_means(rows_v11)
    paired = paired_turn_comparison(per_v1, per_v11)
    turn_curves = aggregate_turn_curves(rows_v11)
    composition = aggregate_composition(rows_v11)
    lobby = aggregate_lobby_dynamics(rows_v11)
    headline = summarize_divergence(turn_curves)
    gates = evaluate_gates(thresholds, paired, turn_curves, lobby)
    contract = build_simulator_v1_1_contract(
        evaluation_seed=seed, lobbies=lobbies,
        success_thresholds_sha256=thresholds_sha,
        success_thresholds_path=thresholds_path)

    ref_label = contract.get("reference_metadata", {}).get(
        "reference_label", "Firestone reference distribution")
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": "2B residual scaling correction",
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "baseline_simulator_version": SIMULATOR_VERSION,
        "reference_label": ref_label,
        "contract": contract,
        "code_commit": contract["code_commit"],
        "working_tree_clean": contract["working_tree_clean"],
        "success_thresholds_sha256": thresholds_sha,
        "success_thresholds_frozen": not freeze_thresholds,
        "runtime_seconds": round(time.time() - t0, 2),
        "runtime_v1_1_rollouts_seconds": runtime_v11,
        "success_thresholds_path": thresholds_path,
        "success_thresholds": thresholds,
        "paired_comparison": paired,
        "gate_evaluation": gates,
        "headline_divergence": headline,
        "turn_curves": turn_curves,
        "composition": composition,
        "lobby_dynamics": lobby,
        "note_composition": "Reported but not optimized in Phase 2B.",
    }
    _write_json(os.path.join(out_dir, "phase_2b_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--thresholds-path", default=THRESHOLDS_PATH)
    ap.add_argument("--freeze-thresholds", action="store_true",
                    help="Regenerate thresholds (initial Phase 2B only; default loads frozen file)")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2B")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")
    result = run_phase_2b(
        lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
        thresholds_path=args.thresholds_path,
        freeze_thresholds=args.freeze_thresholds)

    print_table(result["turn_curves"])
    paired = result["paired_comparison"]
    print("\nPaired sim/real stats ratios (identical lobby set per turn):")
    for t in ("10", "12", "14"):
        row = paired.get(t, {})
        if not row:
            continue
        n = row.get("n_paired_lobbies", 0)
        print(f"  Turn {t}: {row.get('v1_mean_ratio', 0):.2f}× → "
              f"{row.get('v1_1_mean_ratio', 0):.2f}×  "
              f"(paired Δ mean stats {row.get('paired_mean_delta_sim_stats', 0):+.0f}, "
              f"{row.get('v1_lobbies_improved_ratio', 0)}/{n} lobbies improved ratio)")
        unpaired = row.get("unpaired_aggregate", {})
        if unpaired:
            print(f"           unpaired aggregate: v1 {unpaired.get('v1_mean_ratio', 0):.2f}× "
                  f"(n={unpaired.get('n_v1_lobbies', 0)}) → "
                  f"v1.1 {unpaired.get('v1_1_mean_ratio', 0):.2f}× "
                  f"(n={unpaired.get('n_v1_1_lobbies', 0)})")

    gates = result["gate_evaluation"]
    print(f"\nGate evaluation (accept v1.1 = {gates['accept_v1_1']}):")
    for name in ("turn_14_primary", "turn_12_secondary", "turn_10_regression",
                 "tavern_tier_unchanged", "alive_curve_unchanged"):
        g = gates[name]
        status = "PASS" if g.get("passed") else "FAIL"
        print(f"  {name}: {status}  {g}")
    gl = gates.get("game_length", {})
    print(f"  game_length: monitored only — {gl.get('value')} turns  ({gl.get('note')})")

    print(f"\nContract code_commit: {result['code_commit']}")
    print(f"Threshold SHA: {result['success_thresholds_sha256']}")
    print(f"\nSaved -> {args.out_dir}/phase_2b_report.json")
    return 0 if gates["accept_v1_1"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
