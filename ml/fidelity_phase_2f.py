"""Simulator Fidelity Phase 2F — post-purchase core lifecycle diagnosis.

    python -m ml.fidelity_phase_2f
    python -m ml.fidelity_phase_2f --lobbies 200 --seed 1000

Measurement-only: trace each fulfilled seeded exposure in the Phase 2E oracle
treatment from purchase through disappearance or game end. Fresh seeds 1000–1199.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Optional

from hsbg_coach.bg_env import seeded_core_stress_greedy_policy
from hsbg_coach.seeded_core_stress_policy import (PHASE_2E_EVAL_SEED_BASE,
                                                  POLICY_CONFIG_FINGERPRINT)

from .composition_diagnostic import METHODOLOGY_VERSION as PHASE_2C_VERSION
from .composition_trace import run_traced_rollouts
from .core_lifecycle_diagnostic import (METHODOLOGY_VERSION,
                                        analyze_core_lifecycles,
                                        lifecycle_meets_fulfillment_count)
from .fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                 SIMULATOR_V1_1_VERSION,
                                 build_simulator_v1_1_contract, git_commit,
                                 git_working_tree_clean)
from .phase_2d_acceptance import composition_mechanism_summary
from .phase_2f_decision import evaluate_phase_2f_decision

DEFAULT_DIR = "results/sim_fidelity_phase_2f"
DEFAULT_SEED = PHASE_2E_EVAL_SEED_BASE
PHASE = "2F post-purchase core lifecycle diagnosis"


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_config_hash() -> str:
    blob = json.dumps(POLICY_CONFIG_FINGERPRINT, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def build_phase_2f_contract(*, evaluation_seed: int = DEFAULT_SEED,
                            lobbies: int = 200) -> Dict:
    base = build_simulator_v1_1_contract(
        evaluation_seed=evaluation_seed, lobbies=lobbies)
    base["phase"] = PHASE
    base["phase_2c_methodology_version"] = PHASE_2C_VERSION
    base["phase_2f_methodology_version"] = METHODOLOGY_VERSION
    base["policy_config"] = POLICY_CONFIG_FINGERPRINT
    base["policy_config_hash_sha256"] = _policy_config_hash()
    base["policy"] = {
        "name": "hsbg_coach.bg_env.seeded_core_stress_greedy_policy",
        "note": "Same oracle as Phase 2E treatment — measurement only",
    }
    base["evaluation_note"] = (
        f"Lifecycle trace on seeds {evaluation_seed}–"
        f"{evaluation_seed + lobbies - 1}; Phase 2E oracle treatment.")
    return base


def run_phase_2f(*, lobbies: int = 200, seed: int = DEFAULT_SEED,
                 out_dir: str = DEFAULT_DIR,
                 require_clean_tree: bool = True) -> Dict:
    impl_commit = git_commit()
    tree_clean = git_working_tree_clean()
    if require_clean_tree and not tree_clean:
        raise RuntimeError(
            "Working tree is not clean. Commit Phase 2F implementation first.")

    t0 = time.time()
    print(f"Phase 2F lifecycle diagnosis — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}")
    print("  [treatment] composition traces…")
    traces = run_traced_rollouts(
        lobbies, seed=seed, policy=seeded_core_stress_greedy_policy,
        scaling_mode="residual")

    from .composition_diagnostic import aggregate_diagnostics
    diagnostic = aggregate_diagnostics(traces)
    mechanism = composition_mechanism_summary(diagnostic)
    lifecycle = analyze_core_lifecycles(traces)
    latch_ok = lifecycle_meets_fulfillment_count(traces)
    decision = evaluate_phase_2f_decision(lifecycle)
    contract = build_phase_2f_contract(evaluation_seed=seed, lobbies=lobbies)
    contract["working_tree_clean"] = tree_clean

    seeded = mechanism.get("seeded_current_target") or {}
    result = {
        "benchmark": FIDELITY_BENCHMARK_VERSION,
        "phase": PHASE,
        "research_question": (
            "What happened to fulfilled seeded core purchases after acquisition "
            "under oracle stress?"),
        "simulator_version": SIMULATOR_V1_1_VERSION,
        "phase_2c_methodology_version": PHASE_2C_VERSION,
        "phase_2f_methodology_version": METHODOLOGY_VERSION,
        "implementation_commit": impl_commit,
        "code_commit": impl_commit,
        "working_tree_clean": tree_clean,
        "policy_config": POLICY_CONFIG_FINGERPRINT,
        "policy_config_hash_sha256": _policy_config_hash(),
        "contract": contract,
        "runtime_seconds": round(time.time() - t0, 2),
        "n_lobbies": lobbies,
        "evaluation_seed_base": seed,
        "seeded_mechanism_summary": seeded,
        "lifecycle_latch_parity_ok": latch_ok,
        "lifecycle": {
            "n_fulfilled_purchases": lifecycle["n_fulfilled_purchases"],
            "fate_totals": lifecycle["fate_totals"],
            "funnel": lifecycle["funnel"],
            "fate_labels": lifecycle["fate_labels"],
            "board_full_summary": lifecycle["board_full_summary"],
        },
        "lifecycle_purchases": lifecycle["purchases"],
        "decision": decision,
        "phase_2e_context": (
            "Runs oracle treatment only; explains Phase 2E 34/49 fulfillment with "
            "0 end-of-recruit 2+ core assembly."),
    }
    _write_json(os.path.join(out_dir, "phase_2f_report.json"), result)
    _write_json(os.path.join(out_dir, "contract.json"), contract)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lobbies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--allow-dirty-tree", action="store_true")
    args = ap.parse_args(argv)

    print(f"{FIDELITY_BENCHMARK_VERSION} — Phase 2F")
    print(f"Implementation commit: {git_commit()}")
    print(f"Working tree clean: {git_working_tree_clean()}")

    try:
        result = run_phase_2f(
            lobbies=args.lobbies, seed=args.seed, out_dir=args.out_dir,
            require_clean_tree=not args.allow_dirty_tree)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    lc = result["lifecycle"]
    dec = result["decision"]
    bfs = lc.get("board_full_summary") or {}
    print(f"\nFulfilled seeded purchases: {lc['n_fulfilled_purchases']}")
    print(f"Latch parity with 2C fulfilled_exposures: "
          f"{result['lifecycle_latch_parity_ok']}")
    print("Board-full summary:")
    for k, v in bfs.items():
        print(f"  {k}: {v}")
    print("Funnel:")
    for k, v in lc["funnel"].items():
        print(f"  {k}: {v}")
    print("Fate totals:")
    for fate, cnt in sorted(lc["fate_totals"].items()):
        print(f"  {fate}: {cnt}")
    print(f"\nDecision branch: {dec['decision_branch']}")
    print(f"  {dec['recommended_next_step']}")
    print(f"\nSaved -> {args.out_dir}/phase_2f_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
