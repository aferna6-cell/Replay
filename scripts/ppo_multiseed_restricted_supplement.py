"""Supplementary comparisons for a checkpoint that fails the DEV protocol.

    python scripts/ppo_multiseed_restricted_supplement.py

The primary analysis leaves training seed 1's 5,120-episode checkpoint
unscored: it stalls on some of the fixed DEV lobbies, so the frozen protocol
gives it no placement. That is the correct headline treatment and this script
does not change it.

It answers the narrower question the pre-specified protocol still asks —
"iter320 − iter80 for every seed" — on the only footing available: the
lobbies the failing checkpoint did finish. Every row it writes is stamped
restricted and carries the bias note, because the excluded lobbies are
exactly the ones where the policy degenerated.

Reads committed JSON only; writes
``results/ppo_multiseed_v1/aggregate/restricted_supplement.json``.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dev_partial import BIAS_NOTE, restricted_pairs      # noqa: E402
from ml.multiseed_analysis import (MULTI_DIR, PRIMARY_ITERS,  # noqa: E402
                                   load_dev_result, load_json, write_json)

REFERENCES = [0, 40, 80, 160]


def supplement() -> dict:
    rows = []
    for path in sorted(glob.glob(
            f"{MULTI_DIR}/seed_*/dev/*.protocol_failure.json")):
        diagnostic = load_json(path)
        directory = os.path.dirname(os.path.dirname(path))
        field = diagnostic["field"]
        scored = {it: load_dev_result(directory, it, field)
                  for it in PRIMARY_ITERS
                  if it != diagnostic["ppo_iteration"]}
        pairs = restricted_pairs(diagnostic, scored, REFERENCES)
        rows.append({
            "training_seed": diagnostic["training_seed"],
            "iteration": diagnostic["ppo_iteration"],
            "field": field,
            "games_attempted": diagnostic["games_attempted"],
            "games_non_terminating": diagnostic["n_non_terminating"],
            "non_terminating_game_seeds":
                diagnostic["non_terminating_game_seeds"],
            "completed_subset_avg_placement":
                diagnostic["completed_games_diagnostic"]["avg_placement"],
            "pairs": pairs,
            "source": os.path.relpath(path),
        })
    return {
        "kind": "RESTRICTED SUPPLEMENT — not a benchmark or DEV result",
        "why": ("the primary analysis leaves an unscoreable checkpoint "
                "unscored; these paired differences exist only on the "
                "lobbies it finished and are reported so the pre-specified "
                "per-seed questions have a stated answer rather than a gap"),
        "bias": BIAS_NOTE,
        "checkpoints": rows,
    }


def main() -> int:
    blob = supplement()
    out = os.path.join(MULTI_DIR, "aggregate", "restricted_supplement.json")
    write_json(out, blob)
    for row in blob["checkpoints"]:
        print(f"seed {row['training_seed']} iter{row['iteration']} "
              f"{row['field']}: {row['games_non_terminating']} of "
              f"{row['games_attempted']} lobbies unfinished, completed-subset "
              f"avg {row['completed_subset_avg_placement']:.3f}")
        for key, p in row["pairs"].items():
            print(f"    {key:<16} {p['mean_diff']:+8.3f} "
                  f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  "
                  f"paired on {p['games_paired']}/{p['games_attempted']}"
                  f"{'  (CI excludes 0)' if p['ci_excludes_zero'] else ''}")
    print(f"Saved -> {out}")
    print(json.dumps({"bias": blob["bias"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
