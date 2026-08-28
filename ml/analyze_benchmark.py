"""Paired analysis over Replay Benchmark v1 result files.

Kept separate from ``ml/benchmark.py`` on purpose: the benchmark stays the
measurement instrument; this is the experiment-analysis layer on top of its
saved JSON. Nothing here re-runs games or touches raw result files.

Why pairing is valid: two Benchmark v1 runs with the same benchmark version,
field, game count, base seed, and environment evaluate game ``i`` on the same
evaluation seed (``base_seed + i``), so the per-game placement lists line up
sample-by-sample and per-game differences are paired observations. That
soaks up shared lobby-difficulty variance and gives a much tighter CI than
comparing two independent means.

    python -m ml.analyze_benchmark results/benchmark_v1/*.json

For every unordered pair of input files it reports the mean paired placement
difference (A - B; placement is LOWER-is-better, so a negative difference
means A finished better) with a deterministic paired percentile-bootstrap 95%
CI over the per-game differences. When the CI includes zero the benchmark
does not establish a clear difference — that is reported verbatim, never
upgraded to a significance claim.
"""

import argparse
import itertools
import json
import random
import sys
from typing import Dict, List, Optional, Sequence

BOOT_RESAMPLES = 10_000

# A pair of results is comparable only when the runs were identical in
# everything except the tested agent.
_PAIR_KEYS = ("benchmark_version", "field", "games", "base_seed",
              "seed_range", "environment")


def load_result(path: str) -> Dict:
    """One single-result Benchmark v1 JSON with per-game placements."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or "metrics" not in blob:
        raise ValueError(
            f"{path}: not a single-result Benchmark v1 file (suite wrappers "
            f"and other shapes aren't paired-comparable — pass the individual "
            f"result files)")
    if "placements" not in blob:
        raise ValueError(
            f"{path}: no per-game placements — re-run the benchmark with a "
            f"version that saves them (Benchmark v1, PR #8+)")
    if len(blob["placements"]) != blob.get("games"):
        raise ValueError(f"{path}: placements length "
                         f"{len(blob['placements'])} != games {blob.get('games')}")
    return blob


def verify_paired(a: Dict, b: Dict) -> None:
    """Raise unless the two runs are pairwise comparable (same seeds/config)."""
    bad = [k for k in _PAIR_KEYS if a.get(k) != b.get(k)]
    if bad:
        raise ValueError(
            f"results are not paired-comparable ({a.get('agent')} vs "
            f"{b.get('agent')}): mismatched {', '.join(bad)} — paired "
            f"analysis requires identical seeds and run configuration")


def paired_diff(pa: Sequence[int], pb: Sequence[int], seed: int = 0,
                resamples: int = BOOT_RESAMPLES) -> Dict:
    """Mean of per-game differences (a_i - b_i) with a deterministic paired
    percentile-bootstrap 95% CI (resample the difference list with
    replacement, take the 2.5th/97.5th percentiles of resampled means)."""
    if len(pa) != len(pb) or not pa:
        raise ValueError(f"placement lists must be equal-length and "
                         f"non-empty (got {len(pa)} and {len(pb)})")
    diffs = [x - y for x, y in zip(pa, pb)]
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    lo = means[max(0, int(0.025 * resamples) - 1)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return {"n": n, "mean_diff": mean, "ci95": [lo, hi],
            "method": "paired percentile bootstrap",
            "resamples": resamples, "bootstrap_seed": seed}


def compare_pair(a: Dict, b: Dict, seed: int = 0) -> Dict:
    verify_paired(a, b)
    d = paired_diff(a["placements"], b["placements"], seed=seed)
    lo, hi = d["ci95"]
    if hi < 0:
        verdict = f"{a['agent']} places better"          # lower is better
    elif lo > 0:
        verdict = f"{b['agent']} places better"
    else:
        verdict = "no clear difference (CI includes 0)"
    return {"a": a["agent"], "b": b["agent"], **d, "verdict": verdict}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="ml.analyze_benchmark",
        description="Paired comparisons over Benchmark v1 result JSONs "
                    "(placement is lower-is-better)")
    p.add_argument("files", nargs="+", help="single-result benchmark JSONs")
    p.add_argument("--boot-seed", type=int, default=0,
                   help="bootstrap seed (deterministic CI; default 0)")
    p.add_argument("--json-out", help="also write the pair table as JSON")
    a = p.parse_args(argv)

    results = [load_result(f) for f in a.files]
    rows: List[Dict] = []
    print(f"Paired comparisons ({results[0]['games']} games each, "
          f"seeds {results[0]['seed_range'][0]}-{results[0]['seed_range'][1]}; "
          f"diff = row A - row B, negative favors A; lower placement is better)")
    fmt = "{:<14} {:<14} {:>10} {:>20}   {}"
    print(fmt.format("A", "B", "mean diff", "95% CI (paired)", "verdict"))
    print("-" * 86)
    for x, y in itertools.combinations(results, 2):
        row = compare_pair(x, y, seed=a.boot_seed)
        rows.append(row)
        print(fmt.format(row["a"][:14], row["b"][:14],
                         f"{row['mean_diff']:+.3f}",
                         f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]",
                         row["verdict"]))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"method": rows[0]["method"] if rows else None,
                       "bootstrap_seed": a.boot_seed, "pairs": rows}, f,
                      indent=2)
        print(f"\nSaved -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
