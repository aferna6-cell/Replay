"""Replay Benchmark v1 — the canonical, reproducible agent evaluation.

One command compares every Replay agent from the same deterministic initial
conditions: identical evaluation seed per game, identical environment
configuration, identical opponent field, identical tested seat (seat 0 of
``BGEnv``). Because the tested agent's own actions feed back into the shared
pool and combat pairings, trajectories legitimately diverge between agents
after the first differing decision — what is fixed is the seeded starting
point and every rule around it, so any outcome difference is attributable to
the agent, not the setup.

    python -m ml.benchmark --games 1000                    # full default suite
    python -m ml.benchmark --agent greedy --games 200 --field greedy
    python -m ml.benchmark --agent policy --checkpoint ml/policy_ppo.pt \
        --name PPO --games 1000 --field greedy --json-out results/ppo.json
    python -m ml.benchmark compare results/*.json

SEED DISCIPLINE — see ``ml/seeds.py`` (the single authority on seed policy).
Evaluation owns the finite reserved interval [EVAL_SEED_START, EVAL_SEED_END]
= [10_250_000, 10_299_999]; game ``i`` runs on ``base_seed + i`` and the
benchmark REFUSES a run whose seed range leaves that interval. The interval
was placed after auditing every BGEnv-seeding training scheme (BC, DAgger,
PPO's base*1000003+k episodes, the midgame dataset's base*100003+i lobbies,
the legacy 9000+ eval): no current default or reasonable configuration can
reach it, and the exact bounds of that claim — not a mathematical guarantee —
are documented in ``ml/seeds.py``. Training entry points warn loudly if a
planned run would touch the interval.

INTEGRITY — a benchmark that silently mishandles episodes or actions can make
a model look better than it is, so both are hard failures here: an episode
that does not terminate within the decision cap raises (it is never scored as
8th), and an agent action is validated as an in-range, legal action index
before use (no Python negative indexing making -1 "legal").

The tested agent and every scripted opponent see the exact same observation
contract (``BGEnv.observe`` dict + legal mask); the learned policy additionally
runs that dict through ``ml.env_obs.encode_obs``. No agent gets privileged
engine state. The 4.5 average-placement threshold only means "outperformed the
average of the comparison field" (8 players average 4.5 by construction) — it
is not evidence of optimal play.
"""

import argparse
import hashlib
import json
import math
import operator
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from hsbg_coach.bg_env import BGEnv, MAX_TURNS, greedy_policy, random_policy
from .rl_common import MAX_DECISIONS
from .seeds import (EVAL_SEED_END, EVAL_SEED_START, eval_game_seed,
                    validate_eval_range)

BENCHMARK_VERSION = "Replay Benchmark v1"
EVAL_SEED_BASE = EVAL_SEED_START           # default --seed; policy: ml/seeds.py
FIELD_SIZE = 7                     # opponents; 8-player lobby, agent in seat 0
BEAT_FIELD_THRESHOLD = 4.5         # lobby-average placement
BOOTSTRAP_RESAMPLES = 2_000

_FIELDS: Dict[str, Callable] = {"greedy": greedy_policy, "random": random_policy}


class BenchmarkIntegrityError(RuntimeError):
    """A condition that would silently corrupt benchmark results — always
    fail the run loudly instead of scoring around it."""


# --- agents -------------------------------------------------------------------
@dataclass
class Agent:
    """A named (obs, legal_mask, rng) -> action policy — the env's native
    scripted-seat contract, which random/greedy already satisfy and which
    ``ml.policy_net.as_env_policy`` adapts PolicyNet checkpoints to."""
    name: str
    kind: str                       # "random" | "greedy" | "policy"
    policy: Callable
    checkpoint: Optional[str] = None          # basename only, never a path
    checkpoint_sha256: Optional[str] = None   # fingerprint of the .pt bytes


def make_agent(kind: str, checkpoint: Optional[str] = None,
               name: Optional[str] = None) -> Agent:
    if kind == "random":
        return Agent(name or "Random", "random", random_policy)
    if kind == "greedy":
        return Agent(name or "Greedy", "greedy", greedy_policy)
    if kind == "policy":
        if not checkpoint:
            raise ValueError("--agent policy requires --checkpoint")
        if not os.path.isfile(checkpoint):
            raise ValueError(f"checkpoint not found: {checkpoint}")
        # The filename alone can't identify a model (different weights can
        # share "policy_ppo.pt") — fingerprint the actual bytes evaluated.
        with open(checkpoint, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        # torch stays a lazy import so scripted-agent benchmarks run without it
        from hsbg_coach.synergy import load_embeddings
        from .policy_net import as_env_policy, load_policy
        from .rl_common import kb_byname
        try:
            net = load_policy(checkpoint)
        except Exception as e:
            raise ValueError(f"could not load policy checkpoint "
                             f"{checkpoint}: {e}") from e
        net.eval()
        # greedy=True → argmax decisions, so evaluation is deterministic
        policy = as_env_policy(net, load_embeddings(), kb_byname(), greedy=True)
        return Agent(name or os.path.basename(checkpoint), "policy", policy,
                     checkpoint=os.path.basename(checkpoint),
                     checkpoint_sha256=sha256)
    raise ValueError(f"unknown agent kind: {kind!r} "
                     f"(expected random | greedy | policy)")


def _validated_action(action, mask: Sequence[bool], agent: Agent,
                      seed: int) -> int:
    """The action must be an integer index into the mask, in range, and
    legal — checked explicitly so Python negative indexing can never make -1
    read as the (legal) last action."""
    where = f"agent {agent.name!r} on seed {seed}"
    try:
        idx = operator.index(action)
    except TypeError:
        raise BenchmarkIntegrityError(
            f"{where} returned non-integer action {action!r}") from None
    if idx < 0 or idx >= len(mask):
        raise BenchmarkIntegrityError(
            f"{where} returned out-of-range action {idx} "
            f"(valid indices 0..{len(mask) - 1}, "
            f"legal: {[i for i, ok in enumerate(mask) if ok]})")
    if not mask[idx]:
        raise BenchmarkIntegrityError(
            f"{where} chose illegal (masked) action {idx} "
            f"(legal: {[i for i, ok in enumerate(mask) if ok]})")
    return idx


# --- single game --------------------------------------------------------------
def run_game(agent: Agent, field_policy: Callable, seed: int) -> Dict:
    """One seeded lobby: tested agent in seat 0 vs 7 copies of the field
    policy. Returns the placement plus per-decision latencies (seconds),
    timing ONLY the agent's decision function — for a learned policy that is
    observation encoding + network forward, for scripted agents the heuristic
    itself — never ``env.step``. An episode that does not terminate within
    MAX_DECISIONS raises instead of being scored (a silent 8th would corrupt
    the numbers)."""
    env = BGEnv(seed=seed, opponent_policies=[field_policy] * FIELD_SIZE)
    obs = env.reset(seed=seed)
    rng = random.Random(seed)       # the tested agent's private rng
    latencies: List[float] = []
    for _ in range(MAX_DECISIONS):
        mask = env.legal_mask(0)
        t0 = time.perf_counter()
        action = agent.policy(obs, mask, rng)
        latencies.append(time.perf_counter() - t0)
        action = _validated_action(action, mask, agent, seed)
        obs, _, done, info = env.step(action)
        if done:
            return {"seed": seed, "placement": info.get("placement", 8),
                    "latencies": latencies}
    raise BenchmarkIntegrityError(
        f"episode did not terminate: agent {agent.name!r}, seed {seed}, "
        f"{len(latencies)} decisions (cap MAX_DECISIONS={MAX_DECISIONS}) — "
        f"refusing to score an unfinished game")


# --- metrics ------------------------------------------------------------------
def compute_metrics(placements: Sequence[int]) -> Dict:
    n = len(placements)
    if n == 0:
        raise ValueError("no games played")
    counts = {str(p): 0 for p in range(1, 9)}
    for p in placements:
        counts[str(p)] += 1
    return {
        "games": n,
        "avg_placement": sum(placements) / n,
        "median_placement": statistics.median(placements),
        "std_placement": statistics.pstdev(placements) if n > 1 else 0.0,
        "top4_rate": sum(1 for p in placements if p <= 4) / n,
        "win_rate": counts["1"] / n,
        "placement_counts": counts,
    }


def bootstrap_ci(placements: Sequence[int], seed: int,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 alpha: float = 0.05) -> Dict:
    """95% CI for the mean placement via a seeded percentile bootstrap
    (deterministic for a given base seed). Documented method: resample the
    placement list with replacement `resamples` times, take the 2.5th/97.5th
    percentiles of the resampled means."""
    rng = random.Random(seed)
    n = len(placements)
    means = sorted(sum(rng.choices(placements, k=n)) / n
                   for _ in range(resamples))
    lo = means[max(0, math.floor((alpha / 2) * resamples) - 1)]
    hi = means[min(resamples - 1, math.ceil((1 - alpha / 2) * resamples) - 1)]
    return {"method": "percentile bootstrap",
            "resamples": resamples, "level": 0.95,
            "low": lo, "high": hi}


def latency_stats(latencies: Sequence[float]) -> Dict:
    if not latencies:
        return {"decisions": 0}
    s = sorted(latencies)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"decisions": len(s),
            "mean_ms": 1000.0 * sum(s) / len(s),
            "p50_ms": 1000.0 * pick(0.50),
            "p95_ms": 1000.0 * pick(0.95)}


# --- full benchmark -----------------------------------------------------------
@dataclass
class BenchmarkResult:
    agent: Agent
    field: str
    games: int
    base_seed: int
    metrics: Dict = dc_field(default_factory=dict)
    ci95: Dict = dc_field(default_factory=dict)
    latency: Dict = dc_field(default_factory=dict)
    placements: List[int] = dc_field(default_factory=list)


def run_benchmark(agent: Agent, field: str, games: int,
                  base_seed: int = EVAL_SEED_BASE,
                  progress: bool = False) -> BenchmarkResult:
    """Deterministic TEST evaluation: game i uses seed base_seed + i,
    validated to sit inside the reserved Benchmark v1 TEST interval
    (ml/seeds.py). Development evaluation lives in ml/dev_benchmark.py on
    the separate DEV interval — this entry point is the held-out test set,
    used sparingly for final confirmation, never for model iteration.
    Single process on purpose — parallelism is a documented future
    optimization so v1 stays trivially reproducible."""
    validate_eval_range(base_seed, games)
    return _run_games(agent, field, games, base_seed, progress)


def _run_games(agent: Agent, field: str, games: int, base_seed: int,
               progress: bool = False) -> BenchmarkResult:
    """The shared evaluation loop; callers are responsible for validating
    the seed range against the right reserved interval first."""
    if field not in _FIELDS:
        raise ValueError(f"unknown field {field!r} "
                         f"(expected one of {sorted(_FIELDS)})")
    placements: List[int] = []
    latencies: List[float] = []
    for i in range(games):
        g = run_game(agent, _FIELDS[field], eval_game_seed(base_seed, i))
        placements.append(g["placement"])
        latencies.extend(g["latencies"])
        if progress and (i + 1) % 50 == 0:
            print(f"  {agent.name}: {i + 1}/{games} games "
                  f"(running avg {sum(placements) / len(placements):.2f})",
                  file=sys.stderr)
    return BenchmarkResult(
        agent=agent, field=field, games=games, base_seed=base_seed,
        metrics=compute_metrics(placements),
        ci95=bootstrap_ci(placements, seed=base_seed),
        latency=latency_stats(latencies),
        placements=placements)


def git_commit() -> Optional[str]:
    """The repo HEAD the benchmark ran from, "-dirty" suffixed when the
    working tree has uncommitted changes; None (stored as JSON null) when Git
    metadata is unavailable — e.g. outside a checkout. Never fails the run."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None
        commit = sha.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=10)
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return None


def result_to_json(res: BenchmarkResult) -> Dict:
    """Machine-readable record. Model identity is explicit (checkpoint
    basename + sha256 of its bytes, repo commit) and no absolute paths are
    included, so results identify what ran and compare across machines."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "evaluation_split": "test",        # ml/dev_benchmark.py stamps "dev"
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": res.agent.name,
        "agent_kind": res.agent.kind,
        "checkpoint": res.agent.checkpoint,
        "checkpoint_sha256": res.agent.checkpoint_sha256,
        "git_commit": git_commit(),
        "field": res.field,
        "games": res.games,
        "base_seed": res.base_seed,
        "seed_range": [res.base_seed, res.base_seed + res.games - 1],
        "seed_policy": (f"evaluation-only seeds from the reserved interval "
                        f"[{EVAL_SEED_START}, {EVAL_SEED_END}]; never reuse "
                        f"for training (see ml/seeds.py)"),
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS,
                        "agent_seat": 0},
        "metrics": res.metrics,
        # Per-game raw placements; index i is the game on seed base_seed+i,
        # so equal-config runs are paired sample-by-sample (ml/analyze_benchmark).
        "placements": list(res.placements),
        "avg_placement_ci95": res.ci95,
        "decision_latency": res.latency,
        "beat_field_threshold": BEAT_FIELD_THRESHOLD,
        "beats_field": res.metrics["avg_placement"] < BEAT_FIELD_THRESHOLD,
    }


def suite_to_json(results: Sequence[BenchmarkResult]) -> Dict:
    """Suite files use a versioned wrapper — {"benchmark_version", …,
    "results": [<result objects>]} — and compare mode understands both this
    wrapper and bare single-result files."""
    return {"benchmark_version": BENCHMARK_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": [result_to_json(r) for r in results]}


def _write_json(blob: Dict, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def save_json(res: BenchmarkResult, path: str) -> None:
    _write_json(result_to_json(res), path)


# --- reporting ----------------------------------------------------------------
_ROW = "{:<20} {:>9} {:>9} {:>7} {:>9}"


def _table(rows: List[Dict]) -> str:
    lines = [_ROW.format("Agent", "Avg Place", "Top 4 %", "Win %", "Std Dev"),
             "-" * 58]
    for r in sorted(rows, key=lambda r: r["metrics"]["avg_placement"]):
        m = r["metrics"]
        lines.append(_ROW.format(
            r["agent"][:20], f"{m['avg_placement']:.2f}",
            f"{100 * m['top4_rate']:.1f}", f"{100 * m['win_rate']:.1f}",
            f"{m['std_placement']:.2f}"))
    return "\n".join(lines)


def _identity_lines(rows: List[Dict]) -> List[str]:
    """Model identity per row — a differing checkpoint hash is the point of
    a comparison, so it is surfaced, never treated as an error."""
    out = ["", "Identities:"]
    for r in sorted(rows, key=lambda r: r["metrics"]["avg_placement"]):
        sha = r.get("checkpoint_sha256")
        ident = (f"{r.get('checkpoint')} sha256:{sha[:12]}" if sha
                 else "(scripted, no checkpoint)")
        out.append(f"  {r['agent'][:20]:<20} {ident}   "
                   f"git:{r.get('git_commit') or 'unknown'}")
    return out


def print_summary(res: BenchmarkResult) -> None:
    m, ci = res.metrics, res.ci95
    print(f"\n{res.agent.name} vs {FIELD_SIZE}x {res.field} "
          f"({res.games} games, seeds {res.base_seed}-"
          f"{res.base_seed + res.games - 1})")
    if res.agent.checkpoint_sha256:
        print(f"Checkpoint: {res.agent.checkpoint} "
              f"sha256:{res.agent.checkpoint_sha256[:12]}")
    print(f"Avg placement: {m['avg_placement']:.2f}")
    print(f"95% CI: [{ci['low']:.2f}, {ci['high']:.2f}]  ({ci['method']})")
    print(f"Median: {m['median_placement']:g}   Std dev: {m['std_placement']:.2f}")
    print(f"Top-4: {100 * m['top4_rate']:.1f}%")
    print(f"Win rate: {100 * m['win_rate']:.1f}%")
    dist = "  ".join(f"{p}:{m['placement_counts'][str(p)]}"
                     for p in range(1, 9))
    print(f"Placements: {dist}")
    if res.latency.get("decisions"):
        print(f"Decision latency: mean {res.latency['mean_ms']:.2f}ms  "
              f"p50 {res.latency['p50_ms']:.2f}ms  "
              f"p95 {res.latency['p95_ms']:.2f}ms")
    if m["avg_placement"] < BEAT_FIELD_THRESHOLD:
        print(f"\nPASS: beats {BEAT_FIELD_THRESHOLD} average placement "
              f"threshold (outperforms the field's average — "
              f"not proof of optimal play)")
    else:
        print(f"\nFAIL: does not beat {BEAT_FIELD_THRESHOLD} average "
              f"placement threshold")


# --- compare mode -------------------------------------------------------------
def _flatten_results(blob) -> List[Dict]:
    """Accept a single-result file, a suite wrapper ({"results": [...]}), or
    a bare list of results — every JSON shape this CLI can write."""
    if isinstance(blob, dict) and "results" in blob:
        return list(blob["results"])
    if isinstance(blob, dict):
        return [blob]
    if isinstance(blob, list):
        return list(blob)
    raise ValueError(f"unrecognized result JSON shape: {type(blob).__name__}")


def compare_files(paths: Sequence[str]) -> str:
    """Table over saved result JSONs (single-result and suite files alike),
    sorted by average placement. Warns when the run conditions —
    version/field/games/seed/environment — differ, because those results
    aren't apples-to-apples. Differing checkpoint hashes are expected (that
    is what's being compared) and shown in the identity block, not warned."""
    rows: List[Dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.extend(_flatten_results(json.load(f)))
    keys = {(r.get("benchmark_version"), r.get("field"), r.get("games"),
             r.get("base_seed"), json.dumps(r.get("environment"),
                                            sort_keys=True)) for r in rows}
    out = []
    if len(keys) > 1:
        out.append("WARNING: results differ in version/field/games/seed/"
                   "environment — comparison is not apples-to-apples:")
        for r in rows:
            out.append(f"  {r.get('agent')}: {r.get('benchmark_version')}, "
                       f"field={r.get('field')}, games={r.get('games')}, "
                       f"base_seed={r.get('base_seed')}")
        out.append("")
    out.append(_table(rows))
    out.extend(_identity_lines(rows))
    return "\n".join(out)


# --- CLI ----------------------------------------------------------------------
_DIR = os.path.dirname(__file__)
# The repo's real checkpoint conventions (ml/bc.py + ml/train_ppo.py defaults).
DEFAULT_SUITE = [("random", None, "Random"),
                 ("greedy", None, "Greedy"),
                 ("policy", os.path.join(_DIR, "policy_bc.pt"), "BC"),
                 ("policy", os.path.join(_DIR, "policy_ppo.pt"), "PPO")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        cp = argparse.ArgumentParser(prog="ml.benchmark compare",
                                     description="Compare saved result JSONs")
        cp.add_argument("files", nargs="+")
        ca = cp.parse_args(argv[1:])
        print(compare_files(ca.files))
        return 0

    p = argparse.ArgumentParser(
        prog="ml.benchmark",
        description=f"{BENCHMARK_VERSION} — deterministic agent evaluation. "
                    f"Seeds live in the reserved evaluation interval "
                    f"[{EVAL_SEED_START}, {EVAL_SEED_END}] (ml/seeds.py); "
                    f"never train on them.")
    p.add_argument("--agent", choices=["random", "greedy", "policy"],
                   help="agent to test (omit to run the default suite: "
                        "random, greedy, plus BC/PPO checkpoints if present)")
    p.add_argument("--checkpoint", help="PolicyNet .pt file (with --agent policy)")
    p.add_argument("--name", help="display name for the tested agent")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=EVAL_SEED_BASE,
                   help=f"base evaluation seed (default {EVAL_SEED_BASE}; "
                        "game i uses seed+i; the whole range must stay "
                        "inside the reserved evaluation interval)")
    p.add_argument("--field", choices=sorted(_FIELDS), default="greedy",
                   help="opponent field: 7 copies of this policy (default greedy)")
    p.add_argument("--json-out", help="write machine-readable results here")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    a = p.parse_args(argv)

    try:
        validate_eval_range(a.seed, a.games)
    except ValueError as e:
        p.error(str(e))

    print(f"{BENCHMARK_VERSION}")
    print(f"Evaluation games: {a.games}")
    print(f"Seed range: {a.seed}-{a.seed + a.games - 1}")
    print(f"Field: {FIELD_SIZE}x {a.field}")

    if a.agent:
        try:
            agent = make_agent(a.agent, a.checkpoint, a.name)
        except ValueError as e:
            p.error(str(e))
        res = run_benchmark(agent, a.field, a.games, a.seed,
                            progress=not a.quiet)
        print_summary(res)
        if a.json_out:
            save_json(res, a.json_out)
            print(f"\nSaved -> {a.json_out}")
        return 0

    # Default suite: every agent evaluated from the same deterministic
    # initial conditions (same seeds, env config, field, seat).
    results = []
    for kind, ckpt, name in DEFAULT_SUITE:
        if kind == "policy" and not (ckpt and os.path.isfile(ckpt)):
            print(f"  (skipping {name}: no checkpoint at {ckpt})")
            continue
        agent = make_agent(kind, ckpt, name)
        results.append(run_benchmark(agent, a.field, a.games, a.seed,
                                     progress=not a.quiet))
    rows = [result_to_json(r) for r in results]
    print()
    print(_table(rows))
    print("\n".join(_identity_lines(rows)))
    if a.json_out:
        _write_json(suite_to_json(results), a.json_out)
        print(f"\nSaved -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
