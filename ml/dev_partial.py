"""DEV evaluation that survives a policy which cannot finish a lobby.

``ml.benchmark.run_game`` refuses to score an episode that has not
terminated within ``MAX_DECISIONS`` — scoring it as a silent 8th place would
corrupt the numbers. That guard is correct and is NOT relaxed here.

Experiment 3 ran into it: one training seed's 5,120-episode policy stalls on
a handful of the 1000 fixed DEV lobbies, so the whole evaluation aborted and
that checkpoint had no measurement at all. This module runs the identical
protocol lobby by lobby and records the stalls as data:

  * completed lobbies are scored exactly as ``ml.dev_benchmark`` scores them
    (same seeds, same seat assignment, same metrics, same bootstrap CI);
  * lobbies the policy could not finish are recorded by seed and excluded;
  * the resulting JSON is stamped ``complete: false`` so nothing downstream
    can mistake it for a full 1000-game result.

A restricted result is **biased in the policy's favour**: the excluded
lobbies are exactly the ones where its behavior degenerated. Comparisons
against it must go through ``paired_common_games``, which pairs only the
lobbies both checkpoints finished and reports how many were dropped.

DEV split only; the seed range is validated against the reserved DEV
interval by the same helper the normal path uses.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .benchmark import (Agent, BenchmarkIntegrityError, BenchmarkResult,
                        bootstrap_ci, compute_metrics, latency_stats,
                        run_game)
from .analyze_benchmark import paired_diff
from .dev_benchmark import dev_field_seats, dev_result_to_json
from .seeds import DEV_SEED_START, eval_game_seed, validate_dev_range

RESTRICTED_NOTE = (
    "INCOMPLETE EVALUATION. The policy failed to terminate some of the fixed "
    "DEV lobbies within ml.benchmark.MAX_DECISIONS, so those lobbies carry no "
    "placement. The metrics below cover only the lobbies that finished and "
    "are therefore optimistic: the excluded lobbies are the ones where the "
    "policy's behavior degenerated. Pair this result against others only on "
    "the lobbies both finished.")


def run_dev_benchmark_tolerant(agent: Agent, field: str, games: int,
                               base_seed: int = DEV_SEED_START,
                               progress: bool = False
                               ) -> Tuple[BenchmarkResult, List[Dict], List[int]]:
    """(result over completed lobbies, stalled lobbies, completed indices).

    Identical to ``ml.dev_benchmark.run_dev_benchmark`` when nothing stalls.
    """
    validate_dev_range(base_seed, games)
    seats = dev_field_seats(field)
    placements: List[int] = []
    latencies: List[float] = []
    completed: List[int] = []
    stalled: List[Dict] = []
    for i in range(games):
        seed = eval_game_seed(base_seed, i)
        try:
            g = run_game(agent, seats, seed)
        except BenchmarkIntegrityError as e:
            stalled.append({"index": i, "seed": seed, "reason": str(e)})
            continue
        placements.append(g["placement"])
        latencies.extend(g["latencies"])
        completed.append(i)
        if progress and (i + 1) % 100 == 0:
            print(f"  {agent.name}: {i + 1}/{games} lobbies "
                  f"({len(stalled)} did not terminate)")
    if not placements:
        raise BenchmarkIntegrityError(
            f"agent {agent.name!r} finished 0 of {games} DEV lobbies — "
            f"there is nothing to score")
    res = BenchmarkResult(agent=agent, field=field, games=len(placements),
                          base_seed=base_seed,
                          metrics=compute_metrics(placements),
                          ci95=bootstrap_ci(placements, seed=base_seed),
                          latency=latency_stats(latencies),
                          placements=placements)
    return res, stalled, completed


def partial_result_to_json(res: BenchmarkResult, stalled: Sequence[Dict],
                           completed: Sequence[int],
                           games_requested: int) -> Dict:
    """The DEV result schema plus an unmissable completeness record.

    ``games`` stays equal to ``len(placements)`` so the file remains a valid
    single-result JSON, while ``games_requested`` and ``seed_range`` keep the
    protocol's fixed 1000-lobby (or 500-lobby) block visible.
    """
    blob = dev_result_to_json(res)
    blob["games_requested"] = games_requested
    blob["seed_range"] = [res.base_seed, res.base_seed + games_requested - 1]
    blob["complete"] = not stalled
    blob["games_non_terminating"] = len(stalled)
    blob["non_termination_rate"] = len(stalled) / games_requested
    if stalled:
        blob["non_terminating_seeds"] = [s["seed"] for s in stalled]
        blob["completed_game_indices"] = list(completed)
        blob["restricted_note"] = RESTRICTED_NOTE
        blob["beats_field"] = None       # an incomplete run decides nothing
    return blob


def _index_map(blob) -> Dict[int, int]:
    """game index -> position in the placement list."""
    idx = blob.get("completed_game_indices")
    if idx is None:
        idx = range(blob["games"])
    return {int(g): pos for pos, g in enumerate(idx)}


def is_complete(blob) -> bool:
    return bool(blob.get("complete", True))


def paired_common_games(a, b, seed: int = 0) -> Dict:
    """Paired difference (a - b) over the lobbies BOTH runs finished.

    Falls back to the ordinary full pairing when both runs are complete, so
    the number is identical to ``ml.analyze_benchmark.compare_pair`` in the
    normal case. When either run is restricted, the dropped lobbies are
    reported alongside the estimate — they are never silently discarded.
    """
    for blob in (a, b):
        if blob.get("base_seed") != a.get("base_seed"):
            raise ValueError("results use different DEV base seeds")
        if blob.get("field") != a.get("field"):
            raise ValueError("results use different opponent fields")
    requested = max(a.get("games_requested", a["games"]),
                    b.get("games_requested", b["games"]))
    ia, ib = _index_map(a), _index_map(b)
    common = sorted(set(ia) & set(ib))
    if not common:
        raise ValueError("the two runs share no completed lobby")
    pa = [a["placements"][ia[g]] for g in common]
    pb = [b["placements"][ib[g]] for g in common]
    out = paired_diff(pa, pb, seed=seed)
    out.update({
        "a": a.get("agent"), "b": b.get("agent"),
        "games_requested": requested,
        "games_paired": len(common),
        "games_dropped": requested - len(common),
        "restricted": len(common) != requested,
        "a_complete": is_complete(a), "b_complete": is_complete(b),
    })
    lo, hi = out["ci95"]
    out["ci_excludes_zero"] = not (lo <= 0 <= hi)
    if out["restricted"]:
        out["note"] = (
            f"paired on the {len(common)} of {requested} DEV lobbies both "
            f"checkpoints finished; the {requested - len(common)} dropped "
            f"lobbies are ones a policy could not terminate, so this "
            f"estimate is optimistic for that policy")
    return out


def scan_non_termination(checkpoint: str, field: str, games: int,
                         base_seed: int = DEV_SEED_START,
                         name: Optional[str] = None) -> Dict:
    """Standalone count of the DEV lobbies a checkpoint cannot finish."""
    from .benchmark import make_agent
    agent = make_agent("policy", checkpoint, name)
    res, stalled, completed = run_dev_benchmark_tolerant(agent, field, games,
                                                         base_seed)
    return {"checkpoint": agent.checkpoint, "field": field,
            "games_requested": games,
            "games_completed": len(completed),
            "games_non_terminating": len(stalled),
            "non_termination_rate": len(stalled) / games,
            "non_terminating_seeds": [s["seed"] for s in stalled],
            "completed_avg_placement": res.metrics["avg_placement"],
            "note": RESTRICTED_NOTE}
