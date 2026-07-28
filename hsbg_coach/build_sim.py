"""Board-building simulation — build boards, reward size, grade against winners.

Runs the recruit-phase env repeatedly and asks two questions of every board it
produces:

  1. **How big is it?** Total stats on board — the "build the biggest board you
     can, turn by turn" objective.
  2. **Does it look like a board that actually wins?** Firestone gives us 32 real
     winning archetypes with their core cards weighted by how often they appear in
     winning boards (`data/stats/firestone_final_boards.json`). `build_path`
     scores how much of one a board has assembled.

Keeping both is the point. Board size alone is satisfiable by buying the biggest
stat line every turn, which is exactly the play we suspect is wrong. If maximizing
size produced boards that resembled real winners, size would be a sufficient
training reward — so measuring the gap between the two tells us whether it is.

    python -m hsbg_coach.build_sim --lobbies 40

Reports board size against the measured top-10% scaling curve, and archetype
coverage against the real winning boards.
"""

import argparse
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .bg_env import BGEnv, greedy_policy
from .build_path import infer_target, load_archetypes
from .pace import load_pace
from .synergy import load_embeddings


@dataclass
class BuiltBoard:
    """One seat's board at the end of a game."""
    turn: int
    placement: int
    stats: int                              # total attack + health
    minions: int
    archetype: Optional[str] = None
    coverage: float = 0.0                   # 0..1 of a real winning comp assembled


@dataclass
class BuildReport:
    boards: List[BuiltBoard] = field(default_factory=list)
    stats_by_turn: Dict[int, List[int]] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        if not self.boards:
            return {}
        cov = [b.coverage for b in self.boards]
        return {
            "boards": len(self.boards),
            "mean_stats": statistics.mean(b.stats for b in self.boards),
            "mean_minions": statistics.mean(b.minions for b in self.boards),
            "mean_coverage": statistics.mean(cov),
            "median_coverage": statistics.median(cov),
            "pct_half_built": 100.0 * sum(1 for c in cov if c >= 0.5) / len(cov),
        }

    def archetype_spread(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for b in self.boards:
            if b.archetype:
                out[b.archetype] = out.get(b.archetype, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def board_stats(board) -> int:
    """Total attack + health — 'how big is the board'."""
    total = 0
    for m in board:
        atk = getattr(m, "attack", None)
        hp = getattr(m, "health", None)
        if atk is None and isinstance(m, dict):
            atk, hp = m.get("attack", 0), m.get("health", 0)
        total += int(atk or 0) + int(hp or 0)
    return total


def _grade(board, archetypes, emb) -> tuple:
    """(archetype key, coverage) against the real winning boards."""
    if not board:
        return None, 0.0
    fit = infer_target(board, archetypes=archetypes, emb=emb)
    if fit is None:
        return None, 0.0
    return fit.arch.key, fit.coverage


def run_builds(lobbies: int = 40, seed: int = 0,
               policy: Optional[Callable] = None) -> BuildReport:
    """Play `lobbies` games; grade every seat's board at the end of its game."""
    policy = policy or greedy_policy
    archetypes = load_archetypes()
    emb = load_embeddings()
    report = BuildReport()

    for g in range(lobbies):
        rng = random.Random(seed + g)
        env = BGEnv(seed=seed + g)
        env.reset()
        while not env.done:
            mask = env.legal_mask()
            if not any(mask):
                break
            env.step(policy(env.observe(0), mask, rng))
            report.stats_by_turn.setdefault(env.turn, []).append(
                board_stats(env.players[0].board))

        for p in env.players:
            board = list(p.board) or list(p.last_board)
            if not board:
                continue
            key, cov = _grade(board, archetypes, emb)
            report.boards.append(BuiltBoard(
                turn=env.turn, placement=getattr(p, "placement", 0) or 0,
                stats=board_stats(board), minions=len(board),
                archetype=key, coverage=cov))
    return report


def reference_coverage(archetypes=None, emb=None) -> float:
    """Grade the *real* winning boards with the same grader.

    Without this the simulated coverage number means nothing — it fixes the top of
    the scale, and proves a low sim score is the build's fault and not the grader's.
    """
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "stats",
                        "firestone_final_boards.json")
    if not os.path.isfile(path):
        return 0.0
    archetypes = archetypes if archetypes is not None else load_archetypes()
    emb = emb if emb is not None else load_embeddings()
    covs = []
    for b in json.load(open(path, encoding="utf-8")).get("boards", []):
        board = [{"name": c["name"], "attack": 1, "health": 1}
                 for c in (b.get("coreCards") or [])[:7]]
        if not board:
            continue
        _, cov = _grade(board, archetypes, emb)
        covs.append(cov)
    return statistics.mean(covs) if covs else 0.0


def _pace_row(turn: int) -> Optional[float]:
    scaling = (load_pace() or {}).get("scaling") or {}
    row = scaling.get(str(turn)) or scaling.get(turn)
    if isinstance(row, dict):
        return row.get("stats") or row.get("value")
    return row if isinstance(row, (int, float)) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build boards and grade them against "
                                             "the real winning boards")
    ap.add_argument("--lobbies", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    report = run_builds(lobbies=a.lobbies, seed=a.seed)
    s = report.summary()
    if not s:
        print("No boards produced.")
        return 1

    print(f"Built {int(s['boards'])} boards over {a.lobbies} lobbies\n")
    print(f"  board size    mean {s['mean_stats']:.0f} total stats "
          f"across {s['mean_minions']:.1f} minions")
    ref = reference_coverage()
    print(f"  winning-board fit   mean coverage {s['mean_coverage']:.2f}  "
          f"median {s['median_coverage']:.2f}  "
          f"{s['pct_half_built']:.0f}% at least half-built")
    print(f"                      real winning boards score {ref:.2f} "
          f"through the same grader\n")

    print("  board size by turn (sim vs measured top-10% curve):")
    for turn in sorted(report.stats_by_turn)[:14]:
        vals = report.stats_by_turn[turn]
        real = _pace_row(turn)
        ref = f"{real:6.1f}" if isinstance(real, (int, float)) else "     -"
        print(f"    T{turn:<3d} sim {statistics.mean(vals):6.1f}   real {ref}")

    spread = report.archetype_spread()
    print("\n  archetypes the built boards land on:")
    for key, n in list(spread.items())[:8]:
        print(f"    {key:22s} {n:4d}  ({100.0 * n / len(report.boards):4.1f}%)")
    if len(spread) <= 3:
        print("    ^ collapsed onto very few comps — the build is not comp-driven")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
