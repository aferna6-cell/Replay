"""Combat calibration — how far the fast sim is from ground truth.

We have two combat engines and they are not interchangeable:

  * **Firestone** (`bridge/`, `@firestone-hs/simulate-bgs-battle`) — the real
    thing. Every deathrattle, battlecry-summon and hero power, maintained against
    the live patch by the community that ships the overlay. Measured here at
    ~2900 ms per call, because each one spawns a Node process.
  * **`sim.py`** — pure Python, representative card coverage, ~0.4 ms per call.

That is a ~7000x gap, so Firestone can't be the rollout engine: one game is about
56 combats, which is 161 seconds through Firestone against 0.02 through `sim.py`.
Millions of self-play games is only reachable on the fast one.

So the fast sim carries the rollouts and Firestone is the *ruler*. This module
measures the distance between them on boards drawn from real env play, which
turns "is our combat accurate?" from an opinion into a number — and names the
cards that break it.

    python -m hsbg_coach.combat_calib --pairs 40

Agreement on the *outcome* is what matters. Being wrong about damage on a fight
you'd win anyway does not change a decision; flipping win to loss does.
"""

import argparse
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import firestone_bridge as fb
from .bg_env import BGEnv, greedy_policy
from .sim import Combatant, simulate


@dataclass
class PairResult:
    fast_win: float
    true_win: float
    fast_outcome: str                      # win | tie | loss (most likely)
    true_outcome: str
    names: List[str] = field(default_factory=list)

    @property
    def agrees(self) -> bool:
        return self.fast_outcome == self.true_outcome

    @property
    def win_err(self) -> float:
        return abs(self.fast_win - self.true_win)


def _outcome(res) -> Tuple[str, float]:
    """(most likely outcome, win probability) from either engine's result."""
    win = float(getattr(res, "win_pct", None) or getattr(res, "wins", 0) or 0)
    tie = float(getattr(res, "tie_pct", None) or getattr(res, "ties", 0) or 0)
    loss = float(getattr(res, "loss_pct", None) or getattr(res, "losses", 0) or 0)
    total = win + tie + loss
    if total > 1.5:                        # percentages, not fractions
        win, tie, loss = win / total, tie / total, loss / total
    best = max((win, "win"), (tie, "tie"), (loss, "loss"))[1]
    return best, win


def collect_pairs(lobbies: int = 3, seed: int = 0,
                  max_pairs: int = 40) -> List[Tuple[List, List, List[str]]]:
    """Board pairs drawn from actual play, not synthetic ones.

    Synthetic boards would under-sample exactly what breaks the fast sim: the
    minions the env's own shop keeps offering.
    """
    pairs: List[Tuple[List, List, List[str]]] = []
    for g in range(lobbies):
        rng = random.Random(seed + g)
        env = BGEnv(seed=seed + g)
        env.reset()
        while not env.done and len(pairs) < max_pairs:
            mask = env.legal_mask()
            if not any(mask):
                break
            env.step(greedy_policy(env.observe(0), mask, rng))
            alive = [p for p in env.players if p.alive and p.board]
            if len(alive) >= 2:
                a, b = rng.sample(alive, 2)
                pairs.append((
                    [Combatant.from_minion(m.view()) for m in a.board],
                    [Combatant.from_minion(m.view()) for m in b.board],
                    [m.name for m in a.board] + [m.name for m in b.board],
                ))
        if len(pairs) >= max_pairs:
            break
    return pairs[:max_pairs]


def compare(pairs, runs: int = 30) -> List[PairResult]:
    out: List[PairResult] = []
    for mine, enemy, names in pairs:
        try:
            true = fb.simulate(mine, enemy, runs=runs)
        except Exception:
            continue
        if true is None:
            continue
        fast = simulate(mine, enemy, runs=runs, seed=0)
        t_out, t_win = _outcome(true)
        f_out, f_win = _outcome(fast)
        out.append(PairResult(fast_win=f_win, true_win=t_win, fast_outcome=f_out,
                              true_outcome=t_out, names=names))
    return out


def disagreeing_cards(results: List[PairResult], top: int = 10) -> Dict[str, int]:
    """Cards over-represented on fights the fast sim gets wrong — the suspects."""
    bad: Dict[str, int] = {}
    for r in results:
        if r.agrees:
            continue
        for n in set(r.names):
            bad[n] = bad.get(n, 0) + 1
    return dict(sorted(bad.items(), key=lambda kv: -kv[1])[:top])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the fast sim against Firestone")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if not fb.is_available():
        print("Firestone bridge unavailable — run: cd bridge && npm install")
        return 1

    pairs = collect_pairs(seed=a.seed, max_pairs=a.pairs)
    print(f"Comparing {len(pairs)} board pairs drawn from real env play "
          f"(~{2.9 * len(pairs):.0f}s)…\n")
    res = compare(pairs, runs=a.runs)
    if not res:
        print("No comparable pairs.")
        return 1

    agree = sum(1 for r in res if r.agrees)
    print(f"  outcome agreement   {agree}/{len(res)}  "
          f"({100.0 * agree / len(res):.0f}%)")
    print(f"  win%% mean abs error {statistics.mean(r.win_err for r in res):.3f}")
    print(f"  win%% worst case     {max(r.win_err for r in res):.3f}")

    bad = disagreeing_cards(res)
    if bad:
        print("\n  cards most present on disagreeing fights:")
        for name, n in bad.items():
            print(f"    {name:28s} {n:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
