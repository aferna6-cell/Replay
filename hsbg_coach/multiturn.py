"""Multi-turn lookahead planner — reason several turns ahead about tempo vs greed.

The single-turn planner (`advisor.plan_turn`) is greedy *within* a turn. This one
looks K turns ahead to answer strategic questions: "do I spend on my board now, or
sacrifice this turn to tier up for a bigger spike in two turns?"

We can't simulate exact future shops (they're random draws), so instead of
card-by-card simulation this projects the **economy trajectory** — tier, board
strength relative to the top-10% pace curve, and HP — under a handful of candidate
strategies, and scores each. The model is grounded in the real pace curves
(`data/stats/firestone_pace.json`): stats compound ~1.5–2×/turn, so falling behind
snowballs and tiering up raises your ceiling.

It's an honest economic approximation, not an oracle: it ranks *strategies*
(tempo / level-now / double-level), and the single-turn planner fills the concrete
buys for whichever this-turn intent it picks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pace import load_pace, board_stats as _board_stats, _at as _curve_at

MAX_TIER = 6

# Candidate strategies = per-turn intent sequences (padded with "tempo").
STRATEGIES = {
    "Tempo": ["tempo"],
    "Level now": ["level", "tempo"],
    "Double level": ["level", "level", "tempo"],
    "Level next turn": ["tempo", "level", "tempo"],
}

# --- model constants (approximate; tuned for sensible tempo/greed ordering) ---
_ON_TIER_GAIN = 1.06        # tempo on-tier slightly beats the curve
_UNDER_TIER_GAIN = 0.84     # tempo while under-tiered can't keep pace
_LEVEL_TURN_GAIN = 0.70     # the turn you tier up, your board falls behind
_DMG_TIER_BASE = 4.0        # below-curve damage scales with tier
_CHIP_DMG = 1.0             # minimal damage when at/above curve


@dataclass
class TurnProjection:
    turn: int
    tier: int
    stats: float
    ratio: float            # stats / top-10% curve (1.0 = on pace)
    hp: float
    intent: str

    def line(self) -> str:
        return (f"  T{self.turn}: {self.intent:11s} tier {self.tier} · "
                f"{self.ratio:.0%} of pace · hp {self.hp:.0f}")


@dataclass
class StrategyPlan:
    name: str
    intents: List[str]
    projection: List[TurnProjection]
    value: float
    died: bool = False

    @property
    def this_turn(self) -> str:
        return self.intents[0] if self.intents else "tempo"

    def summary(self) -> str:
        head = f"{self.name}  (value {self.value:.1f}{' · DIES' if self.died else ''})"
        return "\n".join([head] + [p.line() for p in self.projection])


def _exp_tier(lev, turn):
    v = _curve_at(lev, turn) if lev else None
    return v if v else min(MAX_TIER, 1 + (turn - 1) // 2)


def _exp_stats(sca, turn):
    v = _curve_at(sca, turn) if sca else None
    return v if v else max(4.0, 6.0 * (turn - 1))


def _project(turn, tier, stats, hp, intents, pace, horizon) -> StrategyPlan:
    lev = pace.get("leveling", {}) if pace else {}
    sca = pace.get("scaling", {}) if pace else {}
    proj: List[TurnProjection] = []
    cum_dmg = 0.0
    died = False
    for k in range(horizon):
        t = turn + k
        intent = intents[k] if k < len(intents) else "tempo"
        prev_curve = _exp_stats(sca, t - 1) or 1.0
        curve = _exp_stats(sca, t)
        c_growth = curve / prev_curve if prev_curve else 1.0   # how fast the field grows
        exp_tier = _exp_tier(lev, t)

        if intent == "level" and tier < MAX_TIER:
            tier += 1
            g = c_growth * _LEVEL_TURN_GAIN                    # board lags this turn
        else:
            if tier + 0.5 >= exp_tier:
                # On or above the tier curve: a tier *lead* raises your ceiling,
                # so you scale faster — this is what makes tiering up pay off.
                g = c_growth * (_ON_TIER_GAIN + 0.05 * max(0.0, tier - exp_tier))
            else:
                g = c_growth * _UNDER_TIER_GAIN                # under-tiered: can't keep pace

        stats = max(stats, 1.0) * g
        ratio = stats / curve if curve else 1.0
        dmg = (max(0.0, 1.0 - ratio) * (tier + _DMG_TIER_BASE)) or _CHIP_DMG
        dmg = max(dmg, _CHIP_DMG)
        hp -= dmg
        cum_dmg += dmg
        if hp <= 0:
            died = True
        proj.append(TurnProjection(t, tier, round(stats, 1), ratio, round(hp, 1), intent))

    term = proj[-1]
    value = (term.ratio * 10.0 + term.tier * 1.5
             - cum_dmg * 0.4 - (60.0 if died else 0.0)
             - max(0.0, 12.0 - term.hp) * 0.4)
    return StrategyPlan("", list(intents), proj, round(value, 2), died)


def plan_multiturn(snapshot, pace: Optional[Dict] = None, horizon: int = 3
                   ) -> List[StrategyPlan]:
    """Rank strategies for the next `horizon` turns, best first."""
    pace = pace if pace is not None else load_pace()
    turn = (snapshot.get("turn") if isinstance(snapshot, dict)
            else getattr(snapshot, "turn", None)) or 1
    tier = (snapshot.get("tavern_tier") if isinstance(snapshot, dict)
            else getattr(snapshot, "tavern_tier", None)) or 1
    hp = (snapshot.get("hero_health") if isinstance(snapshot, dict)
          else getattr(snapshot, "hero_health", None))
    hp = 30.0 if hp is None else float(hp)
    stats = float(_board_stats(snapshot))

    out = []
    for name, intents in STRATEGIES.items():
        plan = _project(turn, tier, stats, hp, intents, pace, horizon)
        plan.name = name
        out.append(plan)
    out.sort(key=lambda p: p.value, reverse=True)
    return out


def best_plan(snapshot, pace: Optional[Dict] = None, horizon: int = 3
              ) -> Optional[StrategyPlan]:
    plans = plan_multiturn(snapshot, pace, horizon)
    return plans[0] if plans else None
