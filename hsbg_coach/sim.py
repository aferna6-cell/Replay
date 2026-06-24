"""Monte Carlo Battlegrounds combat simulator.

Given two boards, simulate the auto-battle many times and report win/tie/loss
probabilities plus expected damage. This is the engine behind "you're 60% to win
this fight" — no ML, pure rules.

SCOPE (honest): this models the *core* combat loop and the four keywords that
dominate most fights — Divine Shield, Taunt, Poisonous, Reborn. It does NOT yet
model deathrattles, battlecries, auras, or minion-specific text. Those are the
long tail that a full port of Bob's Buddy (HDT's simulator) handles; wiring that
in is milestone 3b. Until then, treat odds as a strong approximation, most
accurate on stat-stick boards and least accurate on deathrattle/scam comps.

Input is anything with ``attack``/``health`` and a ``tags`` dict (e.g. the
``MinionView`` produced by bg.py), or build ``Combatant`` directly.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Tag names that flag keywords on a MinionView. Values are truthy when set.
_KW_TAGS = {
    "divine_shield": "DIVINE_SHIELD",
    "taunt": "TAUNT",
    "poisonous": "POISONOUS",
    "reborn": "REBORN",
}


@dataclass
class Combatant:
    attack: int
    health: int
    divine_shield: bool = False
    taunt: bool = False
    poisonous: bool = False
    reborn: bool = False
    name: str = ""

    def copy(self) -> "Combatant":
        return Combatant(self.attack, self.health, self.divine_shield,
                         self.taunt, self.poisonous, self.reborn, self.name)

    @classmethod
    def from_minion(cls, m) -> "Combatant":
        tags = getattr(m, "tags", {}) or {}

        def flag(key: str) -> bool:
            return str(tags.get(_KW_TAGS[key], "")).strip() not in ("", "0", "False")

        return cls(
            attack=int(getattr(m, "attack", 0) or 0),
            health=int(getattr(m, "health", 0) or 0),
            divine_shield=flag("divine_shield"),
            taunt=flag("taunt"),
            poisonous=flag("poisonous"),
            reborn=flag("reborn"),
            name=getattr(m, "name", "") or getattr(m, "card_id", "") or "",
        )


@dataclass
class SimResult:
    runs: int
    wins: int
    ties: int
    losses: int
    avg_damage_dealt: float       # to enemy hero when we win
    avg_damage_taken: float       # to our hero when we lose

    @property
    def win_pct(self) -> float:
        return self.wins / self.runs if self.runs else 0.0

    @property
    def tie_pct(self) -> float:
        return self.ties / self.runs if self.runs else 0.0

    @property
    def loss_pct(self) -> float:
        return self.losses / self.runs if self.runs else 0.0

    def summary(self) -> str:
        return (f"win {self.win_pct:.0%} / tie {self.tie_pct:.0%} / "
                f"loss {self.loss_pct:.0%}  "
                f"(avg dmg dealt {self.avg_damage_dealt:.1f}, "
                f"taken {self.avg_damage_taken:.1f})")


def _living(board: List[Combatant]) -> List[Combatant]:
    return [m for m in board if m.health > 0]


def _next_attacker(board: List[Combatant], last_idx: int) -> Optional[int]:
    """Next living minion at-or-after (last_idx+1), wrapping once."""
    n = len(board)
    for step in range(1, n + 1):
        idx = (last_idx + step) % n
        if board[idx].health > 0:
            return idx
    return None


def _pick_defender(defenders: List[Combatant], rng: random.Random) -> int:
    living_idx = [i for i, m in enumerate(defenders) if m.health > 0]
    taunts = [i for i in living_idx if defenders[i].taunt]
    pool = taunts or living_idx
    return rng.choice(pool)


def _strike(attacker: Combatant, defender: Combatant) -> None:
    """Apply attacker -> defender damage, honoring divine shield + poisonous."""
    dmg = attacker.attack
    if dmg <= 0:
        return
    if defender.divine_shield:
        defender.divine_shield = False  # shield eats the hit
        return
    defender.health -= dmg
    if attacker.poisonous and dmg > 0:
        defender.health = min(defender.health, 0)


def _resolve_deaths(board: List[Combatant]) -> None:
    """Reborn minions respawn with 1 health; others are removed by the caller."""
    for m in board:
        if m.health <= 0 and m.reborn:
            m.health = 1
            m.reborn = False
            m.divine_shield = False


def _damage_to_hero(winner: List[Combatant], tavern_tier: int) -> int:
    """BG: damage = sum of surviving minions' tiers + winner's tavern tier.
    We approximate per-minion tier as 1 (true tiers need card data); this gives
    a rough magnitude, refined once card->tier data is wired in (milestone 5)."""
    return len(_living(winner)) + max(tavern_tier, 1)


def simulate_once(
    board_a: Sequence[Combatant],
    board_b: Sequence[Combatant],
    rng: random.Random,
    tier_a: int = 1,
    tier_b: int = 1,
    max_steps: int = 500,
) -> int:
    """Return signed damage: >0 if A wins (dmg to B's hero), <0 if B wins, 0 tie."""
    a = [m.copy() for m in board_a]
    b = [m.copy() for m in board_b]

    # Side with more minions attacks first; tie broken randomly.
    if len(_living(a)) > len(_living(b)):
        turn = 0
    elif len(_living(b)) > len(_living(a)):
        turn = 1
    else:
        turn = rng.randint(0, 1)

    last = {0: -1, 1: -1}
    boards = {0: a, 1: b}

    for _ in range(max_steps):
        if not _living(a) or not _living(b):
            break
        atk_board = boards[turn]
        def_board = boards[1 - turn]
        ai = _next_attacker(atk_board, last[turn])
        if ai is None:
            turn ^= 1
            continue
        last[turn] = ai
        di = _pick_defender(def_board, rng)

        attacker, defender = atk_board[ai], def_board[di]
        # Simultaneous exchange.
        _strike(attacker, defender)
        _strike(defender, attacker)
        _resolve_deaths(atk_board)
        _resolve_deaths(def_board)
        boards[0][:] = [m for m in boards[0] if m.health > 0]
        boards[1][:] = [m for m in boards[1] if m.health > 0]
        # Removing dead shifts indices; reset the attacker pointers to be safe.
        last = {0: -1, 1: -1}
        turn ^= 1

    a_alive, b_alive = bool(_living(boards[0])), bool(_living(boards[1]))
    if a_alive and not b_alive:
        return _damage_to_hero(boards[0], tier_a)
    if b_alive and not a_alive:
        return -_damage_to_hero(boards[1], tier_b)
    return 0


def simulate(
    board_a: Sequence,
    board_b: Sequence,
    runs: int = 1000,
    seed: Optional[int] = None,
    tier_a: int = 1,
    tier_b: int = 1,
) -> SimResult:
    """Run the matchup ``runs`` times. Accepts Combatants or MinionView-likes."""
    ca = [m if isinstance(m, Combatant) else Combatant.from_minion(m) for m in board_a]
    cb = [m if isinstance(m, Combatant) else Combatant.from_minion(m) for m in board_b]
    rng = random.Random(seed)

    wins = ties = losses = 0
    dmg_dealt = dmg_taken = 0
    for _ in range(runs):
        result = simulate_once(ca, cb, rng, tier_a=tier_a, tier_b=tier_b)
        if result > 0:
            wins += 1
            dmg_dealt += result
        elif result < 0:
            losses += 1
            dmg_taken += -result
        else:
            ties += 1
    return SimResult(
        runs=runs, wins=wins, ties=ties, losses=losses,
        avg_damage_dealt=(dmg_dealt / wins) if wins else 0.0,
        avg_damage_taken=(dmg_taken / losses) if losses else 0.0,
    )
