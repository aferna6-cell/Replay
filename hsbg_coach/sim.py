"""Monte Carlo Battlegrounds combat simulator.

Given two boards, simulate the auto-battle many times and report win/tie/loss
probabilities plus expected damage — the engine behind "you're 60% to win this
fight." No ML, pure rules.

Modeled now:
- Core attack loop (alternating sides, more-minions-attacks-first, random tie).
- Keywords: Divine Shield, Taunt, Poisonous, Reborn, **Windfury**, **Cleave**.
- **Start-of-combat** effects (deal damage to random enemies — Red Whelp etc.).
- **Deathrattle summons** (Harvest Golem, Rat Pack, Scallywag's immediate-attack
  token, etc.), driven by the registry in ``effects.py``.

Still partial: the *card list* in effects.py is representative, not exhaustive,
and a few effects (stat-buff deathrattles, damage-deathrattles, golden/triples)
are placeholders. The engine handles the mechanics; completing coverage is a
data task (HearthstoneJSON / BG JSON) or a bridge to Firestone's open-source sim.
See specs §6. Odds are most accurate on boards built from modeled cards.

Input is anything with ``attack``/``health`` and a ``tags`` dict (e.g. the
``MinionView`` from bg.py), or build ``Combatant`` directly.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .effects import Summon, StartOfCombat, effects_for

BOARD_CAP = 7

# Tag names that flag keywords on a MinionView. Values are truthy when set.
_KW_TAGS = {
    "divine_shield": "DIVINE_SHIELD",
    "taunt": "TAUNT",
    "poisonous": "POISONOUS",
    "reborn": "REBORN",
    "windfury": "WINDFURY",
    "cleave": "CLEAVE",
}


@dataclass
class Combatant:
    attack: int
    health: int
    divine_shield: bool = False
    taunt: bool = False
    poisonous: bool = False
    reborn: bool = False
    windfury: bool = False
    cleave: bool = False
    name: str = ""
    deathrattle: Optional[Summon] = None
    start_of_combat: Optional[StartOfCombat] = None
    card_id: str = ""                 # needed by the Firestone bridge for effects
    death_burst: Optional["object"] = None   # AOE-damage deathrattle (Tunnel Blaster)

    def copy(self) -> "Combatant":
        return Combatant(
            self.attack, self.health, self.divine_shield, self.taunt,
            self.poisonous, self.reborn, self.windfury, self.cleave,
            self.name, self.deathrattle, self.start_of_combat, self.card_id,
            self.death_burst)

    @classmethod
    def from_minion(cls, m) -> "Combatant":
        # Minions arrive as either attribute objects (MinionView) or plain dicts
        # (live snapshots, discover candidates); read both the same way, else a
        # dict silently becomes a 0/0 with no keywords.
        def get(key, default=None):
            if isinstance(m, dict):
                return m.get(key, default)
            return getattr(m, key, default)

        tags = get("tags", {}) or {}
        name = get("name", "") or get("card_id", "") or ""
        eff = effects_for(get("name", None), get("card_id", None))
        grants = set(getattr(eff, "grants", ()) or ()) if eff else set()

        def flag(key: str) -> bool:
            if key in grants:                       # keyword granted by the card itself
                return True
            return str(tags.get(_KW_TAGS[key], "")).strip() not in ("", "0", "False")

        return cls(
            attack=int(get("attack", 0) or 0),
            health=int(get("health", 0) or 0),
            divine_shield=flag("divine_shield"),
            taunt=flag("taunt"),
            poisonous=flag("poisonous"),
            reborn=flag("reborn"),
            windfury=flag("windfury"),
            cleave=flag("cleave"),
            name=name,
            deathrattle=eff.deathrattle if eff else None,
            start_of_combat=eff.start_of_combat if eff else None,
            card_id=get("card_id", "") or "",
            death_burst=eff.death_burst if eff else None,
        )


@dataclass
class SimResult:
    runs: int
    wins: int
    ties: int
    losses: int
    avg_damage_dealt: float
    avg_damage_taken: float

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


def _apply_damage(target: Combatant, dmg: int, poison: bool = False) -> None:
    if dmg <= 0:
        return
    if target.divine_shield:
        target.divine_shield = False     # shield eats the hit (and any poison)
        return
    target.health -= dmg
    if poison:
        target.health = min(target.health, 0)


def _make_token(s: Summon) -> Combatant:
    return Combatant(s.attack, s.health, s.divine_shield, s.taunt, s.poisonous,
                     s.reborn, name=s.name)


def _pick_defender(defenders: List[Combatant], rng: random.Random) -> Combatant:
    taunts = [m for m in defenders if m.taunt]
    return rng.choice(taunts or defenders)


def _resolve_start_of_combat(side: List[Combatant], enemy: List[Combatant],
                             rng: random.Random) -> None:
    for m in list(side):
        soc = m.start_of_combat
        if not soc or m.health <= 0:
            continue
        for _ in range(soc.targets):
            living = _living(enemy)
            if not living:
                break
            _apply_damage(rng.choice(living), soc.damage)


def _exchange(attacker: Combatant, defender: Combatant,
              def_board: List[Combatant]) -> None:
    """One simultaneous strike: attacker<->defender, plus cleave to neighbors."""
    a_dmg, d_dmg = attacker.attack, defender.attack
    try:
        di = def_board.index(defender)
    except ValueError:
        di = -1
    _apply_damage(defender, a_dmg, attacker.poisonous)
    if attacker.cleave and di >= 0:
        for nb in (di - 1, di + 1):
            if 0 <= nb < len(def_board) and def_board[nb] is not defender:
                _apply_damage(def_board[nb], a_dmg, attacker.poisonous)
    _apply_damage(attacker, d_dmg, defender.poisonous)


def _resolve_deaths(board: List[Combatant], enemy: List[Combatant],
                    rng: random.Random, process_immediates: bool = True) -> None:
    """Rebuild a board: drop dead minions, fire deathrattle summons + reborn.

    Immediate-attack tokens (Scallywag) strike right away. process_immediates is
    set False on the recursive resolve so chains terminate."""
    if all(m.health > 0 for m in board):
        return
    new: List[Combatant] = []
    immediates: List[Combatant] = []
    bursts: List = []                                  # AOE-damage deathrattles
    for m in board:
        if m.health > 0:
            new.append(m)
            continue
        if m.death_burst:
            bursts.append(m.death_burst)
        dr = m.deathrattle
        if dr and dr.count > 0:
            for _ in range(dr.count):
                if len(new) >= BOARD_CAP:
                    break
                tok = _make_token(dr)
                new.append(tok)
                if dr.attack_immediately:
                    immediates.append(tok)
        if m.reborn and len(new) < BOARD_CAP:
            rb = m.copy()
            rb.health = 1
            rb.reborn = False
            rb.divine_shield = False
            new.append(rb)
    board[:] = new

    # Fire AOE-damage deathrattles (Tunnel Blaster: 3 to all enemies) — pops Divine
    # Shields and clears swarms, then resolve the deaths that causes.
    for burst in bursts:
        targets = _living(enemy)
        if not getattr(burst, "hits_all", True):
            rng.shuffle(targets)
            targets = targets[:max(1, getattr(burst, "targets", 1))]
        for t in targets:
            _apply_damage(t, burst.damage)
        if targets:
            _resolve_deaths(enemy, board, rng, process_immediates=False)

    if not process_immediates:
        return
    for tok in immediates:
        if tok.health <= 0:
            continue
        defenders = _living(enemy)
        if not defenders:
            continue
        _exchange(tok, _pick_defender(defenders, rng), enemy)
        _resolve_deaths(enemy, board, rng, process_immediates=False)
        _resolve_deaths(board, enemy, rng, process_immediates=False)


def _do_attack(attacker: Combatant, atk_board: List[Combatant],
               def_board: List[Combatant], rng: random.Random) -> None:
    swings = 2 if attacker.windfury else 1
    for _ in range(swings):
        if attacker.health <= 0:
            return
        defenders = _living(def_board)
        if not defenders:
            return
        _exchange(attacker, _pick_defender(defenders, rng), def_board)
        _resolve_deaths(def_board, atk_board, rng)
        _resolve_deaths(atk_board, def_board, rng)


def _damage_to_hero(winner: List[Combatant], tavern_tier: int) -> int:
    """Approx BG bonus damage: surviving minion count + winner's tavern tier.
    (True per-minion tier needs card data — milestone 5.)"""
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
    boards = {0: a, 1: b}

    la, lb = len(_living(a)), len(_living(b))
    if la > lb:
        turn = 0
    elif lb > la:
        turn = 1
    else:
        turn = rng.randint(0, 1)

    # Start-of-combat resolves for the first-attacking side first.
    for side in (turn, 1 - turn):
        _resolve_start_of_combat(boards[side], boards[1 - side], rng)
    _resolve_deaths(a, b, rng)
    _resolve_deaths(b, a, rng)

    pos = {0: 0, 1: 0}
    for _ in range(max_steps):
        if not _living(a) or not _living(b):
            break
        atk_board, def_board = boards[turn], boards[1 - turn]
        living = _living(atk_board)
        idx = pos[turn] % len(living)
        attacker = living[idx]
        pos[turn] = idx + 1
        _do_attack(attacker, atk_board, def_board, rng)
        turn ^= 1

    a_alive, b_alive = bool(_living(a)), bool(_living(b))
    if a_alive and not b_alive:
        return _damage_to_hero(a, tier_a)
    if b_alive and not a_alive:
        return -_damage_to_hero(b, tier_b)
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
