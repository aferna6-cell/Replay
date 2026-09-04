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
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

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
    # Observational tavern tier. Combat RNG/outcomes never read this.
    # Carried so a resolved fight can report actual survivor identities/tiers.
    tier: int = 1
    # Observational body identity / composition. Combat never reads these.
    body_id: str = ""
    origin: str = "starting"          # starting | token | reborn
    golden: bool = False
    tribes: tuple = ()
    recruit_attack: Optional[int] = None
    recruit_health: Optional[int] = None
    board_slot: Optional[int] = None
    generated: bool = False

    def copy(self) -> "Combatant":
        return replace(self)

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

        raw_tier = get("tier", None)
        if raw_tier is None and isinstance(tags, dict):
            raw_tier = tags.get("TECH_LEVEL")
        try:
            tier = int(raw_tier) if raw_tier not in (None, "") else 1
        except (TypeError, ValueError):
            tier = 1

        golden = bool(get("golden", False))
        if not golden and isinstance(tags, dict):
            golden = str(tags.get("PREMIUM", "")).strip() not in ("", "0", "False")
        tribes_raw = get("tribes", None) or ()
        try:
            tribes = tuple(str(t) for t in tribes_raw)
        except TypeError:
            tribes = ()

        def _opt_int(key, fallback):
            raw = get(key, None)
            if raw in (None, ""):
                return int(fallback)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return int(fallback)

        atk = int(get("attack", 0) or 0)
        hp = int(get("health", 0) or 0)
        return cls(
            attack=atk,
            health=hp,
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
            tier=max(1, tier),
            golden=golden,
            tribes=tribes,
            recruit_attack=_opt_int("recruit_attack", atk),
            recruit_health=_opt_int("recruit_health", hp),
            origin="starting",
            generated=False,
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


# Observational only. simulate_once sets this when ``trace`` is provided.
_TRACE_CTX: Optional[Dict] = None


def _make_token(s: Summon) -> Combatant:
    return Combatant(
        s.attack, s.health, s.divine_shield, s.taunt, s.poisonous,
        s.reborn, name=s.name, tier=int(getattr(s, "tier", 1) or 1),
        origin="token", generated=True,
    )


def _record_created(tok: Combatant, board: List[Combatant]) -> None:
    """Stamp a token/reborn body id. No-ops unless a fight trace is active."""
    ctx = _TRACE_CTX
    if tok.origin == "starting" or not tok.origin:
        tok.origin = "token"
    tok.generated = True
    if ctx is None:
        return
    ctx["n"] = int(ctx.get("n") or 0) + 1
    if board is ctx.get("a"):
        side = "a"
    elif board is ctx.get("b"):
        side = "b"
    else:
        side = "?"
    tok.body_id = f"{side}:{tok.origin}:{ctx['n']}"
    row = combatant_trace_row(tok)
    row["side"] = side
    ctx.setdefault("created", []).append(row)


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
                _record_created(tok, board)
                new.append(tok)
                if dr.attack_immediately:
                    immediates.append(tok)
        if m.reborn and len(new) < BOARD_CAP:
            rb = m.copy()
            rb.health = 1
            rb.reborn = False
            rb.divine_shield = False
            rb.origin = "reborn"
            rb.generated = True
            rb.body_id = ""
            _record_created(rb, board)
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


def _record_attack(attacker: Combatant) -> None:
    """Count a combat-loop swing. No-ops unless a fight trace is active."""
    ctx = _TRACE_CTX
    if ctx is None:
        return
    bid = str(getattr(attacker, "body_id", "") or "")
    if not bid:
        return
    attacks = ctx.setdefault("attacks", {})
    attacks[bid] = int(attacks.get(bid) or 0) + 1


def _annotate_attack_rows(rows: Sequence[dict], attacks: Dict) -> None:
    """Stamp observational attacked/n_attacks onto already-built trace rows."""
    for row in rows:
        bid = str(row.get("body_id") or "")
        n = int(attacks.get(bid) or 0)
        row["n_attacks"] = n
        row["attacked"] = n > 0


def _do_attack(attacker: Combatant, atk_board: List[Combatant],
               def_board: List[Combatant], rng: random.Random) -> None:
    swings = 2 if attacker.windfury else 1
    for _ in range(swings):
        if attacker.health <= 0:
            return
        defenders = _living(def_board)
        if not defenders:
            return
        _record_attack(attacker)
        _exchange(attacker, _pick_defender(defenders, rng), def_board)
        _resolve_deaths(def_board, atk_board, rng)
        _resolve_deaths(atk_board, def_board, rng)


def _damage_to_hero(winner: List[Combatant], tavern_tier: int) -> int:
    """Approx BG bonus damage: surviving minion count + winner's tavern tier.
    (True per-minion tier needs card data — milestone 5.)"""
    return len(_living(winner)) + max(tavern_tier, 1)


def combatant_trace_row(m: Combatant) -> dict:
    """Slim observational identity for a living combatant. No RNG."""
    atk = int(getattr(m, "attack", 0) or 0)
    hp = int(getattr(m, "health", 0) or 0)
    ra = getattr(m, "recruit_attack", None)
    rh = getattr(m, "recruit_health", None)
    try:
        recruit_attack = int(ra) if ra not in (None, "") else atk
        recruit_health = int(rh) if rh not in (None, "") else hp
    except (TypeError, ValueError):
        recruit_attack, recruit_health = atk, hp
    origin = str(getattr(m, "origin", "") or "starting")
    generated = bool(getattr(m, "generated", False) or origin in ("token", "reborn"))
    tribes = tuple(getattr(m, "tribes", ()) or ())
    return {
        "body_id": str(getattr(m, "body_id", "") or ""),
        "name": str(getattr(m, "name", "") or ""),
        "card_id": str(getattr(m, "card_id", "") or ""),
        "tier": int(getattr(m, "tier", 1) or 1),
        "attack": atk,
        "health": hp,
        "recruit_attack": recruit_attack,
        "recruit_health": recruit_health,
        "combat_raw": atk + hp,
        "recruit_raw": recruit_attack + recruit_health,
        "golden": bool(getattr(m, "golden", False)),
        "tribes": list(tribes),
        "archetype": str(tribes[0]) if tribes else (
            "token" if origin == "token" else "tribeless"
        ),
        "origin": origin,
        "generated": generated,
        "token": origin == "token",
        "board_slot": getattr(m, "board_slot", None),
    }


def fill_combat_survivor_trace(
    trace: dict,
    board_a: List[Combatant],
    board_b: List[Combatant],
    result: int,
    tier_a: int,
    tier_b: int,
) -> dict:
    """Populate ``trace`` with actual combat survivors. Observational; no RNG."""
    survivors_a = [combatant_trace_row(m) for m in _living(board_a)]
    survivors_b = [combatant_trace_row(m) for m in _living(board_b)]
    if result > 0:
        survivors = survivors_a
        winner_tavern = int(tier_a)
        winner_side = "a"
    elif result < 0:
        survivors = survivors_b
        winner_tavern = int(tier_b)
        winner_side = "b"
    else:
        survivors = []
        winner_tavern = None
        winner_side = None
    tier_sum = int(sum(int(s["tier"]) for s in survivors))
    ctx = _TRACE_CTX or {}
    created = list(ctx.get("created") or [])
    attacks = dict(ctx.get("attacks") or {})
    _annotate_attack_rows(survivors_a, attacks)
    _annotate_attack_rows(survivors_b, attacks)
    _annotate_attack_rows(trace.get("starting_a") or [], attacks)
    _annotate_attack_rows(trace.get("starting_b") or [], attacks)
    _annotate_attack_rows(created, attacks)
    if winner_side == "a":
        starting_winner = list(trace.get("starting_a") or [])
        created_winner = [c for c in created if c.get("side") == "a"]
    elif winner_side == "b":
        starting_winner = list(trace.get("starting_b") or [])
        created_winner = [c for c in created if c.get("side") == "b"]
    else:
        starting_winner = []
        created_winner = []
    trace.update({
        "winner_side": winner_side,
        "survivors": survivors,
        "survivors_a": survivors_a,
        "survivors_b": survivors_b,
        "survivor_count": len(survivors),
        "survivor_tier_sum": tier_sum,
        "winner_tavern_tier": winner_tavern,
        "rules_faithful_damage": (
            None if winner_tavern is None else int(winner_tavern) + tier_sum
        ),
        "created": created,
        "starting_winner": starting_winner,
        "created_winner": created_winner,
        "attacks": attacks,
    })
    return trace


def simulate_once(
    board_a: Sequence[Combatant],
    board_b: Sequence[Combatant],
    rng: random.Random,
    tier_a: int = 1,
    tier_b: int = 1,
    max_steps: int = 500,
    trace: Optional[dict] = None,
) -> int:
    """Return signed damage: >0 if A wins (dmg to B's hero), <0 if B wins, 0 tie.

    Optional ``trace`` is filled after the fight with actual surviving minion
    identities/tiers plus starting/created body ids. Must not consume RNG or
    change the returned damage.
    """
    global _TRACE_CTX
    a = [m.copy() for m in board_a]
    b = [m.copy() for m in board_b]
    _stamp_starting_bodies(a, "a")
    _stamp_starting_bodies(b, "b")
    boards = {0: a, 1: b}
    prev_ctx = _TRACE_CTX
    if trace is not None:
        _TRACE_CTX = {"a": a, "b": b, "created": [], "n": 0, "attacks": {}}
        trace["starting_a"] = [combatant_trace_row(m) for m in a]
        trace["starting_b"] = [combatant_trace_row(m) for m in b]
    else:
        _TRACE_CTX = None

    try:
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
            result = _damage_to_hero(a, tier_a)
        elif b_alive and not a_alive:
            result = -_damage_to_hero(b, tier_b)
        else:
            result = 0
        if trace is not None:
            fill_combat_survivor_trace(trace, a, b, result, tier_a, tier_b)
        return result
    finally:
        _TRACE_CTX = prev_ctx


def _stamp_starting_bodies(board: List[Combatant], side: str) -> None:
    """Assign stable observational ids to starting combat copies. No RNG."""
    for i, m in enumerate(board):
        m.body_id = f"{side}:start:{i}"
        m.origin = "starting"
        m.generated = False
        if m.board_slot is None:
            m.board_slot = i


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
