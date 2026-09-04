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

from .effects import (
    Summon,
    StartOfCombat,
    effects_for,
    APPROXIMATE_DEATHRATTLE_NAMES,
    PLACEHOLDER_DEATHRATTLE_NAMES,
    is_placeholder_summon,
)

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


def classify_effect_status(m: Combatant) -> str:
    """Mark represented vs unsupported effects. Never invent a missing mechanic."""
    name = str(getattr(m, "name", "") or "")
    if name in APPROXIMATE_DEATHRATTLE_NAMES:
        return "represented_approximate"
    if name in PLACEHOLDER_DEATHRATTLE_NAMES:
        return "unsupported_placeholder"
    dr = getattr(m, "deathrattle", None)
    if is_placeholder_summon(dr):
        return "unsupported_placeholder"
    if dr is not None and int(getattr(dr, "count", 0) or 0) > 0:
        return "represented"
    if getattr(m, "death_burst", None) is not None:
        return "represented"
    if getattr(m, "start_of_combat", None) is not None:
        return "represented"
    if bool(getattr(m, "reborn", False)):
        return "represented"
    return "unregistered"


def has_represented_generated_effect(m: Combatant) -> bool:
    """Faithful deathrattle-summon / death-burst / reborn — not SOC, not stubs."""
    status = classify_effect_status(m)
    if status != "represented":
        return False
    dr = getattr(m, "deathrattle", None)
    if dr is not None and int(getattr(dr, "count", 0) or 0) > 0:
        return True
    if getattr(m, "death_burst", None) is not None:
        return True
    return bool(getattr(m, "reborn", False))


def _bump_map(ctx: Dict, name: str, bid: str, n: int = 1) -> None:
    bucket = ctx.setdefault(name, {})
    bucket[bid] = int(bucket.get(bid) or 0) + n


def _trace_hit(
    ctx: Dict,
    bid: str,
    *,
    cause: str,
    poison: bool,
    cleave_role: Optional[str],
    ds_before: bool,
    ds_after: bool,
    hp_before: int,
    hp_after: int,
    lethal: bool,
) -> None:
    """Observational per-swing / per-body lethal-cause tags. No RNG."""
    ctx["n_damage"] = int(ctx.get("n_damage") or 0) + 1
    if not bid:
        return
    ctx.setdefault("end_health", {})[bid] = int(hp_after)
    ctx.setdefault("end_divine_shield", {})[bid] = bool(ds_after)
    ctx.setdefault("ds_before_last_hit", {})[bid] = bool(ds_before)
    ctx.setdefault("ds_after_last_hit", {})[bid] = bool(ds_after)
    _bump_map(ctx, "n_hits", bid)
    if ds_before and not ds_after:
        ctx["n_shield_pops"] = int(ctx.get("n_shield_pops") or 0) + 1
        _bump_map(ctx, "n_shield_pops_body", bid)
        pops = ctx.setdefault("shield_pop_cause", {})
        if bid not in pops:
            pops[bid] = cause
    if poison:
        ctx["n_poison_hits"] = int(ctx.get("n_poison_hits") or 0) + 1
        _bump_map(ctx, "n_hits_poison", bid)
        if lethal:
            ctx.setdefault("poison_lethal", {})[bid] = True
    if cleave_role == "primary":
        ctx["n_cleave_primary"] = int(ctx.get("n_cleave_primary") or 0) + 1
        _bump_map(ctx, "n_cleave_primary_body", bid)
    elif cleave_role == "secondary" or cause == "cleave":
        ctx["n_cleave_secondary"] = int(ctx.get("n_cleave_secondary") or 0) + 1
        _bump_map(ctx, "n_cleave_secondary_body", bid)
    if (cleave_role or cause == "cleave") and lethal:
        ctx.setdefault("cleave_lethal", {})[bid] = True
    if cause == "start_of_combat":
        ctx["n_soc_hits"] = int(ctx.get("n_soc_hits") or 0) + 1
        _bump_map(ctx, "n_soc_hits_body", bid)
        if lethal:
            ctx.setdefault("soc_lethal", {})[bid] = True
    if cause == "attack":
        ctx["n_ordinary_attack"] = int(ctx.get("n_ordinary_attack") or 0) + 1
        _bump_map(ctx, "n_ordinary_attack_body", bid)
    elif cause == "counterattack":
        ctx["n_ordinary_counter"] = int(ctx.get("n_ordinary_counter") or 0) + 1
        _bump_map(ctx, "n_ordinary_counter_body", bid)
    if lethal and cause in ("attack", "counterattack"):
        ctx.setdefault("ordinary_lethal", {})[bid] = True
    if lethal:
        ctx.setdefault("death_cause", {})[bid] = cause
        ctx.setdefault("killed_by", {})[bid] = ctx.get("pending_attacker_id")
        ctx["n_deaths"] = int(ctx.get("n_deaths") or 0) + 1


def _apply_damage(target: Combatant, dmg: int, poison: bool = False,
                  cause: str = "attack", cleave_role: Optional[str] = None) -> None:
    if dmg <= 0:
        return
    ctx = _TRACE_CTX
    pre = int(target.health)
    bid = str(getattr(target, "body_id", "") or "")
    ds_before = bool(target.divine_shield)
    if target.divine_shield:
        target.divine_shield = False     # shield eats the hit (and any poison)
        if ctx is not None:
            _trace_hit(
                ctx, bid,
                cause=cause, poison=poison, cleave_role=cleave_role,
                ds_before=True, ds_after=False,
                hp_before=pre, hp_after=int(target.health), lethal=False,
            )
        return
    target.health -= dmg
    if poison:
        target.health = min(target.health, 0)
    if ctx is not None:
        lethal = pre > 0 and target.health <= 0
        _trace_hit(
            ctx, bid,
            cause=cause, poison=poison, cleave_role=cleave_role,
            ds_before=ds_before, ds_after=bool(target.divine_shield),
            hp_before=pre, hp_after=int(target.health), lethal=lethal,
        )


# Observational only. simulate_once sets this when ``trace`` is provided.
_TRACE_CTX: Optional[Dict] = None


def _make_token(s: Summon) -> Combatant:
    return Combatant(
        s.attack, s.health, s.divine_shield, s.taunt, s.poisonous,
        s.reborn, name=s.name, tier=int(getattr(s, "tier", 1) or 1),
        origin="token", generated=True,
    )


def _record_created(tok: Combatant, board: List[Combatant],
                    parent: Optional[Combatant] = None) -> None:
    """Stamp a token/reborn body id. No-ops unless a fight trace is active."""
    ctx = _TRACE_CTX
    if tok.origin == "starting" or not tok.origin:
        tok.origin = "token"
    tok.generated = True
    if ctx is None:
        return
    ctx["n"] = int(ctx.get("n") or 0) + 1
    ctx["n_created"] = int(ctx.get("n_created") or 0) + 1
    if board is ctx.get("a"):
        side = "a"
    elif board is ctx.get("b"):
        side = "b"
    else:
        side = "?"
    tok.body_id = f"{side}:{tok.origin}:{ctx['n']}"
    row = combatant_trace_row(tok)
    row["side"] = side
    parent_id = str(getattr(parent, "body_id", "") or "") if parent else ""
    row["parent_body_id"] = parent_id
    parent_status = classify_effect_status(parent) if parent else "unregistered"
    represented = parent_status == "represented"
    row["represented_generated"] = represented
    row["parent_effect_status"] = parent_status
    if parent_id and represented:
        spawned = ctx.setdefault("spawned_represented", {})
        spawned[parent_id] = int(spawned.get(parent_id) or 0) + 1
    ctx.setdefault("created", []).append(row)


def _pick_defender(defenders: List[Combatant], rng: random.Random) -> Combatant:
    taunts = [m for m in defenders if m.taunt]
    forced = bool(taunts)
    chosen = rng.choice(taunts or defenders)
    # Observational targeting exposure. Choice already consumed RNG.
    ctx = _TRACE_CTX
    if ctx is not None:
        bid = str(getattr(chosen, "body_id", "") or "")
        ctx["n_targets"] = int(ctx.get("n_targets") or 0) + 1
        if forced:
            ctx["n_targets_forced"] = int(ctx.get("n_targets_forced") or 0) + 1
        else:
            ctx["n_targets_open"] = int(ctx.get("n_targets_open") or 0) + 1
        if bid:
            targeted = ctx.setdefault("targeted", {})
            targeted[bid] = int(targeted.get(bid) or 0) + 1
            bucket_name = "targeted_forced" if forced else "targeted_open"
            bucket = ctx.setdefault(bucket_name, {})
            bucket[bid] = int(bucket.get(bid) or 0) + 1
            ctx.setdefault("last_attacker", {})[bid] = ctx.get("pending_attacker_id")
    return chosen


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
            _apply_damage(rng.choice(living), soc.damage, cause="start_of_combat")


def _exchange(attacker: Combatant, defender: Combatant,
              def_board: List[Combatant]) -> None:
    """One simultaneous strike: attacker<->defender, plus cleave to neighbors."""
    a_dmg, d_dmg = attacker.attack, defender.attack
    try:
        di = def_board.index(defender)
    except ValueError:
        di = -1
    cause = "poison" if attacker.poisonous else "attack"
    primary_role = "primary" if attacker.cleave else None
    _apply_damage(
        defender, a_dmg, attacker.poisonous, cause=cause, cleave_role=primary_role,
    )
    if attacker.cleave and di >= 0:
        for nb in (di - 1, di + 1):
            if 0 <= nb < len(def_board) and def_board[nb] is not defender:
                _apply_damage(
                    def_board[nb], a_dmg, attacker.poisonous, cause="cleave",
                    cleave_role="secondary",
                )
    back = "poison" if defender.poisonous else "counterattack"
    _apply_damage(attacker, d_dmg, defender.poisonous, cause=back)


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
                _record_created(tok, board, parent=m)
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
            _record_created(rb, board, parent=m)
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
            _apply_damage(t, burst.damage, cause="death_burst")
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
        ctx = _TRACE_CTX
        if ctx is not None:
            ctx["pending_attacker_id"] = str(getattr(tok, "body_id", "") or "")
            ctx["n_immediate_attacks"] = int(ctx.get("n_immediate_attacks") or 0) + 1
        _record_attack(tok)
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
    seq = int(ctx.get("n_swings") or 0)
    ctx["n_swings"] = seq + 1
    first = ctx.setdefault("first_attack_index", {})
    if bid not in first:
        first[bid] = seq


def _annotate_attack_rows(
    rows: Sequence[dict],
    attacks: Dict,
    first_attack: Optional[Dict] = None,
    targeted: Optional[Dict] = None,
    ctx: Optional[Dict] = None,
) -> None:
    """Stamp observational attack/target fields onto already-built trace rows."""
    first_attack = first_attack or {}
    targeted = targeted or {}
    ctx = ctx or {}
    forced = ctx.get("targeted_forced") or {}
    opened = ctx.get("targeted_open") or {}
    death_cause = ctx.get("death_cause") or {}
    killed_by = ctx.get("killed_by") or {}
    end_health = ctx.get("end_health") or {}
    spawned = ctx.get("spawned_represented") or {}
    wrap_before = ctx.get("wrap_before_first") or {}
    first_label = str(ctx.get("first_side_label") or "")
    last_attacker = ctx.get("last_attacker") or {}
    for row in rows:
        bid = str(row.get("body_id") or "")
        n = int(attacks.get(bid) or 0)
        row["n_attacks"] = n
        row["attacked"] = n > 0
        idx = first_attack.get(bid)
        row["first_attack_index"] = int(idx) if idx is not None else None
        nt = int(targeted.get(bid) or 0)
        row["n_targeted"] = nt
        row["was_targeted"] = nt > 0
        nf = int(forced.get(bid) or 0)
        no = int(opened.get(bid) or 0)
        row["n_targeted_forced"] = nf
        row["n_targeted_open"] = no
        row["taunt_forced_target"] = nf > 0
        row["open_target"] = no > 0 and nf == 0
        row["death_cause"] = death_cause.get(bid)
        row["killed_by_body_id"] = killed_by.get(bid)
        row["last_attacker_id"] = last_attacker.get(bid)
        start_hp = int(row.get("health") or 0)
        row["start_health"] = start_hp
        if bid in end_health:
            row["end_health"] = int(end_health[bid])
        elif row.get("death_cause"):
            row["end_health"] = 0
        else:
            row["end_health"] = start_hp
        row["spawned_represented"] = int(spawned.get(bid) or 0)
        row["cursor_wrapped_before_first"] = bool(wrap_before.get(bid))
        side = bid.split(":")[0] if bid else ""
        row["side_first"] = bool(first_label) and side == first_label
        row["effect_status"] = row.get("effect_status") or "unregistered"
        row["has_unsupported_effect"] = row.get("effect_status") in (
            "unsupported_placeholder", "represented_approximate",
        )
        row["has_represented_generated_effect"] = bool(
            row.get("has_represented_generated_effect")
        )
        row["n_hits"] = int((ctx.get("n_hits") or {}).get(bid) or 0)
        row["n_shield_pops"] = int(
            (ctx.get("n_shield_pops_body") or {}).get(bid) or 0
        )
        row["shield_pop_cause"] = (ctx.get("shield_pop_cause") or {}).get(bid)
        row["ds_before_last_hit"] = (ctx.get("ds_before_last_hit") or {}).get(bid)
        row["ds_after_last_hit"] = (ctx.get("ds_after_last_hit") or {}).get(bid)
        if bid in (ctx.get("end_divine_shield") or {}):
            row["end_divine_shield"] = bool(ctx["end_divine_shield"][bid])
        else:
            row["end_divine_shield"] = bool(row.get("start_divine_shield"))
        row["n_hits_poison"] = int((ctx.get("n_hits_poison") or {}).get(bid) or 0)
        row["poison_lethal"] = bool((ctx.get("poison_lethal") or {}).get(bid))
        row["n_cleave_primary"] = int(
            (ctx.get("n_cleave_primary_body") or {}).get(bid) or 0
        )
        row["n_cleave_secondary"] = int(
            (ctx.get("n_cleave_secondary_body") or {}).get(bid) or 0
        )
        row["cleave_lethal"] = bool((ctx.get("cleave_lethal") or {}).get(bid))
        row["n_soc_hits"] = int((ctx.get("n_soc_hits_body") or {}).get(bid) or 0)
        row["soc_lethal"] = bool((ctx.get("soc_lethal") or {}).get(bid))
        row["n_ordinary_attack_hits"] = int(
            (ctx.get("n_ordinary_attack_body") or {}).get(bid) or 0
        )
        row["n_ordinary_counter_hits"] = int(
            (ctx.get("n_ordinary_counter_body") or {}).get(bid) or 0
        )
        row["ordinary_lethal"] = bool((ctx.get("ordinary_lethal") or {}).get(bid))


def _do_attack(attacker: Combatant, atk_board: List[Combatant],
               def_board: List[Combatant], rng: random.Random) -> None:
    swings = 2 if attacker.windfury else 1
    for _ in range(swings):
        if attacker.health <= 0:
            return
        defenders = _living(def_board)
        if not defenders:
            return
        ctx = _TRACE_CTX
        if ctx is not None:
            ctx["pending_attacker_id"] = str(getattr(attacker, "body_id", "") or "")
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
    status = classify_effect_status(m)
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
        "taunt": bool(getattr(m, "taunt", False)),
        "divine_shield": bool(getattr(m, "divine_shield", False)),
        "poisonous": bool(getattr(m, "poisonous", False)),
        "cleave": bool(getattr(m, "cleave", False)),
        "start_divine_shield": bool(getattr(m, "divine_shield", False)),
        "effect_status": status,
        "has_unsupported_effect": status in (
            "unsupported_placeholder", "represented_approximate",
        ),
        "has_represented_generated_effect": has_represented_generated_effect(m),
        "start_health": hp,
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
    first_attack = dict(ctx.get("first_attack_index") or {})
    targeted = dict(ctx.get("targeted") or {})
    _annotate_attack_rows(survivors_a, attacks, first_attack, targeted, ctx)
    _annotate_attack_rows(survivors_b, attacks, first_attack, targeted, ctx)
    _annotate_attack_rows(trace.get("starting_a") or [], attacks, first_attack, targeted, ctx)
    _annotate_attack_rows(trace.get("starting_b") or [], attacks, first_attack, targeted, ctx)
    _annotate_attack_rows(created, attacks, first_attack, targeted, ctx)
    if winner_side == "a":
        starting_winner = list(trace.get("starting_a") or [])
        created_winner = [c for c in created if c.get("side") == "a"]
    elif winner_side == "b":
        starting_winner = list(trace.get("starting_b") or [])
        created_winner = [c for c in created if c.get("side") == "b"]
    else:
        starting_winner = []
        created_winner = []
    start_all = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    all_bodies = start_all + created
    n_attacks_sum = int(sum(int(r.get("n_attacks") or 0) for r in all_bodies))
    n_targets_sum = int(sum(int(r.get("n_targeted") or 0) for r in all_bodies))
    n_forced_sum = int(sum(int(r.get("n_targeted_forced") or 0) for r in all_bodies))
    n_open_sum = int(sum(int(r.get("n_targeted_open") or 0) for r in all_bodies))
    n_living_end = len(_living(board_a)) + len(_living(board_b))
    n_start = len(start_all)
    n_created = len(created)
    n_deaths_expected = n_start + n_created - n_living_end
    n_attacks_events = int(ctx.get("n_swings") or 0)
    n_targets_events = int(ctx.get("n_targets") or 0)
    n_forced_events = int(ctx.get("n_targets_forced") or 0)
    n_open_events = int(ctx.get("n_targets_open") or 0)
    n_death_events = int(ctx.get("n_deaths") or 0)
    n_created_events = int(ctx.get("n_created") or 0)
    n_cursor_advance = int(ctx.get("n_cursor_advance") or 0)
    n_cursor_wrap = int(ctx.get("n_cursor_wrap") or 0)
    n_placeholder = sum(
        1 for r in start_all
        if r.get("effect_status") in (
            "unsupported_placeholder", "represented_approximate",
        )
    )
    n_created_represented = sum(
        1 for c in created if c.get("represented_generated")
    )
    n_hits_sum = int(sum(int(r.get("n_hits") or 0) for r in all_bodies))
    n_shield_sum = int(sum(int(r.get("n_shield_pops") or 0) for r in all_bodies))
    n_poison_sum = int(sum(int(r.get("n_hits_poison") or 0) for r in all_bodies))
    n_cleave_p_sum = int(sum(int(r.get("n_cleave_primary") or 0) for r in all_bodies))
    n_cleave_s_sum = int(sum(int(r.get("n_cleave_secondary") or 0) for r in all_bodies))
    n_soc_sum = int(sum(int(r.get("n_soc_hits") or 0) for r in all_bodies))
    n_ord_atk_sum = int(
        sum(int(r.get("n_ordinary_attack_hits") or 0) for r in all_bodies)
    )
    n_ord_ctr_sum = int(
        sum(int(r.get("n_ordinary_counter_hits") or 0) for r in all_bodies)
    )
    n_poison_lethal = sum(1 for r in all_bodies if r.get("poison_lethal"))
    n_cleave_lethal = sum(1 for r in all_bodies if r.get("cleave_lethal"))
    n_soc_lethal = sum(1 for r in all_bodies if r.get("soc_lethal"))
    n_ordinary_lethal = sum(1 for r in all_bodies if r.get("ordinary_lethal"))
    n_death_causes = sum(1 for r in all_bodies if r.get("death_cause"))
    n_hits_events = int(ctx.get("n_damage") or 0)
    n_shield_events = int(ctx.get("n_shield_pops") or 0)
    n_poison_events = int(ctx.get("n_poison_hits") or 0)
    n_cleave_p_events = int(ctx.get("n_cleave_primary") or 0)
    n_cleave_s_events = int(ctx.get("n_cleave_secondary") or 0)
    n_soc_events = int(ctx.get("n_soc_hits") or 0)
    n_ord_atk_events = int(ctx.get("n_ordinary_attack") or 0)
    n_ord_ctr_events = int(ctx.get("n_ordinary_counter") or 0)
    event_counts = {
        "n_attacks_events": n_attacks_events,
        "n_attacks_sum": n_attacks_sum,
        "n_targets_events": n_targets_events,
        "n_targets_sum": n_targets_sum,
        "n_targets_forced_events": n_forced_events,
        "n_targets_forced_sum": n_forced_sum,
        "n_targets_open_events": n_open_events,
        "n_targets_open_sum": n_open_sum,
        "n_deaths_events": n_death_events,
        "n_deaths_expected": n_deaths_expected,
        "n_created_events": n_created_events,
        "n_created": n_created,
        "n_created_represented": n_created_represented,
        "n_cursor_advance": n_cursor_advance,
        "n_cursor_wrap": n_cursor_wrap,
        "n_cursor_reset": n_cursor_wrap,
        "n_damage": n_hits_events,
        "n_hits_events": n_hits_events,
        "n_hits_sum": n_hits_sum,
        "n_shield_pops_events": n_shield_events,
        "n_shield_pops_sum": n_shield_sum,
        "n_poison_hits_events": n_poison_events,
        "n_poison_hits_sum": n_poison_sum,
        "n_cleave_primary_events": n_cleave_p_events,
        "n_cleave_primary_sum": n_cleave_p_sum,
        "n_cleave_secondary_events": n_cleave_s_events,
        "n_cleave_secondary_sum": n_cleave_s_sum,
        "n_soc_hits_events": n_soc_events,
        "n_soc_hits_sum": n_soc_sum,
        "n_ordinary_attack_events": n_ord_atk_events,
        "n_ordinary_attack_sum": n_ord_atk_sum,
        "n_ordinary_counter_events": n_ord_ctr_events,
        "n_ordinary_counter_sum": n_ord_ctr_sum,
        "n_poison_lethal": n_poison_lethal,
        "n_cleave_lethal": n_cleave_lethal,
        "n_soc_lethal": n_soc_lethal,
        "n_ordinary_lethal": n_ordinary_lethal,
        "n_death_causes": n_death_causes,
        "n_immediate_attacks": int(ctx.get("n_immediate_attacks") or 0),
        "n_unsupported_placeholders": n_placeholder,
        "n_living_end": n_living_end,
        "n_start": n_start,
        "attacks_reconcile": n_attacks_events == n_attacks_sum,
        "targets_reconcile": n_targets_events == n_targets_sum,
        "forced_open_reconcile": (n_forced_events + n_open_events) == n_targets_events,
        "created_reconcile": n_created_events == n_created,
        "deaths_reconcile": n_death_events == n_deaths_expected,
        "hits_reconcile": n_hits_events == n_hits_sum,
        "shield_pops_reconcile": n_shield_events == n_shield_sum,
        "poison_hits_reconcile": n_poison_events == n_poison_sum,
        "cleave_primary_reconcile": n_cleave_p_events == n_cleave_p_sum,
        "cleave_secondary_reconcile": n_cleave_s_events == n_cleave_s_sum,
        "soc_hits_reconcile": n_soc_events == n_soc_sum,
        "ordinary_attack_reconcile": n_ord_atk_events == n_ord_atk_sum,
        "ordinary_counter_reconcile": n_ord_ctr_events == n_ord_ctr_sum,
        "death_causes_reconcile": n_death_causes == n_death_events,
    }
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
        "side_first": ctx.get("first_side_label"),
        "event_counts": event_counts,
        "n_board_generated_represented": n_created_represented,
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
        _TRACE_CTX = {
            "a": a, "b": b, "created": [], "n": 0,
            "attacks": {}, "first_attack_index": {},
            "targeted": {}, "targeted_forced": {}, "targeted_open": {},
            "n_swings": 0, "n_targets": 0, "n_targets_forced": 0,
            "n_targets_open": 0, "n_damage": 0, "n_deaths": 0,
            "n_created": 0, "n_cursor_advance": 0, "n_cursor_wrap": 0,
            "n_immediate_attacks": 0,
            "n_shield_pops": 0, "n_poison_hits": 0,
            "n_cleave_primary": 0, "n_cleave_secondary": 0,
            "n_soc_hits": 0, "n_ordinary_attack": 0, "n_ordinary_counter": 0,
            "death_cause": {}, "killed_by": {}, "end_health": {},
            "spawned_represented": {}, "wrap_before_first": {},
            "last_attacker": {},
        }
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
        if trace is not None:
            _TRACE_CTX["first_side"] = int(turn)
            _TRACE_CTX["first_side_label"] = "a" if turn == 0 else "b"
        for _ in range(max_steps):
            if not _living(a) or not _living(b):
                break
            atk_board, def_board = boards[turn], boards[1 - turn]
            living = _living(atk_board)
            n_liv = len(living)
            pos_before = pos[turn]
            wrapped = pos_before >= n_liv
            idx = pos_before % n_liv
            attacker = living[idx]
            pos[turn] = idx + 1
            if trace is not None:
                _TRACE_CTX["n_cursor_advance"] = (
                    int(_TRACE_CTX.get("n_cursor_advance") or 0) + 1
                )
                if wrapped:
                    _TRACE_CTX["n_cursor_wrap"] = (
                        int(_TRACE_CTX.get("n_cursor_wrap") or 0) + 1
                    )
                    bid = str(getattr(attacker, "body_id", "") or "")
                    first = _TRACE_CTX.setdefault("first_attack_index", {})
                    if bid and bid not in first:
                        _TRACE_CTX.setdefault("wrap_before_first", {})[bid] = True
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
