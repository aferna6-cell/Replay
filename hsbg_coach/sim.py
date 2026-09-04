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
from typing import Dict, List, Optional, Sequence, Tuple

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
    # Observational combat-start attack. Combat RNG/outcomes never read this.
    start_attack: Optional[int] = None

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


def _hit_overkill(incoming: int, hp_before: int, *, poison: bool, lethal: bool) -> int:
    """Damage beyond the death threshold. Poison leftover HP counts as overkill."""
    if not lethal:
        return max(0, int(incoming) - int(hp_before))
    needed = max(0, int(hp_before))
    extra = max(0, int(incoming) - needed)
    if poison:
        extra += max(0, needed - int(incoming))
    return int(extra)


def _opt_stat(raw, fallback: int) -> int:
    try:
        if raw in (None, ""):
            return int(fallback)
        return int(raw)
    except (TypeError, ValueError):
        return int(fallback)


def _combat_start_attack(m: Combatant) -> int:
    """Observational combat-start (or spawn-time) attack. Combat never reads this."""
    stamped = getattr(m, "start_attack", None)
    atk = int(getattr(m, "attack", 0) or 0)
    return _opt_stat(stamped, atk)


def _side_board_attack_stats(board: Sequence[Combatant]) -> Dict:
    """Opposing-board recruit/pool attack totals at combat start. No RNG."""
    rec = 0
    pool = 0
    size = 0
    tiers: List[int] = []
    shares: List[Tuple[int, str, int]] = []
    for i, m in enumerate(board):
        size += 1
        atk0 = _combat_start_attack(m)
        recruit = _opt_stat(getattr(m, "recruit_attack", None), atk0)
        rec += recruit
        pool += atk0 - recruit
        try:
            tier = int(getattr(m, "tier", 1) or 1)
        except (TypeError, ValueError):
            tier = 1
        tiers.append(tier)
        bid = str(getattr(m, "body_id", "") or "")
        shares.append((atk0 - recruit, bid, i))
    order = sorted(range(size), key=lambda j: (-shares[j][0], shares[j][2]))
    rank = {shares[idx][1]: r for r, idx in enumerate(order, start=1) if shares[idx][1]}
    hist = {str(t): 0 for t in range(1, 7)}
    for t in tiers:
        tt = min(6, max(1, t))
        hist[str(tt)] += 1
    return {
        "recruit_attack": rec,
        "pool_attack": pool,
        "size": size,
        "mean_tier": (float(sum(tiers)) / float(size)) if size else 0.0,
        "tier_hist": hist,
        "rank": rank,
    }


def _finalize_attacking_pool(rows: Sequence[dict], stats: Dict) -> Dict:
    """After n_attacks is known: pool sitting on bodies that actually swung."""
    attacking_pool = 0
    n_attacked = 0
    for row in rows:
        if int(row.get("n_attacks") or 0) <= 0:
            continue
        n_attacked += 1
        atk0 = int(row.get("start_attack") if row.get("start_attack") not in (None, "") else row.get("attack") or 0)
        rec = int(row.get("recruit_attack") or 0)
        attacking_pool += atk0 - rec
    board_pool = int(stats.get("pool_attack") or 0)
    out = dict(stats)
    out["attacking_pool_attack"] = int(attacking_pool)
    out["n_attacked"] = int(n_attacked)
    out["pool_on_attackers_share"] = (
        (float(attacking_pool) / float(board_pool)) if board_pool else 0.0
    )
    return out


def _dealer_snapshot(dealer: Optional[Combatant], ctx: Optional[Dict] = None) -> Dict:
    """Observational dealer identity/stats at impact. Combat never reads this."""
    if dealer is None:
        return {}
    atk = int(getattr(dealer, "attack", 0) or 0)
    ra = getattr(dealer, "recruit_attack", None)
    recruit_attack = _opt_stat(ra, atk)
    synth = max(0, atk - recruit_attack)
    bid = str(getattr(dealer, "body_id", "") or "")
    idx = None
    if ctx is not None and bid:
        raw = (ctx.get("first_attack_index") or {}).get(bid)
        if raw is not None:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                idx = None
    slot = getattr(dealer, "board_slot", None)
    try:
        slot_i = None if slot is None else int(slot)
    except (TypeError, ValueError):
        slot_i = None
    origin = str(getattr(dealer, "origin", "") or "")
    generated = bool(
        getattr(dealer, "generated", False) or origin in ("token", "reborn")
    )
    stamped = getattr(dealer, "start_attack", None)
    start_atk = _combat_start_attack(dealer)
    start_recruit = recruit_attack
    start_pool = int(start_atk) - int(start_recruit)
    combat_delta = int(atk) - int(start_atk)
    represented = stamped is not None and origin == "starting" and not generated
    identity_ok = int(atk) == int(start_recruit) + int(start_pool) + int(combat_delta)
    side = bid.split(":")[0] if bid else ""
    side_stats = ((ctx or {}).get("side_board_stats") or {}).get(side) or {}
    board_pool = int(side_stats.get("pool_attack") or 0)
    board_recruit = int(side_stats.get("recruit_attack") or 0)
    board_size = int(side_stats.get("size") or 0)
    share = (float(start_pool) / float(board_pool)) if board_pool else 0.0
    rank_map = side_stats.get("rank") or {}
    rank = rank_map.get(bid)
    try:
        rank_i = None if rank is None else int(rank)
    except (TypeError, ValueError):
        rank_i = None
    return {
        "attacker_id": bid,
        "attacker_name": str(getattr(dealer, "name", "") or ""),
        "attacker_card_id": str(getattr(dealer, "card_id", "") or ""),
        "attacker_tier": int(getattr(dealer, "tier", 1) or 1),
        "attacker_slot": slot_i,
        "attacker_attack": atk,
        "attacker_recruit_attack": recruit_attack,
        "attacker_synth_attack": synth,
        "attacker_synth_share": (float(synth) / float(atk)) if atk else 0.0,
        "attacker_start_attack": int(start_atk),
        "attacker_start_recruit_attack": int(start_recruit),
        "attacker_start_pool_attack": int(start_pool),
        "attacker_combat_delta": int(combat_delta),
        "attacker_attack_identity_ok": bool(identity_ok),
        "attacker_start_represented": bool(represented),
        "attacker_board_pool_attack": board_pool,
        "attacker_board_recruit_attack": board_recruit,
        "attacker_board_size": board_size,
        "attacker_board_mean_tier": float(side_stats.get("mean_tier") or 0.0),
        "attacker_pool_share_of_board": float(share),
        "attacker_pool_rank": rank_i,
        "attacker_golden": bool(getattr(dealer, "golden", False)),
        "attacker_generated": bool(
            getattr(dealer, "generated", False)
            or str(getattr(dealer, "origin", "") or "") in ("token", "reborn")
        ),
        "attacker_origin": str(getattr(dealer, "origin", "") or ""),
        "attacker_poisonous": bool(getattr(dealer, "poisonous", False)),
        "attacker_cleave": bool(getattr(dealer, "cleave", False)),
        "attacker_divine_shield": bool(getattr(dealer, "divine_shield", False)),
        "attacker_windfury": bool(getattr(dealer, "windfury", False)),
        "attacker_taunt": bool(getattr(dealer, "taunt", False)),
        "attacker_health_at_impact": int(getattr(dealer, "health", 0) or 0),
        "attacker_first_attack_index": idx,
        "attacker_has_represented_generated": has_represented_generated_effect(dealer),
        "attacker_survived_swing": int(getattr(dealer, "health", 0) or 0) > 0,
    }


def _classify_hit_kind(
    *,
    cause: str,
    poison: bool,
    cleave_role: Optional[str],
    ds_before: bool,
) -> str:
    """Partition ordinary vs shield / poison / cleave / SOC / other. No combat effect."""
    if ds_before:
        return "shield"
    if poison or cause == "poison":
        return "poison"
    if cleave_role or cause == "cleave":
        return "cleave"
    if cause == "start_of_combat":
        return "start_of_combat"
    if cause == "death_burst":
        return "death_burst"
    if cause in ("attack", "counterattack"):
        return "ordinary"
    return "other"


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
    incoming: int,
    dealer: Optional[Combatant] = None,
    defender_slot: Optional[int] = None,
) -> None:
    """Observational per-swing / per-body lethal-cause and HP-flow tags. No RNG."""
    incoming = int(incoming)
    hp_before = int(hp_before)
    hp_after = int(hp_after)
    applied = max(0, hp_before - max(hp_after, 0))
    damaging = hp_after < hp_before
    hp_delta = hp_before - hp_after
    kind = _classify_hit_kind(
        cause=cause, poison=poison, cleave_role=cleave_role, ds_before=ds_before,
    )
    ordinary_expected = min(hp_before, incoming)
    ordinary_ok = (applied == ordinary_expected) if kind == "ordinary" else None
    if kind == "ordinary":
        ctx["n_ordinary_kind"] = int(ctx.get("n_ordinary_kind") or 0) + 1
        if ordinary_ok:
            ctx["n_ordinary_ok"] = int(ctx.get("n_ordinary_ok") or 0) + 1
    elif kind == "shield":
        ctx["n_shield_kind"] = int(ctx.get("n_shield_kind") or 0) + 1
    elif kind == "poison":
        ctx["n_poison_kind"] = int(ctx.get("n_poison_kind") or 0) + 1
    else:
        ctx["n_non_ordinary_kind"] = int(ctx.get("n_non_ordinary_kind") or 0) + 1
    dealer_snap = _dealer_snapshot(dealer, ctx)
    if damaging or lethal:
        overkill = _hit_overkill(
            incoming, hp_before, poison=bool(poison), lethal=bool(lethal),
        )
    else:
        overkill = 0
    ctx["n_damage"] = int(ctx.get("n_damage") or 0) + 1
    ctx["n_incoming"] = int(ctx.get("n_incoming") or 0) + incoming
    ctx["n_applied"] = int(ctx.get("n_applied") or 0) + applied
    ctx["n_hp_delta"] = int(ctx.get("n_hp_delta") or 0) + hp_delta
    if damaging:
        ctx["n_damaging"] = int(ctx.get("n_damaging") or 0) + 1
    if lethal:
        ctx["n_overkill"] = int(ctx.get("n_overkill") or 0) + overkill
    if not bid:
        return
    ctx.setdefault("end_health", {})[bid] = hp_after
    ctx.setdefault("end_divine_shield", {})[bid] = bool(ds_after)
    ctx.setdefault("ds_before_last_hit", {})[bid] = bool(ds_before)
    ctx.setdefault("ds_after_last_hit", {})[bid] = bool(ds_after)
    ctx.setdefault("last_hp_before", {})[bid] = hp_before
    ctx.setdefault("last_hp_after", {})[bid] = hp_after
    ctx.setdefault("last_incoming", {})[bid] = incoming
    ctx.setdefault("last_cause", {})[bid] = cause
    _bump_map(ctx, "n_hits", bid)
    _bump_map(ctx, "cumulative_incoming", bid, incoming)
    _bump_map(ctx, "cumulative_applied", bid, applied)
    _bump_map(ctx, "hp_delta_sum", bid, hp_delta)
    if damaging:
        _bump_map(ctx, "n_damaging_hits", bid)
    if poison:
        _bump_map(ctx, "incoming_poison", bid, incoming)
    if cleave_role == "primary" or cleave_role == "secondary" or cause == "cleave":
        _bump_map(ctx, "incoming_cleave", bid, incoming)
    if cause == "start_of_combat":
        _bump_map(ctx, "incoming_soc", bid, incoming)
    if cause in ("attack", "counterattack"):
        _bump_map(ctx, "incoming_ordinary", bid, incoming)
    if cause == "death_burst":
        _bump_map(ctx, "incoming_other", bid, incoming)
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
        ctx.setdefault("overkill_on_death", {})[bid] = overkill
        ctx["n_deaths"] = int(ctx.get("n_deaths") or 0) + 1
    events = ctx.setdefault("hit_events", {})
    lst = events.setdefault(bid, [])
    event = {
        "cause": cause,
        "incoming": incoming,
        "hp_before": hp_before,
        "hp_after": hp_after,
        "poison": bool(poison),
        "cleave_role": cleave_role,
        "damaging": bool(damaging),
        "applied": applied,
        "overkill": overkill,
        "lethal": bool(lethal),
        "hit_kind": kind,
        "ordinary_expected": ordinary_expected,
        "ordinary_ok": ordinary_ok,
        "defender_id": bid,
        "defender_slot": defender_slot,
        "hit_was_counterattack": cause == "counterattack",
    }
    event.update(dealer_snap)
    lst.append(event)


def _apply_damage(target: Combatant, dmg: int, poison: bool = False,
                  cause: str = "attack", cleave_role: Optional[str] = None,
                  dealer: Optional[Combatant] = None) -> None:
    if dmg <= 0:
        return
    ctx = _TRACE_CTX
    pre = int(target.health)
    bid = str(getattr(target, "body_id", "") or "")
    ds_before = bool(target.divine_shield)
    slot = getattr(target, "board_slot", None)
    try:
        defender_slot = None if slot is None else int(slot)
    except (TypeError, ValueError):
        defender_slot = None
    if target.divine_shield:
        target.divine_shield = False     # shield eats the hit (and any poison)
        if ctx is not None:
            _trace_hit(
                ctx, bid,
                cause=cause, poison=poison, cleave_role=cleave_role,
                ds_before=True, ds_after=False,
                hp_before=pre, hp_after=int(target.health), lethal=False,
                incoming=int(dmg), dealer=dealer, defender_slot=defender_slot,
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
            incoming=int(dmg), dealer=dealer, defender_slot=defender_slot,
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
    if getattr(tok, "start_attack", None) is None:
        tok.start_attack = int(getattr(tok, "attack", 0) or 0)
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
            _apply_damage(
                rng.choice(living), soc.damage, cause="start_of_combat", dealer=m,
            )


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
    ctx = _TRACE_CTX
    before_lens: Dict[str, int] = {}
    if ctx is not None:
        for bid, lst in (ctx.get("hit_events") or {}).items():
            before_lens[bid] = len(lst)
    _apply_damage(
        defender, a_dmg, attacker.poisonous, cause=cause, cleave_role=primary_role,
        dealer=attacker,
    )
    if attacker.cleave and di >= 0:
        for nb in (di - 1, di + 1):
            if 0 <= nb < len(def_board) and def_board[nb] is not defender:
                _apply_damage(
                    def_board[nb], a_dmg, attacker.poisonous, cause="cleave",
                    cleave_role="secondary", dealer=attacker,
                )
    back = "poison" if defender.poisonous else "counterattack"
    _apply_damage(attacker, d_dmg, defender.poisonous, cause=back, dealer=defender)
    if ctx is not None:
        a_alive = int(attacker.health) > 0
        d_alive = int(defender.health) > 0
        a_id = str(getattr(attacker, "body_id", "") or "")
        d_id = str(getattr(defender, "body_id", "") or "")
        for bid, lst in (ctx.get("hit_events") or {}).items():
            start = before_lens.get(bid, 0)
            for ev in lst[start:]:
                dealer_id = str(ev.get("attacker_id") or "")
                if dealer_id == a_id:
                    ev["attacker_survived_swing"] = a_alive
                elif dealer_id == d_id:
                    ev["attacker_survived_swing"] = d_alive
                ev["hit_was_counterattack"] = ev.get("cause") == "counterattack"


def _resolve_deaths(board: List[Combatant], enemy: List[Combatant],
                    rng: random.Random, process_immediates: bool = True) -> None:
    """Rebuild a board: drop dead minions, fire deathrattle summons + reborn.

    Immediate-attack tokens (Scallywag) strike right away. process_immediates is
    set False on the recursive resolve so chains terminate."""
    if all(m.health > 0 for m in board):
        return
    new: List[Combatant] = []
    immediates: List[Combatant] = []
    bursts: List = []                                  # (parent, AOE-damage deathrattle)
    for m in board:
        if m.health > 0:
            new.append(m)
            continue
        if m.death_burst:
            bursts.append((m, m.death_burst))
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
    for parent, burst in bursts:
        targets = _living(enemy)
        if not getattr(burst, "hits_all", True):
            rng.shuffle(targets)
            targets = targets[:max(1, getattr(burst, "targets", 1))]
        for t in targets:
            _apply_damage(t, burst.damage, cause="death_burst", dealer=parent)
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


def _annotate_attacker_punch(row: dict, events: Sequence[dict]) -> None:
    """Roll damaging-hit dealer stats onto a body row. Observational; no RNG."""
    damaging = [e for e in events if e.get("damaging")]
    n_ord = n_ord_ok = n_shield = n_poison = n_cleave = n_soc = n_other = 0
    for e in events:
        kind = str(e.get("hit_kind") or "")
        if kind == "ordinary":
            n_ord += 1
            if e.get("ordinary_ok") is True:
                n_ord_ok += 1
        elif kind == "shield":
            n_shield += 1
        elif kind == "poison":
            n_poison += 1
        elif kind == "cleave":
            n_cleave += 1
        elif kind == "start_of_combat":
            n_soc += 1
        else:
            n_other += 1
    row["n_ordinary_kind"] = n_ord
    row["n_ordinary_ok"] = n_ord_ok
    row["n_shield_kind"] = n_shield
    row["n_poison_kind"] = n_poison
    row["n_cleave_kind"] = n_cleave
    row["n_soc_kind"] = n_soc
    row["n_other_kind"] = n_other
    row["ordinary_hp_loss_ok"] = n_ord == n_ord_ok

    def _mean_key(key: str) -> Optional[float]:
        xs = []
        for e in damaging:
            v = e.get(key)
            if v is None:
                continue
            try:
                xs.append(float(v))
            except (TypeError, ValueError):
                continue
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    defender_slot = row.get("board_slot")
    try:
        d_slot = None if defender_slot is None else int(defender_slot)
    except (TypeError, ValueError):
        d_slot = None
    rels = []
    for e in damaging:
        a_slot = e.get("attacker_slot")
        try:
            a_i = None if a_slot is None else int(a_slot)
        except (TypeError, ValueError):
            a_i = None
        if a_i is not None and d_slot is not None:
            rels.append(float(a_i - d_slot))
    first = damaging[0] if damaging else None
    last = damaging[-1] if damaging else None
    row["n_punch_hits"] = len(damaging)
    row["mean_attacker_attack"] = _mean_key("attacker_attack") or 0.0
    row["mean_attacker_recruit_attack"] = _mean_key("attacker_recruit_attack") or 0.0
    row["mean_attacker_synth_attack"] = _mean_key("attacker_synth_attack") or 0.0
    row["mean_attacker_synth_share"] = _mean_key("attacker_synth_share") or 0.0
    row["mean_attacker_start_attack"] = _mean_key("attacker_start_attack") or 0.0
    row["mean_attacker_start_recruit"] = (
        _mean_key("attacker_start_recruit_attack") or 0.0
    )
    row["mean_attacker_start_pool"] = _mean_key("attacker_start_pool_attack") or 0.0
    row["mean_attacker_combat_delta"] = _mean_key("attacker_combat_delta") or 0.0
    row["mean_attacker_pool_share_of_board"] = (
        _mean_key("attacker_pool_share_of_board") or 0.0
    )
    row["mean_attacker_pool_rank"] = _mean_key("attacker_pool_rank")
    row["mean_attacker_board_pool"] = _mean_key("attacker_board_pool_attack") or 0.0
    row["mean_attacker_board_recruit"] = (
        _mean_key("attacker_board_recruit_attack") or 0.0
    )
    row["mean_attacker_board_size"] = _mean_key("attacker_board_size")
    row["mean_attacker_board_mean_tier"] = _mean_key("attacker_board_mean_tier")
    n_id = n_id_ok = n_repr = 0
    for e in damaging:
        n_id += 1
        if e.get("attacker_attack_identity_ok") is True:
            n_id_ok += 1
        if e.get("attacker_start_represented"):
            n_repr += 1
    row["n_attack_identity"] = n_id
    row["n_attack_identity_ok"] = n_id_ok
    row["n_attacker_start_represented"] = n_repr
    row["attack_identity_ok"] = (n_id == n_id_ok) if n_id else True
    row["mean_attacker_tier"] = _mean_key("attacker_tier")
    row["mean_attacker_slot"] = _mean_key("attacker_slot")
    row["mean_relative_slot"] = (
        float(sum(rels) / len(rels)) if rels else 0.0
    )
    row["mean_attacker_first_attack_index"] = _mean_key("attacker_first_attack_index")
    row["mean_defender_hp_before"] = _mean_key("hp_before")
    row["mean_hp_loss"] = _mean_key("applied") or 0.0
    survived = [
        1.0 if e.get("attacker_survived_swing") else 0.0 for e in damaging
    ]
    counters = [
        1.0 if e.get("hit_was_counterattack") else 0.0 for e in damaging
    ]
    goldens = [
        1.0 if e.get("attacker_golden") else 0.0 for e in damaging
    ]
    generated = [
        1.0 if e.get("attacker_generated") else 0.0 for e in damaging
    ]
    n_d = float(len(damaging)) if damaging else 0.0
    row["p_attacker_survived_swing"] = (
        (sum(survived) / n_d) if n_d else 0.0
    )
    row["p_hit_was_counterattack"] = (sum(counters) / n_d) if n_d else 0.0
    row["p_attacker_golden"] = (sum(goldens) / n_d) if n_d else 0.0
    row["p_attacker_generated"] = (sum(generated) / n_d) if n_d else 0.0
    row["first_attacker_id"] = (first or {}).get("attacker_id")
    row["first_attacker_name"] = (first or {}).get("attacker_name")
    row["last_attacker_id"] = (last or {}).get("attacker_id")
    row["last_attacker_attack"] = (last or {}).get("attacker_attack")
    idx = row.get("mean_attacker_first_attack_index")
    rel = float(row.get("mean_relative_slot") or 0)
    if int(row.get("n_punch_hits") or 0) <= 0:
        row["pairing_order_value"] = -1.0
    else:
        row["pairing_order_value"] = (
            (0.0 if idx is None else float(idx)) + 0.05 * rel
        )


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
        n_hits = int(row.get("n_hits") or 0)
        incoming = int((ctx.get("cumulative_incoming") or {}).get(bid) or 0)
        applied = int((ctx.get("cumulative_applied") or {}).get(bid) or 0)
        hp_delta = int((ctx.get("hp_delta_sum") or {}).get(bid) or 0)
        row["cumulative_incoming"] = incoming
        row["cumulative_applied"] = applied
        row["hp_delta_sum"] = hp_delta
        row["incoming_ordinary"] = int(
            (ctx.get("incoming_ordinary") or {}).get(bid) or 0
        )
        row["incoming_poison"] = int(
            (ctx.get("incoming_poison") or {}).get(bid) or 0
        )
        row["incoming_cleave"] = int(
            (ctx.get("incoming_cleave") or {}).get(bid) or 0
        )
        row["incoming_soc"] = int((ctx.get("incoming_soc") or {}).get(bid) or 0)
        row["incoming_other"] = int(
            (ctx.get("incoming_other") or {}).get(bid) or 0
        )
        row["n_damaging_hits"] = int(
            (ctx.get("n_damaging_hits") or {}).get(bid) or 0
        )
        row["overkill_on_death"] = int(
            (ctx.get("overkill_on_death") or {}).get(bid) or 0
        )
        last_before = (ctx.get("last_hp_before") or {}).get(bid)
        last_after = (ctx.get("last_hp_after") or {}).get(bid)
        row["last_hp_before"] = (
            int(last_before) if last_before is not None else start_hp
        )
        row["last_hp_after"] = (
            int(last_after) if last_after is not None else int(row["end_health"])
        )
        row["last_incoming"] = int((ctx.get("last_incoming") or {}).get(bid) or 0)
        row["last_cause"] = (ctx.get("last_cause") or {}).get(bid)
        row["mean_incoming_dmg"] = (
            float(incoming) / float(n_hits) if n_hits else 0.0
        )
        mean_in = float(row["mean_incoming_dmg"])
        row["hp_depletion_margin"] = float(start_hp) / max(mean_in, 1.0)
        row["hp_flow_ok"] = (start_hp - int(row["end_health"])) == hp_delta
        events = (ctx.get("hit_events") or {}).get(bid) or []
        row["hit_events"] = list(events)
        _annotate_attacker_punch(row, events)


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
        "start_attack": _combat_start_attack(m),
        "start_pool_attack": _combat_start_attack(m) - recruit_attack,
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
        starting_loser = list(trace.get("starting_b") or [])
        created_winner = [c for c in created if c.get("side") == "a"]
    elif winner_side == "b":
        starting_winner = list(trace.get("starting_b") or [])
        starting_loser = list(trace.get("starting_a") or [])
        created_winner = [c for c in created if c.get("side") == "b"]
    else:
        starting_winner = []
        starting_loser = []
        created_winner = []
    stats_a = _finalize_attacking_pool(
        list(trace.get("starting_a") or []),
        dict((ctx.get("side_board_stats") or {}).get("a") or {}),
    )
    stats_b = _finalize_attacking_pool(
        list(trace.get("starting_b") or []),
        dict((ctx.get("side_board_stats") or {}).get("b") or {}),
    )
    if winner_side == "a":
        opp_stats = stats_b
    elif winner_side == "b":
        opp_stats = stats_a
    else:
        opp_stats = {}
    for row in starting_winner:
        row["opp_board_pool_attack"] = int(opp_stats.get("pool_attack") or 0)
        row["opp_board_recruit_attack"] = int(opp_stats.get("recruit_attack") or 0)
        row["opp_board_size"] = int(opp_stats.get("size") or 0)
        row["opp_board_mean_tier"] = float(opp_stats.get("mean_tier") or 0.0)
        row["opp_board_tier_hist"] = dict(opp_stats.get("tier_hist") or {})
        row["opp_attacking_pool_attack"] = int(
            opp_stats.get("attacking_pool_attack") or 0
        )
        row["opp_n_attacked"] = int(opp_stats.get("n_attacked") or 0)
        row["opp_pool_on_attackers_share"] = float(
            opp_stats.get("pool_on_attackers_share") or 0.0
        )
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
    n_incoming_events = int(ctx.get("n_incoming") or 0)
    n_applied_events = int(ctx.get("n_applied") or 0)
    n_hp_delta_events = int(ctx.get("n_hp_delta") or 0)
    n_damaging_events = int(ctx.get("n_damaging") or 0)
    n_overkill_events = int(ctx.get("n_overkill") or 0)
    n_incoming_sum = int(sum(int(r.get("cumulative_incoming") or 0) for r in all_bodies))
    n_applied_sum = int(sum(int(r.get("cumulative_applied") or 0) for r in all_bodies))
    n_hp_delta_sum = int(sum(int(r.get("hp_delta_sum") or 0) for r in all_bodies))
    n_damaging_sum = int(sum(int(r.get("n_damaging_hits") or 0) for r in all_bodies))
    n_overkill_sum = int(sum(int(r.get("overkill_on_death") or 0) for r in all_bodies))
    n_start_minus_end = int(sum(
        int(r.get("start_health") or 0) - int(r.get("end_health") or 0)
        for r in all_bodies
    ))
    n_hp_flow_ok = sum(1 for r in all_bodies if r.get("hp_flow_ok") is False)
    n_shield_events = int(ctx.get("n_shield_pops") or 0)
    n_poison_events = int(ctx.get("n_poison_hits") or 0)
    n_cleave_p_events = int(ctx.get("n_cleave_primary") or 0)
    n_cleave_s_events = int(ctx.get("n_cleave_secondary") or 0)
    n_soc_events = int(ctx.get("n_soc_hits") or 0)
    n_ord_atk_events = int(ctx.get("n_ordinary_attack") or 0)
    n_ord_ctr_events = int(ctx.get("n_ordinary_counter") or 0)
    n_ordinary_kind_events = int(ctx.get("n_ordinary_kind") or 0)
    n_ordinary_ok_events = int(ctx.get("n_ordinary_ok") or 0)
    n_shield_kind_events = int(ctx.get("n_shield_kind") or 0)
    n_poison_kind_events = int(ctx.get("n_poison_kind") or 0)
    n_non_ordinary_kind_events = int(ctx.get("n_non_ordinary_kind") or 0)
    n_ordinary_kind_sum = int(sum(int(r.get("n_ordinary_kind") or 0) for r in all_bodies))
    n_ordinary_ok_sum = int(sum(int(r.get("n_ordinary_ok") or 0) for r in all_bodies))
    n_shield_kind_sum = int(sum(int(r.get("n_shield_kind") or 0) for r in all_bodies))
    n_poison_kind_sum = int(sum(int(r.get("n_poison_kind") or 0) for r in all_bodies))
    n_ordinary_mismatch_bodies = sum(
        1 for r in all_bodies if r.get("ordinary_hp_loss_ok") is False
    )
    n_identity_sum = int(sum(int(r.get("n_attack_identity") or 0) for r in all_bodies))
    n_identity_ok_sum = int(
        sum(int(r.get("n_attack_identity_ok") or 0) for r in all_bodies)
    )
    n_identity_repr_sum = int(
        sum(int(r.get("n_attacker_start_represented") or 0) for r in all_bodies)
    )
    n_identity_mismatch_bodies = sum(
        1 for r in all_bodies if r.get("attack_identity_ok") is False
    )
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
        "n_incoming_events": n_incoming_events,
        "n_incoming_sum": n_incoming_sum,
        "n_applied_events": n_applied_events,
        "n_applied_sum": n_applied_sum,
        "n_hp_delta_events": n_hp_delta_events,
        "n_hp_delta_sum": n_hp_delta_sum,
        "n_start_minus_end": n_start_minus_end,
        "n_damaging_events": n_damaging_events,
        "n_damaging_sum": n_damaging_sum,
        "n_overkill_events": n_overkill_events,
        "n_overkill_sum": n_overkill_sum,
        "n_hp_flow_mismatch_bodies": n_hp_flow_ok,
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
        "incoming_reconcile": n_incoming_events == n_incoming_sum,
        "applied_reconcile": n_applied_events == n_applied_sum,
        "hp_delta_reconcile": n_hp_delta_events == n_hp_delta_sum,
        "hp_flow_reconcile": (
            n_hp_delta_events == n_start_minus_end and n_hp_flow_ok == 0
        ),
        "damaging_hits_reconcile": n_damaging_events == n_damaging_sum,
        "overkill_reconcile": n_overkill_events == n_overkill_sum,
        "n_ordinary_kind_events": n_ordinary_kind_events,
        "n_ordinary_kind_sum": n_ordinary_kind_sum,
        "n_ordinary_ok_events": n_ordinary_ok_events,
        "n_ordinary_ok_sum": n_ordinary_ok_sum,
        "n_shield_kind_events": n_shield_kind_events,
        "n_shield_kind_sum": n_shield_kind_sum,
        "n_poison_kind_events": n_poison_kind_events,
        "n_poison_kind_sum": n_poison_kind_sum,
        "n_non_ordinary_kind_events": n_non_ordinary_kind_events,
        "n_ordinary_mismatch_bodies": n_ordinary_mismatch_bodies,
        "ordinary_kind_reconcile": n_ordinary_kind_events == n_ordinary_kind_sum,
        "ordinary_ok_reconcile": n_ordinary_ok_events == n_ordinary_ok_sum,
        "shield_kind_reconcile": n_shield_kind_events == n_shield_kind_sum,
        "poison_kind_reconcile": n_poison_kind_events == n_poison_kind_sum,
        "ordinary_hp_loss_reconcile": (
            n_ordinary_kind_events == n_ordinary_ok_events
            and n_ordinary_mismatch_bodies == 0
        ),
        "n_attack_identity_sum": n_identity_sum,
        "n_attack_identity_ok_sum": n_identity_ok_sum,
        "n_attacker_start_represented_sum": n_identity_repr_sum,
        "n_attack_identity_mismatch_bodies": n_identity_mismatch_bodies,
        "attack_identity_reconcile": (
            n_identity_sum == n_identity_ok_sum
            and n_identity_mismatch_bodies == 0
        ),
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
        "starting_loser": starting_loser,
        "starting_a": list(trace.get("starting_a") or []),
        "starting_b": list(trace.get("starting_b") or []),
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
        _TRACE_CTX["side_board_stats"] = {
            "a": _side_board_attack_stats(a),
            "b": _side_board_attack_stats(b),
        }
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
        m.start_attack = int(getattr(m, "attack", 0) or 0)
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
