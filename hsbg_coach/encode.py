"""The ONE state/option encoder for the behaviour-cloned NEXT policy.

Shared by training (ml/train_bc_policy.py) and the live overlay
(hsbg_coach/live.py behind HSBG_NEXT_POLICY=1). There is no second copy: if a
feature changes, bump ENCODER_VERSION. Checkpoints record it, and a mismatched
checkpoint refuses to load.

Input is a bg.Snapshot or its dict form (Snapshot.to_dict(), dataclasses.asdict
or a JSON row); every form encodes to the identical vector. Optional State
Builder fields (hero_armor, shop_frozen) are read defensively and default to
absent. anomaly and hand_spells are ignored on purpose: State Builder rows have
anomaly=None and keep spells in `hand`.

Canonical option dict (the Labeler emits exactly this shape)::

    {"type": "buy|sell|reroll|freeze|level|hero_power|play|discover|end_turn|reposition",
     "card_id": str | None,
     "source": {"zone": "board|shop|hand|none", "slot": int | None},
     "target": {"zone": "board|shop|hand|none", "slot": int | None},
     "position": int | None,      # board index for play / reposition
     "choice_kind": str | None}

plus optional debug fields src_entity / target_entity, which are ignored here
(the live overlay has no entity ids for its options, so encoding them would
skew train vs live; the trainer merges options whose vectors coincide, e.g. a
spell aimed at our hero vs at Bob). Slots are 0-based indexes into the snapshot
lists (board / shop / hand), not ZONE_POSITION. Shop spells take the shop slots
after the minions: len(shop) + index into shop_spells. A `play` from zone board
is an Activate; a `play` with source none is the Dark Gift button; a `play`
whose card_id differs from the hand card at its source slot is a Choose One
variant (e.g. BG31_880 -> BG31_880t / BG31_880t2).

Training and held-out scoring use each label row's own `options` list (the
server's legal options); legal_options() below is only the live overlay's
candidate generator. scripts/legal_options_parity.py measures how far apart the
two are.

bc-enc-v2 (from v1): option vector gains [choose-one variant flag, variant
index] so Choose One variants no longer collapse onto one vector; option_cost
knows Activate / Dark Gift costs.

Needs numpy (the ml extra). The stdlib-only core never imports this module
unless the policy flag is on.
"""

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from .actions import BUY_COST, MAX_BOARD, MAX_TIER, ROLL_COST, SELL_VALUE, UPGRADE_COST

ENCODER_VERSION = "bc-enc-v2"

MAX_HAND = 10
EMB_DIM = 48                            # card2vec width (data/cards/card2vec.json)
OPTION_TYPES = ("buy", "sell", "reroll", "freeze", "level", "hero_power",
                "play", "discover", "end_turn", "reposition")
ZONES = ("board", "shop", "hand", "none")
OPTION_KEYS = ("type", "card_id", "source", "target", "position", "choice_kind")

_N_TRIBES = 11                          # len(ml.board_features.TRIBES)
_ZONE_DIM = EMB_DIM + _N_TRIBES + 11    # board_vector: card2vec | tribes | 11 scalars
# Fixed scale for board_vector's scalars (size, sum atk/hp, mean atk/hp,
# max/mean tier, golden, divine, reborn, taunt) so the MLP sees ~unit inputs.
_ZONE_SCALE = np.array([7, 50, 50, 10, 10, 6, 6, 7, 7, 7, 7], dtype=np.float64)
STATE_SCALARS = ("turn", "tier", "gold", "health", "armor", "n_board", "n_shop",
                 "n_hand", "level_cost", "can_level", "hp_usable", "hp_cost",
                 "n_shop_spells", "shop_frozen", "n_hand_spells")
STATE_DIM = 3 * _ZONE_DIM + len(STATE_SCALARS)       # board | shop | hand | scalars
_CARD_DIM = EMB_DIM + 7 + _N_TRIBES + 1              # card2vec | stats | tribes | spell
OPTION_DIM = len(OPTION_TYPES) + _CARD_DIM + 2 * (len(ZONES) + 1) + 2 + 2 + 1 + 2

_SPELL_TYPES = ("BATTLEGROUND_SPELL", "SPELL")
_NOT_A_CARD = ("ENCHANTMENT", "HERO", "HERO_POWER", "GAME_MODE_BUTTON")


# --- small readers (Snapshot object or dict) --------------------------------
def _get(obj, key, default=None):
    v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if v is None else v


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_dict(m) -> Dict:
    if isinstance(m, dict):
        return m
    return {"entity_id": getattr(m, "entity_id", None),
            "card_id": getattr(m, "card_id", None), "name": getattr(m, "name", None),
            "attack": getattr(m, "attack", None), "health": getattr(m, "health", None),
            "position": getattr(m, "position", None),
            "tags": getattr(m, "tags", None) or {}}


def _tags(card) -> Dict:
    return (card.get("tags") or {}) if isinstance(card, dict) else {}


def _zone(snapshot, key) -> List[Dict]:
    return [_as_dict(m) for m in (_get(snapshot, key) or [])]


@lru_cache(maxsize=1)
def _resources():
    """(card2vec by name, CardKnowledge by name, CardKnowledge by card id)."""
    from hsbg_coach.cards import by_name, load_kb
    from hsbg_coach.synergy import load_embeddings
    emb = {k: v for k, v in load_embeddings().items() if len(v) == EMB_DIM}
    kb = load_kb()
    return emb, by_name(kb), kb


def is_spell(card) -> bool:
    return _tags(card).get("CARDTYPE") in _SPELL_TYPES


def needs_board_slot(card) -> bool:
    """Minions (CARDTYPE MINION or untagged) need a free board slot to play."""
    return _tags(card).get("CARDTYPE") in (None, "", "MINION")


def playable_hand(snapshot) -> List[tuple]:
    """(hand index, card) for real cards in hand; skips DragBuy/placeholder chrome."""
    out = []
    for i, m in enumerate(_zone(snapshot, "hand")):
        cid = m.get("card_id") or ""
        if not cid or "DragBuy" in cid or "UNKNOWN ENTITY" in (m.get("name") or ""):
            continue
        if _tags(m).get("CARDTYPE") in _NOT_A_CARD:
            continue
        out.append((i, m))
    return out


def level_cost(snapshot) -> Optional[int]:
    """Gold to tier up now: the live discounted cost, else the base cost."""
    tier = _int(_get(snapshot, "tavern_tier"), 1) or 1
    if tier >= MAX_TIER:
        return None
    cost = _get(snapshot, "level_cost")
    return _int(cost) if cost is not None else UPGRADE_COST.get(tier)


def _spell_cost(spell) -> int:
    c = _get(spell, "cost")
    return BUY_COST if c is None else _int(c, BUY_COST)


def _hero_power(snapshot) -> Optional[Dict]:
    hp = _get(snapshot, "hero_power")
    return hp if isinstance(hp, dict) else None


def _dark_gift(snapshot) -> Optional[Dict]:
    dg = _get(snapshot, "dark_gift")
    return dg if isinstance(dg, dict) and dg.get("card_id") else None


def is_unplayable(card) -> bool:
    """The game flags the card as unplayable right now (LITERALLY_UNPLAYABLE=1,
    e.g. a Lockbox or a spell whose condition isn't met)."""
    return str(_tags(card).get("LITERALLY_UNPLAYABLE", "")) == "1"


def activation(snapshot, slot: int, minion: Dict) -> Optional[Dict]:
    """{cost, usable} if the board minion at `slot` has an Activate ability.

    The raw tag INTERACTABLE_OBJECT, when present, is the game's own signal:
    1 = clickable now; legal if gold >= INTERACTABLE_OBJECT_COST (absent = 0).
    Without that tag, the minion must be listed in snapshot.activatable (by
    entity id, else card id; the tracker / State Builder fill it from the
    Activate allowlist in hsbg_coach/activate_cards.py) and its entry's `usable`
    flag decides. Pilot check (5 games, targets ignored): 499 server-listed
    Activates, 2 missed, 63 extra (scripts/legal_options_parity.py)."""
    tags = _tags(minion)
    entries = [a for a in (_get(snapshot, "activatable") or []) if isinstance(a, dict)]
    eid, cid = minion.get("entity_id"), minion.get("card_id")
    entry = next((a for a in entries if eid is not None and a.get("entity_id") == eid), None)
    if entry is None:
        entry = next((a for a in entries if cid and a.get("card_id") == cid
                      and a.get("entity_id") is None), None)
    io = tags.get("INTERACTABLE_OBJECT")
    if io is not None:
        if str(io) != "1" and entry is None:
            return None
        raw = tags.get("INTERACTABLE_OBJECT_COST")
        need = _int(raw, 0) if raw is not None else 0
        cost = need if raw is not None or entry is None else _int(entry.get("cost"), 0)
        gold = _int(_get(snapshot, "gold"), 0)
        return {"cost": cost, "usable": str(io) == "1" and gold >= need}
    if entry is None:
        return None
    return {"cost": _int(entry.get("cost"), 0), "usable": bool(entry.get("usable"))}


# --- state ------------------------------------------------------------------
def _zone_block(cards: List[Dict]) -> np.ndarray:
    from ml.board_features import board_vector, minion_from_snapshot
    emb, byname, _ = _resources()
    minions = [x for x in (minion_from_snapshot(c, byname) for c in cards) if x]
    v = board_vector(minions, emb or {"": [0.0] * EMB_DIM})
    v[-11:] = v[-11:] / _ZONE_SCALE
    return v


def encode_state(snapshot) -> np.ndarray:
    """Fixed STATE_DIM float32 vector: board | shop | hand blocks + scalars."""
    board, shop = _zone(snapshot, "board"), _zone(snapshot, "shop")
    hand = [m for _, m in playable_hand(snapshot)]
    gold = _int(_get(snapshot, "gold"), 0)
    lc = level_cost(snapshot)
    hp = _hero_power(snapshot) or {}
    scalars = np.array([
        _int(_get(snapshot, "turn"), 0) / 20.0,
        _int(_get(snapshot, "tavern_tier"), 1) / 6.0,
        gold / 10.0,
        _int(_get(snapshot, "hero_health"), 0) / 40.0,
        _int(_get(snapshot, "hero_armor"), 0) / 20.0,
        len(board) / MAX_BOARD,
        len(shop) / MAX_BOARD,
        len(hand) / MAX_HAND,
        (lc or 0) / 10.0,
        1.0 if lc is not None and gold >= lc else 0.0,
        1.0 if hp.get("usable") else 0.0,
        _int(hp.get("cost"), 0) / 10.0,
        len(_get(snapshot, "shop_spells") or []) / 3.0,
        1.0 if _get(snapshot, "shop_frozen") else 0.0,
        sum(1 for m in hand if is_spell(m)) / MAX_HAND,
    ], dtype=np.float64)
    return np.concatenate([_zone_block(board), _zone_block(shop), _zone_block(hand),
                           scalars]).astype(np.float32)


# --- options ----------------------------------------------------------------
def make_option(type_: str, card_id: Optional[str] = None, source=("none", None),
                target=("none", None), position: Optional[int] = None,
                choice_kind: Optional[str] = None) -> Dict:
    return {"type": type_, "card_id": card_id,
            "source": {"zone": source[0], "slot": source[1]},
            "target": {"zone": target[0], "slot": target[1]},
            "position": position, "choice_kind": choice_kind}


def option_key(option: Dict) -> tuple:
    """Hashable identity of an option (debug fields ignored), for matching."""
    src, tgt = option.get("source") or {}, option.get("target") or {}
    return (option.get("type"), option.get("card_id"),
            src.get("zone") or "none", src.get("slot"),
            tgt.get("zone") or "none", tgt.get("slot"),
            option.get("position"), option.get("choice_kind"))


def legal_options(snapshot) -> List[Dict]:
    """Legal actions at a recruit-phase decision point, in a stable order.

    Live-overlay candidate generator only: training and scoring use each label
    row's own server option list. Gold / space rules plus what the Snapshot
    carries (card-specific exceptions are not modelled):
      buy        3g per shop minion (shop spell: its own cost, default 3); needs
                 a free hand slot (< 10 cards). A full board does NOT block buys.
                 (The server's per-slot price, 3/2/1/0, is not in the Snapshot.)
      sell       every board minion, always
      play       hand card not flagged LITERALLY_UNPLAYABLE; a spell with a
                 COST tag needs gold >= COST. Minion: board < 7, one option per
                 insertion index 0..len(board); spell / non-minion card: one
                 option, position None. Always target none (targets need card
                 data the repo does not have); a Choose One card is emitted as
                 its parent (variant ids are not in the Snapshot).
      activate   `play` from zone board for each board minion whose activation()
                 is usable; target none
      dark gift  `play` of snapshot.dark_gift's card_id with source none, when
                 it is usable
      reposition board >= 2: every (slot, new index) pair with index != slot
      reroll     gold >= 1 (free rerolls are not in the Snapshot)
      freeze     always (it toggles)
      level      tier < 6 and gold >= level_cost (live discounted, else base)
      hero_power hero_power present, usable, and gold >= its cost; no target
      end_turn   always
      discover   never enumerated: a Snapshot carries no choice offer
    """
    if _get(snapshot, "phase") not in (None, "recruit", "unknown"):
        return []
    gold = _int(_get(snapshot, "gold"), 0)
    board, shop = _zone(snapshot, "board"), _zone(snapshot, "shop")
    hand = playable_hand(snapshot)
    opts: List[Dict] = []
    if len(hand) < MAX_HAND:
        if gold >= BUY_COST:
            for i, m in enumerate(shop):
                opts.append(make_option("buy", m.get("card_id"), ("shop", i)))
        for j, sp in enumerate(_get(snapshot, "shop_spells") or []):
            if gold >= _spell_cost(sp):
                opts.append(make_option("buy", _get(sp, "card_id"), ("shop", len(shop) + j)))
    for i, m in enumerate(board):
        opts.append(make_option("sell", m.get("card_id"), ("board", i)))
    for i, m in hand:
        if is_unplayable(m):
            continue
        if needs_board_slot(m):
            if len(board) < MAX_BOARD:
                for p in range(len(board) + 1):
                    opts.append(make_option("play", m.get("card_id"), ("hand", i), position=p))
        else:
            cost = _tags(m).get("COST")
            if is_spell(m) and cost is not None and gold < _int(cost, 0):
                continue
            opts.append(make_option("play", m.get("card_id"), ("hand", i)))
    for i, m in enumerate(board):
        act = activation(snapshot, i, m)
        if act and act["usable"]:
            opts.append(make_option("play", m.get("card_id"), ("board", i)))
    dg = _dark_gift(snapshot)
    if dg and dg.get("usable"):
        opts.append(make_option("play", dg.get("card_id")))
    if len(board) >= 2:
        for i, m in enumerate(board):
            for p in range(len(board)):
                if p != i:
                    opts.append(make_option("reposition", m.get("card_id"), ("board", i),
                                            position=p))
    if gold >= ROLL_COST:
        opts.append(make_option("reroll"))
    opts.append(make_option("freeze"))
    lc = level_cost(snapshot)
    if lc is not None and gold >= lc:
        opts.append(make_option("level"))
    hp = _hero_power(snapshot)
    if hp and hp.get("usable") and gold >= _int(hp.get("cost"), 0):
        opts.append(make_option("hero_power", hp.get("card_id")))
    opts.append(make_option("end_turn"))
    return opts


def option_card(snapshot, option: Dict) -> Optional[Dict]:
    """The card an option acts on: the snapshot entry at source zone/slot, else
    the hero power, else a KB lookup of card_id (e.g. discover picks)."""
    src = option.get("source") or {}
    zone, slot = src.get("zone"), src.get("slot")
    card = None
    if isinstance(slot, int) and slot >= 0 and zone in ("board", "shop", "hand"):
        cards = _zone(snapshot, zone)
        if slot < len(cards):
            card = cards[slot]
        elif zone == "shop":
            spells = _get(snapshot, "shop_spells") or []
            k = slot - len(cards)
            if k < len(spells):
                sp = dict(_as_dict(spells[k]))
                sp["tags"] = dict(sp.get("tags") or {}, CARDTYPE="BATTLEGROUND_SPELL")
                card = sp
    if card is None and option.get("type") == "hero_power":
        hp = _hero_power(snapshot)
        if hp:
            card = {"name": hp.get("name"), "card_id": hp.get("card_id"), "tags": {}}
    if card is None and is_dark_gift(option):
        dg = _dark_gift(snapshot) or {}
        card = {"name": dg.get("name") or "Dark Gift",
                "card_id": option.get("card_id") or dg.get("card_id"), "tags": {}}
    cid = option.get("card_id")
    if card is None and cid:
        ck = _resources()[2].get(cid)
        card = {"card_id": cid, "name": ck.name if ck else cid,
                "attack": ck.attack if ck else None,
                "health": ck.health if ck else None, "tags": {}}
    return card


def is_activate(option: Dict) -> bool:
    return option.get("type") == "play" and (option.get("source") or {}).get("zone") == "board"


def is_dark_gift(option: Dict) -> bool:
    return (option.get("type") == "play"
            and ((option.get("source") or {}).get("zone") or "none") == "none")


def variant_index(snapshot, option: Dict) -> int:
    """0 unless this is a Choose One variant play (card_id differs from the hand
    card at its source slot); then the variant's ordinal from its card-id
    suffix: <parent>t -> 1, <parent>t2 -> 2, <parent>a / b -> 1 / 2, else 1."""
    src = option.get("source") or {}
    cid = option.get("card_id")
    if option.get("type") != "play" or src.get("zone") != "hand" or not cid:
        return 0
    card = option_card(snapshot, option) or {}
    parent = card.get("card_id")
    if not parent or parent == cid:
        return 0
    suffix = cid[len(parent):] if cid.startswith(parent) else cid
    digits = ""
    while suffix and suffix[-1].isdigit():
        digits = suffix[-1] + digits
        suffix = suffix[:-1]
    if digits:
        return max(1, _int(digits, 1))
    if len(suffix) == 1 and suffix.isalpha() and suffix.lower() != "t":
        return max(1, ord(suffix.lower()) - ord("a") + 1)
    return 1


def option_cost(snapshot, option: Dict) -> int:
    """Gold the option spends (negative = gained)."""
    t = option.get("type")
    if is_activate(option):
        slot = (option.get("source") or {}).get("slot")
        board = _zone(snapshot, "board")
        if isinstance(slot, int) and 0 <= slot < len(board):
            act = activation(snapshot, slot, board[slot])
            return act["cost"] if act else 0
        return 0
    if is_dark_gift(option):
        return _int((_dark_gift(snapshot) or {}).get("cost"), 0)
    if t == "buy":
        card = option_card(snapshot, option)
        return _spell_cost(card) if card is not None and is_spell(card) else BUY_COST
    if t == "sell":
        return -SELL_VALUE
    if t == "reroll":
        return ROLL_COST
    if t == "level":
        return level_cost(snapshot) or 0
    if t == "hero_power":
        return _int((_hero_power(snapshot) or {}).get("cost"), 0)
    return 0


def _onehot(value, choices) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def _slot(v) -> float:
    return (v + 1) / 10.0 if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0.0


def _card_block(card: Optional[Dict]) -> np.ndarray:
    from ml.board_features import TRIBES, minion_from_snapshot
    out = np.zeros(_CARD_DIM, dtype=np.float64)
    if not card:
        return out
    emb, byname, _ = _resources()
    m = minion_from_snapshot(card, byname)
    if m is None:
        return out
    if m["name"] in emb:
        out[:EMB_DIM] = emb[m["name"]]
    out[EMB_DIM:EMB_DIM + 7] = [m["atk"] / 10.0, m["health"] / 10.0, m["tier"] / 6.0,
                                float(m["golden"]), float(m["divine"]),
                                float(m["reborn"]), float(m["taunt"])]
    for tr in m["tribes"]:
        if tr in TRIBES:
            out[EMB_DIM + 7 + TRIBES.index(tr)] = 1.0
    out[-1] = 1.0 if is_spell(card) else 0.0
    return out


def encode_option(snapshot, option: Dict) -> np.ndarray:
    """Fixed OPTION_DIM float32 vector for one option dict."""
    src, tgt = option.get("source") or {}, option.get("target") or {}
    pos = option.get("position")
    gold = _int(_get(snapshot, "gold"), 0)
    cost = option_cost(snapshot, option)
    vidx = min(variant_index(snapshot, option), 4)
    parts = [
        _onehot(option.get("type"), OPTION_TYPES),
        _card_block(option_card(snapshot, option)),
        _onehot(src.get("zone") or "none", ZONES), [_slot(src.get("slot"))],
        _onehot(tgt.get("zone") or "none", ZONES), [_slot(tgt.get("slot"))],
        [1.0 if _slot(pos) else 0.0, _slot(pos)],
        [cost / 10.0, (gold - cost) / 10.0],
        [1.0 if option.get("choice_kind") else 0.0],
        [1.0 if vidx else 0.0, vidx / 4.0],
    ]
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts]).astype(np.float32)


def describe_option(snapshot, option: Dict) -> str:
    """One NEXT line in the advisor's wording (actions.Action.describe style)."""
    t = option.get("type")
    card = option_card(snapshot, option)
    name = (card or {}).get("name") or option.get("card_id") or "?"
    pos = option.get("position")
    if t == "buy":
        if card is not None and is_spell(card):
            return f"Buy spell: {name} ({option_cost(snapshot, option)}g)"
        return f"Buy {name}"
    if t == "sell":
        return f"Sell {name}"
    if t == "reroll":
        return "Roll the shop"
    if t == "freeze":
        return "Freeze the shop"
    if t == "level":
        tier = _int(_get(snapshot, "tavern_tier"), 1) or 1
        return f"Tier up to {tier + 1} ({option_cost(snapshot, option)}g)"
    if t == "hero_power":
        cost = option_cost(snapshot, option)
        return f"Use hero power: {name}" + (f" ({cost}g)" if cost else "")
    if t == "play" and is_dark_gift(option):
        cost = option_cost(snapshot, option)
        return "Dark Gift" + (f" ({cost}g)" if cost else "")
    if t == "play" and is_activate(option):
        cost = option_cost(snapshot, option)
        return f"Activate {name}" + (f" ({cost}g)" if cost else "")
    if t == "play":
        where = (f" at slot {pos + 1}"
                 if isinstance(pos, int) and _zone(snapshot, "board") else "")
        variant = option.get("card_id")
        choice = (f" (Choose One: {variant})"
                  if variant_index(snapshot, option) and variant else "")
        return f"Play {name} from hand{where}{choice}"
    if t == "reposition":
        return f"Move {name} to slot {pos + 1}" if isinstance(pos, int) else f"Move {name}"
    if t == "discover":
        return f"PICK {name}"
    return "End turn"