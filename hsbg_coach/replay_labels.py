"""Label ``states.v2`` decision points with the action the player took.

Turns State Builder's per-decision-point states (``replay_states``) into
weighted training rows ``(state, every legal option, chosen option)``. See
``docs/labels_schema.md``.

    python -m hsbg_coach.replay_labels --states data/firestone/states/v2 \\
        --replays data/firestone/raw --manifest data/firestone/raw/manifest.json \\
        --out data/firestone/labels/v2 --workers 6

States carry the legal options but not the choice, so the replay XML is read once
more:

- ``kind="options"`` rows: the player's action between one ``Options`` block and
  the next, i.e. a ``PLAY`` block (buy/sell helpers, reroll/freeze/level buttons,
  hero power, cards) or a ``MOVE_MINION`` block (reposition). No action + the turn
  advanced = end turn.
- ``kind="choice"`` rows (hero pick, discover, Dark Discovery, trinket, quest, ...):
  the ``ChosenEntities`` that answers the matching ``Choices`` block.

Writes ``<out>/<reviewId>.jsonl.gz`` (labeled rows), ``<out>/quarantine/<reviewId>.jsonl``
(decision points that could not be labeled, with a reason) and ``<out>/summary.json``.
Everything under ``data/firestone/`` is gitignored.
"""

import argparse
import bisect
import gzip
import json
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .hs_enums import ENUM_VALUES
from .hsreplay_xml import _int, _open

SCHEMA_VERSION = "labels.v2"
CURRENT_BUILD = 253216
WEIGHT_FLOOR = 0.2
MAX_BOARD = 7
MAX_EXAMPLES = 5

REROLL = "TB_BaconShop_8p_Reroll_Button"
FREEZE = "TB_BaconShopLockAll_Button"
LEVEL_PREFIX = "TB_BaconShopTechUp"
DRAG_BUY = ("TB_BaconShop_DragBuy", "TB_BaconShop_DragBuy_Spell")
DRAG_SELL = "TB_BaconShop_DragSell"
DARK_DISCOVERY = "BG36_Button_DarkGift"
BUTTONS = {REROLL: "reroll", FREEZE: "freeze"}

BLOCK_PLAY, BLOCK_MOVE_MINION = 7, 12
TAG_ZONE, TAG_ZONE_POSITION, TAG_CARDTYPE = 49, 263, 202
ZONE_PLAY, ZONE_HAND = 1, 3
CARDTYPES = ENUM_VALUES["CARDTYPE"]
ACTION_TYPES = ("buy", "sell", "reroll", "freeze", "level", "hero_power", "play",
                "discover", "end_turn", "reposition")
CHOICE_KINDS = ("hero", "discover", "dark_discovery", "trinket", "quest", "other")
FLAG_GOLDEN_GIFT = "golden_dark_gift_first_only"
# Quarantine reasons that are not decisions: empty drags (a card dragged and
# dropped back, or a shop/hand card reordered) and a Choices offer re-sent
# unchanged after a re-dump. ``decision_match_rate`` leaves them out.
NON_DECISION_REASONS = ("noop_move", "choice_offer_resent")
NON_DECISION_PREFIX = "move_minion_in_"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# --- raw XML pass ---------------------------------------------------------
def scan_replay(path) -> Dict:
    """One pass over the raw XML. ``dps`` has one entry per ``Options`` block,
    counted exactly like ``replay_states`` counts options rows (a re-sent
    ``Options`` id replaces the previous entry), with the top-level PLAY /
    MOVE_MINION blocks that follow it and the SubOptions of its legal options.
    ``choices`` lists every ``Choices`` block in order with the
    ``ChosenEntities`` that answered it and the offered cards that moved to hand
    before the next ``Options``."""
    card: Dict[int, str] = {}
    cardtype: Dict[int, int] = {}
    dps: List[Dict] = []
    choices: List[Dict] = []
    stats: Counter = Counter()
    cur = act = last_id = None
    depth = block_depth = 0
    fh = _open(path)
    try:
        for ev, el in ET.iterparse(fh, events=("start", "end")):
            tag = el.tag
            if ev == "start":
                depth += 1
                if tag == "Block":
                    block_depth += 1
                    if block_depth == 1:
                        act = _start_action(el) if cur is not None else None
                        if act is not None:
                            cur["acts"].append(act)
                continue
            depth -= 1
            if tag in ("FullEntity", "ShowEntity", "ChangeEntity"):
                eid = _int(el.get("id") or el.get("entity"))
                if el.get("cardID"):
                    card[eid] = el.get("cardID")
                for t in el.findall("Tag"):
                    if _int(t.get("tag")) == TAG_CARDTYPE:
                        cardtype[eid] = _int(t.get("value"))
            elif tag == "TagChange":
                _tag_change(el, act, choices, cardtype)
            elif tag == "Block":
                block_depth -= 1
                if block_depth == 0:
                    act = None
            elif tag == "Options":
                oid = el.get("id")
                if oid == last_id and dps:
                    stats["options_resent"] += 1
                    stats["acts_dropped_on_resend"] += len(dps[-1]["acts"])
                    dps.pop()
                last_id = oid
                cur = {"options_id": oid, "acts": [], "suboptions": _suboptions(el, card)}
                dps.append(cur)
                for ch in choices:
                    ch["open"] = False
            elif tag == "Choices":
                choices.append({"after_dp": len(dps) - 1, "type": _int(el.get("type")),
                                "source": _int(el.get("source")),
                                "offered": [_int(c.get("entity")) for c in el.findall("Choice")],
                                "chosen": [], "moved_to_hand": [], "open": True})
            elif tag == "ChosenEntities":
                ids = [_int(c.get("entity")) for c in el.findall("Choice")]
                for ch in reversed(choices):
                    if set(ids) & set(ch["offered"]):
                        ch["chosen"] = ids
                        break
                else:
                    stats["chosen_entities_unmatched"] += 1
            if depth <= 2:
                el.clear()
    finally:
        if fh is not path:
            fh.close()
    return {"dps": dps, "choices": choices, "stats": stats,
            "entities": {"card": card, "cardtype": cardtype}}


def _start_action(el) -> Optional[Dict]:
    btype = _int(el.get("type"))
    if btype not in (BLOCK_PLAY, BLOCK_MOVE_MINION):
        return None
    return {"block": "PLAY" if btype == BLOCK_PLAY else "MOVE_MINION",
            "entity": _int(el.get("entity")), "target": _int(el.get("target"), 0) or None,
            "sub_option": _int(el.get("subOption"), -1),
            "entered_play": False, "play_position": None, "last_position": None}


def _tag_change(el, act, choices, cardtype) -> None:
    eid, tag, value = _int(el.get("entity")), _int(el.get("tag")), _int(el.get("value"), 0)
    if tag == TAG_CARDTYPE:
        cardtype[eid] = value
    if act is not None and eid == act["entity"]:
        if tag == TAG_ZONE:
            act["entered_play"] = value == ZONE_PLAY
        elif tag == TAG_ZONE_POSITION and value > 0:
            act["last_position"] = value
            if act["entered_play"] and act["play_position"] is None:
                act["play_position"] = value
    if tag == TAG_ZONE and value == ZONE_HAND:
        for ch in choices:
            if ch["open"] and eid in ch["offered"]:
                ch["moved_to_hand"].append(eid)


def _suboptions(el, card) -> Dict[int, List[Dict]]:
    """XML SubOptions per legal option index (states.v1 fallback; states.v2 rows
    carry ``options[].sub_options``). ``index`` = the SubOption index the
    ``Block subOption`` attribute refers to."""
    out = {}
    for o in el.findall("Option"):
        if _int(o.get("error"), -1) != -1:
            continue
        subs = [{"index": _int(s.get("index"), i), "entity_id": _int(s.get("entity")),
                 "card_id": card.get(_int(s.get("entity"))),
                 "targets": [_int(t.get("entity")) for t in s.findall("Target")]}
                for i, s in enumerate(o.findall("SubOption"))
                if _int(s.get("error"), -1) == -1]
        if subs:
            out[_int(o.get("index"))] = subs
    return out


# --- options --------------------------------------------------------------
def zone_map(snap: Dict) -> Dict[int, Tuple[str, int, Dict]]:
    """entity_id -> (zone, slot, view). slot = 0-based index into the snapshot
    list (State Builder sorts board/hand/shop by ZONE_POSITION). Shop spells
    take slots len(shop) + j (``shop_spells`` is sorted by shop position)."""
    out = {}
    for zone in ("board", "hand", "shop"):
        for slot, m in enumerate(snap.get(zone) or []):
            out[m["entity_id"]] = (zone, slot, m)
    base = len(snap.get("shop") or [])
    for j, sp in enumerate(snap.get("shop_spells") or []):
        if sp.get("entity_id") is not None:
            out[sp["entity_id"]] = ("shop", base + j, sp)
    return out


def make_option(type_, card_id=None, source=None, target=None, position=None,
                src_entity=None, target_entity=None, choice_kind=None) -> Dict:
    src, tgt = source or ("none", None), target or ("none", None)
    return {"type": type_, "card_id": card_id,
            "source": {"zone": src[0], "slot": src[1]},
            "target": {"zone": tgt[0], "slot": tgt[1]},
            "position": position, "src_entity": src_entity,
            "target_entity": target_entity, "choice_kind": choice_kind}


def option_key(o: Dict) -> tuple:
    """Identity of an action (entity ids are debug only, except to tell apart
    targets outside board/shop/hand, e.g. the hero or Bob)."""
    tgt = o["target"]
    return (o["type"], o["card_id"], o["source"]["zone"], o["source"]["slot"],
            tgt["zone"], tgt["slot"], o["position"],
            o["target_entity"] if tgt["zone"] == "none" else None)


def _target(zones, t) -> Tuple[Optional[tuple], Optional[int]]:
    if not t:
        return None, None
    where = zones.get(t)
    return (where[:2] if where else ("none", None)), t


def _entity_desc(entities: Optional[Dict], eid) -> Tuple[Optional[str], Optional[str]]:
    """(card_id, CARDTYPE name) of a raw XML entity, from ``scan_replay``."""
    if not entities or eid is None:
        return None, None
    return entities["card"].get(eid), CARDTYPES.get(entities["cardtype"].get(eid))


def _is_hero_power(row: Dict, eid, cid, entities) -> bool:
    if _entity_desc(entities, eid)[1] == "HERO_POWER":
        return True
    hp = [(row.get("hero_power") or {}).get("card_id"),
          ((row.get("snapshot") or {}).get("hero_power") or {}).get("card_id")]
    return cid is not None and cid in hp


def _outside_reason(kind: str, t, zones, entities) -> str:
    """Why a buy/sell helper target was not expanded into an option. The
    helpers list every drop spot (Bob, our hero, board and hand cards) as
    targets; only the card actually bought (shop) / sold (board) is an action."""
    if t in zones:
        return f"{kind}_drop_target_{zones[t][0]}"
    cid, ctype = _entity_desc(entities, t)
    if cid == "TB_BaconShopBob":
        return f"{kind}_drop_target_bob"
    if ctype == "HERO":
        return f"{kind}_drop_target_hero"
    return f"{kind}_target_not_in_state:{ctype}"


def _subs(raw: Dict, suboptions: Dict) -> List[Dict]:
    """Legal Choose One variants: the row's ``sub_options`` (states.v2), else
    the XML SubOptions (states.v1 rows)."""
    if raw.get("sub_options") is not None:
        return raw["sub_options"]
    return suboptions.get(raw["index"]) or []


def _sub_card(s: Dict, i: int, cid: Optional[str]) -> str:
    return s.get("card_id") or f"{cid}#sub{s.get('index', i)}"


def _variants(raw: Dict, suboptions: Dict, cid: Optional[str]) -> List[Tuple[str, List[int]]]:
    subs = _subs(raw, suboptions)
    if not subs:
        return [(cid, raw.get("targets") or [])]
    return [(_sub_card(s, i, cid), s.get("targets") or []) for i, s in enumerate(subs)]


def enumerate_options(row: Dict, dp: Optional[Dict] = None,
                      entities: Optional[Dict] = None) -> Tuple[List[Dict], Counter]:
    """Expand an options row's legal options into labeler options (one per
    distinct target / board position / Choose One variant). ``dp`` / ``entities``
    come from ``scan_replay``. Returns (options, skipped raw parts)."""
    snap = row["snapshot"]
    zones = zone_map(snap)
    board_n = len(snap.get("board") or [])
    suboptions = (dp or {}).get("suboptions") or {}
    out, seen, skipped, movable = [], set(), Counter(), set()

    def add(opt):
        k = option_key(opt)
        if k not in seen:
            seen.add(k)
            out.append(opt)

    for raw in row["options"]:
        eid, cid, targets = raw["entity_id"], raw["card_id"], raw.get("targets") or []
        where = zones.get(eid)
        if raw["type"] == "END_TURN":
            add(make_option("end_turn"))
        elif cid in DRAG_BUY:
            for t in targets:
                tz = zones.get(t)
                if tz and tz[0] == "shop":
                    add(make_option("buy", tz[2]["card_id"], source=tz[:2], src_entity=t))
                else:
                    skipped[_outside_reason("buy", t, zones, entities)] += 1
        elif cid == DRAG_SELL:
            for t in targets:
                tz = zones.get(t)
                if tz and tz[0] == "board":
                    add(make_option("sell", tz[2]["card_id"], source=tz[:2], src_entity=t))
                else:
                    skipped[_outside_reason("sell", t, zones, entities)] += 1
        elif cid in BUTTONS:
            add(make_option(BUTTONS[cid], None, src_entity=eid))
        elif cid and cid.startswith(LEVEL_PREFIX):
            add(make_option("level", None, src_entity=eid))
        elif where is None:
            if _is_hero_power(row, eid, cid, entities):
                for t in targets or [None]:
                    tgt, te = _target(zones, t)
                    add(make_option("hero_power", cid, target=tgt, src_entity=eid, target_entity=te))
            elif cid == DARK_DISCOVERY:
                add(make_option("play", cid, src_entity=eid))
            else:
                skipped[f"entity_not_in_state:{_entity_desc(entities, eid)[1]}:{cid}"] += 1
        elif where[0] == "shop":
            skipped["shop_card_drag_handle"] += 1
        elif where[0] == "board" and not targets and not _subs(raw, suboptions) \
                and eid not in movable:
            movable.add(eid)             # first untargeted board entry = drag to reorder
            for p in range(board_n):
                if p != where[1]:
                    add(make_option("reposition", cid, source=where[:2], position=p,
                                    src_entity=eid))
        else:                            # card in hand, or a board minion's activation
            zone, slot, m = where
            minion = zone == "hand" and (m.get("tags") or {}).get("CARDTYPE") == "MINION"
            positions = list(range(board_n + 1)) if minion and board_n < MAX_BOARD else [None]
            for vcid, vtargets in _variants(raw, suboptions, cid):
                for t in vtargets or [None]:
                    tgt, te = _target(zones, t)
                    for p in positions:
                        add(make_option("play", vcid, source=(zone, slot), target=tgt,
                                        position=p, src_entity=eid, target_entity=te))
    return out, skipped


def choice_options(row: Dict) -> List[Dict]:
    """One ``discover`` option per offered card of a choice row, in offer
    order: source = ("choice", offer index)."""
    ch = row["choice"]
    kind = ch.get("choice_kind")
    return [make_option("discover", c.get("card_id"), source=("choice", i),
                        src_entity=c.get("entity_id"), choice_kind=kind)
            for i, c in enumerate(ch.get("cards") or [])]


# --- chosen action --------------------------------------------------------
def _ok(option, method, confidence) -> Dict:
    return {"option": option, "method": method, "confidence": confidence,
            "reason": None, "detail": None}


def _bad(reason, **detail) -> Dict:
    return {"option": None, "method": None, "confidence": None,
            "reason": reason, "detail": detail or None}


def _chosen_sub_card(raws: List[Dict], dp: Dict, k: int, cid) -> Optional[str]:
    for o in raws:
        for i, s in enumerate(_subs(o, dp.get("suboptions") or {})):
            if s.get("index", i) == k:
                return _sub_card(s, i, cid)
    return None


def infer_choice(row: Dict, nxt: Optional[Dict], dp: Dict,
                 entities: Optional[Dict] = None) -> Dict:
    """The action taken at an options row, or a quarantine reason. ``nxt`` is
    the next options row."""
    acts = dp["acts"]
    if not acts:
        if nxt is None:
            return _ok(make_option("end_turn"), "no_action_last_decision_point", "medium")
        if (nxt.get("raw_turn") or 0) > (row.get("raw_turn") or 0):
            return _ok(make_option("end_turn"), "no_action_turn_advanced", "high")
        return _bad("no_action_same_turn")
    if len(acts) > 1:
        return _bad("multiple_actions", blocks=[a["block"] for a in acts])
    act = acts[0]
    snap = row["snapshot"]
    zones = zone_map(snap)
    eid, t = act["entity"], act["target"]
    detail = {"block": act["block"], "entity": eid, "target": t, "sub_option": act["sub_option"]}
    raws = [o for o in row["options"] if o["entity_id"] == eid]
    if not raws:
        return _bad("action_entity_not_a_legal_option", **detail)
    cid, where = raws[0]["card_id"], zones.get(eid)
    detail["card_id"] = cid

    if act["block"] == "MOVE_MINION":
        noop = act["last_position"] is None or (where and act["last_position"] - 1 == where[1])
        if where is None or where[0] != "board":
            zone = where[0] if where else "unknown"
            return _bad(f"move_minion_in_{zone}_{'noop' if noop else 'reorder'}", **detail)
        if noop:
            return _bad("noop_move", **detail)
        return _ok(make_option("reposition", cid, source=where[:2],
                               position=act["last_position"] - 1, src_entity=eid),
                   "move_minion_block", "high")
    if cid in DRAG_BUY or cid == DRAG_SELL:
        kind, zone = ("buy", "shop") if cid in DRAG_BUY else ("sell", "board")
        tz = zones.get(t)
        if not tz or tz[0] != zone:
            detail["target_card_id"], detail["target_cardtype"] = _entity_desc(entities, t)
            return _bad(f"{kind}_target_not_in_state_{zone}", **detail)
        return _ok(make_option(kind, tz[2]["card_id"], source=tz[:2], src_entity=t),
                   "play_block", "high")
    if cid in BUTTONS:
        return _ok(make_option(BUTTONS[cid], None, src_entity=eid), "play_block", "high")
    if cid and cid.startswith(LEVEL_PREFIX):
        return _ok(make_option("level", None, src_entity=eid), "play_block", "high")
    if where is None:
        if _is_hero_power(row, eid, cid, entities):
            tgt, te = _target(zones, t)
            return _ok(make_option("hero_power", cid, target=tgt, src_entity=eid,
                                   target_entity=te), "play_block", "high")
        if cid == DARK_DISCOVERY:
            return _ok(make_option("play", cid, src_entity=eid), "play_block", "high")
        detail["cardtype"] = _entity_desc(entities, eid)[1]
        return _bad("action_entity_not_in_state", **detail)
    zone, slot, m = where
    if zone == "shop":
        return _bad("play_from_shop", **detail)
    card = cid
    if act["sub_option"] >= 0:
        card = _chosen_sub_card(raws, dp, act["sub_option"], cid)
        if card is None:
            return _bad("suboption_unresolved", **detail)
    tgt, te = _target(zones, t)
    position, method = None, "play_block"
    minion = zone == "hand" and (m.get("tags") or {}).get("CARDTYPE") == "MINION"
    if minion and len(snap.get("board") or []) < MAX_BOARD:
        if act["play_position"] is None:
            return _bad("minion_play_position_unknown", **detail)
        position, method = act["play_position"] - 1, "play_block+zone_position"
    return _ok(make_option("play", card, source=(zone, slot), target=tgt, position=position,
                           src_entity=eid, target_entity=te), method, "high")


def _offer_ids(row: Optional[Dict]) -> Optional[List[int]]:
    if not row or row.get("kind") != "choice":
        return None
    return [c.get("entity_id") for c in row["choice"].get("cards") or []]


def match_choices(choice_rows: List[Dict], xml_choices: List[Dict]) -> List[Optional[Dict]]:
    """Pair each choice row (in order) with the next unused XML ``Choices`` block
    that offers exactly the same entity ids (ids are consistent within the
    snapshot the offer was made in). None = no matching block."""
    out, p = [], 0
    for r in choice_rows:
        ids = _offer_ids(r)
        q = p
        while q < len(xml_choices) and xml_choices[q]["offered"] != ids:
            q += 1
        if q < len(xml_choices):
            out.append(xml_choices[q])
            p = q + 1
        else:
            out.append(None)
    return out


def infer_pick(row: Dict, xch: Optional[Dict], next_row: Optional[Dict]) -> Dict:
    """The card picked at a choice row, from the ``ChosenEntities`` that answered
    its ``Choices`` block. An offer re-sent unchanged (after a re-dump) before
    any pick is one decision: the pick is labeled on the re-sent row and the
    earlier row is quarantined as ``choice_offer_resent``."""
    ids = _offer_ids(row) or []
    opts = choice_options(row)
    if xch is None:
        return _bad("choice_not_in_replay", offered=ids)
    picked = [e for e in xch["chosen"] if e in ids]
    if len(picked) == 1:
        return _ok(opts[ids.index(picked[0])], "chosen_entities", "high")
    if len(picked) > 1:
        return _bad("choice_multiple_picked", chosen=picked)
    if xch["chosen"]:
        return _bad("chosen_entities_not_offered", chosen=xch["chosen"], offered=ids)
    if _offer_ids(next_row) == ids:
        return _bad("choice_offer_resent", offered=ids)
    moved = sorted(set(xch["moved_to_hand"]) & set(ids))
    if len(moved) == 1:
        return _ok(opts[ids.index(moved[0])], "moved_to_hand", "medium")
    return _bad("choice_pick_unresolved", offered=ids, moved_to_hand=moved)


# --- transition check -----------------------------------------------------
def _ids(snap: Dict, zone: str) -> List[int]:
    return [m["entity_id"] for m in snap.get(zone) or []]


def _shop_ids(snap: Dict) -> List[int]:
    return _ids(snap, "shop") + [s.get("entity_id") for s in snap.get("shop_spells") or []]


def _fingerprint(row: Dict) -> tuple:
    s = row["snapshot"]
    views = tuple((z, m["entity_id"], m["card_id"], m.get("attack"), m.get("health"))
                  for z in ("board", "hand", "shop") for m in s.get(z) or [])
    return (s.get("gold"), s.get("tavern_tier"), s.get("shop_frozen"), views,
            tuple(_shop_ids(s)), json.dumps(row.get("hero_power"), sort_keys=True),
            json.dumps(s.get("trinkets"), sort_keys=True))


def _new_golden(a: Dict, b: Dict, card_id: str) -> bool:
    before = {m["entity_id"] for z in ("hand", "board") for m in a.get(z) or []}
    return any(m["entity_id"] not in before
               and (m["card_id"] == f"{card_id}_G" or m["card_id"].startswith("TB_BaconUps"))
               for z in ("hand", "board") for m in b.get(z) or [])


def _order_kept(old: List[int], new: List[int]) -> bool:
    return [x for x in new if x in old] == [x for x in old if x in new]


def _magnetic(a: Dict, opt: Dict) -> bool:
    """A MAGNETIC minion played from hand. Dropped at position p to the left of
    a Mech it magnetizes onto board[p] instead of entering the board."""
    if opt["type"] != "play" or opt["source"]["zone"] != "hand" or opt["position"] is None:
        return False
    hand = a.get("hand") or []
    slot = opt["source"]["slot"]
    return slot < len(hand) and str((hand[slot].get("tags") or {}).get("MAGNETIC")) == "1"


def _magnetized(a: Dict, b: Dict, opt: Dict, by_card: bool = False) -> bool:
    board_a = a.get("board") or []
    if not _magnetic(a, opt) or opt["position"] >= len(board_a):
        return False
    if by_card:
        return (len(b.get("board") or []) == len(board_a)
                and board_a[opt["position"]]["card_id"] in _cards(b, "board"))
    return (opt["src_entity"] not in _ids(b, "board")
            and board_a[opt["position"]]["entity_id"] in _ids(b, "board"))


def check_transition(row: Dict, nxt: Optional[Dict], opt: Dict) -> str:
    """'pass' / 'fail' / 'na': is the chosen shopping action consistent with
    State Builder's next options row?"""
    if nxt is None:
        return "na"
    a, b = row["snapshot"], nxt["snapshot"]
    typ, e = opt["type"], opt["src_entity"]
    if typ != "end_turn" and (nxt.get("raw_turn") or 0) > (row.get("raw_turn") or 0):
        return _check_across_turn(a, b, opt)
    if typ == "end_turn":
        ok = (nxt.get("raw_turn") or 0) > (row.get("raw_turn") or 0)
    elif typ == "buy":
        ok = e not in _shop_ids(b) and (e in _ids(b, "hand") or e in _ids(b, "board")
                                        or _new_golden(a, b, opt["card_id"]))
    elif typ == "sell":
        ok = e in _ids(a, "board") and e not in _ids(b, "board")
    elif typ == "reroll":
        ok = bool(_shop_ids(a) or _shop_ids(b)) and set(_shop_ids(a)) != set(_shop_ids(b))
    elif typ == "freeze":
        ok = bool(a.get("shop_frozen")) != bool(b.get("shop_frozen"))
    elif typ == "level":
        ok = (b.get("tavern_tier") or 0) == (a.get("tavern_tier") or 0) + 1
    elif typ == "reposition":
        new = _ids(b, "board")
        ok = (opt["position"] < len(new) and new[opt["position"]] == e
              and set(new) == set(_ids(a, "board")))
    elif typ == "play" and opt["source"]["zone"] == "hand":
        ok = e not in _ids(b, "hand")
        if ok and opt["position"] is not None:
            new = _ids(b, "board")
            ok = ((opt["position"] < len(new) and new[opt["position"]] == e
                   and _order_kept(_ids(a, "board"), new)) or _magnetized(a, b, opt))
    else:                     # hero power, board activation, Dark Discovery button
        ok = _fingerprint(row) != _fingerprint(nxt)
    return "pass" if ok else "fail"


def _cards(snap: Dict, zone: str) -> List[str]:
    return [m["card_id"] for m in snap.get(zone) or []]


def _check_across_turn(a: Dict, b: Dict, opt: Dict) -> str:
    """The turn ended before the next decision point (timer, or combat right
    after the action). Minion entity ids are re-issued between turns, so compare
    card ids; actions whose effect cannot be seen that way are 'na'."""
    typ, cid = opt["type"], opt["card_id"]
    if typ == "play" and opt["source"]["zone"] == "hand" and opt["position"] is not None:
        new = _cards(b, "board")
        ok = ((opt["position"] < len(new) and new[opt["position"]].startswith(cid))
              or _magnetized(a, b, opt, by_card=True))
    elif typ == "buy":
        ok = any(c.startswith(cid) for c in _cards(b, "hand") + _cards(b, "board"))
    elif typ == "sell":
        ok = _cards(b, "board").count(cid) < _cards(a, "board").count(cid)
    elif typ == "reposition":
        new = _cards(b, "board")
        ok = opt["position"] < len(new) and new[opt["position"]] == cid
    else:
        return "na"
    return "pass" if ok else "fail"


def check_pick_transition(nxt: Optional[Dict], opt: Dict) -> str:
    """'pass' / 'unverified' / 'na' for a choice row: does the picked card show
    up in the next options row (entity in hand/board, or its card id, golden
    or upgraded version as hero, hand/board card, trinket, quest or hero
    power)? The pick itself is the server's ``ChosenEntities``, so a pick
    with no visible trace (spent on pick, tripled, transformed) is
    'unverified', never a failure."""
    if nxt is None:
        return "na"
    s = nxt["snapshot"]
    if opt["src_entity"] in _ids(s, "hand") + _ids(s, "board"):
        return "pass"
    cid = opt["card_id"] or ""
    seen = _cards(s, "hand") + _cards(s, "board") + [s.get("hero")]
    seen += [t.get("card_id") if isinstance(t, dict) else t for t in s.get("trinkets") or []]
    seen += [q.get("card_id") for q in s.get("quests") or []]
    seen += [(s.get("hero_power") or {}).get("card_id"),
             (nxt.get("hero_power") or {}).get("card_id")]
    if cid and any(c and c.startswith(cid) for c in seen):
        return "pass"
    return "unverified"


# --- per game -------------------------------------------------------------
def mmr_weight_fn(corpus_mmrs):
    """weight = 0.2 + 0.8 * percentile_rank(mmr within the corpus), mid-rank
    percentile (ties count half): monotonic in MMR, always in (0.2, 1.0)."""
    vals = sorted(m for m in corpus_mmrs if m is not None)

    def weight(mmr):
        if mmr is None or not vals:
            return None
        lo, hi = bisect.bisect_left(vals, mmr), bisect.bisect_right(vals, mmr)
        pct = (lo + 0.5 * (hi - lo)) / len(vals)
        return round(WEIGHT_FLOOR + (1 - WEIGHT_FLOOR) * pct, 6)
    return weight


def created_fields(meta: Dict) -> Tuple[Optional[str], Optional[int]]:
    """(created_at, created_ts): manifest ``creationDate`` as UTC ISO-8601 with
    milliseconds and ``Z``, and the same instant in epoch milliseconds. Falls
    back to ``creationTimestamp`` (ms) when there is no ``creationDate``."""
    iso, ts = meta.get("creationDate"), meta.get("creationTimestamp")
    dt = None
    if iso:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    elif ts is not None:
        dt = EPOCH + timedelta(milliseconds=int(ts))
    if dt is None:
        return None, None
    dt = dt.astimezone(timezone.utc)
    ms = (dt - EPOCH) // timedelta(milliseconds=1)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z", ms


def _gifted(snap: Dict, card_id: str, zones=("hand", "board", "shop")) -> int:
    return sum(1 for z in zones for m in snap.get(z) or []
               if m["card_id"] == card_id and m.get("dark_gift"))


def gift_flags(states: List[Dict]) -> Dict[int, List[str]]:
    """dp_index -> flags for a known State Builder limit that does not block
    labeling: a golden tripled from 2+ Dark-Gifted copies carries every copy's
    gift, but ``dark_gift`` names only the first one found. A golden ``X_G``
    that appears while the previous row held 2+ gifted ``X`` (and fewer gifted
    ``X`` remain) is tracked by card id for as long as a gifted ``X_G`` is held."""
    out, tracked, prev = {}, set(), None
    for r in states:
        s = r["snapshot"]
        held = [m for z in ("hand", "board") for m in s.get(z) or []]
        if prev is not None:
            before = Counter(m["card_id"] for z in ("hand", "board") for m in prev.get(z) or [])
            for cid, n in Counter(m["card_id"] for m in held).items():
                if cid and cid.endswith("_G") and n > before.get(cid, 0):
                    base = cid[:-2]
                    if _gifted(prev, base) >= 2 and _gifted(s, base) < _gifted(prev, base):
                        tracked.add(cid)
        tracked = {c for c in tracked if any(m["card_id"] == c and m.get("dark_gift") for m in held)}
        if tracked:
            out[r["dp_index"]] = [FLAG_GOLDEN_GIFT]
        prev = s
    return out


def _next_options_index(states: List[Dict]) -> List[Optional[int]]:
    out, nxt = [None] * len(states), None
    for i in range(len(states) - 1, -1, -1):
        out[i] = nxt
        if states[i].get("kind", "options") == "options":
            nxt = i
    return out


def label_game(states: List[Dict], replay_path: str, meta: Dict, weight: Optional[float]) -> Dict:
    """Label one game's states.v2 rows (states.v1 rows = options rows only).
    Returns rows, quarantine, transitions and stats."""
    states = sorted(states, key=lambda r: r["dp_index"])
    scan = scan_replay(replay_path)
    gid = meta.get("reviewId") or (states[0]["game_id"] if states else None)
    stats: Counter = Counter(scan["stats"])
    transitions: Dict[str, Counter] = defaultdict(Counter)
    rows, quarantine = [], []
    created_at, created_ts = created_fields(meta)
    is_opt = [r.get("kind", "options") == "options" for r in states]
    opt_rows = [r for r, o in zip(states, is_opt) if o]
    ch_rows = [r for r, o in zip(states, is_opt) if not o]
    base = {"game_id": gid, "decision_points": len(states), "rows": rows,
            "quarantine": quarantine, "transitions": transitions, "stats": stats}
    if len(scan["dps"]) != len(opt_rows):
        quarantine.extend({"game_id": gid, "dp_index": r["dp_index"], "turn": r.get("turn"),
                           "kind": r.get("kind", "options"), "reason": "dp_count_mismatch",
                           "detail": {"xml_options": len(scan["dps"]),
                                      "options_rows": len(opt_rows)}} for r in states)
        return base
    dp_of = {id(r): dp for r, dp in zip(opt_rows, scan["dps"])}
    xch_of = {id(r): x for r, x in zip(ch_rows, match_choices(ch_rows, scan["choices"]))}
    stats["xml_choices"] += len(scan["choices"])
    stats["choice_rows_unmatched"] += sum(x is None for x in xch_of.values())
    next_opt = _next_options_index(states)
    flags_of = gift_flags(states)
    for i, row in enumerate(states):
        nxt = states[next_opt[i]] if next_opt[i] is not None else None
        if is_opt[i]:
            dp = dp_of[id(row)]
            options, skipped = enumerate_options(row, dp, scan["entities"])
            stats.update({f"skipped:{k}": v for k, v in skipped.items()})
            res = infer_choice(row, nxt, dp, scan["entities"])
        else:
            options = choice_options(row)
            nrow = states[i + 1] if i + 1 < len(states) else None
            res = infer_pick(row, xch_of[id(row)], nrow)
        stats["options_total"] += len(options)
        chosen = None
        if res["option"] is not None:
            keys = [option_key(o) for o in options]
            k = option_key(res["option"])
            if k not in keys:
                res = _bad("chosen_not_in_legal_options", chosen=res["option"])
            else:
                chosen = keys.index(k)
                opt = options[chosen]
                if is_opt[i]:
                    tkey, verdict = opt["type"], check_transition(row, nxt, opt)
                else:
                    tkey, verdict = f"discover:{opt['choice_kind']}", check_pick_transition(nxt, opt)
                transitions[tkey][verdict] += 1
                if verdict == "fail":
                    res = _bad(f"transition_failed:{opt['type']}", chosen=opt)
                    chosen = None
        if chosen is None:
            quarantine.append({"game_id": gid, "dp_index": row["dp_index"], "turn": row.get("turn"),
                               "kind": row.get("kind", "options"), "reason": res["reason"],
                               "detail": res["detail"], "n_options": len(options)})
            continue
        rows.append({
            "game_id": row["game_id"], "build": row["build"],
            "current_patch": row["build"] == CURRENT_BUILD,
            "turn": row["turn"], "dp_index": row["dp_index"],
            "placement": _int(meta.get("placement")), "mmr": row["mmr"],
            "created_at": created_at, "created_ts": created_ts,
            "state": row["snapshot"], "options": options, "chosen": chosen,
            "inference_method": res["method"], "confidence": res["confidence"],
            "weight": weight, "flags": flags_of.get(row["dp_index"], []),
        })
    for r in rows:                       # hard contract: chosen is a legal option
        if not (isinstance(r["chosen"], int) and 0 <= r["chosen"] < len(r["options"])):
            raise RuntimeError(f"chosen not in legal options: {gid} dp {r['dp_index']}")
    return base


# --- corpus run -----------------------------------------------------------
def _load_games(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("games", []) if isinstance(data, dict) else data


def _read_jsonl_gz(path: str) -> List[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: List[Dict], compress: bool) -> int:
    opener = gzip.open if compress else open
    with opener(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return os.path.getsize(path)


def _dist(vals: List[float]) -> Optional[Dict]:
    if not vals:
        return None
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}


def _label_one(job: Tuple[Dict, str, str, str, Optional[float]]) -> Dict:
    """Label one game and write its files; return only aggregates (runs in a
    worker process)."""
    g, states_path, replay_path, out, weight = job
    gid = g["reviewId"]
    res = label_game(_read_jsonl_gz(states_path), replay_path, g, weight)
    rows, quarantine = res["rows"], res["quarantine"]
    _write_jsonl(os.path.join(out, f"{gid}.jsonl.gz"), rows, compress=True)
    _write_jsonl(os.path.join(out, "quarantine", f"{gid}.jsonl"), quarantine, compress=False)
    agg = {"types": Counter(), "choice_kinds": Counter(), "conf": Counter(),
           "methods": Counter(), "flags": Counter(), "n_options": []}
    for r in rows:
        opt = r["options"][r["chosen"]]
        agg["types"][opt["type"]] += 1
        if opt["type"] == "discover":
            agg["choice_kinds"][opt["choice_kind"]] += 1
        agg["conf"][r["confidence"]] += 1
        agg["methods"][r["inference_method"]] += 1
        agg["flags"].update(r["flags"])
        agg["n_options"].append(len(r["options"]))
    dps = res["decision_points"]
    non_dec = sum(_non_decision(q["reason"]) for q in quarantine)
    return {"game_id": gid, "mmr": _int(g.get("mmr")), "weight": weight,
            "build": _int(g.get("buildNumber")), "created_at": created_fields(g)[0],
            "decision_points": dps, "rows": len(rows), "quarantined": len(quarantine),
            "choice_rows": sum(r["options"][r["chosen"]]["type"] == "discover" for r in rows),
            "match_rate": round(len(rows) / dps, 4) if dps else None,
            "non_decisions": non_dec,
            "decision_match_rate": round(len(rows) / (dps - non_dec), 4) if dps > non_dec else None,
            "quarantine": quarantine, "transitions": res["transitions"],
            "stats": res["stats"], **agg}


def run(states_dir: str, replays_dir: str, manifest: str, out: str,
        corpus_manifest: Optional[str] = None, limit: Optional[int] = None,
        workers: int = 1) -> Dict:
    t0 = time.time()
    games = _load_games(manifest)[:limit]
    corpus = _load_games(corpus_manifest or manifest)
    weight_of = mmr_weight_fn([_int(g.get("mmr")) for g in corpus])
    for sub in ("", "quarantine"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    jobs, dropped = [], []
    for g in games:
        gid = g["reviewId"]
        states_path = os.path.join(states_dir, f"{gid}.jsonl.gz")
        replay_path = os.path.join(replays_dir, f"{gid}.xml.gz")
        weight = weight_of(_int(g.get("mmr")))
        reason = ("placement_not_1" if _int(g.get("placement")) != 1 else
                  "states_quarantined" if os.path.isfile(
                      os.path.join(states_dir, "quarantine", f"{gid}.json")) else
                  "missing_states" if not os.path.isfile(states_path) else
                  "missing_replay" if not os.path.isfile(replay_path) else
                  "missing_mmr" if weight is None else None)
        if reason:
            dropped.append({"game_id": gid, "reason": reason, "placement": g.get("placement")})
            continue
        jobs.append((g, states_path, replay_path, out, weight))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = []
            for n, r in enumerate(ex.map(_label_one, jobs, chunksize=1), 1):
                results.append(r)
                _progress(n, len(jobs), r)
    else:
        results = []
        for n, job in enumerate(jobs, 1):
            results.append(_label_one(job))
            _progress(n, len(jobs), results[-1])
    summary = summarize(results, dropped, corpus, weight_of)
    summary["inputs"] = {"states": states_dir, "replays": replays_dir, "manifest": manifest,
                         "corpus_manifest": corpus_manifest or manifest,
                         "corpus_games_with_mmr": sum(_int(g.get("mmr")) is not None
                                                      for g in corpus)}
    summary["runtime_s"] = round(time.time() - t0, 2)
    summary["workers"] = workers
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _non_decision(reason: Optional[str]) -> bool:
    return bool(reason) and (reason in NON_DECISION_REASONS
                             or reason.startswith(NON_DECISION_PREFIX))


def _progress(n: int, total: int, r: Dict) -> None:
    print(f"[{n}/{total}] {r['game_id']} dps={r['decision_points']} rows={r['rows']} "
          f"quarantined={r['quarantined']} match={r['match_rate']}", flush=True)


def summarize(results: List[Dict], dropped: List[Dict], corpus: List[Dict], weight_of) -> Dict:
    reasons, types, kinds, conf, methods, flags, stats = (Counter() for _ in range(7))
    reasons_by_kind: Dict[str, Counter] = defaultdict(Counter)
    transitions: Dict[str, Counter] = defaultdict(Counter)
    q_examples: Dict[str, list] = defaultdict(list)
    weights, n_options, per_game = [], [], []
    for r in results:
        for q in r["quarantine"]:
            reasons[q["reason"]] += 1
            reasons_by_kind[q.get("kind", "options")][q["reason"]] += 1
            if len(q_examples[q["reason"]]) < MAX_EXAMPLES:
                q_examples[q["reason"]].append({"game_id": r["game_id"], "dp_index": q["dp_index"],
                                                "detail": q["detail"]})
        types.update(r["types"])
        kinds.update(r["choice_kinds"])
        conf.update(r["conf"])
        methods.update(r["methods"])
        flags.update(r["flags"])
        stats.update(r["stats"])
        for typ, c in r["transitions"].items():
            transitions[typ].update(c)
        weights.extend([r["weight"]] * r["rows"])
        n_options.extend(r["n_options"])
        top = Counter(q["reason"] for q in r["quarantine"]).most_common(3)
        per_game.append({k: r[k] for k in ("game_id", "mmr", "weight", "build", "created_at",
                                           "decision_points", "rows", "choice_rows",
                                           "quarantined", "non_decisions", "match_rate",
                                           "decision_match_rate")}
                        | {"top_quarantine_reasons": dict(top)})
    total_dps = sum(g["decision_points"] for g in per_game)
    total_rows = sum(g["rows"] for g in per_game)
    created = sorted(g["created_at"] for g in per_game if g["created_at"])
    worst = sorted((g for g in per_game if g["match_rate"] is not None),
                   key=lambda g: (g["match_rate"], g["game_id"]))[:10]
    worst_dec = sorted((g for g in per_game if g["decision_match_rate"] is not None),
                       key=lambda g: (g["decision_match_rate"], g["game_id"]))[:10]
    non_dec = sum(g["non_decisions"] for g in per_game)

    def rate(c):
        judged = c["pass"] + c["fail"] + c["unverified"]
        return round(c["pass"] / judged, 4) if judged else None

    return {
        "schema_version": SCHEMA_VERSION,
        "games_labeled": len(per_game), "games_dropped": dropped,
        "games_dropped_by_reason": dict(Counter(d["reason"] for d in dropped)),
        "decision_points": total_dps, "rows": total_rows,
        "rows_options": total_rows - sum(kinds.values()), "rows_choice": sum(kinds.values()),
        "match_rate": round(total_rows / total_dps, 4) if total_dps else None,
        "match_rate_high_only": round(conf["high"] / total_dps, 4) if total_dps else None,
        "non_decisions": non_dec,
        "decision_match_rate": round(total_rows / (total_dps - non_dec), 4)
        if total_dps > non_dec else None,
        "worst_games": worst, "worst_games_decision_match": worst_dec,
        "rows_per_action_type": {t: types[t] for t in ACTION_TYPES},
        "rows_per_choice_kind": {k: kinds[k] for k in CHOICE_KINDS},
        "confidence": dict(conf), "inference_methods": dict(methods.most_common()),
        "chosen_in_legal": {"rows_checked": total_rows,
                            "in_legal": total_rows,   # enforced: label_game raises otherwise
                            "rate": 1.0 if total_rows else None,
                            "inferred_but_not_in_legal_quarantined":
                                reasons.get("chosen_not_in_legal_options", 0)},
        "quarantine": {"count": sum(reasons.values()), "policy": "excluded from rows",
                       "reasons": dict(reasons.most_common()),
                       "reasons_by_row_kind": {k: dict(c.most_common())
                                               for k, c in reasons_by_kind.items()},
                       "examples": q_examples},
        "transition_check": {t: {**dict(c), "pass_rate": rate(c)}
                             for t, c in sorted(transitions.items())},
        "weight": {"curve": "0.2 + 0.8 * midrank_percentile(mmr in corpus manifest)",
                   "rows": _dist(weights),
                   "corpus_examples": _weight_examples(corpus, weight_of)},
        "created_at_range": [created[0], created[-1]] if created else None,
        "builds": dict(Counter(str(g["build"]) for g in per_game)),
        "flags": dict(flags),
        "options_per_row": _dist(n_options),
        "options_skipped": {k[len("skipped:"):]: v for k, v in sorted(stats.items())
                            if k.startswith("skipped:")},
        "xml_stats": {k: v for k, v in sorted(stats.items())
                      if not k.startswith("skipped:") and k != "options_total"},
        "games": per_game,
    }


def _weight_examples(corpus: List[Dict], weight_of) -> Dict[str, Optional[float]]:
    mmrs = sorted(m for m in (_int(g.get("mmr")) for g in corpus) if m is not None)
    if not mmrs:
        return {}
    picks = {"min": mmrs[0], "p25": mmrs[len(mmrs) // 4], "median": mmrs[len(mmrs) // 2],
             "p75": mmrs[(3 * len(mmrs)) // 4], "max": mmrs[-1]}
    return {f"{k}_mmr_{v}": weight_of(v) for k, v in picks.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, help="states.v2 folder (<reviewId>.jsonl.gz)")
    ap.add_argument("--replays", required=True, help="folder of <reviewId>.xml.gz")
    ap.add_argument("--manifest", required=True, help="manifest of the games to label")
    ap.add_argument("--corpus-manifest", default=None,
                    help="manifest used for MMR percentile weights (default: --manifest)")
    ap.add_argument("--out", default=os.path.join("data", "firestone", "labels", "v2"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    s = run(args.states, args.replays, args.manifest, args.out, args.corpus_manifest,
            args.limit, args.workers)
    print(f"rows={s['rows']}/{s['decision_points']} match_rate={s['match_rate']} "
          f"quarantined={s['quarantine']['count']} runtime={s['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())