"""Build ``states.v2`` decision-point rows from Firestone HSReplay XML replays.

One row per ``Options`` block offered to the local player (``kind="options"``)
and one per ``Choices`` block offered to the local player (``kind="choice"``:
hero pick, discovers, trinkets, Dark Discovery, Sire quests, ...), in game
order. Each row carries the ``BGTracker.snapshot()`` of that moment
(``Snapshot.to_dict()``, rebuildable with ``Snapshot.from_dict``) so training
and the live overlay share one encoder, plus the legal options or the offered
cards. See ``docs/replay_states_v1.md``.

    python -m hsbg_coach.replay_states --replays data/firestone/raw \\
        --manifest data/firestone/raw/pilot_manifest.json \\
        --out data/firestone/states/v1

Writes ``<out>/<reviewId>.jsonl.gz`` for games that pass every fidelity check,
``<out>/quarantine/<reviewId>.json`` (reasons + examples) for games that do
not, and ``<out>/report.json``. Every manifest entry and every replay in the
folder is accounted for in the report; nothing is dropped silently.
"""

import argparse
import copy
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .bg import BGTracker, DARK_GIFT_PREFIX, SIRE_HERO_RE
from .hsreplay_xml import iter_events

SCHEMA_VERSION = "states.v2"
DARK_DISCOVERY_BUTTON = "BG36_Button_DarkGift"
DARK_DISCOVERY_EFFECT = "BG36_MidGameEffect_010"   # CREATOR of the offered minions
# Shady Aristocrat, Sire Denathrius's buddy: "When you sell this, Discover a
# Quest". Other heroes can get it too (e.g. from a Wisdomball refresh).
SIRE_BUDDY_PREFIX = "BG24_HERO_100_Buddy"
# Pressure the Authorities: "Get your warband to {0} total Attack."
# QUEST_PROGRESS is the live sum of board attack, so it falls when the board
# does (sells, or an empty board between combats). The offered goal is
# QUEST_PROGRESS_TOTAL, which can differ from the card's baseline script
# numbers (28 on BG27_Quest_801; this game's tag is 20).
ATTACK_TOTAL_QUEST = "BG27_Quest_801"
MAX_EXAMPLES = 5
DEFAULT_CARDS = os.path.join("data", "firestone", "cards.json")


# --- card names -----------------------------------------------------------
def load_card_names(path: str = DEFAULT_CARDS) -> Dict[str, str]:
    """cardId -> English name from a HearthstoneJSON cards.json (downloaded
    once to ``path`` if missing). XML replays carry no entity names."""
    from .firestone_stats import CARDS_URL, _card_meta, _fetch_url
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(_fetch_url(CARDS_URL))
    return {cid: m["name"] for cid, m in _card_meta(path).items()}


# --- per-game build -------------------------------------------------------
class _Fail:
    """Failure reasons with a count and a few examples each."""

    def __init__(self):
        self.count: Counter = Counter()
        self.examples: Dict[str, list] = defaultdict(list)

    def add(self, reason: str, example=None) -> None:
        self.count[reason] += 1
        if example is not None and len(self.examples[reason]) < MAX_EXAMPLES:
            self.examples[reason].append(example)


def _legal(options: List[Dict]) -> List[Dict]:
    return [o for o in options if o["error"] == -1]


# The hero-pick check compares the chosen card with the played hero. These
# are the same hero on a different card id: a `_SKIN` cosmetic of the same
# stem, Aranna Starseeker becoming Aranna Unleashed, or any played hero the
# log tags BACON_SKIN (legacy skin ids do not share the base card's stem).
_HERO_PLACEHOLDER = "TB_BaconShop_HERO_PH"
_HERO_SWAP_STEMS = {
    "TB_BaconShop_HERO_59": "aranna",     # Aranna Starseeker
    "TB_BaconShop_HERO_59t": "aranna",    # Aranna, Unleashed
}
_ROW_HP_KEYS = ("card_id", "name", "cost", "used", "activatable")


def _hero_power_entries(tracker: BGTracker, legal_ids: set) -> List[Dict]:
    """Every HERO_POWER we control in PLAY, passives included.

    Order is the index used to choose the row's ``hero_power``: lowest
    ZONE_POSITION, then lowest entity id. ``activatable`` is membership in
    the legal options, not the snapshot's gold check. ``passive`` is
    HIDE_COST=1 (Illidan, Morchie, Drek'Thar, ...)."""
    ents = [e for e in tracker.state.in_zone("PLAY", tracker.local_player)
            if e.tags.get("CARDTYPE") == "HERO_POWER"]
    ents.sort(key=lambda e: ((e.tag_int("ZONE_POSITION") or 0), e.id))
    out = []
    for ent in ents:
        out.append({
            "card_id": ent.card_id,
            "name": tracker._display_name(ent.card_id, ent.name),
            "cost": ent.tag_int("COST") or 0,
            "used": ent.tags.get("EXHAUSTED") == "1",
            "activatable": ent.id in legal_ids,
            "passive": ent.tags.get("HIDE_COST") == "1",
        })
    return out


def _select_row_hero_power(entries: List[Dict]) -> Optional[Dict]:
    """The legal power if any (lowest index), else the first non-passive,
    else the passive. The row field keeps the historical shape (no
    ``passive`` key)."""
    if not entries:
        return None
    legal = [e for e in entries if e["activatable"]]
    if legal:
        chosen = legal[0]
    else:
        active = [e for e in entries if not e["passive"]]
        chosen = active[0] if active else entries[0]
    return {k: chosen[k] for k in _ROW_HP_KEYS}


def _hero_power(tracker: BGTracker, legal_ids: set) -> Optional[Dict]:
    """Our hero power in PLAY, including a passive when it is the only one
    (Snapshot.hero_power is None for passives). ``activatable`` = it is a
    legal option right now."""
    return _select_row_hero_power(_hero_power_entries(tracker, legal_ids))


def _options_rows(tracker: BGTracker, legal: List[Dict]) -> List[Dict]:
    out = []
    for o in legal:
        ent = tracker.state.entities.get(o["entity"])
        out.append({"index": o["index"], "type": o["type"], "entity_id": o["entity"],
                    "card_id": ent.card_id if ent else None,
                    "zone": ent.zone if ent else None,
                    "targets": o["targets"],
                    "sub_options": _sub_options(tracker, o)})
    return out


def _sub_options(tracker: BGTracker, option: Dict) -> List[Dict]:
    """Legal Choose One variants of an option (``error == -1``), card ids
    resolved through the tracker at row time."""
    out = []
    for so in option.get("sub_options") or []:
        if so["error"] != -1:
            continue
        ent = tracker.state.entities.get(so["entity"])
        out.append({"index": so["index"], "entity_id": so["entity"],
                    "card_id": ent.card_id if ent else None,
                    "targets": so["targets"]})
    return out


def build_game(path: str, meta: Dict, card_names: Optional[Dict[str, str]] = None,
               track_opponents: bool = True) -> Dict:
    """Replay one game; return {rows, failures, warnings, game_warnings, stats}."""
    tracker = BGTracker()
    tracker.track_opponents = track_opponents
    if card_names:
        tracker.state.card_names.update(card_names)
    fail = _Fail()
    rows: List[Dict] = []
    last_options_id = None
    game_entity_id = None
    main_player_entity = None
    stats = Counter()
    pending_picks: List[Dict] = []
    quests_seen: Dict[int, Dict] = {}
    combat_boards: Dict[int, List] = {}   # turn -> board at that turn's first attack
    await_combat = None                   # turn of the last Options row, until combat
    prev_choice = None                 # entity ids of the last unanswered Choices
    open_hero = None                   # hero MULLIGAN awaiting ChosenEntities
    hero_pick = None                   # picked card vs the hero that was then played

    for ev in iter_events(path):
        tracker.feed(ev)
        if open_hero is not None:
            _note_hero_reroll(open_hero, ev)
        if ev.kind == "FULL_ENTITY" and ev.entity and ev.entity.name == "GameEntity":
            game_entity_id = ev.entity.id
        elif ev.kind == "PLAYER" and ev.fields.get("hi") == "1":
            main_player_entity = _int(ev.fields.get("entity_id"))
        elif (await_combat is not None and ev.kind == "RAW"
              and "BlockType=ATTACK" in ev.text):
            # Board at the first attack after a turn's decision points: the
            # start-of-combat board, which is what Firestone's finalComp shows.
            combat_boards[await_combat] = [
                (m.card_id, m.tags.get("PREMIUM") == "1", m.attack, m.health)
                for m in tracker.snapshot().board]
            await_combat = None
        elif ev.kind == "CHOSEN":
            prev_choice = None
            _record_dark_discovery_picks(tracker, ev.items, pending_picks, fail, stats)
            if open_hero is not None:
                picked = _close_hero_choice(tracker, open_hero, ev.items, fail)
                if picked is not _NOT_HERO_CHOSEN:
                    hero_pick = {"card_id": picked, "row": open_hero["row"],
                                 "seen_card": None, "seen_tags": None,
                                 "seen_entity": None}
                    open_hero = None
        elif ev.kind == "CHOICES":
            if ev.fields.get("player_id") != main_player_entity:
                stats["choice_blocks_other_player"] += 1
                continue
            stats["choice_blocks"] += 1
            if prev_choice == ev.items:
                stats["choices_repeated"] += 1    # same offer again, no pick between
            prev_choice = list(ev.items)
            row = _choice_row(tracker, ev, meta, len(rows), game_entity_id)
            if row["choice"]["choice_kind"] == "hero":
                # cards[] is rewritten at the pick. Keep the offer as first seen.
                row["choice"]["offer_initial"] = copy.deepcopy(row["choice"]["cards"])
                open_hero = {
                    "row": row,
                    "rerolls": [],
                    "latest": {c["entity_id"]: c["card_id"]
                               for c in row["choice"]["cards"]},
                }
            _check_turn(row, rows, fail)
            _check_choice(row, fail, stats)
            _check_quests(row, quests_seen, fail, stats)
            rows.append(row)
            _maybe_observe_hero(tracker, hero_pick)
        elif ev.kind == "OPTIONS":
            stats["options_blocks"] += 1
            if ev.fields.get("id") == last_options_id:
                # Same Options id re-sent right after a re-dump: one decision
                # point. Keep the post-re-dump state (authoritative).
                stats["options_resent"] += 1
                last = max(i for i, r in enumerate(rows) if r["kind"] == "options")
                if last != len(rows) - 1:
                    stats["options_resent_after_choice"] += 1
                rows.pop(last)
            last_options_id = ev.fields.get("id")
            row = _row(tracker, ev.items, meta, len(rows), game_entity_id)
            _check_turn(row, rows, fail)
            _check_row(tracker, row, fail, stats)
            _check_picks(row, pending_picks, fail, stats)
            _check_quests(row, quests_seen, fail, stats)
            rows.append(row)
            await_combat = row["turn"]
            _maybe_observe_hero(tracker, hero_pick)

    # No later decision point (the pick was the last thing logged): the hero
    # entity, if the replay has one, is what we can check against.
    _maybe_observe_hero(tracker, hero_pick)

    # dp_index: chronological over both kinds; options_index: Options rows only
    # (the states.v1 dp_index).
    n_options = 0
    for i, r in enumerate(rows):
        r["dp_index"] = i
        if r["kind"] == "options":
            r["options_index"] = n_options
            n_options += 1
    _check_hero_pick(hero_pick, fail)
    options_rows = [r for r in rows if r["kind"] == "options"]
    choice_rows = [r for r in rows if r["kind"] == "choice"]
    stats["dark_discovery_picks_unverifiable"] += len(pending_picks)
    stats["quests_seen"] = len(quests_seen)
    stats["quests_completed"] = sum(q["completed"] for q in quests_seen.values())
    if any(SIRE_HERO_RE.match(r["snapshot"]["hero"] or "") for r in rows) and not quests_seen:
        fail.add("sire_quests_missing")
    warnings = _final_checks(tracker, options_rows, meta, fail, combat_boards, stats)
    stats["rows"] = len(rows)
    stats["options_rows"] = len(options_rows)
    stats["choice_rows"] = len(choice_rows)
    for r in choice_rows:
        stats["choice_kind_" + r["choice"]["choice_kind"]] += 1
    if stats["options_blocks"] - stats["options_resent"] != len(options_rows):
        fail.add("row_count_mismatch", {"options": stats["options_blocks"],
                                        "resent": stats["options_resent"],
                                        "rows": len(options_rows)})
    if stats["choice_blocks"] != len(choice_rows):
        fail.add("choice_row_count_mismatch", {"choices": stats["choice_blocks"],
                                               "rows": len(choice_rows)})
    game_warnings = []
    if not stats["choice_kind_hero"]:
        game_warnings.append("hero_pick_missing")
    return {"rows": rows, "failures": fail, "warnings": warnings,
            "game_warnings": game_warnings, "stats": stats}


def _base_row(tracker, meta, kind, dp_index, game_entity_id):
    snap = tracker.snapshot()
    ge = tracker.state.entities.get(game_entity_id)
    raw_turn = ge.tag_int("TURN") if ge is not None else snap.turn
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": meta["reviewId"],
        "build": _int(meta.get("buildNumber")),
        "mmr": _int(meta.get("mmr")),
        "lobby_tribes": [t["name"] for t in (meta.get("tribes") or {}).get("available", [])],
        "kind": kind,
        "dp_index": dp_index,
        "turn": (raw_turn + 1) // 2 if raw_turn is not None else None,
        "raw_turn": raw_turn,
        "snapshot": snap.to_dict(),
    }


def _row(tracker, options, meta, dp_index, game_entity_id) -> Dict:
    row = _base_row(tracker, meta, "options", dp_index, game_entity_id)
    legal = _legal(options)
    legal_ids = {o["entity"] for o in legal}
    dd_available = any(
        (tracker.state.entities.get(i) is not None
         and tracker.state.entities[i].card_id == DARK_DISCOVERY_BUTTON)
        for i in legal_ids)
    row["options_index"] = None          # set once the game is complete
    entries = _hero_power_entries(tracker, legal_ids)
    row.update({
        "hero_power": _select_row_hero_power(entries),
        "hero_powers": entries,
        "dark_discovery": {"available": dd_available},
        "options": _options_rows(tracker, legal),
    })
    return row


def _choice_row(tracker, ev, meta, dp_index, game_entity_id) -> Dict:
    """A Choices block offered to us. The pick is NOT recorded here: match
    ``cards[].entity_id`` against the following ChosenEntities.

    Hero MULLIGAN rows are the exception to "cards as first seen". A reroll
    keeps the slot's entity id and changes its card (ChangeEntity /
    ShowEntity) before ChosenEntities, so the row's ``cards`` are replaced
    with the pick-time cards in ``_close_hero_choice``. ``offer_initial``
    keeps the offer from this moment. Other choice kinds are not rewritten:
    discovers, trinkets, quests and Dark Discovery name the cards in the
    Choices block, and a later ChangeEntity on a shop slot is not one of
    those offers."""
    row = _base_row(tracker, meta, "choice", dp_index, game_entity_id)
    entities = tracker.state.entities
    src = entities.get(ev.fields.get("source") or -1)
    cards = [_choice_card(tracker, eid) for eid in ev.items]
    src_card = src.card_id if src is not None else None
    row["choice"] = {
        "choice_id": _int(ev.fields.get("id")),
        "choice_type": ev.fields.get("type"),
        "choice_kind": _choice_kind(ev.fields.get("type"), src_card, cards),
        "source_entity_id": ev.fields.get("source"),
        "source_card_id": src_card,
        "source_name": (tracker._display_name(src_card, src.name)
                        if src is not None else None),
        "min": ev.fields.get("min"),
        "max": ev.fields.get("max"),
        "cards": cards,
    }
    return row


def _choice_card(tracker, eid) -> Dict:
    ent = tracker.state.entities.get(eid)
    if ent is None:
        return {"entity_id": eid, "card_id": None, "name": None, "cardtype": None,
                "tags": {}, "dark_gift": None}
    gift = tracker._minion_dark_gift(ent)
    if gift is None:            # offered by Dark Discovery: DARK_GIFT_ENTITY only
        ref = tracker.state.card_id_of(ent.tag_int("DARK_GIFT_ENTITY")) or ""
        if ref.startswith(DARK_GIFT_PREFIX):
            gift = {"card_id": ref, "name": tracker._display_name(ref)}
    return {"entity_id": eid, "card_id": ent.card_id,
            "name": tracker._display_name(ent.card_id, ent.name),
            "cardtype": ent.tags.get("CARDTYPE"), "tags": dict(ent.tags),
            "dark_gift": gift}


# Returned by _close_hero_choice when ChosenEntities is some other offer.
_NOT_HERO_CHOSEN = object()


def _note_hero_reroll(open_hero: Dict, ev) -> None:
    """Record a ShowEntity / ChangeEntity / re-dump that changes an offered
    hero slot's card. The slot id stays; the card does not."""
    if ev.kind not in ("SHOW_ENTITY", "FULL_ENTITY") or ev.entity is None:
        return
    eid = ev.entity.id
    latest = open_hero["latest"]
    if eid not in latest:
        return
    new = ev.entity.card_id
    prev = latest[eid]
    if new and new != prev:
        open_hero["rerolls"].append({
            "entity_id": eid,
            "from_card_id": prev,
            "to_card_id": new,
        })
        latest[eid] = new


def _close_hero_choice(tracker, open_hero, chosen_ids, fail):
    """Rewrite a hero row's cards to the offer as it stands at the pick.

    Returns ``_NOT_HERO_CHOSEN`` when this ChosenEntities is a different
    offer, else the picked card id (None if that entity has no card)."""
    row = open_hero["row"]
    offered = [c["entity_id"] for c in row["choice"]["offer_initial"]]
    chosen = [i for i in chosen_ids if i in set(offered)]
    if not chosen:
        return _NOT_HERO_CHOSEN
    row["choice"]["cards"] = [_choice_card(tracker, eid) for eid in offered]
    if open_hero["rerolls"]:
        row["choice"]["rerolls"] = list(open_hero["rerolls"])
    for c in row["choice"]["cards"]:
        if not c["card_id"]:
            fail.add("choice_card_missing", {
                "dp": row["dp_index"], "entity_id": c["entity_id"],
                "source": row["choice"]["source_card_id"]})
    picked_id = chosen[0]
    for c in row["choice"]["cards"]:
        if c["entity_id"] == picked_id:
            return c["card_id"]
    return None


def _maybe_observe_hero(tracker, hero_pick) -> None:
    """Lock the first real hero played after the pick.

    The placeholder (TB_BaconShop_HERO_PH) is not that hero. Later swaps
    (Aranna Unleashed, a skin applied after the first decision point) are
    not what the pick has to match; the first non-placeholder hero is."""
    if not hero_pick or hero_pick.get("seen_card"):
        return
    hero = tracker._hero_entity()
    if hero is None or not hero.card_id or hero.card_id == _HERO_PLACEHOLDER:
        return
    hero_pick["seen_card"] = hero.card_id
    hero_pick["seen_tags"] = dict(hero.tags)
    hero_pick["seen_entity"] = hero.id


def _hero_stem(card_id: Optional[str]) -> str:
    if not card_id:
        return ""
    stem = card_id.split("_SKIN")[0]
    return _HERO_SWAP_STEMS.get(stem, stem)


def _hero_cards_match(picked: Optional[str], actual: Optional[str], tags: Optional[Dict]) -> bool:
    """True when the played hero is the picked card or a known swap of it."""
    if not picked or not actual or actual == _HERO_PLACEHOLDER:
        return False
    if picked == actual:
        return True
    if _hero_stem(picked) and _hero_stem(picked) == _hero_stem(actual):
        return True
    # Legacy skins (TB_BaconShop_HERO_44_SKIN_* for Sylvanas BG23_HERO_306)
    # do not share a card-id stem. The played entity is tagged BACON_SKIN.
    if (tags or {}).get("BACON_SKIN") == "1":
        return True
    return False


def _check_hero_pick(hero_pick, fail) -> None:
    """Quarantine when the picked hero card is not the hero that was played."""
    if not hero_pick or not hero_pick.get("seen_card"):
        return
    if _hero_cards_match(hero_pick.get("card_id"), hero_pick["seen_card"],
                         hero_pick.get("seen_tags")):
        return
    row = hero_pick.get("row") or {}
    fail.add("hero_pick_mismatch", {
        "dp": row.get("dp_index"),
        "picked": hero_pick.get("card_id"),
        "hero": hero_pick["seen_card"],
        "entity_id": hero_pick.get("seen_entity"),
    })


CHOICE_KINDS = ("hero", "dark_discovery", "quest", "trinket", "discover", "other")


def _choice_kind(choice_type, source_card_id, cards) -> str:
    """hero: the MULLIGAN hero pick only; dark_discovery: the
    Dark Discovery effect's offer; quest: only QUEST=1 cards (Sire); trinket:
    only BATTLEGROUND_TRINKET cards; discover: only minions / spells (triple
    rewards, discover effects); other: anything else (e.g. hero-power offers,
    Friendly Wager (TB_BaconShop_HP_081) combat guesses)."""
    types = {c["cardtype"] for c in cards}
    if choice_type == "MULLIGAN":
        return "hero"
    if source_card_id == DARK_DISCOVERY_EFFECT:
        return "dark_discovery"
    if cards and all(c["tags"].get("QUEST") == "1" for c in cards):
        return "quest"
    if types == {"BATTLEGROUND_TRINKET"}:
        return "trinket"
    if types and types <= {"MINION", "SPELL", "BATTLEGROUND_SPELL"}:
        return "discover"
    return "other"


def _check_turn(row, rows, fail) -> None:
    if rows and (row["raw_turn"] or 0) < (rows[-1]["raw_turn"] or 0):
        fail.add("invariant_turn_regressed", {"dp": row["dp_index"], "kind": row["kind"],
                                              "raw_turn": row["raw_turn"],
                                              "prev": rows[-1]["raw_turn"]})


def _check_choice(row, fail, stats) -> None:
    """Every choice row offers >= 1 card and every card entity is known."""
    ch, dp = row["choice"], row["dp_index"]
    if not ch["cards"]:
        fail.add("choice_no_cards", {"dp": dp, "source": ch["source_card_id"]})
    for c in ch["cards"]:
        stats["choice_cards"] += 1
        if not c["card_id"]:
            fail.add("choice_card_missing", {"dp": dp, "entity_id": c["entity_id"],
                                             "source": ch["source_card_id"]})
        if c["dark_gift"]:
            stats["choice_cards_with_dark_gift"] += 1


def _check_row(tracker, row, fail, stats) -> None:
    snap, dp = row["snapshot"], row["dp_index"]
    # gold is None until the player's RESOURCES tag is first set (the first
    # decision point(s) of turn 1-2 in some replays): unknown, not an error.
    # Once known it must stay known and never go negative.
    if snap["gold"] is None:
        stats["gold_unknown_rows"] += 1
        if stats["gold_known_rows"]:
            fail.add("invariant_gold", {"dp": dp, "gold": None})
    elif snap["gold"] < 0:
        fail.add("invariant_gold", {"dp": dp, "gold": snap["gold"]})
    else:
        stats["gold_known_rows"] += 1
    if len(snap["board"]) > 7:
        fail.add("invariant_board_gt_7", {"dp": dp, "board": len(snap["board"])})
    if len(snap["hand"]) > 10:
        fail.add("invariant_hand_gt_10", {"dp": dp, "hand": len(snap["hand"])})
    entities = tracker.state.entities
    for o in row["options"]:
        missing = [i for i in [o["entity_id"]] + o["targets"] if i and i not in entities]
        if missing:
            fail.add("invariant_option_entity_missing", {"dp": dp, "option": o["index"],
                                                         "missing": missing})
        if o["sub_options"]:
            stats["options_with_sub_options"] += 1
            stats["sub_options"] += len(o["sub_options"])
        for so in o["sub_options"]:
            missing = [i for i in [so["entity_id"]] + so["targets"] if i and i not in entities]
            if missing:
                fail.add("invariant_option_entity_missing", {
                    "dp": dp, "option": o["index"], "sub_option": so["index"],
                    "missing": missing})
    for zone in ("board", "hand", "shop"):
        for m in snap[zone]:
            if m["tags"].get("HAS_DARK_GIFT") != "1":
                continue
            stats["gifted_minion_snapshots"] += 1
            stats[f"gifted_{zone}"] += 1
            if m["dark_gift"] is None:
                fail.add("dark_gift_unresolved", {"dp": dp, "zone": zone,
                                                  "entity_id": m["entity_id"],
                                                  "card_id": m["card_id"]})
    _check_hero_power_options(tracker, row, fail)


def _check_hero_power_options(tracker, row, fail) -> None:
    """Every legal option that is a PLAY hero power is in ``hero_powers``
    with ``activatable`` true, and the row's ``hero_power`` is activatable
    when any such option exists."""
    legal = []
    for o in row["options"]:
        ent = tracker.state.entities.get(o["entity_id"])
        if ent is None:
            continue
        if ent.tags.get("CARDTYPE") != "HERO_POWER" or ent.zone != "PLAY":
            continue
        legal.append(ent)
    if not legal:
        return
    dp = row["dp_index"]
    hp = row.get("hero_power") or {}
    if not hp.get("activatable"):
        fail.add("hero_power_option_mismatch", {
            "dp": dp, "hero_power": hp.get("card_id"),
            "legal": [e.card_id for e in legal]})
    pool = [dict(p) for p in (row.get("hero_powers") or [])]
    for ent in legal:
        idx = next((i for i, p in enumerate(pool)
                    if p.get("card_id") == ent.card_id and p.get("activatable") is True),
                   None)
        if idx is None:
            fail.add("hero_power_option_mismatch", {
                "dp": dp, "entity_id": ent.id, "card_id": ent.card_id})
        else:
            pool.pop(idx)


def _record_dark_discovery_picks(tracker, chosen, pending, fail, stats) -> None:
    entities = tracker.state.entities
    for eid in chosen:
        ent = entities.get(eid)
        if ent is None:
            continue
        if tracker.state.card_id_of(ent.tag_int("CREATOR")) != DARK_DISCOVERY_EFFECT:
            continue
        stats["dark_discovery_picks"] += 1
        gift = tracker.state.card_id_of(ent.tag_int("DARK_GIFT_ENTITY")) or ""
        persistent = ent.tags.get("HAS_DARK_GIFT") == "1"
        if not gift.startswith(DARK_GIFT_PREFIX):
            if not persistent:
                fail.add("dark_discovery_pick_without_gift", {"entity_id": eid,
                                                             "card_id": ent.card_id})
                continue
            # No DARK_GIFT_ENTITY on the offer (a3252d4e): the gift is only
            # known later, from the minion's enchantment.
            gift = None
            stats["dark_discovery_picks_gift_unnamed"] += 1
        pending.append({"entity_id": eid, "card_id": ent.card_id, "gift": gift,
                        "persistent": persistent})


def _check_picks(row, pending, fail, stats) -> None:
    """A Dark Discovery pick that is still in our hand or on our board at the
    next decision point must carry the picked gift. A pick that is already
    gone by then (sold, or tripled: the golden carries every copy's gift but
    ``dark_gift`` names only one) counts as verified if its gift is on
    another held minion, otherwise it is counted as gone, not failed. One-shot gifts (Double Vision: "get an extra
    copy of this") are spent on pick: the offered minion never gets
    HAS_DARK_GIFT, so they are counted, not checked. A persistent pick whose
    offer did not name its gift only needs to carry some gift."""
    if not pending:
        return
    snap = row["snapshot"]
    held = {m["entity_id"]: m for z in ("board", "hand") for m in snap[z]}
    gifts = {m["dark_gift"]["card_id"] for m in held.values() if m.get("dark_gift")}
    for p in pending:
        m = held.get(p["entity_id"])
        if not p["persistent"]:
            stats["dark_discovery_picks_one_shot"] += 1
        elif m is None:
            # gone; if its gift is on another minion (e.g. the golden it was
            # tripled into) that still counts as verified
            stats["dark_discovery_picks_verified" if p["gift"] in gifts
                  else "dark_discovery_picks_gone"] += 1
        elif (m.get("dark_gift") or {}).get("card_id") in (
                {p["gift"]} if p["gift"] else set(gifts)):
            stats["dark_discovery_picks_verified"] += 1
        else:
            fail.add("dark_discovery_pick_missing", dict(p, dp=row["dp_index"]))
    pending.clear()


def _attack_quest_tracks_board(q, snap) -> bool:
    """A drop on Pressure the Authorities is the warband's current attack, not
    a lost counter, when the new progress equals the board's total attack."""
    if q.get("card_id") != ATTACK_TOTAL_QUEST:
        return False
    board = snap.get("board")
    if board is None:
        return False
    total = sum((m.get("attack") or 0) for m in board)
    return q.get("progress") == total


def _check_quests(row, seen, fail, stats) -> None:
    """Quests come from Sire Denathrius (any skin) or from his buddy, Shady
    Aristocrat (sell it: Discover a Quest), which any hero can get (e.g. a
    Wisdomball refresh offers it). A non-Sire row may only hold quests whose
    source (CREATOR) is Shady Aristocrat. For each quest entity: progress
    never decreases (except Pressure the Authorities, whose progress is the
    warband's current total attack), never exceeds the goal while active, the
    goal is known, and a completed quest stays completed. ``seen`` maps quest
    entity_id -> last observed entry."""
    snap, dp = row["snapshot"], row["dp_index"]
    quests = snap.get("quests") or []
    if not quests:
        return
    if not SIRE_HERO_RE.match(snap.get("hero") or ""):
        bad = [q for q in quests
               if not (q.get("source_card_id") or "").startswith(SIRE_BUDDY_PREFIX)]
        if bad:
            fail.add("quests_on_non_sire_hero", {
                "dp": dp, "hero": snap.get("hero"),
                "quests": [[q["card_id"], q.get("source_card_id")] for q in bad]})
            return
        stats["quest_snapshots_non_sire"] += len(quests)
    stats["quest_snapshots"] += len(quests)
    for q in quests:
        eid, prev = q["entity_id"], seen.get(q["entity_id"])
        ex = {"dp": dp, "entity_id": eid, "card_id": q["card_id"],
              "progress": q["progress"], "goal": q["goal"]}
        if prev is not None and prev["completed"] and not q["completed"]:
            fail.add("quest_uncompleted", ex)
        if not q["completed"]:
            if q["goal"] is None or q["goal"] <= 0:
                fail.add("quest_goal_missing", ex)
            elif q["progress"] > q["goal"]:
                fail.add("quest_progress_over_goal", ex)
            if prev is not None and not prev["completed"]:
                if (q["progress"] < prev["progress"]
                        and not _attack_quest_tracks_board(q, snap)):
                    fail.add("quest_progress_decreased", dict(ex, prev=prev["progress"]))
                if q["goal"] != prev["goal"]:
                    stats["quest_goal_changed"] += 1
        seen[eid] = q


def _final_checks(tracker, rows, meta, fail, combat_boards=None, stats=None) -> List[Dict]:
    """Final board vs manifest finalComp (card ids + golden, in order; atk/health
    only warns) and final placement vs manifest placement.

    Firestone's finalComp is the start-of-combat board of turn
    ``finalComp.turn``, which is often one turn before the game's last turn.
    So we compare against our last decision point of that turn, and accept
    the board at that turn's first attack when the player changed the board
    after the last decision point. A manifest board that matches another turn
    of ours is labelled ``manifest_final_comp_suspect``."""
    warnings: List[Dict] = []
    stats = stats if stats is not None else Counter()
    combat_boards = combat_boards or {}
    if not rows:
        fail.add("no_decision_points")
        return warnings
    fc = meta.get("finalComp") or {}
    fc_turn = _int(fc.get("turn")) if isinstance(fc, dict) else None
    fc = fc.get("board", []) if isinstance(fc, dict) else fc
    theirs = [(m["cardId"], bool(m.get("golden"))) for m in fc]
    board = lambda r: [(m["card_id"], m["tags"].get("PREMIUM") == "1")
                       for m in r["snapshot"]["board"]]
    at_turn = [r for r in rows if fc_turn is not None and r["turn"] == fc_turn]
    row = at_turn[-1] if at_turn else rows[-1]
    if row is not rows[-1]:
        stats["final_comp_earlier_turn"] = 1
    ours = board(row)
    combat = combat_boards.get(row["turn"])
    slot_stats = [(m["attack"], m["health"]) for m in row["snapshot"]["board"]]
    if ours != theirs and combat is not None and [c[:2] for c in combat] == theirs:
        stats["final_comp_from_combat"] = 1   # board changed after the last DP
        ours, slot_stats = theirs, [c[2:] for c in combat]
    if ours != theirs:
        other = [r["turn"] for r in rows if board(r) == theirs] + [
            t for t, cb in combat_boards.items() if [c[:2] for c in cb] == theirs]
        if other:
            reason = "manifest_final_comp_suspect"
        elif Counter(ours) == Counter(theirs):
            reason = "final_board_order"
        else:
            reason = "final_board_mismatch"
        fail.add(reason, {"ours": ours, "manifest": theirs, "turn": row["turn"],
                          "manifest_turn": fc_turn, "matches_our_turns": sorted(set(other))})
    else:
        for slot, ((atk, hp), f) in enumerate(zip(slot_stats, fc)):
            if (atk, hp) != (f.get("atk"), f.get("health")):
                warnings.append({"slot": slot, "card_id": theirs[slot][0],
                                 "ours": [atk, hp],
                                 "manifest": [f.get("atk"), f.get("health")],
                                 "first_attack": (list(combat[slot][2:])
                                                  if combat is not None
                                                  and [c[:2] for c in combat] == theirs
                                                  else None)})
    place = _placement(tracker)
    if place != _int(meta.get("placement")):
        # Independent evidence: our player entity's final PLAYSTATE. A WON
        # game is 1st place, so a different manifest placement is suspect.
        pe = tracker._player_entity()
        won = pe is not None and pe.tags.get("PLAYSTATE") == "WON"
        reason = ("manifest_placement_suspect" if won and place == 1
                  else "placement_mismatch")
        fail.add(reason, {"ours": place, "manifest": meta.get("placement"),
                          "playstate": pe.tags.get("PLAYSTATE") if pe is not None else None})
    return warnings


def _placement(tracker) -> Optional[int]:
    hero = tracker._hero_entity()
    if hero is not None and hero.tag_int("PLAYER_LEADERBOARD_PLACE") is not None:
        return hero.tag_int("PLAYER_LEADERBOARD_PLACE")
    return tracker.placement()


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --- corpus run -----------------------------------------------------------
def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def run(replays: str, manifest: str, out: str, cards: str = DEFAULT_CARDS,
        limit: Optional[int] = None, track_opponents: bool = True) -> Dict:
    t_start = time.time()
    games = json.load(open(manifest, encoding="utf-8"))
    games = games.get("games", games) if isinstance(games, dict) else games
    by_id = {g["reviewId"]: g for g in games}
    on_disk = {f[:-len(".xml.gz")] for f in os.listdir(replays) if f.endswith(".xml.gz")}
    ids = sorted(set(by_id) | on_disk)[:limit]
    qdir = os.path.join(out, "quarantine")
    os.makedirs(qdir, exist_ok=True)
    card_names = load_card_names(cards)

    per_game, by_reason, warn_by_reason = [], defaultdict(list), defaultdict(list)
    totals = Counter()
    for gid in ids:
        t0 = time.time()
        rec = {"game_id": gid}
        failures, warnings, stats, rows = _Fail(), [], Counter(), []
        game_warnings: List[str] = []
        path = os.path.join(replays, gid + ".xml.gz")
        if gid not in by_id:
            failures.add("not_in_manifest")
        elif gid not in on_disk:
            failures.add("missing_replay")
        else:
            try:
                res = build_game(path, by_id[gid], card_names, track_opponents)
                failures, warnings, stats, rows = (res["failures"], res["warnings"],
                                                  res["stats"], res["rows"])
                game_warnings = res["game_warnings"]
            except Exception as exc:     # a crash quarantines the game, never drops it
                failures.add("exception", f"{type(exc).__name__}: {exc}")
        states_path = os.path.join(out, gid + ".jsonl.gz")
        q_path = os.path.join(qdir, gid + ".json")
        rec.update(passed=not failures.count, reasons=sorted(failures.count),
                   rows=len(rows), atk_health_warnings=len(warnings), bytes=0,
                   game_warnings=game_warnings,
                   stats=dict(stats))
        if rec["passed"]:
            with gzip.open(states_path + ".tmp", "wt", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            os.replace(states_path + ".tmp", states_path)
            rec["bytes"] = os.path.getsize(states_path)
            if os.path.exists(q_path):
                os.remove(q_path)
        else:
            _write_json(q_path, {"game_id": gid, "reasons": dict(failures.count),
                                 "examples": dict(failures.examples),
                                 "atk_health_warnings": warnings, "stats": dict(stats)})
            if os.path.exists(states_path):
                os.remove(states_path)
            for r in failures.count:
                by_reason[r].append(gid)
        for w in game_warnings:
            warn_by_reason[w].append(gid)
        rec["warnings"] = warnings
        rec["runtime_s"] = round(time.time() - t0, 2)
        totals.update(stats)
        per_game.append(rec)
        print(f"{gid} {'PASS' if rec['passed'] else 'FAIL ' + ','.join(rec['reasons'])} "
              f"rows={rec['rows']} bytes={rec['bytes']} {rec['runtime_s']}s", flush=True)

    passed = sum(g["passed"] for g in per_game)
    report = {
        "schema_version": SCHEMA_VERSION,
        "games_processed": len(per_game),
        "passed": passed,
        "pass_rate": round(passed / len(per_game), 4) if per_game else None,
        "failures_by_reason": dict(by_reason),
        "game_warnings_by_reason": dict(warn_by_reason),
        "rows_written": sum(g["rows"] for g in per_game if g["passed"]),
        "bytes_written": sum(g["bytes"] for g in per_game),
        "runtime_s": round(time.time() - t_start, 2),
        "track_opponents": track_opponents,
        "totals": dict(totals),
        "games": per_game,
    }
    _write_json(os.path.join(out, "report.json"), report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replays", required=True, help="folder of <reviewId>.xml.gz")
    ap.add_argument("--manifest", required=True, help="manifest JSON ({games:[...]})")
    ap.add_argument("--out", default=os.path.join("data", "firestone", "states", "v1"))
    ap.add_argument("--cards", default=DEFAULT_CARDS,
                    help="HearthstoneJSON cards.json (downloaded here if missing)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-opponents", action="store_true",
                    help="skip per-event opponent capture (faster; opponents_seen/"
                         "opponent_profiles stay empty)")
    args = ap.parse_args(argv)
    rep = run(args.replays, args.manifest, args.out, args.cards, args.limit,
              track_opponents=not args.no_opponents)
    print(f"passed {rep['passed']}/{rep['games_processed']} rows={rep['rows_written']} "
          f"bytes={rep['bytes_written']} runtime={rep['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
