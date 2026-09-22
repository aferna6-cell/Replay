"""Ingest HSReplay XML / Tier7 structured boards into trajectory JSONL.

Output schema matches ``TrajectoryRecorder`` rows consumed by
``ml.train_eval_net --trajectories``, plus:

  source:    "hsreplay_expert" (so retrain can upweight vs personal games)
  game_id:   stable id for group_split
  weight_hint: optional float (default 1.0; trainers may amplify further)

BG reality check (2026-09):
  * No public bulk trajectory API.
  * HSReplay does not host Battlegrounds replay pages / My Replays for BG.
  * Tier7 unlocks structured stats + perfect-game / inspiration boards — those
    become expert endgame snapshots (placement=1 when "perfect").
  * Per-click mid-game actions still come from local ``.hsreplay`` / ``.xml``
    files (HDT export / manual download) when you have them.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bg import ActionType

EXPERT_SOURCE = "hsreplay_expert"
PERSONAL_SOURCE = "personal"

# Hearthstone numeric tags used in HSReplay XML (hearthstone package / GameTag).
TAG_ZONE = 49
TAG_ZONE_POSITION = 263
TAG_CONTROLLER = 50
TAG_ATK = 47
TAG_HEALTH = 45
TAG_DAMAGE = 44
TAG_CARDTYPE = 202
TAG_TURN = 20
TAG_STEP = 19
TAG_PLAYER_LEADERBOARD_PLACE = 3679  # PLAYER_LEADERBOARD_PLACE
TAG_TECH_LEVEL = 471                # TECH_LEVEL (minion tier) / also tavern on player
TAG_PLAYER_TECH_LEVEL = 464         # PLAYER_TECH_LEVEL (tavern tier)
TAG_RESOURCES = 26                  # RESOURCES (gold)
TAG_BACON_DUMMY = 4190              # BACON_DUMMY_PLAYER-ish; treated soft

ZONE_PLAY = 1
ZONE_HAND = 3
ZONE_DECK = 2
ZONE_GRAVEYARD = 4
ZONE_SETASIDE = 6
ZONE_SHOP = 7                       # often used for Bob's tavern in BG

CARDTYPE_MINION = 4
CARDTYPE_HERO = 3

# Block type ids (approx; used for action tagging when present)
BLOCK_POWER = 3
BLOCK_ATTACK = 1
BLOCK_TRIGGER = 5
BLOCK_DEATHS = 6
BLOCK_PLAY = 7
BLOCK_FATIGUE = 8
BLOCK_RITUAL = 9
BLOCK_REVEAL_CARD = 10
BLOCK_GAME_RESET = 11
BLOCK_MOVE_MINION = 12

# Card ids / names that signal common BG actions when seen in options/blocks.
_ROLL_HINTS = ("BACONSHOP8", "TB_BaconShop_8", "Refresh")  # refresh button entity
_TIER_HINTS = ("BACONSHOP8", "TB_BaconShopLock",)  # weak; also TagChange tech level


@dataclass
class _Entity:
    eid: int
    card_id: Optional[str] = None
    name: Optional[str] = None
    tags: Dict[int, int] = field(default_factory=dict)

    def tag(self, t: int, default: int = 0) -> int:
        return int(self.tags.get(t, default))


@dataclass
class IngestStats:
    files: int = 0
    games: int = 0
    rows: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _as_int(val: Optional[str], default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _minion_view(ent: _Entity) -> Dict[str, Any]:
    hp = ent.tag(TAG_HEALTH) - ent.tag(TAG_DAMAGE)
    return {
        "entity_id": ent.eid,
        "card_id": ent.card_id,
        "name": ent.name or ent.card_id,
        "attack": ent.tag(TAG_ATK) or None,
        "health": hp or None,
        "position": ent.tag(TAG_ZONE_POSITION) or None,
        "tags": {str(k): str(v) for k, v in ent.tags.items()},
    }


def _snapshot(
    *,
    game_counter: int,
    turn: Optional[int],
    phase: str,
    tavern_tier: Optional[int],
    gold: Optional[int],
    hero_health: Optional[int],
    board: List[Dict[str, Any]],
    shop: List[Dict[str, Any]],
    hero: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "game_counter": game_counter,
        "turn": turn,
        "phase": phase,
        "tavern_tier": tavern_tier,
        "gold": gold,
        "hero_health": hero_health,
        "board": board,
        "shop": shop,
        "shop_spells": [],
        "hand_spells": [],
        "hero_power": None,
        "anomaly": None,
        "level_cost": None,
        "trinkets": [],
        "opponent_profiles": [],
        "hero": hero,
        "hero_name": None,
        "hand": [],
        "opponents_seen": [],
        "notes": notes or [],
    }


def _decision(
    state: Dict[str, Any],
    action_type: str,
    *,
    placement: Optional[int],
    game_id: str,
    source: str = EXPERT_SOURCE,
    action_detail: Optional[Dict[str, Any]] = None,
    weight_hint: float = 1.0,
) -> Dict[str, Any]:
    return {
        "state": state,
        "action_type": action_type,
        "action_detail": action_detail or {},
        "placement": placement,
        "wall_clock": time.time(),
        "game_id": game_id,
        "source": source,
        "weight_hint": weight_hint,
    }


# ---------------------------------------------------------------------------
# XML (.hsreplay) ingest
# ---------------------------------------------------------------------------

def parse_hsreplay_xml(xml_text: str, *, game_id: Optional[str] = None,
                       source: str = EXPERT_SOURCE) -> List[Dict[str, Any]]:
    """Parse an HSReplay XML document into trajectory decision rows.

    Strategy (BG-oriented, intentionally forgiving):
      * Track FullEntity / ShowEntity / TagChange into an entity table.
      * Infer the friendly player as the lowest playerID that is not a dummy.
      * On TURN TagChange (odd → even recruit end / combat start heuristic) and at
        end of Game, emit an END_TURN (or last known action) board snapshot.
      * Detect BUY/SELL/ROLL/TIER_UP/HERO_POWER when Block/Option card movements
        are unambiguous; otherwise label END_TURN.
      * Read PLAYER_LEADERBOARD_PLACE for placement when present.
    """
    root = ET.fromstring(xml_text)
    # Allow either <HSReplay><Game>… or bare <Game>…
    games = []
    if _local(root.tag) == "HSReplay":
        games = [c for c in root if _local(c.tag) == "Game"]
    elif _local(root.tag) == "Game":
        games = [root]
    else:
        games = [c for c in root.iter() if _local(c.tag) == "Game"]

    rows: List[Dict[str, Any]] = []
    for gi, game_el in enumerate(games):
        gid = game_id or game_el.attrib.get("id") or f"hsreplay-{gi+1}"
        rows.extend(_parse_game_element(game_el, game_id=str(gid), source=source))
    return rows


def _parse_game_element(game_el: ET.Element, *, game_id: str,
                        source: str) -> List[Dict[str, Any]]:
    entities: Dict[int, _Entity] = {}
    players: Dict[int, int] = {}          # playerID -> entity id
    player_names: Dict[int, str] = {}
    friendly_player: Optional[int] = None
    turn: Optional[int] = None
    placement: Optional[int] = None
    last_action = ActionType.END_TURN.value
    last_detail: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    pending_snapshot = False

    def ensure(eid: int) -> _Entity:
        if eid not in entities:
            entities[eid] = _Entity(eid=eid)
        return entities[eid]

    def apply_tags(eid: int, tag_els: Iterable[ET.Element],
                   card_id: Optional[str] = None) -> None:
        ent = ensure(eid)
        if card_id:
            ent.card_id = card_id
        for t in tag_els:
            if _local(t.tag) != "Tag":
                continue
            tag_id = _as_int(t.attrib.get("tag"))
            val = _as_int(t.attrib.get("value"))
            ent.tags[tag_id] = val

    def board_of(controller: int) -> List[Dict[str, Any]]:
        mins = []
        for ent in entities.values():
            if ent.tag(TAG_CONTROLLER) != controller:
                continue
            if ent.tag(TAG_ZONE) != ZONE_PLAY:
                continue
            if ent.tag(TAG_CARDTYPE) not in (0, CARDTYPE_MINION):
                # allow missing cardtype if it has ATK/HEALTH and a card id
                if not (ent.card_id and (ent.tag(TAG_ATK) or ent.tag(TAG_HEALTH))):
                    continue
            if ent.tag(TAG_CARDTYPE) == CARDTYPE_HERO:
                continue
            mins.append(_minion_view(ent))
        mins.sort(key=lambda m: (m["position"] or 99, m["entity_id"]))
        return mins

    def shop_of(bob_controller: Optional[int]) -> List[Dict[str, Any]]:
        if bob_controller is None:
            return []
        out = []
        for ent in entities.values():
            if ent.tag(TAG_CONTROLLER) != bob_controller:
                continue
            # Bob's tavern cards often sit in PLAY or a shop zone under Bob.
            if ent.tag(TAG_ZONE) not in (ZONE_PLAY, ZONE_SHOP):
                continue
            if ent.tag(TAG_CARDTYPE) == CARDTYPE_HERO:
                continue
            if not ent.card_id:
                continue
            out.append(_minion_view(ent))
        out.sort(key=lambda m: (m["position"] or 99, m["entity_id"]))
        return out

    def hero_of(controller: int) -> Tuple[Optional[str], Optional[int]]:
        for ent in entities.values():
            if ent.tag(TAG_CONTROLLER) != controller:
                continue
            if ent.tag(TAG_CARDTYPE) == CARDTYPE_HERO and ent.tag(TAG_ZONE) == ZONE_PLAY:
                hp = ent.tag(TAG_HEALTH) - ent.tag(TAG_DAMAGE)
                return ent.card_id, hp or None
        return None, None

    def tavern_tier(controller: int) -> Optional[int]:
        for ent in entities.values():
            if ent.tag(TAG_CONTROLLER) != controller:
                continue
            tl = ent.tag(TAG_PLAYER_TECH_LEVEL) or ent.tag(TAG_TECH_LEVEL)
            if tl:
                return tl
        return None

    def gold(controller: int) -> Optional[int]:
        for ent in entities.values():
            if ent.tag(TAG_CONTROLLER) != controller:
                continue
            if TAG_RESOURCES in ent.tags:
                return ent.tag(TAG_RESOURCES)
        return None

    def emit(phase: str = "combat") -> None:
        nonlocal pending_snapshot
        if friendly_player is None:
            return
        hero_id, hero_hp = hero_of(friendly_player)
        # Bob is usually another player controller; pick the highest playerID
        # that isn't friendly as a weak shop heuristic.
        bob = None
        for pid in sorted(players.keys(), reverse=True):
            if pid != friendly_player:
                bob = pid
                break
        state = _snapshot(
            game_counter=1,
            turn=turn,
            phase=phase,
            tavern_tier=tavern_tier(friendly_player),
            gold=gold(friendly_player),
            hero_health=hero_hp,
            board=board_of(friendly_player),
            shop=shop_of(bob),
            hero=hero_id,
            notes=["hsreplay_xml"],
        )
        # Only keep rows that carry a usable board (eval net needs >=2 minions),
        # or keep early rows with explicit actions for policy BC later.
        if len(state["board"]) < 1 and last_action == ActionType.END_TURN.value:
            return
        rows.append(_decision(
            state, last_action, placement=placement, game_id=game_id,
            source=source, action_detail=dict(last_detail),
        ))
        pending_snapshot = False

    # Walk game children in order.
    for el in game_el:
        tag = _local(el.tag)
        if tag == "GameEntity":
            eid = _as_int(el.attrib.get("id"))
            apply_tags(eid, el)
        elif tag == "Player":
            eid = _as_int(el.attrib.get("id"))
            pid = _as_int(el.attrib.get("playerID") or el.attrib.get("PlayerID"))
            apply_tags(eid, el)
            if pid:
                players[pid] = eid
                player_names[pid] = el.attrib.get("name") or f"Player{pid}"
                # First non-empty named human-ish player becomes friendly.
                if friendly_player is None and el.attrib.get("name"):
                    friendly_player = pid
            # Some exports set friendlyPlayerID on Game attrib instead.
        elif tag in ("FullEntity", "ShowEntity"):
            eid = _as_int(el.attrib.get("id"))
            apply_tags(eid, el, card_id=el.attrib.get("cardID"))
            name = el.attrib.get("name")
            if name:
                ensure(eid).name = name
        elif tag == "TagChange":
            eid = _as_int(el.attrib.get("entity") or el.attrib.get("id"))
            t = _as_int(el.attrib.get("tag"))
            v = _as_int(el.attrib.get("value"))
            ent = ensure(eid)
            prev = ent.tag(t)
            ent.tags[t] = v
            if t == TAG_TURN:
                # Odd tavern / even combat is the usual BG parity; snapshot when
                # entering an even turn (combat) so the board is post-recruit.
                turn = v
                if v % 2 == 0 and friendly_player is not None:
                    last_action = ActionType.END_TURN.value
                    last_detail = {}
                    emit(phase="combat")
            if t == TAG_PLAYER_LEADERBOARD_PLACE and v >= 1:
                # Prefer the friendly player's placement when we can attribute.
                ctrl = ent.tag(TAG_CONTROLLER)
                if friendly_player is None or ctrl in (0, friendly_player):
                    placement = v
            # Tier-up detection: PLAYER_TECH_LEVEL increase on friendly.
            if t == TAG_PLAYER_TECH_LEVEL and v > prev:
                ctrl = ent.tag(TAG_CONTROLLER)
                if ctrl == friendly_player or friendly_player is None:
                    last_action = ActionType.TIER_UP.value
                    last_detail = {"to_tier": v}
                    pending_snapshot = True
            # Zone transitions → buy/sell heuristics.
            if t == TAG_ZONE and friendly_player is not None:
                ctrl = ent.tag(TAG_CONTROLLER)
                if ctrl == friendly_player:
                    if prev == ZONE_PLAY and v != ZONE_PLAY:
                        last_action = ActionType.SELL.value
                        last_detail = {"card_id": ent.card_id}
                        pending_snapshot = True
                    elif prev != ZONE_PLAY and v == ZONE_PLAY and ent.card_id:
                        # Could be play-from-hand or buy; treat as BUY if came
                        # from another controller's shop-ish zone is hard — label PLAY.
                        last_action = ActionType.PLAY.value
                        last_detail = {"card_id": ent.card_id}
                        pending_snapshot = True
        elif tag == "Block":
            btype = _as_int(el.attrib.get("type"))
            card = el.attrib.get("cardID")
            effect = (el.attrib.get("effectCardId") or "") + (card or "")
            if any(h in effect for h in _ROLL_HINTS) or "Refresh" in (el.attrib.get("entity") or ""):
                last_action = ActionType.ROLL.value
                last_detail = {}
                pending_snapshot = True
            elif btype == BLOCK_PLAY and card:
                # Buying from Bob often looks like a PLAY block with a BG card.
                last_action = ActionType.BUY.value
                last_detail = {"card_id": card}
                pending_snapshot = True
            # Recurse into block children for nested TagChanges / entities.
            for child in el:
                ctag = _local(child.tag)
                if ctag in ("FullEntity", "ShowEntity"):
                    eid = _as_int(child.attrib.get("id"))
                    apply_tags(eid, child, card_id=child.attrib.get("cardID"))
                elif ctag == "TagChange":
                    eid = _as_int(child.attrib.get("entity") or child.attrib.get("id"))
                    t = _as_int(child.attrib.get("tag"))
                    v = _as_int(child.attrib.get("value"))
                    ensure(eid).tags[t] = v
                    if t == TAG_PLAYER_LEADERBOARD_PLACE and v >= 1:
                        placement = v
        elif tag == "Options":
            # Look for refresh / hero power option text in attributes if present.
            for opt in el:
                if _local(opt.tag) != "Option":
                    continue
                info = json.dumps(opt.attrib)
                if "BACONSHOP8" in info or "Refresh" in info:
                    last_action = ActionType.ROLL.value
                if "HERO_POWER" in info or opt.attrib.get("type") == "2":
                    last_action = ActionType.HERO_POWER.value

    # Friendly fallback: lowest playerID.
    if friendly_player is None and players:
        friendly_player = sorted(players.keys())[0]

    # Final snapshot / backfill placement.
    if friendly_player is not None:
        emit(phase="game_over")
    if placement is not None:
        for r in rows:
            r["placement"] = placement
    return rows


# ---------------------------------------------------------------------------
# Tier7 structured boards → trajectory rows
# ---------------------------------------------------------------------------

def boards_from_tier7_payload(
    payload: Any,
    *,
    game_id_prefix: str = "tier7",
    source: str = EXPERT_SOURCE,
    default_placement: int = 1,
) -> List[Dict[str, Any]]:
    """Convert Tier7 perfect_games / inspiration-like JSON into trajectory rows.

    Accepts several observed/expected shapes without requiring a live token:
      * list of games/boards
      * ``{"data": [...]}`` / ``{"games": [...]}`` / ``{"results": [...]}``
      * each item may carry ``board`` / ``final_board`` / ``minions`` / ``cards``
        as a list of {card_id|cardId|dbf_id, atk|attack, health|hp, position?}
    """
    items = _unwrap_list(payload)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        board = _extract_board(item)
        if len(board) < 1:
            continue
        placement = (
            item.get("placement")
            or item.get("final_placement")
            or item.get("avg_final_placement")
            or default_placement
        )
        try:
            placement_i = int(placement)
        except (TypeError, ValueError):
            placement_i = default_placement
        hero = item.get("hero_card_id") or item.get("heroCardId") or item.get("hero")
        if hero is not None:
            hero = str(hero)
        gid = str(item.get("id") or item.get("shortid")
                  or f"{game_id_prefix}-{i+1}")
        state = _snapshot(
            game_counter=1,
            turn=item.get("turn") or 13,
            phase="game_over",
            tavern_tier=item.get("tavern_tier") or item.get("tech_level") or 6,
            gold=0,
            hero_health=item.get("hero_health") or 1,
            board=board,
            shop=[],
            hero=hero,
            notes=["tier7_structured"],
        )
        rows.append(_decision(
            state, ActionType.END_TURN.value, placement=placement_i,
            game_id=gid, source=source,
            action_detail={"tier7": True},
            weight_hint=float(item.get("weight_hint") or 1.5),
        ))
    return rows


def _unwrap_list(payload: Any) -> List[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "games", "results", "perfect_games", "boards",
                    "items", "rows"):
            if isinstance(payload.get(key), list):
                return payload[key]
        # Single board object
        if _extract_board(payload):
            return [payload]
    return []


def _extract_board(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = None
    for key in ("board", "final_board", "finalBoard", "minions", "cards",
                "final_comp", "finalComp"):
        if key in item:
            raw = item[key]
            break
    if isinstance(raw, dict):
        raw = raw.get("board") or raw.get("minions") or raw.get("cards")
    if not isinstance(raw, list):
        return []
    out = []
    for i, m in enumerate(raw):
        if isinstance(m, str):
            out.append({
                "entity_id": i + 1, "card_id": m, "name": m,
                "attack": None, "health": None, "position": i + 1, "tags": {},
            })
            continue
        if not isinstance(m, dict):
            continue
        cid = (m.get("card_id") or m.get("cardId") or m.get("cardID")
               or m.get("id") or (str(m["dbf_id"]) if m.get("dbf_id") else None)
               or (str(m["dbfId"]) if m.get("dbfId") else None))
        out.append({
            "entity_id": int(m.get("entity_id") or m.get("id") or i + 1),
            "card_id": cid,
            "name": m.get("name") or cid,
            "attack": m.get("attack") if m.get("attack") is not None else m.get("atk"),
            "health": m.get("health") if m.get("health") is not None else m.get("hp"),
            "position": m.get("position") or m.get("zone_position") or i + 1,
            "tags": {},
        })
    return out


# ---------------------------------------------------------------------------
# File / directory drivers
# ---------------------------------------------------------------------------

_XML_SUFFIXES = (".hsreplay", ".xml")


def iter_hsreplay_paths(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in files:
                    if name.lower().endswith(_XML_SUFFIXES):
                        out.append(os.path.join(root, name))
        elif os.path.isfile(p) and p.lower().endswith(_XML_SUFFIXES):
            out.append(p)
    return sorted(out)


def ingest_files(
    paths: Sequence[str],
    out_dir: str,
    *,
    source: str = EXPERT_SOURCE,
) -> IngestStats:
    stats = IngestStats()
    os.makedirs(out_dir, exist_ok=True)
    for path in iter_hsreplay_paths(paths):
        stats.files += 1
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            base = os.path.splitext(os.path.basename(path))[0]
            rows = parse_hsreplay_xml(text, game_id=f"hsreplay-{base}", source=source)
            if not rows:
                stats.skipped += 1
                continue
            out_path = os.path.join(out_dir, f"game-hsreplay-{base}.jsonl")
            with open(out_path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            stats.games += 1
            stats.rows += len(rows)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"{path}: {exc}")
    return stats


def write_rows(rows: List[Dict[str, Any]], out_dir: str,
               filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return path
