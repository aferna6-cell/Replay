"""Stream a Firestone HSReplay XML replay as parser ``Event`` objects.

Firestone stores games as HSReplay XML (``<reviewId>.xml.gz``) rather than raw
``Power.log`` text. This adapter walks the XML and yields the same ``Event``
kinds ``parser.parse_line`` produces, so ``BGTracker`` rebuilds state exactly
as it does for a live log. Numeric tags/values are mapped to the names the
client prints in Power.log (``ZONE=PLAY``, ``CARDTYPE=MINION``) using the
vendored HearthSim enums in ``hs_enums``; tags with no name stay numeric,
which is also what Power.log does.

XML-only quirks handled here:

* **Re-dumps.** Firestone re-emits ``GameEntity`` plus the full entity set
  (same ids) 1-10x per game. Entities that died in between are *not*
  re-emitted, so every ``GameEntity`` after the first yields a
  ``RESET_ENTITIES`` event first (``GameState`` drops everything except the
  Player entities). A re-dumped ``<GameEntity>`` also carries both Player
  entities' tags appended to its own (no ``<Player>`` elements are written);
  ``_split_redump`` splits them back out by each segment's ENTITY_ID.
* **Local player.** Account ids are zeroed; the local player is the
  ``<Player isMainPlayer="true">``. It is emitted as a ``PLAYER`` event with a
  non-zero ``hi`` (the signal ``BGTracker`` uses) plus a ``PLAYER_NAME`` so the
  player entity can be found the same way as with battletags.
* **Options / ChosenEntities** become ``OPTIONS`` / ``CHOSEN`` events carrying
  their parsed payload in ``Event.items``. ``BGTracker`` ignores both.

Stdlib only (``xml.etree.ElementTree.iterparse`` + ``gzip``).
"""

import gzip
import xml.etree.ElementTree as ET
from typing import Iterator

from .hs_enums import BLOCK_TYPES, ENUM_VALUES, GAME_TAGS, OPTION_TYPES
from .parser import EntityRef, Event


def tag_name(tag) -> str:
    return GAME_TAGS.get(int(tag), str(int(tag)))


def tag_value(name: str, value) -> str:
    enum = ENUM_VALUES.get(name)
    if enum is None:
        return str(int(value))
    return enum.get(int(value), str(int(value)))


def _int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _open(source):
    if hasattr(source, "read"):
        return source
    fh = open(source, "rb")
    if fh.read(2) == b"\x1f\x8b":
        fh.close()
        return gzip.open(source, "rb")
    fh.seek(0)
    return fh


def _entity_events(kind, ref, tags) -> Iterator[Event]:
    yield Event(kind=kind, logger="GameState", entity=ref)
    for tag, value in tags:
        name = tag_name(tag)
        yield Event(kind="TAG", logger="GameState", tag=name, value=tag_value(name, value))


def _tags(elem):
    return [(_int(t.get("tag")), _int(t.get("value"), 0)) for t in elem.iter("Tag")]


def _split_redump(tags):
    """Split a re-dumped GameEntity's tag list into (entity_id, tags) segments.
    Each appended Player block starts with CONTROLLER (50) then CARDTYPE (202)
    = PLAYER (2) and contains its own ENTITY_ID (53). The GameEntity's own
    segment comes first and is returned with entity_id None."""
    segments = [[]]
    for tag, value in tags:
        cur = segments[-1]
        if tag == 202 and value == 2 and cur and cur[-1][0] == 50:
            segments.append([cur.pop()])
        segments[-1].append((tag, value))
    out = [(None, segments[0])]
    for seg in segments[1:]:
        ids = [v for t, v in seg if t == 53]
        if ids:                        # unattributable segment: skip, never misfile
            out.append((ids[0], seg))
    return out


def iter_events(source) -> Iterator[Event]:
    """Yield parser Events for one HSReplay XML game (path, .gz path, or file)."""
    fh = _open(source)
    try:
        yield from _iter(fh)
    finally:
        if fh is not source:
            fh.close()


def _iter(fh) -> Iterator[Event]:
    depth = 0            # element depth; direct children of <Game> are depth 2
    game_entities = 0
    for ev, elem in ET.iterparse(fh, events=("start", "end")):
        tag = elem.tag
        if ev == "start":
            depth += 1
            if tag == "Block":
                btype = _int(elem.get("type"), 0)
                yield Event(kind="RAW", logger="GameState",
                            text=f"BLOCK_START BlockType={BLOCK_TYPES.get(btype, btype)} "
                                 f"Entity={elem.get('entity')}")
            elif tag == "Game":
                yield Event(kind="CREATE_GAME", logger="GameState", text="CREATE_GAME")
            continue

        depth -= 1
        if tag == "GameEntity":
            game_entities += 1
            geid = _int(elem.get("id"))
            segments = _split_redump(_tags(elem))
            if game_entities > 1:
                yield Event(kind="RESET_ENTITIES", logger="GameState")
            # Player segments first so the GameEntity's TURN is applied last.
            for eid, seg in segments[1:]:
                yield from _entity_events("FULL_ENTITY", EntityRef(id=eid), seg)
            yield from _entity_events(
                "FULL_ENTITY", EntityRef(id=geid, name="GameEntity"), segments[0][1])
        elif tag == "Player":
            pid = elem.get("playerID")
            name = f"{elem.get('name') or 'Player'}#{pid}"
            main = elem.get("isMainPlayer") == "true"
            yield Event(kind="PLAYER", logger="GameState", fields={
                "entity_id": elem.get("id"), "player_id": pid,
                "hi": "1" if main else "0", "lo": "0"})
            yield Event(kind="PLAYER_NAME", logger="GameState",
                        fields={"player_id": pid, "name": name})
            yield from _entity_events(
                "FULL_ENTITY", EntityRef(id=_int(elem.get("id")), name=name), _tags(elem))
        elif tag == "FullEntity":
            yield from _entity_events("FULL_ENTITY", EntityRef(
                id=_int(elem.get("id")), card_id=elem.get("cardID") or None), _tags(elem))
        elif tag in ("ShowEntity", "ChangeEntity"):
            yield from _entity_events("SHOW_ENTITY", EntityRef(
                id=_int(elem.get("entity")), card_id=elem.get("cardID") or None), _tags(elem))
        elif tag == "TagChange":
            name = tag_name(elem.get("tag"))
            yield Event(kind="TAG_CHANGE", logger="GameState",
                        entity=EntityRef(id=_int(elem.get("entity"))),
                        tag=name, value=tag_value(name, elem.get("value")))
        elif tag == "HideEntity":
            yield Event(kind="TAG_CHANGE", logger="GameState",
                        entity=EntityRef(id=_int(elem.get("entity"))),
                        tag="ZONE", value=tag_value("ZONE", elem.get("zone")))
        elif tag == "Options":
            opts = []
            for o in elem.findall("Option"):
                opts.append({
                    "index": _int(o.get("index")),
                    "type": OPTION_TYPES.get(_int(o.get("type")), o.get("type")),
                    "entity": _int(o.get("entity"), 0),
                    "error": _int(o.get("error"), -1),
                    "targets": [_int(t.get("entity")) for t in o.findall("Target")],
                })
            yield Event(kind="OPTIONS", logger="GameState",
                        fields={"id": elem.get("id")}, items=opts)
        elif tag == "ChosenEntities":
            yield Event(kind="CHOSEN", logger="GameState",
                        fields={"player_id": elem.get("playerID")},
                        items=[_int(c.get("entity")) for c in elem.findall("Choice")])
        if depth <= 2:   # finished a direct child of <Game>: free its subtree
            elem.clear()
