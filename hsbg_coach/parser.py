"""Tokenize Hearthstone ``Power.log`` lines into structured events.

The log is undocumented and indentation-significant. A typical block looks like::

    D 09:25:43.123 GameState.DebugPrintPower() - CREATE_GAME
    D 09:25:43.123 GameState.DebugPrintPower() -     GameEntity EntityID=1
    D 09:25:43.123 GameState.DebugPrintPower() -         tag=TURN value=1
    D 09:25:43.123 GameState.DebugPrintPower() - FULL_ENTITY - Updating [entityName=Foo id=27 zone=PLAY zonePos=2 cardId=BGS_039 player=1] CardID=BGS_039
    D 09:25:43.123 GameState.DebugPrintPower() - TAG_CHANGE Entity=[...id=27...] tag=ZONE value=PLAY
    D 09:25:43.123 LoadingScreen.OnSceneLoaded() - prevMode=HUB currMode=BACON

This module is the **stable, unit-tested** core. It deliberately knows nothing
about Battlegrounds semantics — that lives in ``bg.py`` and is calibrated
against real logs. Here we only turn text into typed events.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Prefix: level, timestamp, "Class.Method()", " - ", payload (payload keeps indentation)
_LINE_RE = re.compile(
    r"^[DWIEV]\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+(\w+)\.(\w+)\(\)\s+-\s(.*)$"
)

# Bracketed entity descriptor: [entityName=X id=27 zone=PLAY zonePos=2 cardId=Y player=1]
_BRACKET_RE = re.compile(r"\[(?P<body>[^\]]*)\]")

# Generic key=value scanner. Values run until the next " key=" or end of string,
# so entity names with spaces survive.
_KV_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")


@dataclass
class EntityRef:
    """A reference to a game entity, as it appears inside a log line."""
    id: Optional[int] = None
    name: Optional[str] = None
    card_id: Optional[str] = None
    zone: Optional[str] = None
    zone_pos: Optional[int] = None
    player: Optional[int] = None
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    kind: str                       # CREATE_GAME | FULL_ENTITY | SHOW_ENTITY | TAG_CHANGE | SCENE | TAG | RAW
    logger: str = ""                # e.g. GameState
    method: str = ""                # e.g. DebugPrintPower
    indent: int = 0                 # leading-space count of payload (block depth)
    entity: Optional[EntityRef] = None
    tag: Optional[str] = None
    value: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)
    text: str = ""                  # original payload (trimmed of indentation)


def _parse_bracket(body: str) -> EntityRef:
    kv = {k: v.strip() for k, v in _KV_RE.findall(body)}
    return EntityRef(
        id=_as_int(kv.get("id")),
        name=kv.get("entityName") or kv.get("name"),
        card_id=kv.get("cardId") or None,
        zone=kv.get("zone"),
        zone_pos=_as_int(kv.get("zonePos")),
        player=_as_int(kv.get("player")),
        raw=kv,
    )


def _as_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _extract_entity_field(body: str):
    """Return (EntityRef|None, body_without_entity_field).

    Matches ``Entity=[...]`` (bracket descriptor) or ``Entity=<token>`` (bare id
    or name like GameEntity), then strips it so the remaining tag/value pairs
    parse cleanly.
    """
    m = re.search(r"Entity=(\[[^\]]*\]|\S+)", body)
    if not m:
        return None, body
    ent = _entity_from_field(m.group(1))
    rest = body[: m.start()] + " " + body[m.end():]
    return ent, rest


def _entity_from_field(value: str) -> EntityRef:
    """Entity= can be a bracket, a bare id, or a player name."""
    m = _BRACKET_RE.search(value)
    if m:
        return _parse_bracket(m.group("body"))
    as_int = _as_int(value)
    if as_int is not None:
        return EntityRef(id=as_int)
    return EntityRef(name=value.strip())


def parse_line(line: str) -> Optional[Event]:
    """Parse one raw log line into an Event, or None if it isn't a log line."""
    m = _LINE_RE.match(line)
    if not m:
        return None
    logger, method, payload = m.group(1), m.group(2), m.group(3)
    indent = len(payload) - len(payload.lstrip(" "))
    body = payload.strip()

    common = dict(logger=logger, method=method, indent=indent, text=body)

    # Scene transitions (Battlegrounds detection happens in bg.py via currMode=BACON)
    if method == "OnSceneLoaded" or "currMode=" in body:
        fields = {k: v.strip() for k, v in _KV_RE.findall(body)}
        return Event(kind="SCENE", fields=fields, **common)

    if body == "CREATE_GAME":
        return Event(kind="CREATE_GAME", **common)

    if body.startswith("FULL_ENTITY"):
        ent = _bracket_entity(body)
        cid = _trailing_cardid(body)
        if ent and cid and not ent.card_id:
            ent.card_id = cid
        return Event(kind="FULL_ENTITY", entity=ent, **common)

    if body.startswith("SHOW_ENTITY"):
        ent = _bracket_entity(body)
        cid = _trailing_cardid(body)
        if ent and cid:
            ent.card_id = cid
        return Event(kind="SHOW_ENTITY", entity=ent, **common)

    if body.startswith("TAG_CHANGE"):
        # Pull the Entity= field out first: it may be a [..] descriptor whose
        # nested key=values would otherwise confuse the flat KV scanner.
        ent, rest = _extract_entity_field(body)
        fields = {k: v.strip() for k, v in _KV_RE.findall(rest)}
        return Event(
            kind="TAG_CHANGE", entity=ent,
            tag=fields.get("tag"), value=fields.get("value"),
            fields=fields, **common,
        )

    # Indented "tag=X value=Y" inside a CREATE_GAME/entity block.
    if body.startswith("tag=") and "value=" in body:
        fields = {k: v.strip() for k, v in _KV_RE.findall(body)}
        return Event(kind="TAG", tag=fields.get("tag"), value=fields.get("value"),
                     fields=fields, **common)

    return Event(kind="RAW", **common)


def _bracket_entity(body: str) -> Optional[EntityRef]:
    m = _BRACKET_RE.search(body)
    return _parse_bracket(m.group("body")) if m else None


def _trailing_cardid(body: str) -> Optional[str]:
    m = re.search(r"CardID=(\S+)", body)
    return m.group(1) if m else None


def parse_lines(lines) -> List[Event]:
    out = []
    for line in lines:
        ev = parse_line(line)
        if ev is not None:
            out.append(ev)
    return out
