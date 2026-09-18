"""Parse GameState.DebugPrintOptions — the client's live menu of clickable actions.

Power.log emits a fresh Options block whenever the legal action set changes. Each
`option N type=POWER ... error=NONE` is something the player can click right now
(buy, sell, roll, freeze, hero power, Activate minion, Dark Gift button, hand
plays, …). error≠NONE means visible but illegal (not enough gold, wrong phase).

This is the ground truth for "what can I do?" — more reliable than tag heuristics
alone for Season 14 Activate / Dark Gift / clickable hero powers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_OPT_HDR = re.compile(r"DebugPrintOptions\(\)\s*-\s*id=")
_OPT_LINE = re.compile(
    r"DebugPrintOptions\(\)\s*-\s*option\s+(\d+)\s+type=(\w+)\s+mainEntity="
)
_ENTITY = re.compile(
    r"\[entityName=(?P<name>.+?) id=(?P<eid>\d+) zone=(?P<zone>\w+) "
    r"zonePos=(?P<pos>\d+) cardId=(?P<cid>[A-Za-z0-9_]*) player=(?P<player>\d+)\]"
)
_ERROR = re.compile(r"error=(\w+)")
_BTN_GIFT = re.compile(r"dark\s*gift|darkgift|giftdiscover", re.I)


@dataclass
class ClientOption:
    index: int
    type: str
    name: str = ""
    entity_id: Optional[int] = None
    zone: str = ""
    card_id: str = ""
    player: Optional[int] = None
    error: str = ""

    @property
    def legal(self) -> bool:
        return self.error == "NONE"

    @property
    def is_dark_gift(self) -> bool:
        blob = f"{self.name} {self.card_id}"
        if _BTN_GIFT.search(blob):
            return True
        cid = (self.card_id or "").lower()
        return "darkgift" in cid or "dark_gift" in cid or (
            "gift" in cid and "button" in cid
        )

    @property
    def is_hero_power(self) -> bool:
        cid = self.card_id or ""
        if "_HP_" in cid.upper():
            return True
        if "_HERO_" in cid and cid.endswith("p"):
            return True
        return (self.name or "").lower() in ("hero power",)

    @property
    def is_tavern_button(self) -> bool:
        cid = self.card_id or ""
        return any(x in cid for x in (
            "Reroll_Button", "LockAll_Button", "TechUp", "DragBuy", "DragSell",
            "BaconShopBob",
        ))


@dataclass
class OptionsTracker:
    """Feed raw log lines; keeps the latest Options block (legal clicks)."""

    _current: List[ClientOption] = field(default_factory=list)
    _building: List[ClientOption] = field(default_factory=list)
    _open: bool = False
    version: int = 0

    def feed(self, line: str) -> bool:
        """Return True when a new Options block is finalized."""
        if _OPT_HDR.search(line):
            self._building = []
            self._open = True
            return False
        if not self._open:
            return False
        m = _OPT_LINE.search(line)
        if m:
            err_m = _ERROR.search(line)
            err = err_m.group(1) if err_m else ""
            ent = _ENTITY.search(line)
            if ent:
                self._building.append(ClientOption(
                    index=int(m.group(1)),
                    type=m.group(2),
                    name=(ent.group("name") or "").strip(),
                    entity_id=int(ent.group("eid")),
                    zone=ent.group("zone") or "",
                    card_id=ent.group("cid") or "",
                    player=int(ent.group("player")),
                    error=err,
                ))
            else:
                self._building.append(ClientOption(
                    index=int(m.group(1)), type=m.group(2), error=err,
                ))
            return False
        # Nested target lines still belong to the block.
        if "DebugPrintOptions()" in line:
            return False
        if self._building:
            self._current = list(self._building)
            self._building = []
            self._open = False
            self.version += 1
            return True
        self._open = False
        return False

    def legal(self) -> List[ClientOption]:
        return [o for o in self._current if o.legal and o.type == "POWER"]

    def fingerprint(self) -> Tuple:
        return tuple(sorted(
            (o.card_id, o.entity_id, o.name, o.error)
            for o in self._current if o.type == "POWER"
        ))
