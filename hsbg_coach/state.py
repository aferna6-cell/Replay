"""In-memory game state assembled from parser Events.

Provider-agnostic: this tracks entities and their tags exactly as the log
reports them. Battlegrounds-specific interpretation (who is the local player,
what the tavern tier is, board ordering) lives in ``bg.py``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .parser import Event, EntityRef


@dataclass
class Entity:
    id: int
    name: Optional[str] = None
    card_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def zone(self) -> Optional[str]:
        return self.tags.get("ZONE")

    @property
    def controller(self) -> Optional[str]:
        return self.tags.get("CONTROLLER")

    def tag_int(self, name: str) -> Optional[int]:
        try:
            return int(self.tags[name])
        except (KeyError, ValueError):
            return None


class GameState:
    """Accumulates entity/tag state by applying parser Events in order."""

    def __init__(self) -> None:
        self.entities: Dict[int, Entity] = {}
        self.game_counter: int = 0      # increments each CREATE_GAME
        self.current_turn: Optional[int] = None
        self._last_block_entity: Optional[int] = None

    def apply(self, event: Event) -> None:
        if event.kind == "CREATE_GAME":
            self.entities.clear()
            self.current_turn = None
            self.game_counter += 1
            return

        if event.kind in ("FULL_ENTITY", "SHOW_ENTITY") and event.entity:
            self._upsert(event.entity)
            return

        if event.kind == "TAG_CHANGE" and event.entity:
            ent = self._ensure(event.entity)
            if event.tag:
                ent.tags[event.tag] = event.value or ""
                if event.tag == "TURN":
                    self.current_turn = _safe_int(event.value)
            return

        # Indented tag lines attach to the most recently upserted entity. The
        # parser doesn't thread block context, so we keep a light pointer.
        if event.kind == "TAG" and self._last_block_entity is not None:
            ent = self.entities.get(self._last_block_entity)
            if ent and event.tag:
                ent.tags[event.tag] = event.value or ""

    def _upsert(self, ref: EntityRef) -> Entity:
        ent = self._ensure(ref)
        if ref.name and not ent.name:
            ent.name = ref.name
        if ref.card_id:
            ent.card_id = ref.card_id
        if ref.zone:
            ent.tags.setdefault("ZONE", ref.zone)
        if ref.player is not None:
            ent.tags.setdefault("CONTROLLER", str(ref.player))
        self._last_block_entity = ent.id
        return ent

    def _ensure(self, ref: EntityRef) -> Entity:
        if ref.id is None:
            # Entity referenced only by name (e.g. "GameEntity"): synthesize a
            # stable negative id so we still track it.
            synthetic = -(abs(hash(ref.name or "?")) % 10_000_000)
            ref = EntityRef(id=synthetic, name=ref.name)
        ent = self.entities.get(ref.id)
        if ent is None:
            ent = Entity(id=ref.id, name=ref.name, card_id=ref.card_id)
            self.entities[ref.id] = ent
        return ent

    # --- convenience views ------------------------------------------------
    def in_zone(self, zone: str, controller: Optional[int] = None) -> List[Entity]:
        out = []
        for ent in self.entities.values():
            if ent.zone != zone:
                continue
            if controller is not None and ent.controller != str(controller):
                continue
            out.append(ent)
        return sorted(out, key=lambda e: e.tag_int("ZONE_POSITION") or 0)


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
