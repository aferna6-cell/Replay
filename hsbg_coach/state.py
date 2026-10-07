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
        # cardId -> display name, learned from any entity that carries both. Lets
        # us name spells/anomalies (not in the minion KB) whose own entity was
        # created with only a CardID (no entityName).
        self.card_names: Dict[str, str] = {}
        # entity id -> cardId of entities dropped by RESET_ENTITIES. Ids are
        # never reused within a game, so a tag that still points at a dropped
        # entity (CREATOR, DARK_GIFT_ENTITY) can still be resolved to its card.
        self.dropped_card_ids: Dict[int, str] = {}
        # Player entity ids whose next FULL_ENTITY is an HSReplay re-dump.
        # That dump replaces tags (omitted zeros are absent). Live Power.log
        # never emits RESET_ENTITIES, so this set stays empty there.
        self._redump_players: set = set()

    def card_id_of(self, entity_id: Optional[int]) -> Optional[str]:
        """cardId of an entity, including one dropped by a re-dump reset."""
        ent = self.entities.get(entity_id) if entity_id is not None else None
        if ent is not None and ent.card_id:
            return ent.card_id
        return self.dropped_card_ids.get(entity_id)

    def _learn_name(self, card_id: Optional[str], name: Optional[str]) -> None:
        if card_id and name and "UNKNOWN ENTITY" not in name:
            self.card_names.setdefault(card_id, name)

    def apply(self, event: Event) -> None:
        if event.kind == "CREATE_GAME":
            self.entities.clear()
            self.dropped_card_ids.clear()
            self._redump_players.clear()
            self.current_turn = None
            self.game_counter += 1
            return

        # HSReplay XML re-dumps the full entity set mid-game (same ids) but does
        # not re-emit entities that died in between. Drop everything except the
        # Player entities (no <Player> re-dump); the dump that follows refills
        # state. Players are kept so name lookup and the local player survive,
        # but their next FULL_ENTITY replaces tags: dumps omit zeros, and
        # merging would keep a stale RESOURCES_USED / TEMP_RESOURCES.
        if event.kind == "RESET_ENTITIES":
            self.dropped_card_ids.update(
                (i, e.card_id) for i, e in self.entities.items()
                if e.card_id and e.tags.get("CARDTYPE") != "PLAYER")
            self._redump_players = {
                i for i, e in self.entities.items()
                if e.tags.get("CARDTYPE") == "PLAYER"}
            self.entities = {i: e for i, e in self.entities.items()
                             if e.tags.get("CARDTYPE") == "PLAYER"}
            self._last_block_entity = None
            return

        if event.kind in ("FULL_ENTITY", "SHOW_ENTITY") and event.entity:
            if (event.kind == "FULL_ENTITY"
                    and event.entity.id in self._redump_players):
                self._redump_players.discard(event.entity.id)
                ent = self.entities.get(event.entity.id)
                if ent is not None:
                    # id, name and card_id stay. Tags come only from this dump.
                    ent.tags.clear()
            self._upsert(event.entity)
            return

        if event.kind == "TAG_CHANGE" and event.entity:
            ent = self._ensure(event.entity)
            # Fill stable identity (name/cardId) from the descriptor if we don't
            # have it yet — the bracket carries it even when this entity was first
            # seen via a tag change. Zone/controller/position are NOT taken here;
            # the descriptor shows pre-change state and explicit tags drive those.
            ref = event.entity
            if ref.name and not ent.name:
                ent.name = ref.name
            if ref.card_id and not ent.card_id:
                ent.card_id = ref.card_id
            self._learn_name(ref.card_id, ref.name)
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
                if event.tag == "TURN":
                    self.current_turn = _safe_int(event.value)

    def _upsert(self, ref: EntityRef) -> Entity:
        ent = self._ensure(ref)
        if ref.name and not ent.name:
            ent.name = ref.name
        if ref.card_id:
            ent.card_id = ref.card_id
        self._learn_name(ref.card_id, ref.name)
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
