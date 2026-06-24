"""Battlegrounds semantic layer: phases, the local player, and state snapshots.

This is the layer that must be **calibrated against a real captured Power.log**.
The exact tag names/values Hearthstone uses for tavern tier, hero health, and the
recruit<->combat transition vary by build, so anywhere that depends on them is
marked ``# CALIBRATE``. The structure (action space, snapshot shape, phase
machine) is stable; only the constants need confirming on your machine.

Design goal stated by the user: recommend moves for *every* aspect of the game,
always conditioned on full board + game state. So the action space below is
exhaustive, and every snapshot carries the whole observable state, not just the
shop.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .parser import Event
from .state import GameState, Entity


class Phase(str, Enum):
    UNKNOWN = "unknown"
    HERO_SELECT = "hero_select"
    RECRUIT = "recruit"        # the shopping / tavern phase — where most decisions live
    COMBAT = "combat"
    GAME_OVER = "game_over"


class ActionType(str, Enum):
    """Every decision a Battlegrounds player can make. The recorder labels each
    recorded transition with one of these so the eventual policy can recommend
    across the *whole* game, not just buys."""
    HERO_PICK = "hero_pick"
    TRINKET_PICK = "trinket_pick"     # greater / lesser trinkets
    BUY = "buy"                       # recruit a minion from the shop
    SELL = "sell"                     # sell a minion from the board
    PLAY = "play"                     # play a minion/spell from hand to board
    POSITION = "position"             # reorder board (matters for combat)
    ROLL = "roll"                     # refresh the shop
    FREEZE = "freeze"
    UNFREEZE = "unfreeze"
    TIER_UP = "tier_up"               # level the tavern
    HERO_POWER = "hero_power"
    TAVERN_SPELL = "tavern_spell"     # spells / quests offered in tavern
    TARGET = "target"                 # choosing a target for a battlecry/buff
    END_TURN = "end_turn"


# Internal-mode name Hearthstone uses for Battlegrounds in LoadingScreen logs.
BACON_MODE = "BACON"

# --- CALIBRATE: confirm these tag names/values against a real log -----------
TAG_TAVERN_TIER = "PLAYER_TECH_LEVEL"   # CALIBRATE: tavern tier on player/hero
TAG_HEALTH = "HEALTH"                    # CALIBRATE: hero health
TAG_DAMAGE = "DAMAGE"                    # CALIBRATE: damage taken (health - damage = effective)
TAG_RESOURCES = "RESOURCES"              # CALIBRATE: gold available
TAG_STEP = "STEP"                        # GameEntity step; recruit vs combat


@dataclass
class MinionView:
    entity_id: int
    card_id: Optional[str]
    name: Optional[str]
    attack: Optional[int]
    health: Optional[int]
    position: Optional[int]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Snapshot:
    """Full observable state at a decision point. This is the model's input."""
    game_counter: int
    turn: Optional[int]
    phase: str
    tavern_tier: Optional[int]
    gold: Optional[int]
    hero_health: Optional[int]
    board: List[MinionView] = field(default_factory=list)     # your minions in play
    shop: List[MinionView] = field(default_factory=list)      # minions available to buy
    hand: List[MinionView] = field(default_factory=list)
    opponents_seen: List[Dict] = field(default_factory=list)  # last-known enemy boards
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "game_counter": self.game_counter,
            "turn": self.turn,
            "phase": self.phase,
            "tavern_tier": self.tavern_tier,
            "gold": self.gold,
            "hero_health": self.hero_health,
            "board": [m.__dict__ for m in self.board],
            "shop": [m.__dict__ for m in self.shop],
            "hand": [m.__dict__ for m in self.hand],
            "opponents_seen": self.opponents_seen,
            "notes": self.notes,
        }


class BGTracker:
    """Consumes parser Events, maintains GameState, and exposes BG views."""

    def __init__(self) -> None:
        self.state = GameState()
        self.in_bg = False
        self.phase = Phase.UNKNOWN
        self.local_player: Optional[int] = None  # controller id of the human

    def feed(self, event: Event) -> None:
        # Battlegrounds detection via scene transition.
        if event.kind == "SCENE":
            mode = event.fields.get("currMode") or event.fields.get("mode")
            if mode == BACON_MODE:
                self.in_bg = True
                self.phase = Phase.HERO_SELECT
            elif mode and mode != BACON_MODE:
                self.in_bg = False
                self.phase = Phase.UNKNOWN
            return

        self.state.apply(event)
        self._update_phase(event)
        self._maybe_detect_local_player()

    # --- phase machine (CALIBRATE step values) ----------------------------
    def _update_phase(self, event: Event) -> None:
        if not self.in_bg:
            return
        if event.kind in ("TAG", "TAG_CHANGE") and event.tag == TAG_STEP:
            step = (event.value or "").upper()
            # CALIBRATE: confirm the BACON_* step names on a real log.
            if "COMBAT" in step:
                self.phase = Phase.COMBAT
            elif "RECRUIT" in step or "SHOP" in step or "MAIN" in step:
                self.phase = Phase.RECRUIT
            elif "DONE" in step or "FINAL" in step:
                self.phase = Phase.GAME_OVER

    def _maybe_detect_local_player(self) -> None:
        # Heuristic: the local player is the controller whose HAND cards have
        # known card_ids (opponents' hands are hidden in the log). CALIBRATE
        # against a real log; HDT uses a similar revealed-cards approach.
        if self.local_player is not None:
            return
        for ent in self.state.entities.values():
            if ent.zone == "HAND" and ent.card_id and ent.controller:
                self.local_player = int(ent.controller)
                return

    # --- snapshot ---------------------------------------------------------
    def snapshot(self) -> Snapshot:
        pid = self.local_player
        board = [self._minion(e) for e in self.state.in_zone("PLAY", pid)]
        shop = [self._minion(e) for e in self._shop_entities()]
        hand = [self._minion(e) for e in self.state.in_zone("HAND", pid)]
        notes = []
        if self.local_player is None:
            notes.append("local_player not yet identified")
        return Snapshot(
            game_counter=self.state.game_counter,
            turn=self.state.current_turn,
            phase=self.phase.value,
            tavern_tier=self._player_tag_int(TAG_TAVERN_TIER),
            gold=self._player_tag_int(TAG_RESOURCES),
            hero_health=self._hero_health(),
            board=board,
            shop=shop,
            hand=hand,
            notes=notes,
        )

    def _shop_entities(self) -> List[Entity]:
        # CALIBRATE: shop minions live in a BG-specific zone (often "PLAY" under
        # the neutral/tavern controller, or a dedicated SETASIDE bucket). Confirm
        # the zone/controller on a real log; this returns the neutral-controlled
        # play minions as a first approximation.
        out = []
        for ent in self.state.entities.values():
            if ent.zone == "PLAY" and ent.controller not in (
                None, str(self.local_player)
            ):
                out.append(ent)
        return out

    def _minion(self, ent: Entity) -> MinionView:
        return MinionView(
            entity_id=ent.id,
            card_id=ent.card_id,
            name=ent.name,
            attack=ent.tag_int("ATK"),
            health=ent.tag_int("HEALTH"),
            position=ent.tag_int("ZONE_POSITION"),
            tags=dict(ent.tags),
        )

    def _player_tag_int(self, tag: str) -> Optional[int]:
        for ent in self.state.entities.values():
            if ent.controller == str(self.local_player) and tag in ent.tags:
                try:
                    return int(ent.tags[tag])
                except ValueError:
                    pass
        return None

    def _hero_health(self) -> Optional[int]:
        for ent in self.state.entities.values():
            # CALIBRATE: hero entities carry CARDTYPE=HERO; effective health =
            # HEALTH - DAMAGE.
            if ent.tags.get("CARDTYPE") == "HERO" and ent.controller == str(
                self.local_player
            ):
                h = ent.tag_int(TAG_HEALTH)
                d = ent.tag_int(TAG_DAMAGE) or 0
                if h is not None:
                    return h - d
        return None
