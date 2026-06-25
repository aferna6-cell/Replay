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

# --- Calibrated against twanvl/hearthstone-battlegrounds-simulator log_parser
# and the HearthSim tag conventions (2026-06-24). Items still needing a real
# captured log to confirm are marked # CALIBRATE.
TAG_TAVERN_TIER = ("PLAYER_TECH_LEVEL", "TECH_LEVEL")  # reference uses both
TAG_HEALTH = "HEALTH"                    # hero entity; effective = HEALTH - DAMAGE
TAG_DAMAGE = "DAMAGE"
TAG_RESOURCES = "RESOURCES"              # gold pool  # CALIBRATE: confirm tag name
TAG_STEP = "STEP"                        # GameEntity step; MAIN_READY ~ battle start
TAG_PLACEMENT = "PLAYER_LEADERBOARD_PLACE"  # 1..8 final placement  # CALIBRATE
TAG_DUMMY_PLAYER = "BACON_DUMMY_PLAYER"  # excluded from board/shop scans

# In Battlegrounds the turn counter alternates phases: odd turns are the tavern
# (recruit), even turns are combat. This parity is a more reliable phase signal
# than guessing STEP values, per the reference parser. STEP=MAIN_READY is used
# as a secondary combat-start marker.
STEP_COMBAT_START = "MAIN_READY"


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
        # Battlegrounds detection via scene transition (some clients log it)…
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
        if not self.in_bg:
            self._detect_bg()              # …but most clients only reveal BG via cardIds
        self._update_phase(event)
        self._maybe_detect_local_player()

    def _detect_bg(self) -> None:
        """Detect Battlegrounds from entity cardIds — the tavern infrastructure
        (Bartender Bob, the shop, hero placeholders) all carry 'Bacon', and BG
        minions/heroes use BGS_/BG##_ prefixes. Reliable when the scene line isn't
        logged (verified against a real macOS client log)."""
        for ent in self.state.entities.values():
            cid = ent.card_id or ""
            if "Bacon" in cid or cid.startswith(("BGS_", "BG2", "BG3", "BG_")):
                self.in_bg = True
                if self.phase == Phase.UNKNOWN:
                    self.phase = Phase.HERO_SELECT
                return

    # --- phase machine ----------------------------------------------------
    # Primary signal: BG turn parity (odd=recruit, even=combat). Secondary:
    # STEP=MAIN_READY marks a battle starting. Both per the reference parser.
    def _update_phase(self, event: Event) -> None:
        if not self.in_bg:
            return

        if event.kind in ("TAG", "TAG_CHANGE") and event.tag == "TURN":
            turn = _safe_int(event.value)
            if turn is not None and turn >= 1:
                # Turn 1 is the first tavern; alternates from there.
                self.phase = Phase.RECRUIT if turn % 2 == 1 else Phase.COMBAT
            return

        if event.kind in ("TAG", "TAG_CHANGE") and event.tag == TAG_STEP:
            step = (event.value or "").upper()
            if step == STEP_COMBAT_START or "COMBAT" in step:
                self.phase = Phase.COMBAT
            elif "FINAL" in step or "DONE" in step:
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
            if ent.tags.get(TAG_DUMMY_PLAYER):
                continue
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

    def _player_tag_int(self, tag) -> Optional[int]:
        # tag may be a single name or a tuple of candidate names (the reference
        # parser reads tavern tier from either TECH_LEVEL or PLAYER_TECH_LEVEL).
        candidates = (tag,) if isinstance(tag, str) else tuple(tag)
        for ent in self.state.entities.values():
            if ent.controller != str(self.local_player):
                continue
            for name in candidates:
                if name in ent.tags:
                    try:
                        return int(ent.tags[name])
                    except ValueError:
                        pass
        return None

    def placement(self) -> Optional[int]:
        """Final leaderboard place (1..8) of the local player, if reported."""
        for ent in self.state.entities.values():
            if ent.controller == str(self.local_player) and TAG_PLACEMENT in ent.tags:
                return _safe_int(ent.tags[TAG_PLACEMENT])
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


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
