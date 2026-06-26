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
    shop_spells: List[Dict] = field(default_factory=list)     # tavern spells to buy
    hand_spells: List[Dict] = field(default_factory=list)     # targetable spells in hand
    hand: List[MinionView] = field(default_factory=list)
    opponents_seen: List[Dict] = field(default_factory=list)  # last-known enemy boards
    hero_power: Optional[Dict] = None     # {name, card_id, cost, usable}
    anomaly: Optional[str] = None         # active Battlegrounds anomaly name
    level_cost: Optional[int] = None      # discounted gold to tier up right now
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
            "shop_spells": list(self.shop_spells),
            "hand_spells": list(self.hand_spells),
            "hero_power": self.hero_power,
            "anomaly": self.anomaly,
            "level_cost": self.level_cost,
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
        self.player_names: Dict[int, str] = {}   # PlayerID -> battletag
        self.last_opponent_board: List[MinionView] = []  # most recent enemy we fought
        # Tavern-upgrade discount tracking: HS lowers the tier-up cost by 1 for
        # each recruit phase you stay on a tier, so the real cost is below the base
        # (this is what enables 'tier 2 on turn 2'). We count recruit phases and the
        # phase at which the current tier was reached, since the log never prints
        # the discounted cost.
        self._recruit_phases = 0
        self._tier_anchor = 0          # recruit-phase count when current tier reached
        self._anchor_tier: Optional[int] = None

    def feed(self, event: Event) -> None:
        # Hearthstone logs the whole game twice: GameState.* is the authoritative
        # stream, PowerTaskList.* is a task-list echo that repeats CREATE_GAME /
        # FULL_ENTITY / TAG_CHANGE. Feeding both double-applies state and the echo's
        # CREATE_GAME wipes the game mid-stream, so we consume GameState only.
        if event.logger == "PowerTaskList":
            return

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

        # New game: forget the previous game's local player so a fresh lobby
        # (where our PlayerID changes) re-identifies us cleanly.
        if event.kind == "CREATE_GAME":
            self.local_player = None
            self.player_names = {}
            # Reset the tavern-upgrade discount clock so a new game doesn't inherit
            # the previous one's recruit-phase count (which would skew level_cost).
            self._recruit_phases = 0
            self._tier_anchor = 0
            self._anchor_tier = None

        # Player roster line — the reliable way to find the human: the only
        # seat whose GameAccountId hi != 0. PlayerID is the controller id board
        # minions carry, so it's exactly what the snapshot keys on.
        if event.kind == "PLAYER":
            if _safe_int(event.fields.get("hi")):   # non-zero hi == real account
                self.local_player = _safe_int(event.fields.get("player_id"))
            return

        # PlayerID -> battletag. Gold and the hero pointer are logged against the
        # battletag-named player entity, so we need this map to find them.
        if event.kind == "PLAYER_NAME":
            pid = _safe_int(event.fields.get("player_id"))
            name = event.fields.get("name")
            if pid is not None and name:
                self.player_names[pid] = name
            return

        self.state.apply(event)
        if not self.in_bg:
            self._detect_bg()              # …but most clients only reveal BG via cardIds
        prev_phase = self.phase
        self._update_phase(event)
        if prev_phase != Phase.RECRUIT and self.phase == Phase.RECRUIT:
            self._recruit_phases += 1      # a new shopping turn began
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
    # Detect recruit vs combat from DEFINITIVE events, not TURN parity. Turn
    # parity (odd=recruit) holds in some games but is offset in others (anomalies,
    # byes), so it wrongly showed "combat" while the player was shopping. Instead:
    #   * combat  = a minion attacks (BLOCK_START BlockType=ATTACK)
    #   * recruit = the tavern's buy mechanic (TB_BaconShop_DragBuy) is dealt,
    #               which happens at the start of every recruit phase.
    def _update_phase(self, event: Event) -> None:
        if not self.in_bg:
            return

        if event.kind == "RAW" and "BlockType=ATTACK" in (event.text or ""):
            self.phase = Phase.COMBAT
            return

        if event.kind in ("FULL_ENTITY", "SHOW_ENTITY") and event.entity:
            cid = event.entity.card_id or ""
            if "BaconShop_DragBuy" in cid:        # shop is open → recruit
                self.phase = Phase.RECRUIT
                return

        if event.kind in ("TAG", "TAG_CHANGE") and event.tag == TAG_STEP:
            step = (event.value or "").upper()
            if "FINAL" in step or "DONE" in step:
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
        # Board = our minions in PLAY at a real board slot. zonePos 0 is the
        # hero (and tavern fixtures like Bartender Bob / Drag-To-Buy), so the
        # actual minions are zonePos >= 1.
        board = [
            self._minion(e)
            for e in self.state.in_zone("PLAY", pid)
            if (e.tag_int("ZONE_POSITION") or 0) >= 1 and _is_real_minion(e)
        ]
        shop = [self._minion(e) for e in self._shop_entities()]
        shop_spells = self._shop_spells()
        hand = [self._minion(e) for e in self.state.in_zone("HAND", pid)]

        # During combat the enemy board is fully revealed; remember the latest one
        # so recruit-phase positioning advice has a real opponent to optimize
        # against (matters most for attack-order heroes like Al'Akir).
        enemy = self._opponent_board()
        if enemy:
            self.last_opponent_board = enemy
        opponents = ([[m.__dict__ for m in self.last_opponent_board]]
                     if self.last_opponent_board else [])

        notes = []
        if self.local_player is None:
            notes.append("local_player not yet identified")
        return Snapshot(
            game_counter=self.state.game_counter,
            turn=self.state.current_turn,
            phase=self.phase.value,
            tavern_tier=self._tavern_tier(),
            gold=self._gold(),
            hero_health=self._hero_health(),
            board=board,
            shop=shop,
            shop_spells=shop_spells,
            hand_spells=self._hand_spells(),
            hero_power=self._hero_power(),
            anomaly=self._anomaly(),
            level_cost=self._level_cost(),
            hand=hand,
            opponents_seen=opponents,
            notes=notes,
        )

    def _level_cost(self) -> Optional[int]:
        """The CURRENT (discounted) gold to tier up. HS lowers the base cost by 1
        per recruit phase on the tier, so e.g. tier 1→2 is 5 on turn 1 but 4 on
        turn 2 — which is exactly what makes 'tier 2 on turn 2' affordable. The log
        doesn't print this, so we derive it from recruit-phase counts."""
        from .actions import UPGRADE_COST, MAX_TIER
        tier = self._tavern_tier()
        if tier is None or tier >= MAX_TIER:
            return None
        base = UPGRADE_COST.get(tier)
        if base is None:
            return None
        # Keep the anchor current: when the tier goes up, reset the discount clock.
        if self._anchor_tier is None:
            self._anchor_tier = tier
        elif tier > self._anchor_tier:
            self._tier_anchor = self._recruit_phases
            self._anchor_tier = tier
        on_tier = max(1, self._recruit_phases - self._tier_anchor)
        return max(0, base - (on_tier - 1))

    def _hero_power(self) -> Optional[Dict]:
        """The local player's hero power: name, cost, and whether it's usable now
        (active, not exhausted, affordable). Recommendable like any other action.

        Passive / start-of-combat hero powers (e.g. Illidan's Wingmen) can't be
        activated — HAS_ACTIVATE_POWER on the hero/hero-power entity says which, so
        we never tell you to 'use' a passive power."""
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") != "HERO_POWER":
                continue
            if ent.controller != str(self.local_player):
                continue
            # Passive / start-of-combat powers (Illidan's Wingmen) hide their cost
            # (HIDE_COST=1) and have no real COST — you can't click them. Activatable
            # powers (e.g. Marin's) carry a COST. Don't offer "use" on passives.
            if ent.tags.get("HIDE_COST") == "1" or "COST" not in ent.tags:
                return None
            cost = ent.tag_int("COST") or 0
            gold = self._gold()
            usable = (ent.tags.get("EXHAUSTED") not in ("1",)
                      and (gold is None or gold >= cost))
            # Prefer the logged display name; hero powers aren't in the minion KB,
            # so fall back to a clean label rather than a raw cardId.
            name = ent.name
            if not name or name == ent.card_id:
                name = "Hero Power"
            return {
                "name": name,
                "card_id": ent.card_id,
                "cost": cost,
                "usable": bool(usable),
            }
        return None

    def _anomaly(self) -> Optional[str]:
        # Require the cardId to actually be an anomaly (BG##_Anomaly_###) — guards
        # against a mis-tagged/transient entity showing a spell as the anomaly.
        for ent in self.state.entities.values():
            if (ent.tags.get("CARDTYPE") == "BATTLEGROUND_ANOMALY"
                    and "Anomaly" in (ent.card_id or "")):
                return self._display_name(ent.card_id, ent.name)
        return None

    def _hand_spells(self) -> List[Dict]:
        """Targetable tavern spells in your hand (e.g. Tavern Dish Banana = +stats
        to a minion). These are BATTLEGROUND_SPELL entities you control, in HAND.

        Strictly zone==HAND: a spell you already PLAYED moves to PLAY/GRAVEYARD and
        leaves a SETASIDE/pool copy behind, so accepting SETASIDE made the coach
        keep recommending a spell you'd just used ('stuck — already did this'). A
        spell that's truly castable is in your HAND."""
        out = []
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") != "BATTLEGROUND_SPELL":
                continue
            if ent.zone != "HAND" or ent.controller != str(self.local_player):
                continue
            if not ent.card_id or "DragBuy" in ent.card_id:
                continue
            if ent.tag_int("COST") is None:      # not a real, playable spell
                continue
            out.append({
                "name": self._display_name(ent.card_id, ent.name),
                "card_id": ent.card_id,
                "cost": ent.tag_int("COST"),
                "coin": ent.tags.get("COIN_CARD") == "1",   # gold spell, not targeted
            })
        return out

    def _shop_spells(self) -> List[Dict]:
        """Buyable tavern spells in the shop (CARDTYPE=BATTLEGROUND_SPELL).

        Anchored to the *actual tavern row*: a real shop spell sits in zone=PLAY
        under the SAME shop controller as the visible shop minions. Spells that are
        merely set-aside / in the pool (zone=SETASIDE) or that you own — e.g. a
        spellcraft spell like 'Recruit a Trainee' you generated (controller=you,
        played on a minion) — are NOT tavern offerings and must not show as
        'buy spell'. Without a shop minion to anchor the row we return nothing
        rather than risk a phantom spell (the repeated 'wrong spell' reports)."""
        if self.phase != Phase.RECRUIT:
            return []
        shop_controllers = {e.controller for e in self._shop_entities()}
        if not shop_controllers:                 # no anchor → don't guess a spell
            return []
        out = []
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") != "BATTLEGROUND_SPELL":
                continue
            if not ent.card_id or "DragBuy" in ent.card_id:
                continue
            if ent.zone != "PLAY":               # must be in the tavern row
                continue
            if ent.controller not in shop_controllers:   # same row as shop minions
                continue
            out.append({
                "name": self._display_name(ent.card_id, ent.name),
                "card_id": ent.card_id,
                "cost": ent.tag_int("COST"),
            })
        return out

    def _opponent_board(self) -> List[MinionView]:
        """Foreign revealed minions in PLAY — i.e. the board we're fighting. Only
        meaningful during COMBAT (in recruit, foreign minions are the shop)."""
        if self.phase != Phase.COMBAT:
            return []
        out = []
        for ent in self.state.entities.values():
            if (ent.zone == "PLAY" and ent.controller not in (None, str(self.local_player))
                    and ent.tags.get("CARDTYPE") == "MINION" and _is_real_minion(ent)):
                out.append(ent)
        out.sort(key=lambda e: e.tag_int("ZONE_POSITION") or 0)
        return [self._minion(e) for e in out]

    def _shop_entities(self) -> List[Entity]:
        # Shop minions and the opponent's combat board BOTH live in zone=PLAY under
        # a non-local controller, so they're indistinguishable by zone/controller
        # alone. We disambiguate by phase: during RECRUIT the only foreign PLAY
        # minions are the tavern offerings; during COMBAT they're the enemy board
        # (you can't buy then), so the shop is empty. This keeps us from ever
        # recommending a "buy" against an opponent's combat minion.
        if self.phase != Phase.RECRUIT:
            return []
        out = []
        for ent in self.state.entities.values():
            if ent.tags.get(TAG_DUMMY_PLAYER):
                continue
            if ent.tags.get("CARDTYPE") != "MINION":   # skip Bob, spells, trinkets
                continue
            # Crucial discriminator: our tavern's offerings are revealed to the
            # client (they carry a cardId); opponents shop simultaneously and
            # their minions log as hidden (cardId absent). Requiring a cardId
            # keeps the shop to what we can actually see and buy.
            if not ent.card_id:
                continue
            if ent.zone == "PLAY" and ent.controller not in (
                None, str(self.local_player)
            ):
                out.append(ent)
        return sorted(out, key=lambda e: e.tag_int("ZONE_POSITION") or 0)

    def _display_name(self, card_id, name=None):
        """Best display name: the entity's own name, else a name learned from any
        other entity with this cardId (spells/anomalies), else the committed KB,
        else the raw cardId."""
        if name and "UNKNOWN ENTITY" not in name:
            return name
        return (self.state.card_names.get(card_id or "")
                or (_card_name(card_id) if card_id else None) or name)

    def _minion(self, ent: Entity) -> MinionView:
        # Shop minions log without an entityName; resolve a readable name from the
        # cardId so the overlay shows "Southsea Busker", not "BG26_135".
        name = self._display_name(ent.card_id, ent.name)
        return MinionView(
            entity_id=ent.id,
            card_id=ent.card_id,
            name=name,
            attack=ent.tag_int("ATK"),
            health=ent.tag_int("HEALTH"),
            position=ent.tag_int("ZONE_POSITION"),
            tags=dict(ent.tags),
        )

    # --- player / hero entity resolution ----------------------------------
    # Gold, tavern tier and hero health all hang off the player entity, which
    # the log references by battletag (not EntityID). The player carries a
    # HERO_ENTITY tag pointing at its hero — the single source of truth for tier
    # and health, which sidesteps trinkets/candidates that also log a tech level.
    def _player_entity(self) -> Optional[Entity]:
        name = self.player_names.get(self.local_player)
        if not name:
            return None
        for ent in self.state.entities.values():
            if ent.name == name:
                return ent
        return None

    def _hero_entity(self) -> Optional[Entity]:
        pe = self._player_entity()
        if pe is not None:
            hid = pe.tag_int("HERO_ENTITY")
            if hid is not None and hid in self.state.entities:
                return self.state.entities[hid]
        # Fallback: the in-play hero card controlled by us.
        for ent in self.state.entities.values():
            if (ent.controller == str(self.local_player) and ent.zone == "PLAY"
                    and "HERO" in (ent.card_id or "")):
                return ent
        return None

    def _tavern_tier(self) -> Optional[int]:
        hero = self._hero_entity()
        if hero is None:
            return None
        for name in TAG_TAVERN_TIER:
            if name in hero.tags:
                return _safe_int(hero.tags[name])
        return None

    def _gold(self) -> Optional[int]:
        """Spendable gold = RESOURCES - RESOURCES_USED on the player entity."""
        pe = self._player_entity()
        if pe is None:
            return None
        total = pe.tag_int(TAG_RESOURCES)
        if total is None:
            return None
        return total - (pe.tag_int("RESOURCES_USED") or 0)

    def placement(self) -> Optional[int]:
        """Final leaderboard place (1..8) of the local player, if reported."""
        pe = self._player_entity()
        if pe is not None and TAG_PLACEMENT in pe.tags:
            return _safe_int(pe.tags[TAG_PLACEMENT])
        for ent in self.state.entities.values():
            if ent.controller == str(self.local_player) and TAG_PLACEMENT in ent.tags:
                return _safe_int(ent.tags[TAG_PLACEMENT])
        return None

    def _hero_health(self) -> Optional[int]:
        # BG hero survivability = HEALTH + ARMOR - DAMAGE. Heroes start with bonus
        # armor (e.g. Marin 30 health + 12 armor); combat damage eats armor first.
        hero = self._hero_entity()
        if hero is None:
            return None
        h = hero.tag_int(TAG_HEALTH)
        if h is None:
            return None
        armor = hero.tag_int("ARMOR") or 0
        return h + armor - (hero.tag_int(TAG_DAMAGE) or 0)


def _card_name(card_id: str) -> Optional[str]:
    """cardId -> display name via the committed card KB (cached). Falls back to
    the raw id so the overlay still shows something if the card is unknown."""
    try:
        from . import cards
        c = cards.load_kb().get(card_id)
        return c.name if c else card_id
    except Exception:
        return card_id


def _is_real_minion(ent: Entity) -> bool:
    """Keep only actual board minions. Excludes:
    - trinkets / tavern buttons (CARDTYPE=GAME_MODE_BUTTON) that also sit in PLAY,
    - unrevealed placeholders ('UNKNOWN ENTITY [cardType=INVALID]', no cardId)."""
    cardtype = ent.tags.get("CARDTYPE")
    if cardtype and cardtype != "MINION":
        return False
    if not ent.card_id:
        return False
    if ent.name and "UNKNOWN ENTITY" in ent.name:
        return False
    return True


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
