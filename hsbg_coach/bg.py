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

import re
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
    ACTIVATE = "activate"             # Season 14 Activate keyword on board minion
    DARK_GIFT = "dark_gift"           # Dark Gift discover button (typically 3g)
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

# Dark Gift spells (BG36_MidGameEffect_000tNN) and the enchantments they attach
# (..._000tNNe). Dark Paradox tokens (BG36_360t*, golden BG36_360_Gt*) carry
# HAS_DARK_GIFT natively.
DARK_GIFT_PREFIX = "BG36_MidGameEffect_000t"
DARK_PARADOX_PREFIX = ("BG36_360t", "BG36_360_Gt")   # plain / golden tokens
# Gift enchantments without the gift prefix. Harpy's Talons reuses the
# constructed enchantment; no other gift spell creates it.
DARK_GIFT_ENCHANTMENTS = {"EDR_100t13e": "BG36_MidGameEffect_000t13"}
# Enchantment card = "<spell>e" / "<spell>e2", or Persistent Poet's permanent
# copy of a combat enchantment: "<spell>te" / "<spell>te2". The spell id
# itself does not match.
_GIFT_ENCHANT_SUFFIX = re.compile(r"t?e\d*$")
# Sire Denathrius (the only BG hero with quests; any hero can sell his buddy
# Shady Aristocrat for one) and its skins, e.g.
# BG24_HERO_100_SKIN_A 'Sire Melodious', BG24_HERO_100_SKIN_E 'Boss Denathrius'.
SIRE_HERO_RE = re.compile(r"^BG24_HERO_100(_SKIN_[A-Z0-9]+)?$")
QUEST_REWARD_CARDTYPE = "BATTLEGROUND_QUEST_REWARD"
# Tavern spells show up as BATTLEGROUND_SPELL; accept plain SPELL too.
SPELL_CARDTYPES = ("BATTLEGROUND_SPELL", "SPELL")
BOB_PREFIX = "TB_BaconShopBob"                 # Bartender Bob (and skins): shop owner
BUY_HANDLE_PREFIX = "TB_BaconShop_DragBuy"     # per-slot buy handle (minion / _Spell)
# Unnamed tag on a buy handle holding the shop entity it buys. Unnamed tags
# print as their number in Power.log too.
BUY_HANDLE_TARGET_TAG = "2442"
REROLL_BUTTON_SUFFIX = "Reroll_Button"         # TB_BaconShop_8p_Reroll_Button


@dataclass
class MinionView:
    entity_id: int
    card_id: Optional[str]
    name: Optional[str]
    attack: Optional[int]
    health: Optional[int]
    position: Optional[int]
    tags: Dict[str, str] = field(default_factory=dict)
    dark_gift: Optional[Dict] = None      # {card_id, name} when HAS_DARK_GIFT=1
    buy_cost: Optional[int] = None        # shop only: COST of this slot's buy handle


@dataclass
class Snapshot:
    """Full observable state at a decision point. This is the model's input."""
    game_counter: int
    turn: Optional[int]
    phase: str
    tavern_tier: Optional[int]
    gold: Optional[int]
    hero_health: Optional[int]
    hero_armor: Optional[int] = None      # ARMOR (already included in hero_health)
    board: List[MinionView] = field(default_factory=list)     # your minions in play
    shop: List[MinionView] = field(default_factory=list)      # minions available to buy
    shop_spells: List[Dict] = field(default_factory=list)     # tavern spells to buy
    shop_frozen: bool = False             # any shop minion FROZEN
    hand_spells: List[Dict] = field(default_factory=list)     # targetable spells in hand
    hand: List[MinionView] = field(default_factory=list)
    opponents_seen: List[Dict] = field(default_factory=list)  # last-known enemy boards
    hero_power: Optional[Dict] = None     # {name, card_id, cost, usable}
    # Every other non-passive power in PLAY, same shape and order as the
    # primary. None when there are fewer than two, so a single-power
    # to_dict() stays byte-identical. Omitted from to_dict() in that case.
    hero_powers: Optional[List[Dict]] = None
    activatable: List[Dict] = field(default_factory=list)  # Activate-keyword minions
    dark_gift: Optional[Dict] = None      # {name, card_id, cost, usable, entity_id?}
    anomaly: Optional[str] = None         # active Battlegrounds anomaly name
    level_cost: Optional[int] = None      # discounted gold to tier up right now
    reroll_cost: Optional[int] = None     # refresh button COST (missing COST = 0)
    trinkets: List[Dict] = field(default_factory=list)   # your equipped trinkets
    quests: List[Dict] = field(default_factory=list)     # Sire quests, see _quests()
    opponent_profiles: List[Dict] = field(default_factory=list)  # lobby threats
    hero: Optional[str] = None            # our hero cardId (for the eval net + tribes)
    hero_name: Optional[str] = None
    available_tribes: List[str] = field(default_factory=list)  # lobby tribes
    build_tribe: Optional[str] = None  # soft lean for overlay
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {
            "game_counter": self.game_counter,
            "turn": self.turn,
            "phase": self.phase,
            "tavern_tier": self.tavern_tier,
            "gold": self.gold,
            "hero_health": self.hero_health,
            "hero_armor": self.hero_armor,
            "board": [m.__dict__ for m in self.board],
            "shop": [m.__dict__ for m in self.shop],
            "shop_spells": list(self.shop_spells),
            "shop_frozen": self.shop_frozen,
            "hand_spells": list(self.hand_spells),
            "hero_power": self.hero_power,
            "activatable": list(self.activatable),
            "dark_gift": self.dark_gift,
            "anomaly": self.anomaly,
            "level_cost": self.level_cost,
            "reroll_cost": self.reroll_cost,
            "trinkets": list(self.trinkets),
            "quests": list(self.quests),
            "opponent_profiles": list(self.opponent_profiles),
            "hero": self.hero,
            "hero_name": self.hero_name,
            "available_tribes": list(self.available_tribes),
            "build_tribe": self.build_tribe,
            "hand": [m.__dict__ for m in self.hand],
            "opponents_seen": self.opponents_seen,
            "notes": self.notes,
        }
        # Only a second (or later) non-passive power adds a key. One power,
        # or none, keeps the historical dict byte-for-byte.
        if self.hero_powers:
            items = list(d.items())
            at = next(i for i, (k, _) in enumerate(items) if k == "hero_power")
            items.insert(at + 1, ("hero_powers", list(self.hero_powers)))
            return dict(items)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Snapshot":
        """Inverse of to_dict() (also after a JSON round trip)."""
        kw = dict(d)
        for key in ("board", "shop", "hand"):
            kw[key] = [MinionView(**m) for m in kw.get(key) or []]
        return cls(**kw)


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
        self.opponents: Dict[str, Dict] = {}   # controller -> latest profile we've seen
        self._subset_by_entity: Dict[int, set] = {}  # entity -> BACON_SUBSET_* tokens
        self.available_tribes: List[str] = []
        self.manual_tribe_priors: Dict[str, float] = {}
        # Per-event combat capture of opponent boards/profiles (see feed()).
        # Batch replay builders may switch it off: it dominates their runtime.
        self.track_opponents = True
        # entity id -> Dark Gift spell id. Recorded when a gift enchantment is
        # attached (any zone) and inherited across COPIED_FROM while
        # HAS_DARK_GIFT stays set. A hand copy often keeps the flag after the
        # enchantment entities were already removed (see _track_dark_gift).
        self._dark_gift_by_entity: Dict[int, str] = {}

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
            self.opponents = {}            # forget last game's lobby
            self.last_opponent_board = []  # forget last fight (odds + positioning)
            self._subset_by_entity = {}
            self.available_tribes = []
            self.manual_tribe_priors = {}
            self._dark_gift_by_entity = {}

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
        self._track_dark_gift(event)
        # Lobby tribe detection via BACON_SUBSET_* tags on race-banner entities.
        if event.kind in ("TAG", "TAG_CHANGE") and event.tag and event.tag.startswith("BACON_SUBSET_"):
            eid = None
            if event.entity is not None:
                eid = getattr(event.entity, "id", None) or getattr(event.entity, "entity_id", None)
            if eid is None:
                eid = getattr(self.state, "current_entity_id", None)
            if eid is not None:
                self._track_subset_tag(eid, event.tag, event.value or "0")
        if not self.in_bg:
            self._detect_bg()              # …but most clients only reveal BG via cardIds
        prev_phase = self.phase
        self._update_phase(event)
        if prev_phase != Phase.RECRUIT and self.phase == Phase.RECRUIT:
            self._recruit_phases += 1      # a new shopping turn began
        self._maybe_detect_local_player()
        # Remember the enemy board as combat events arrive — do NOT wait for
        # snapshot(). LiveCoach replays from_start faster than the overlay polls,
        # so without this, recruit-phase odds/positioning see an empty opponents_seen.
        if self.phase == Phase.COMBAT and self.track_opponents:
            self._remember_combat_boards()

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

        # HSReplay XML Options block: the game only offers the local player
        # options while shopping, so this is a definitive recruit signal.
        if event.kind == "OPTIONS":
            self.phase = Phase.RECRUIT
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
        buy_costs = self._buy_costs()
        shop = []
        for e in self._shop_entities():
            m = self._minion(e)
            m.buy_cost = buy_costs.get(e.id)
            shop.append(m)
        shop_spells = self._shop_spells(buy_costs)
        hand = [self._minion(e) for e in self.state.in_zone("HAND", pid)]

        # During combat the enemy board is fully revealed; remember the latest one
        # so recruit-phase positioning advice has a real opponent to optimize
        # against (matters most for attack-order heroes like Al'Akir).
        enemy = self._opponent_board()
        if enemy:
            self.last_opponent_board = enemy
        opponents = ([[m.__dict__ for m in self.last_opponent_board]]
                     if self.last_opponent_board else [])

        # Profile every opponent we can see this combat and keep the latest read,
        # so the recommender knows the whole lobby's threats (heroes, tribes,
        # Divine Shields, strength), not just the last board we fought.
        if self.phase == Phase.COMBAT:
            self._update_opponents()

        notes = []
        if self.local_player is None:
            notes.append("local_player not yet identified")
        hero_ent = self._hero_entity()
        powers = self._hero_powers()
        return Snapshot(
            game_counter=self.state.game_counter,
            turn=self.state.current_turn,
            phase=self.phase.value,
            tavern_tier=self._tavern_tier(),
            gold=self._gold(),
            hero_health=self._hero_health(),
            hero_armor=hero_ent.tag_int("ARMOR") if hero_ent is not None else None,
            board=board,
            shop=shop,
            shop_spells=shop_spells,
            shop_frozen=any(m.tags.get("FROZEN") == "1" for m in shop),
            hand_spells=self._hand_spells(),
            hero_power=powers[0] if powers else None,
            hero_powers=powers if len(powers) > 1 else None,
            activatable=self._activatable(),
            dark_gift=self._dark_gift(),
            anomaly=self._anomaly(),
            level_cost=self._level_cost(),
            reroll_cost=self._reroll_cost(),
            trinkets=self._trinkets(),
            quests=self._quests(),
            hand=hand,
            opponents_seen=opponents,
            opponent_profiles=list(self.opponents.values()),
            hero=self._our_hero()[0],
            hero_name=self._our_hero()[1],
            available_tribes=list(self.available_tribes),
            notes=notes,
        )

    def _our_hero(self):
        """(cardId, name) of OUR hero — feeds the eval net (hero-specific board
        value) and the hero→tribe steering. None until the hero is in play."""
        h = self._hero_entity()
        if h is None or not h.card_id or "HERO" not in h.card_id:
            return None, None
        return h.card_id, self._display_name(h.card_id, h.name)

    _kb_cache = None

    @classmethod
    def _kb(cls):
        if BGTracker._kb_cache is None:
            try:
                from . import cards
                BGTracker._kb_cache = cards.load_kb()
            except Exception:
                BGTracker._kb_cache = {}
        return BGTracker._kb_cache

    def _update_opponents(self) -> None:
        """Merge this combat's opponent reads into the persistent lobby map."""
        try:
            from .opponents import build_profiles
            fresh = build_profiles(self.state.entities, self.local_player, self._kb())
            for ctrl, prof in fresh.items():
                self.opponents[ctrl] = prof          # keep the latest read per seat
        except Exception:
            pass

    def _level_cost(self) -> Optional[int]:
        """The CURRENT (discounted) gold to tier up.

        Ground truth first: the tavern-up button (TB_BaconShopTechUp0N_Button) carries
        the live COST the game already discounted, so we read that and never suggest a
        tier-up you can't afford. Only if the button isn't visible do we fall back to
        deriving the discount from recruit-phase counts (base cost minus 1 per turn on
        the tier)."""
        from .actions import UPGRADE_COST, MAX_TIER
        tier = self._tavern_tier()
        if tier is None or tier >= MAX_TIER:
            return None
        target = tier + 1
        button = None
        for ent in self.state.entities.values():
            if f"TechUp0{target}_Button" in (ent.card_id or "") \
                    and ent.controller == str(self.local_player):
                button = ent
                if ent.zone == "PLAY":           # the active upgrade button
                    break
        if button is not None:
            c = button.tag_int("COST")
            if c is not None:
                return c
        base = UPGRADE_COST.get(tier)
        if base is None:
            return None
        if self._anchor_tier is None:            # keep the discount clock current
            self._anchor_tier = tier
        elif tier > self._anchor_tier:
            self._tier_anchor = self._recruit_phases
            self._anchor_tier = tier
        on_tier = max(1, self._recruit_phases - self._tier_anchor)
        return max(0, base - (on_tier - 1))

    def _hero_power_view(self, ent: Entity, gold: Optional[int]) -> Dict:
        """One clickable hero power. A missing COST tag means 0: zero-valued
        tags are not logged, so free powers must not be dropped."""
        cost = ent.tag_int("COST") or 0
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
            "entity_id": ent.id,
        }

    def _hero_powers(self) -> List[Dict]:
        """Every non-passive hero power we control in PLAY.

        Passive / start-of-combat powers (Illidan's Wingmen, Morchie's
        Warped Conflux, Drek'Thar) hide their cost (HIDE_COST=1). Skip each
        one and keep scanning: a passive first power must not hide a second
        power the player can click (Genn, a Timewarp spell, a trinket, ...).
        There is no card-id allowlist. Whatever put the entity in PLAY counts.

        Order: usable first, then lowest ZONE_POSITION, then lowest entity id.
        The primary ``_hero_power`` is the first entry. A missing ZONE is
        tolerated for sparse live logs; old / offered powers in SETASIDE or
        REMOVEDFROMGAME are not powers we can click."""
        gold = self._gold()
        found = []
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") != "HERO_POWER":
                continue
            if ent.controller != str(self.local_player):
                continue
            if ent.zone not in (None, "PLAY"):
                continue
            if ent.tags.get("HIDE_COST") == "1":
                continue
            found.append(ent)

        def sort_key(ent: Entity):
            view = self._hero_power_view(ent, gold)
            pos = ent.tag_int("ZONE_POSITION") or 0
            return (0 if view["usable"] else 1, pos, ent.id)

        found.sort(key=sort_key)
        return [self._hero_power_view(ent, gold) for ent in found]

    def _hero_power(self) -> Optional[Dict]:
        """The hero power to offer as the primary action.

        Usable non-passive powers come first, then the lowest ZONE_POSITION,
        then the lowest entity id. None when every in-play power is passive
        or none is in PLAY."""
        powers = self._hero_powers()
        return powers[0] if powers else None


    def _track_subset_tag(self, entity_id, tag: str, value: str) -> None:
        """Accumulate BACON_SUBSET_* tags; lobby tribes = union of multi-subset ents."""
        if not tag.startswith("BACON_SUBSET_"):
            return
        token = tag[len("BACON_SUBSET_"):]
        bucket = self._subset_by_entity.setdefault(int(entity_id), set())
        if str(value) == "1":
            bucket.add(token)
        else:
            bucket.discard(token)
        # Recompute lobby: union of entities that carry 3+ subset flags (race banners /
        # pool markers). Fall back to union of all subsets if none qualify yet.
        from .tribe_policy import SUBSET_TO_TRIBE, filter_lobby_tribes, canonicalize
        multi = [s for s in self._subset_by_entity.values() if len(s) >= 3]
        tokens = set()
        for s in (multi or list(self._subset_by_entity.values())):
            tokens |= set(s)
        tribes = []
        for tok in tokens:
            name = SUBSET_TO_TRIBE.get(tok.upper()) or canonicalize(tok)
            if name:
                tribes.append(name)
        self.available_tribes = filter_lobby_tribes(tribes)

    def set_manual_tribe_priors(self, priors: Dict[str, float]) -> None:
        """Lobby-start manual first%/weights (e.g. Aberration before HSReplay has data)."""
        self.manual_tribe_priors = {str(k): float(v) for k, v in (priors or {}).items()}

    def _activatable(self) -> List[Dict]:
        """Board minions with the Season 14 Activate keyword that can be clicked.

        Calibrated (Power.log 2026-09): nearly every recruit minion sets
        HAS_ACTIVATE_POWER=1 (buy/sell interactability), including shop UI
        (Refresh, Freeze, DragBuy/DragSell). Do NOT treat that tag alone — or a
        POWER option with error=NONE (sell targets also appear) — as Activate.

        Require ALL of:
          * friendly board minion (PLAY, zonePos>=1, real BG*/BGS* minion)
          * HAS_ACTIVATE_POWER=1
          * cardId in the Activate-keyword allowlist (card text ``Activate (N)``)
        Cost: TAG_SCRIPT_DATA_NUM_1 when present, else COST.
        """
        from .activate_cards import is_activate_minion
        out: List[Dict] = []
        if self.local_player is None:
            return out
        gold = self._gold()
        for ent in self.state.in_zone("PLAY", self.local_player):
            if not _is_real_minion(ent):
                continue
            cid = ent.card_id or ""
            # Shop / hero / button chrome — never Activate targets.
            if cid.startswith("TB_BaconShop") or "Button" in cid:
                continue
            if not (cid.startswith("BG") or cid.startswith("BGS_")):
                continue
            if (ent.tag_int("ZONE_POSITION") or 0) < 1:
                continue
            if ent.tags.get("HAS_ACTIVATE_POWER") != "1":
                continue
            # Critical: ordinary board/shop minions also carry HAS_ACTIVATE_POWER.
            if not is_activate_minion(cid, ent.name):
                continue
            cost = ent.tag_int("TAG_SCRIPT_DATA_NUM_1")
            if cost is None:
                cost = ent.tag_int("COST") or 0
            exhausted = ent.tags.get("EXHAUSTED") in ("1",)
            usable = (not exhausted and (gold is None or gold >= cost))
            out.append({
                "name": self._display_name(ent.card_id, ent.name),
                "card_id": ent.card_id,
                "entity_id": ent.id,
                "cost": int(cost),
                "usable": bool(usable),
            })
        return out

    def _dark_gift(self) -> Optional[Dict]:
        """Dark Gift discover button (Aberration / Season 14): typically 3 gold.

        Calibrated (Power.log 2026-09): button cardId is BG36_Button_DarkGift,
        entityName "Dark Discovery" (NOT TB_BaconShop_DarkGift_Button). Related
        chrome/tags: HAS_DARK_GIFT, DARK_GIFT_ENTITY, BG36_MidGameEffect_*.

        DebugPrintOptions legality: error=NONE when clickable; blocked with
        REQ_ENOUGH_MANA or REQ_NOT_EXHAUSTED_ACTIVATE. Equivalent without Options:
        EXHAUSTED!=1, BACON_DARK_GIFT_PRESSABLE_VFX!=0 when present, gold>=cost.
        """
        if self.local_player is None:
            return None
        for ent in self.state.entities.values():
            if ent.controller != str(self.local_player):
                continue
            if ent.zone != "PLAY":
                continue
            cid = ent.card_id or ""
            name = ent.name or ""
            # MidGameEffect_* are gift enchantments/chrome, not the button.
            if cid.startswith("BG36_MidGameEffect"):
                continue
            name_l = name.lower().replace("-", " ")
            is_button = (
                cid == "BG36_Button_DarkGift"
                or name_l.strip() == "dark discovery"
                or cid == "TB_BaconShop_DarkGift_Button"  # legacy / older builds
                or ("darkgift" in cid.lower() and "button" in cid.lower())
            )
            if not is_button:
                blob = f"{name} {cid}".lower().replace("-", "").replace(" ", "")
                if not (
                    ("dark" in blob and "gift" in blob and "button" in blob)
                    or (name_l.strip() == "dark gift")
                ):
                    continue
            cost = ent.tag_int("COST")
            if cost is None:
                cost = 3
            gold = self._gold()
            turn = self.state.current_turn
            exhausted = ent.tags.get("EXHAUSTED") in ("1",)
            locked = ent.tags.get("LOCK_VISUAL") == "1"
            # VFX=0 tracks REQ_NOT_EXHAUSTED_ACTIVATE; VFX=1 with low gold → REQ_ENOUGH_MANA.
            vfx = ent.tags.get("BACON_DARK_GIFT_PRESSABLE_VFX")
            pressable = vfx is None or vfx == "1"
            usable = (
                not exhausted and not locked and pressable
                and (turn is None or turn >= 3)
                and (gold is None or gold >= cost)
            )
            return {
                "name": name or "Dark Discovery",
                "card_id": cid or "BG36_Button_DarkGift",
                "cost": int(cost),
                "usable": bool(usable),
                "entity_id": ent.id,
            }
        return None

    def _trinkets(self) -> List[Dict]:
        """Your equipped trinkets (CARDTYPE=BATTLEGROUND_TRINKET, in PLAY, yours).
        They shape your whole game — e.g. a spell-reward trinket means you should
        buy/play more tavern spells — so the recommender needs to see them."""
        out, seen = [], set()
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") != "BATTLEGROUND_TRINKET":
                continue
            if ent.controller != str(self.local_player):
                continue
            # An equipped trinket lives in your trinket zone (SETASIDE in BG) and
            # surfaces to PLAY when it triggers — accept either. Real trinkets are
            # MagicItem cards; skip the empty slot placeholders (BG30_Trinket_1st /
            # "Lesser Trinket" / "Greater Trinket") and the in-game graveyard/removed.
            if ent.zone not in ("PLAY", "SETASIDE"):
                continue
            if "MagicItem" not in (ent.card_id or ""):
                continue
            if ent.card_id in seen:
                continue
            seen.add(ent.card_id)
            out.append({"name": self._display_name(ent.card_id, ent.name),
                        "card_id": ent.card_id})
        return out

    def _quests(self) -> List[Dict]:
        """Your quests (Sire Denathrius). Verified on Firestone replays:

        * Active quest: CARDTYPE=SPELL with QUEST=1, yours, in the SECRET zone.
          QUEST_PROGRESS / QUEST_PROGRESS_TOTAL are progress / goal, and
          TAG_SCRIPT_DATA_ENT_1 points at the (SETASIDE) reward entity. Offered
          quests sit in SETASIDE and are skipped until picked.
        * On completion the quest leaves SECRET (SETASIDE -> REMOVEDFROMGAME,
          its tags reset to card defaults) and, for hero-power quests, a
          BATTLEGROUND_QUEST_REWARD entity whose CREATOR is the quest appears in
          PLAY. That reward stands in for the quest: completed=True, with
          progress/goal None (the reset tags no longer carry them).
        Buddy quests reward a Coin Pouch spell instead, so they simply drop out
        when completed. Each entry: {entity_id, card_id, name, progress, goal,
        reward_card_id, source_card_id, completed}."""
        out = []
        me = str(self.local_player)
        entities = self.state.entities
        for ent in entities.values():
            if ent.controller != me:
                continue
            if ent.tags.get("QUEST") == "1" and ent.zone == "SECRET":
                reward = entities.get(ent.tag_int("TAG_SCRIPT_DATA_ENT_1") or -1)
                out.append({"entity_id": ent.id, "card_id": ent.card_id,
                            "name": self._display_name(ent.card_id, ent.name),
                            "progress": ent.tag_int("QUEST_PROGRESS") or 0,
                            "goal": ent.tag_int("QUEST_PROGRESS_TOTAL"),
                            "reward_card_id": reward.card_id if reward is not None else None,
                            "source_card_id": self.state.card_id_of(ent.tag_int("CREATOR")),
                            "completed": False})
            elif ent.tags.get("CARDTYPE") == QUEST_REWARD_CARDTYPE and ent.zone == "PLAY":
                quest = entities.get(ent.tag_int("CREATOR") or -1)
                if quest is None or quest.tags.get("QUEST") != "1":
                    continue
                out.append({"entity_id": quest.id, "card_id": quest.card_id,
                            "name": self._display_name(quest.card_id, quest.name),
                            "progress": None, "goal": None,
                            "reward_card_id": ent.card_id,
                            "source_card_id": self.state.card_id_of(quest.tag_int("CREATOR")),
                            "completed": True})
        out.sort(key=lambda q: q["entity_id"])
        return out

    def _gift_spell_id(self, card_id: Optional[str]) -> Optional[str]:
        """Gift spell id for a spell or enchantment card.

        ``BG36_MidGameEffect_000t64e`` / ``e2`` are the spell's enchantments.
        ``...t64te`` / ``te2`` are Persistent Poet's permanent copies of those
        enchantments ("Adjacent Dragons permanently keep Bonus Keywords and
        stats gained in combat"). Both name the spell ``...000t64``. A card
        that is not a gift spell returns None; the spell id itself is unchanged.
        """
        if not card_id or not card_id.startswith(DARK_GIFT_PREFIX):
            return None
        return _GIFT_ENCHANT_SUFFIX.sub("", card_id)

    def _enchantment_gift_id(self, ent: Entity) -> Optional[str]:
        """Gift spell named by an enchantment entity, or None."""
        cid = ent.card_id or ""
        gift = self._gift_spell_id(cid)
        if gift:
            return gift
        if cid in DARK_GIFT_ENCHANTMENTS:
            return DARK_GIFT_ENCHANTMENTS[cid]
        return self._gift_spell_id(self.state.card_id_of(ent.tag_int("CREATOR")))

    def _track_dark_gift(self, event: Event) -> None:
        """Remember a gift when its enchantment is attached, and copy that
        memory along ``COPIED_FROM_ENTITY_ID``.

        Timewarped Radio Star (``BG34_Giant_330``: "Get a copy of the enemy
        minion that killed this with full Health and enchantments") copies
        after death has already removed the killer's enchantments. The hand
        copy still gets ``HAS_DARK_GIFT=1`` (b138f295, Persistent Poet 9052,
        copied from 9051 copied from 8958) but no gift enchantment is ever
        attached to it. The identity is the enchantment that was on the
        ancestor (``BG36_MidGameEffect_000t64te``). ``COPIED_FROM`` on the
        intermediate is later cleared, so this has to be recorded at the tag
        change. Ids are not reused; the map survives re-dumps.
        """
        if event.kind not in ("TAG", "TAG_CHANGE", "SHOW_ENTITY", "FULL_ENTITY"):
            return
        eid = (self.state._last_block_entity if event.kind == "TAG"
               else (event.entity.id if event.entity is not None else None))
        ent = self.state.entities.get(eid) if eid is not None else None
        if ent is None:
            return
        host = ent.tag_int("ATTACHED")
        if host:
            gift = self._enchantment_gift_id(ent)
            if gift:
                self._dark_gift_by_entity.setdefault(host, gift)
        if (event.tag in ("COPIED_FROM_ENTITY_ID", "HAS_DARK_GIFT")
                and ent.tags.get("HAS_DARK_GIFT") == "1"
                and ent.id not in self._dark_gift_by_entity):
            src = ent.tag_int("COPIED_FROM_ENTITY_ID") or 0
            seen = set()
            while src and src not in seen:
                seen.add(src)
                gift = self._dark_gift_by_entity.get(src)
                if gift:
                    self._dark_gift_by_entity[ent.id] = gift
                    return
                parent = self.state.entities.get(src)
                src = parent.tag_int("COPIED_FROM_ENTITY_ID") if parent else 0

    def _minion_dark_gift(self, ent: Entity) -> Optional[Dict]:
        """The Dark Gift carried by a HAS_DARK_GIFT minion, as {card_id, name}.

        Resolution order (verified on Firestone replays of patch 36.6):
          1. DARK_GIFT_ENTITY -> that entity's card (set while offered by Dark
             Discovery; dropped when the minion is re-created after combat);
          2. an enchantment in PLAY attached to the minion whose own card, or
             whose CREATOR's card (also after a re-dump dropped it), is a
             BG36_MidGameEffect_000t* gift, or a known gift enchantment
             (DARK_GIFT_ENCHANTMENTS); the enchantment follows the minion
             through combat and triples. Poet permanent copies (``te`` /
             ``te2``) name the same spell as ``e`` / ``e2``;
          3. Dark Paradox tokens (BG36_360t*, BG36_360_Gt*) are their own gift;
          4. a gift remembered when an enchantment was attached to this minion,
             or inherited from the minion it was copied from (step 2 misses a
             hand copy whose enchantments were removed before the copy).
        Returns None for ungifted minions or an unresolvable gift."""
        if ent.tags.get("HAS_DARK_GIFT") != "1":
            return None
        entities = self.state.entities
        raw = self.state.card_id_of(ent.tag_int("DARK_GIFT_ENTITY"))
        gift = self._gift_spell_id(raw) or raw
        if gift is None:
            me = str(ent.id)
            for e in entities.values():
                if (e.tags.get("ATTACHED") != me or e.zone != "PLAY"
                        or e.tags.get("CARDTYPE") != "ENCHANTMENT"):
                    continue
                gift = self._enchantment_gift_id(e)
                if gift:
                    break
        if gift is None and (ent.card_id or "").startswith(DARK_PARADOX_PREFIX):
            gift = ent.card_id
        if gift is None:
            gift = self._dark_gift_by_entity.get(ent.id)
        elif ent.id not in self._dark_gift_by_entity:
            self._dark_gift_by_entity[ent.id] = gift
        if gift is None:
            return None
        return {"card_id": gift, "name": self._display_name(gift)}

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
        to a minion). Spells you control in HAND: in-hand tavern spells are
        CARDTYPE=SPELL (BATTLEGROUND_SPELL is also accepted). A missing COST
        means 0 (zero tags are not logged).

        Strictly zone==HAND: a spell you already PLAYED moves to PLAY/GRAVEYARD and
        leaves a SETASIDE/pool copy behind, so accepting SETASIDE made the coach
        keep recommending a spell you'd just used ('stuck — already did this'). A
        spell that's truly castable is in your HAND."""
        out = []
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") not in SPELL_CARDTYPES:
                continue
            if ent.zone != "HAND" or ent.controller != str(self.local_player):
                continue
            if not ent.card_id or "DragBuy" in ent.card_id:
                continue
            out.append({
                "name": self._display_name(ent.card_id, ent.name),
                "card_id": ent.card_id,
                "cost": ent.tag_int("COST") or 0,
                "coin": ent.tags.get("COIN_CARD") == "1",   # gold spell, not targeted
                "entity_id": ent.id,
            })
        return out

    def _shop_spells(self, buy_costs: Optional[Dict[int, int]] = None) -> List[Dict]:
        """Buyable tavern spells in the shop (CARDTYPE BATTLEGROUND_SPELL or SPELL).

        Anchored to the *actual tavern row*: a real shop spell sits in zone=PLAY
        under the shop's controller. Spells that are merely set-aside / in the
        pool (zone=SETASIDE) or that you own (e.g. a spellcraft spell you
        generated) are NOT tavern offerings. The shop controller is Bartender
        Bob's (TB_BaconShopBob*, in PLAY, not ours) or any shop minion's, so a
        shop showing only spells still counts. With no anchor we return nothing
        rather than guess."""
        if self.phase != Phase.RECRUIT:
            return []
        shop_controllers = ({e.controller for e in self._shop_entities()}
                            | self._bob_controllers())
        if not shop_controllers:                 # no anchor: don't guess a spell
            return []
        if buy_costs is None:
            buy_costs = self._buy_costs()
        out = []
        for ent in self.state.entities.values():
            if ent.tags.get("CARDTYPE") not in SPELL_CARDTYPES:
                continue
            if not ent.card_id or "DragBuy" in ent.card_id:
                continue
            if ent.zone != "PLAY":               # must be in the tavern row
                continue
            if ent.controller not in shop_controllers:   # the shop's row
                continue
            out.append({
                "name": self._display_name(ent.card_id, ent.name),
                "card_id": ent.card_id,
                "cost": ent.tag_int("COST"),
                "entity_id": ent.id,
                "buy_cost": buy_costs.get(ent.id),
                "_pos": ent.tag_int("ZONE_POSITION") or 0,
            })
        out.sort(key=lambda d: (d["_pos"], d["entity_id"]))
        for d in out:
            del d["_pos"]
        return out

    def _bob_controllers(self) -> set:
        """Controller id(s) of Bartender Bob in PLAY (not ours): the shop's owner."""
        me = str(self.local_player)
        return {e.controller for e in self.state.entities.values()
                if (e.card_id or "").startswith(BOB_PREFIX) and e.zone == "PLAY"
                and e.controller not in (None, me)}

    def _buy_costs(self) -> Dict[int, int]:
        """Shop entity id -> gold to buy it, from our per-slot buy handles
        (TB_BaconShop_DragBuy / _DragBuy_Spell in PLAY; tag 2442 = the shop
        entity). Usually 3 for minions; effects lower it. Missing COST = 0."""
        me = str(self.local_player)
        out: Dict[int, int] = {}
        for e in self.state.entities.values():
            if (e.controller == me and e.zone == "PLAY"
                    and (e.card_id or "").startswith(BUY_HANDLE_PREFIX)):
                target = e.tag_int(BUY_HANDLE_TARGET_TAG)
                if target:
                    out[target] = e.tag_int("COST") or 0
        return out

    def _reroll_cost(self) -> Optional[int]:
        """COST of our refresh button in PLAY (missing COST = 0: free refresh)."""
        me = str(self.local_player)
        for e in self.state.entities.values():
            if (e.controller == me and e.zone == "PLAY"
                    and (e.card_id or "").endswith(REROLL_BUTTON_SUFFIX)):
                return e.tag_int("COST") or 0
        return None

    def _remember_combat_boards(self) -> None:
        """Capture the revealed enemy board + lobby profiles during combat.

        Called from feed() so a from-start log replay (or a missed overlay poll)
        still leaves last_opponent_board / opponents populated for the next
        recruit phase — that is what combat odds and positioning need.
        """
        enemy = self._opponent_board()
        if enemy:
            self.last_opponent_board = enemy
        try:
            self._update_opponents()
        except Exception:
            pass

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
            dark_gift=self._minion_dark_gift(ent),
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
        """Spendable gold = RESOURCES - RESOURCES_USED + TEMP_RESOURCES on the
        player entity (TEMP_RESOURCES = this-turn-only gold, e.g. from spells)."""
        pe = self._player_entity()
        if pe is None:
            return None
        total = pe.tag_int(TAG_RESOURCES)
        if total is None:
            return None
        return (total - (pe.tag_int("RESOURCES_USED") or 0)
                + (pe.tag_int("TEMP_RESOURCES") or 0))

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
        c = BGTracker._kb().get(card_id)     # cached; load_kb() re-reads the file
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
