"""Live coaching: tail the game log in the background, advise in the foreground.

The overlay runs Tk's mainloop on the main thread; the log is a blocking tail, so
a background thread feeds the tracker while the overlay polls `frame()` for the
current snapshot + ranked recommendations. Advice is recomputed only when the
board/shop/gold actually change (cheap key check), so the eval net runs a few
times per turn, not every poll tick.

The same background thread records your games (state → action → final placement)
via the recorder, so every game you play becomes training data for the eval net
— the continual-learning loop. (Mirrors the recording in cli `_drive`; kept
separate so the live/threaded path doesn't disturb the proven offline path.)
"""

import threading
from typing import List, Optional, Tuple

from . import cards
from .advisor import advise_actions
from .bg import BGTracker, Phase, ActionType
from .board_value import get_scorer
from .economy import HeroContext
from .parser import parse_line
from .tail import tail_lines


def advice_lines(snapshot: dict, kb, scorer=None,
                 hero_ctx: Optional[HeroContext] = None, top: int = 6) -> List[str]:
    """Ranked one-line recommendations for a snapshot, best first — scored by
    expected FINAL placement (whole-game value), so the future is accounted for.

    Returns moves whenever you can act (the recruit phase) — even before the shop
    is parsed, roll / tier up / hero power / end are always legal, so there's
    always a next move to show (fixes the gap right after combat). Empty only in
    combat/hero-select, where the choice path handles it. Pure + synchronous."""
    if snapshot.get("phase") not in ("recruit", "unknown"):
        return []
    from .game_value import rank_actions
    recs, _ = rank_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer)
    out = []
    # Free cards that landed in your hand (often generated during combat) are
    # usually a play-now: a minion to drop, or a Magnetic mech to fuse. Lead with
    # those, then the spell-on-minion advice.
    for line in _hand_play_lines(snapshot, kb):
        out.append(line)
    # A targetable spell in hand (e.g. Tavern Dish Banana) usually wants playing
    # now — lead with where to put it.
    for line in _hand_spell_lines(snapshot):
        out.append(line)
    # Show the *why* (synergy / tribe / positioning order / sell-for-room / tech
    # caveat) next to each move — that reasoning is the point, not just the verb.
    for r in recs[:top]:
        line = f"{r.action.describe()} (finish {r.placement:.1f})"
        if r.reason:
            line += f" — {r.reason}"
        out.append(line)
    return out


_MAX_BG_BOARD = 7


def _hand_play_lines(snapshot, kb) -> List[str]:
    """'Play <minion> from hand' / 'Magnetize <mech> onto <host>' for each free
    minion sitting in your hand (e.g. one a combat effect generated). Magnetic
    mechs prefer fusing onto a board mech (no slot used, buff protected)."""
    hand = snapshot.get("hand") or []
    minions = [m for m in hand if _is_hand_minion(m)]
    if not minions:
        return []
    from .magnetize import is_magnetic, best_magnetize_target
    board = snapshot.get("board", []) or []
    full = len(board) >= _MAX_BG_BOARD
    out = []
    for m in minions:
        name = m.get("name") or m.get("card_id") or "minion"
        if is_magnetic(m, kb):
            tgt = best_magnetize_target(board, kb)
            if tgt is not None:
                host, why = tgt
                hname = host.get("name") if isinstance(host, dict) else getattr(host, "name", None)
                out.append(f"Magnetize {name} onto {hname or 'your best mech'} — {why}")
                continue                                 # fusing uses no board slot
        # Choose-One battlecry (e.g. Intrepid Botanist → Pristine Lilies / Giant
        # Dewdrop): flag that a pick is coming so the user knows to choose the half
        # that fits their board, not just drop the body.
        choose = " · Choose One — take the half that fits your board" if _is_choose_one(m, kb) else ""
        if full:
            out.append(f"Play {name} from hand — sell your weakest first (board is full){choose}")
        else:
            out.append(f"Play {name} from hand — free body, take the tempo{choose}")
    return out


def _is_choose_one(m, kb) -> bool:
    """True if this minion has a Choose-One battlecry (per card knowledge)."""
    if not kb:
        return False
    cid = m.get("card_id") if isinstance(m, dict) else getattr(m, "card_id", None)
    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
    ck = (kb.get(cid) if cid else None)
    if ck is None:
        ck = next((c for c in kb.values() if c.name == name), None)
    return bool(ck and ck.has("CHOOSE_ONE"))


def _is_hand_minion(m) -> bool:
    """A real, playable minion in hand (not a spell / the buy mechanic)."""
    tags = (m.get("tags") if isinstance(m, dict) else getattr(m, "tags", None)) or {}
    ctype = tags.get("CARDTYPE")
    if ctype and ctype != "MINION":
        return False
    cid = m.get("card_id") if isinstance(m, dict) else getattr(m, "card_id", None)
    if not cid or "DragBuy" in cid:
        return False
    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
    if name and "UNKNOWN ENTITY" in name:
        return False
    return True


def _hand_spell_lines(snapshot) -> List[str]:
    """'Play <spell> on <minion>' for each targetable spell in hand."""
    spells = snapshot.get("hand_spells") or []
    if not spells:
        return []
    from .spell_target import best_buff_target, is_targeted
    pick = best_buff_target(snapshot.get("board", []))
    out = []
    for sp in spells:
        name = sp.get("name") or "spell"
        if not is_targeted(sp):
            out.append(f"Play {name}")               # gold/utility — no minion target
        elif pick is not None:
            target, why = pick
            tname = target.get("name") if isinstance(target, dict) else getattr(target, "name", None)
            out.append(f"Play {name} on {tname or 'your best minion'} — {why}")
        else:
            out.append(f"Play {name} (no minion to target yet)")
    return out


def build_note_for(snapshot, kb=None) -> Optional[str]:
    """One-line 'what you're building toward' for the overlay header."""
    try:
        from .build_path import build_note
        return build_note(snapshot.get("board", []), snapshot.get("tavern_tier"))
    except Exception:
        return None


def _key(d: dict):
    board = tuple(m.get("name") for m in d.get("board", []))
    shop = tuple(m.get("name") for m in d.get("shop", []))
    spells = tuple(s.get("name") for s in d.get("shop_spells", []) or [])
    hand = tuple(s.get("name") for s in d.get("hand_spells", []) or [])
    hand_m = tuple(m.get("name") for m in d.get("hand", []) or [])
    hp = (d.get("hero_power") or {}).get("usable")
    return (board, shop, spells, hand, hand_m, d.get("gold"), d.get("tavern_tier"),
            d.get("phase"), hp, d.get("anomaly"))


class LiveCoach:
    """Background log consumer + cached advice provider for the overlay.

    `power_log` may be None — then it auto-detects the newest Hearthstone session
    log and *waits* for one to appear, so you can launch the overlay before the
    game (HDT-style) and it activates once you're in a match."""

    def __init__(self, power_log: Optional[str] = None,
                 hero_ctx: Optional[HeroContext] = None,
                 recorder=None, from_start: bool = False, top: int = 6):
        self.power_log = power_log
        self.hero_ctx = hero_ctx
        self.recorder = recorder
        self.from_start = from_start
        self.top = top
        self.tracker = BGTracker()
        self.kb = cards.load_kb()
        self.scorer = get_scorer()
        from .choices import ChoiceParser
        from .stats import StatsDB
        self.choices = ChoiceParser()
        self.db = StatsDB.load()
        self._offer = None              # active hero/trinket/discover choice
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cache_key = None
        self._cache_lines: List[str] = []
        self._cache_note: Optional[str] = None
        self._sync_seq = 0                # bumps each time the board/shop changes
        self._version = 0                 # bumps each time a log event is fed
        self._snap_version = -1           # version the cached snapshot was built at
        self._snap_cache: Optional[dict] = None
        self._active = False

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._consume, daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()

    def _resolve_log(self) -> Optional[str]:
        from . import config
        return self.power_log or config.newest_power_log(config.log_dir_candidates())

    def _consume(self):
        # Wait for a log to exist (launch overlay first, then Hearthstone).
        path = None
        while not self._stop.is_set():
            path = self._resolve_log()
            if path:
                break
            self._stop.wait(2.0)
        if self._stop.is_set() or not path:
            return
        self._active = True

        prev_phase = self.tracker.phase
        prev_game = self.tracker.state.game_counter
        for line in tail_lines(path, from_start=self.from_start):
            if self._stop.is_set():
                break
            offer = self.choices.feed(line)        # hero/trinket/discover offers
            if offer is not None:
                self._offer = offer
            elif "SendChoices" in line:
                self._offer = None                 # choice resolved
            ev = parse_line(line)
            if ev is None:
                continue
            with self._lock:
                self.tracker.feed(ev)
                self._version += 1            # mark state advanced (poll rebuilds)
            if self.recorder is not None and self.tracker.state.game_counter != prev_game:
                self.recorder.start_game()
                prev_game = self.tracker.state.game_counter
            if self.tracker.phase != prev_phase:
                self._on_phase(prev_phase, self.tracker.phase)
                prev_phase = self.tracker.phase

    def _on_phase(self, old, new):
        if self.recorder is None:
            return
        if new == Phase.COMBAT:                       # end of recruit = a decision made
            with self._lock:
                snap = self.tracker.snapshot()
            self.recorder.record(snap, ActionType.END_TURN)
        elif new == Phase.GAME_OVER:
            self.recorder.finish_game(placement=self.tracker.placement())

    def frame(self) -> Tuple[dict, Optional[str], List[str]]:
        """(snapshot_dict, odds, recommendations) for the overlay to render."""
        if not self._active:
            return ({"phase": "waiting", "turn": None, "tavern_tier": None,
                     "gold": None, "hero_health": None, "board": [], "shop": [],
                     "notes": ["Launch a Battlegrounds game to begin…"]}, None, [])
        # Rebuild the snapshot only when the log actually advanced; idle ticks
        # (between your actions) reuse the cached one, so polling at 20 Hz stays
        # near-free and the panel still refreshes the instant you act.
        with self._lock:
            version = self._version
            if version != self._snap_version or self._snap_cache is None:
                self._snap_cache = self.tracker.snapshot().to_dict()
                self._snap_version = version
            snap = self._snap_cache
        offer = self._offer
        if offer is not None:                       # a choice is on screen
            from .choices import rank_offer
            # For hero picks, only rank heroes the player can actually draft
            # (locked Perks heroes carry BACON_HERO_CAN_BE_DRAFTED=0).
            draftable = None
            if offer.kind == "hero":
                with self._lock:
                    draftable = self.tracker.draftable_hero_ids()
            picks = rank_offer(offer, board=snap.get("board", []), kb=self.kb,
                               scorer=self.scorer, hero_ctx=self.hero_ctx, db=self.db,
                               tier=snap.get("tavern_tier"), draftable_ids=draftable)
            lines = [f"PICK {c.name} — {c.reason}" for c in picks[:6]]
            snap = dict(snap, phase=f"choose {offer.kind}",
                        notes=[f"{offer.kind.upper()} — pick one"])
            return snap, None, lines
        key = _key(snap)
        if key != self._cache_key:                    # recompute advice only on change
            self._cache_lines = advice_lines(snap, self.kb, self.scorer,
                                              self.hero_ctx, self.top)
            self._cache_key = key
            self._sync_seq += 1            # a real state change was ingested
        # Tag the snapshot with the sync counter so the panel can show that the
        # new board was processed after you roll/buy/sell (it ticks up each change).
        snap = dict(snap, sync_seq=self._sync_seq)
        return snap, None, self._cache_lines
