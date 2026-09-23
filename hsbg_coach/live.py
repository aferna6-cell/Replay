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
from .bg import BGTracker, Phase, ActionType
from .board_value import get_scorer
from .economy import HeroContext
from .parser import parse_line
from .tail import tail_latest


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
    # Skip the reposition optimizer (hundreds of thousands of combat sims, ~1.5s)
    # — we don't show reposition advice, and skipping it makes the panel update
    # near-instantly after each action.
    recs, _ = rank_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer,
                           include_reposition=False)
    out = []
    # Strategic signals (threats / forming synergy / comp lean / anomaly) still
    # shape the RANKING inside rank_actions, but we don't print them as lines — the
    # overlay shows only the move(s), nothing else.
    # Hero/hand contextual scripts FIRST so Lens Case / Gallywix cycle can be NEXT
    # (generic shop EV alone left these silent).
    try:
        from .hero_scripts import hero_script_lines
        for line in hero_script_lines(snapshot, kb=kb):
            out.append(line)
    except Exception:
        pass
    # Free cards that landed in your hand (often generated during combat) are
    # usually a play-now: a minion to drop, or a Magnetic mech to fuse. Lead with
    # those, then the spell-on-minion advice.
    for line in _hand_play_lines(snapshot, kb):
        out.append(line)
    # A targetable spell in hand (e.g. Tavern Dish Banana) usually wants playing
    # now — lead with where to put it.
    for line in _hand_spell_lines(snapshot):
        out.append(line)
    # Show the *why* (synergy / tribe / sell-for-room / tech caveat) next to each
    # move. Suppress Reposition and End-turn — you don't need to be told to pass;
    # your turn ends when you hit 0 gold. Only show real actions to take.
    from .actions import REPOSITION, END, FREEZE
    # Drop Reposition always. Drop End-turn unless freeze is not a real keep —
    # otherwise hiding End made mediocre "Freeze the shop" look like NEXT.
    # Drop freeze noise when the advisor buried it (shop not worth keeping).
    _freeze_bury = ("worth keeping", "no need to freeze")
    shown = []
    freeze_kept = False
    for r in recs:
        if r.action.kind == REPOSITION:
            continue
        if r.action.kind == FREEZE:
            reason = (r.reason or "").lower()
            if any(tok in reason for tok in _freeze_bury):
                continue
            freeze_kept = True
        if r.action.kind == END:
            continue  # decide after pass whether to re-add
        shown.append(r)
    gold = snapshot.get("gold")
    if gold is not None and int(gold) <= 0 and not freeze_kept:
        # Prefer End Turn over a weak Sell when the shop is not freeze-worthy.
        end = next((r for r in recs if r.action.kind == END), None)
        if end is not None:
            shown.insert(0, end)
    for r in shown[:top]:
        line = f"{r.action.describe()} (finish {r.placement:.1f})"
        if r.reason:
            line += f" — {r.reason}"
        out.append(line)
    return out


_MAX_BG_BOARD = 7

def _sell_for_room_target(board, snapshot, kb):
    """Minion to cut for room: prefer off-direction / lowest keep — never a core."""
    from .game_value import _keep_value
    from .actions import Action, SELL

    def score(m):
        # Lower = sell first. Direction-cut chaff gets an extra demotion.
        s = float(_keep_value(m, board, kb))
        try:
            from .jeef_priors import direction_cut_sell_adjust
            act = Action(SELL, target=_name_safe(m), cost=0, detail={"minion": m})
            cadj, _ = direction_cut_sell_adjust(act, snapshot, kb)
            if cadj:                       # negative = encourage cut
                s += cadj * 8.0            # keep-value scale ~stats; amplify cut
        except Exception:
            pass
        return s

    def _name_safe(m):
        if isinstance(m, dict):
            return m.get("name") or m.get("card_id") or "?"
        return getattr(m, "name", None) or "?"

    return min(board, key=score)



def _hand_play_lines(snapshot, kb) -> List[str]:
    """'Play <minion> from hand' / 'Magnetize <mech> onto <host>' for each free
    minion sitting in your hand (e.g. one a combat effect generated). Magnetic
    mechs prefer fusing onto a board mech (no slot used, buff protected)."""
    hand = snapshot.get("hand") or []
    minions = [m for m in hand if _is_hand_minion(m)]
    if not minions:
        return []
    from .magnetize import is_magnetic, best_magnetize_target
    from .board_value import _name as _mname
    from .game_value import _keep_value
    board = snapshot.get("board", []) or []
    full = len(board) >= _MAX_BG_BOARD
    weakest = _mname(_sell_for_room_target(board, snapshot, kb)) if board else None
    out = []
    for m in minions:
        # Hand MagicItems / Lens Case are owned by hero_scripts (avoid dup NEXT).
        try:
            from .hero_scripts import is_hand_playable_item
            if is_hand_playable_item(m):
                continue
        except Exception:
            pass
        name = m.get("name") or m.get("card_id") or "minion"
        if is_magnetic(m, kb):
            tgt = best_magnetize_target(board, kb)
            if tgt is not None:
                host, why = tgt
                hname = host.get("name") if isinstance(host, dict) else getattr(host, "name", None)
                out.append(f"Magnetize {name} onto {hname or 'your best mech'} — {why}")
                continue                                 # fusing uses no board slot
        # Choose-One battlecry (e.g. Intrepid Botanist → Pristine Lilies / Giant
        # Dewdrop): name the half to take (+Attack vs +Health) when we have a
        # curated pick; otherwise a generic hint.
        choose = ""
        if _is_choose_one(m, kb):
            from .choose_one import choose_one_advice
            specific = choose_one_advice(m, snapshot)
            choose = f" · {specific}" if specific else " · Choose One — take the half that fits your board"
        if full:
            if weakest:
                # Lead with sell target so overlay NEXT cannot look like play-only.
                out.append(
                    f"Sell {weakest}, then play {name} from hand "
                    f"— board full, cut lowest keep{choose}"
                )
            else:
                out.append(
                    f"Play {name} from hand — make room first (board is full){choose}"
                )
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
    """A real, playable minion OR hand item (Lens Case / MagicItem) in hand.

    Previously CARDTYPE!=MINION dropped MagicItems → Lens Case was never
    suggested. Spells still go through hand_spells; DragBuy is ignored.
    """
    try:
        from .hero_scripts import is_hand_playable_item
        if is_hand_playable_item(m):
            return True
    except Exception:
        pass
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


def _combat_odds_for(snapshot: dict, runs: int = 80, seed: int = 0) -> Optional[str]:
    """Win/tie/loss vs the last revealed opponent board(s), when available.

    Uses opponents_seen from the Snapshot (populated during combat). Returns None
    when we have no enemy board yet — early game / before first fight — so the
    panel stays clean rather than fabricating odds.
    """
    enemies = snapshot.get("opponents_seen") or []
    board = snapshot.get("board") or []
    if not enemies or not board:
        return None
    try:
        from .recommend import combat_odds
        return combat_odds(snapshot, enemies, runs=runs, seed=seed)
    except Exception:
        return None


def build_note_for(snapshot, kb=None, hero_ctx=None) -> Optional[str]:
    """Minimal 'building: Dragons' line from soft lobby lean / board."""
    try:
        from .tribe_policy import build_direction_note, soft_lean_tribe
        board = snapshot.get("board", []) if isinstance(snapshot, dict) else []
        avail = (snapshot.get("available_tribes") if isinstance(snapshot, dict) else None)
        if hero_ctx is not None and getattr(hero_ctx, "available_tribes", None):
            avail = hero_ctx.available_tribes
        lean = getattr(hero_ctx, "target_tribe", None) if hero_ctx else None
        note = build_direction_note(board, avail, kb=kb, hero_target=lean,
                                    shop=(snapshot.get("shop") if isinstance(snapshot, dict) else None))
        if note:
            return note
        # Fall back to soft lean label even with empty board (lobby prior).
        tribe, why = soft_lean_tribe(board, available_tribes=avail, kb=kb,
                                     hero_target=lean,
                                     shop=(snapshot.get("shop") if isinstance(snapshot, dict) else None),
                                     turn=(snapshot.get("turn") if isinstance(snapshot, dict) else None))
        return f"building: {tribe}" if tribe else None
    except Exception:
        return None


def _sig(m):
    """A change-sensitive signature for a minion: identity + stats, so a buff,
    a sell, or a swap all force the advice to recompute (never goes stale)."""
    if not isinstance(m, dict):
        return (getattr(m, "entity_id", None), getattr(m, "name", None))
    return (m.get("entity_id"), m.get("name"), m.get("attack"), m.get("health"))


def _key(d: dict):
    # Key on CONTENT (ids + stats), not just names — so playing a card, rolling,
    # selling, or a stat buff always changes the key and the panel refreshes. A
    # too-coarse key was leaving the overlay stuck on an action already taken.
    board = tuple(_sig(m) for m in d.get("board", []) or [])
    shop = tuple(_sig(m) for m in d.get("shop", []) or [])
    spells = tuple(s.get("name") for s in d.get("shop_spells", []) or [])
    hand = tuple((s.get("entity_id"), s.get("name")) for s in d.get("hand_spells", []) or [])
    hand_m = tuple(_sig(m) for m in d.get("hand", []) or [])
    trinkets = tuple(t.get("name") for t in d.get("trinkets", []) or [])
    opps = tuple((p.get("controller"), p.get("hero"), p.get("strength"))
                 for p in d.get("opponent_profiles", []) or [])
    hp = (d.get("hero_power") or {}).get("usable")
    activate = tuple(
        (a.get("entity_id"), a.get("usable"), a.get("cost"))
        for a in (d.get("activatable") or [])
    )
    dg = d.get("dark_gift") or {}
    dark = (dg.get("usable"), dg.get("cost"), dg.get("entity_id")) if dg else None
    return (board, shop, spells, hand, hand_m, trinkets, opps, d.get("gold"),
            d.get("tavern_tier"), d.get("phase"), hp, activate, dark,
            d.get("hero_health"), d.get("anomaly"), d.get("hero"))


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
        self._hero_ctx_auto = hero_ctx is None   # auto-build from the detected hero
        self._hero_ctx_for = None
        self._hero_ctx_key = None
        self.manual_tribe_priors = {}
        self.recorder = recorder
        self.from_start = from_start
        self.top = top
        self.tracker = BGTracker()
        self.kb = cards.load_kb()
        self.scorer = get_scorer()
        # Continual learning: retrain between games in the background, hot-swap the
        # eval net when a new one lands. Never blocks the live recommendation.
        import os
        from . import config
        self._model_path = os.path.join(os.path.dirname(__file__), "..", "ml",
                                        "eval_net.pt")
        self._model_mtime = self._scorer_mtime()
        self.trainer = None
        if recorder is not None:
            try:
                from .continual import BackgroundTrainer
                self.trainer = BackgroundTrainer(config.DATA_DIR)
            except Exception:
                self.trainer = None
        from .choices import ChoiceParser
        from .stats import StatsDB
        self.choices = ChoiceParser()
        self.db = StatsDB.load()
        self._offer = None              # active hero/trinket/discover choice
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cache_key = None
        self._cache_lines: List[str] = []
        self._cache_odds: Optional[str] = None
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

    def _ensure_hero_ctx(self, snap) -> None:
        """Soft lobby lean from available tribes + tribe win% (not a hard lock).

        Rebuilds when the hero changes OR lobby tribes get detected mid-game.
        Optional ``self.manual_tribe_priors`` (tribe->first%) covers Aberration
        before HSReplay publishes tribe stats — set via LiveCoach.set_tribe_priors.
        """
        if not self._hero_ctx_auto:
            return
        hero = snap.get("hero") or snap.get("hero_name")
        lobby = tuple(snap.get("available_tribes") or ())
        key = (hero, lobby, tuple(sorted((getattr(self, "manual_tribe_priors", {}) or {}).items())))
        if not hero or key == getattr(self, "_hero_ctx_key", None):
            return
        try:
            from .stats import build_hero_context
            self.hero_ctx = build_hero_context(
                hero, self.db,
                available_tribes=list(lobby) if lobby else None,
                manual_tribe_priors=getattr(self, "manual_tribe_priors", None),
                board=snap.get("board"), shop=snap.get("shop"),
                turn=snap.get("turn"), kb=self.kb,
            )
            self._hero_ctx_for = hero
            self._hero_ctx_key = key
        except Exception:
            pass

    def set_tribe_priors(self, priors: dict) -> None:
        """Manual lobby-start tribe first% / weights (e.g. {"Aberration": 0.22})."""
        self.manual_tribe_priors = {str(k): float(v) for k, v in (priors or {}).items()}
        self._hero_ctx_key = None  # force rebuild
        try:
            self.tracker.set_manual_tribe_priors(self.manual_tribe_priors)
        except Exception:
            pass

    def _scorer_mtime(self):
        import os
        try:
            return os.path.getmtime(self._model_path)
        except OSError:
            return None

    def _maybe_reload_scorer(self) -> None:
        """If a background retrain wrote a new eval net, hot-swap it. Only the cheap
        mtime check runs each recompute; the actual reload happens rarely (after a
        retrain), never on a per-poll basis."""
        m = self._scorer_mtime()
        if m is not None and m != self._model_mtime:
            try:
                self.scorer = get_scorer()
                self._model_mtime = m
            except Exception:
                pass

    def _resolve_log(self) -> Optional[str]:
        from . import config
        return self.power_log or config.newest_power_log(config.log_dir_candidates())

    def _consume(self):
        # Keep discovering sessions for the lifetime of the overlay. Hearthstone
        # writes a new timestamped Power.log on every launch, so pinning the first
        # path found makes a long-running coach silently miss later games.
        prev_phase = self.tracker.phase
        prev_game = self.tracker.state.game_counter
        for line in tail_latest(self._resolve_log, from_start=self.from_start,
                                stop_event=self._stop):
            if self._stop.is_set():
                break
            self._active = True
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
            # Game's over → fold it into the model in the background (low priority,
            # separate process). Doesn't touch the live path.
            if self.trainer is not None:
                try:
                    self.trainer.maybe_train()
                except Exception:
                    pass

    def frame(self) -> Tuple[dict, Optional[str], List[str]]:
        """(snapshot_dict, odds, recommendations) for the overlay to render."""
        if not self._active:
            return ({"phase": "waiting", "turn": None, "tavern_tier": None,
                     "gold": None, "hero_health": None, "board": [], "shop": [],
                     "notes": ["Launch a Battlegrounds game to begin…"]}, None, [])
        if not self.tracker.in_bg:
            return ({"phase": "waiting", "turn": None, "tavern_tier": None,
                     "gold": None, "hero_health": None, "board": [], "shop": [],
                     "notes": ["Hearthstone detected — waiting for Battlegrounds…"]},
                    None, [])
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
            from .choices import offer_advice_lines
            # Heroes: advisory reroll of the weakest, then pick ranking of the rest.
            # (Hero-select reroll ≠ tavern Refresh / TB_BaconShop_8p_Reroll_Button.)
            lines = offer_advice_lines(
                offer, board=snap.get("board", []), kb=self.kb,
                scorer=self.scorer, hero_ctx=self.hero_ctx, db=self.db,
                tier=snap.get("tavern_tier"),
            )
            snap = dict(snap, phase=f"choose {offer.kind}",
                        notes=[f"{offer.kind.upper()} — pick one"])
            return snap, None, lines
        self._ensure_hero_ctx(snap)       # steer toward our hero's best tribes
        key = _key(snap)
        if key != self._cache_key:                    # recompute advice only on change
            self._maybe_reload_scorer()   # hot-swap a freshly retrained model (cheap)
            self._cache_lines = advice_lines(snap, self.kb, self.scorer,
                                              self.hero_ctx, self.top)
            self._cache_odds = _combat_odds_for(snap)
            self._cache_note = build_note_for(snap, self.kb, self.hero_ctx)
            self._cache_key = key
            self._sync_seq += 1            # a real state change was ingested
        # Tag the snapshot with the sync counter so the panel can show that the
        # new board was processed after you roll/buy/sell (it ticks up each change).
        snap = dict(snap, sync_seq=self._sync_seq)
        if self._cache_note:
            snap = dict(snap, build_note=self._cache_note)
        return snap, self._cache_odds, self._cache_lines
