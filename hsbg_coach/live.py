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
    """Ranked one-line recommendations for a snapshot, best first.

    Empty unless we're shopping (a shop is present) — there's nothing to advise
    mid-combat. Pure + synchronous, so it's unit-testable without a display."""
    if not snapshot.get("shop"):
        return []
    plan = advise_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer)
    out = []
    for a in plan.ranked[:top]:
        delta = f" ({a.delta:+.0%})" if a.delta is not None else ""
        out.append(f"{a.action.describe()}{delta}")
    return out


def _key(d: dict):
    board = tuple(m.get("name") for m in d.get("board", []))
    shop = tuple(m.get("name") for m in d.get("shop", []))
    return board, shop, d.get("gold"), d.get("tavern_tier"), d.get("phase")


class LiveCoach:
    """Background log consumer + cached advice provider for the overlay."""

    def __init__(self, power_log: str, hero_ctx: Optional[HeroContext] = None,
                 recorder=None, from_start: bool = False, top: int = 6):
        self.power_log = power_log
        self.hero_ctx = hero_ctx
        self.recorder = recorder
        self.from_start = from_start
        self.top = top
        self.tracker = BGTracker()
        self.kb = cards.load_kb()
        self.scorer = get_scorer()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cache_key = None
        self._cache_lines: List[str] = []

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._consume, daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()

    def _consume(self):
        prev_phase = self.tracker.phase
        prev_game = self.tracker.state.game_counter
        for line in tail_lines(self.power_log, from_start=self.from_start):
            if self._stop.is_set():
                break
            ev = parse_line(line)
            if ev is None:
                continue
            with self._lock:
                self.tracker.feed(ev)
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
        with self._lock:
            snap = self.tracker.snapshot().to_dict()
        key = _key(snap)
        if key != self._cache_key:                    # recompute advice only on change
            self._cache_lines = advice_lines(snap, self.kb, self.scorer,
                                              self.hero_ctx, self.top)
            self._cache_key = key
        return snap, None, self._cache_lines
