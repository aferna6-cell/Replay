"""The beam-search teacher: env observation -> env action.

`turn_search` + the set-net scorer is the strongest player in the repo — it
plans whole turns globally. This module turns it into a per-decision policy
the RL side can imitate (distillation) and measure against: at every decision
it re-searches from the current state (MPC-style — always self-consistent
after RNG events like rolls) and executes the first action of the best line.

Hand play isn't part of the searched space (playing a minion you own is
strictly good in the Phase 0 env), so it's handled by rule before searching.
"""

from typing import Dict, List, Optional

from hsbg_coach.bg_env import (
    A_BUY0, A_PLAY0, A_SELL0, A_ROLL, A_LEVEL, A_END, N_PLAY,
)
from hsbg_coach.turn_search import plan_turn_search


class SearchExpert:
    """Callable with the bg_env scripted-policy signature (obs, mask, rng)."""

    def __init__(self, kb=None, scorer=None, pace=None, beam: int = 5,
                 depth: int = 6, env_mode: bool = False):
        """env_mode=True searches on pure state value: no real-meta buy
        knowledge, no synergy keep-values (kb=None → keep-value = raw stats).
        The right teacher INSIDE the Phase 0 env, where comp churn resets a
        minion's compounded scaling and the meta layer doesn't exist. The
        default (env_mode=False) is the real-game advisor configuration."""
        from hsbg_coach import cards
        from hsbg_coach.board_value import get_scorer
        from hsbg_coach.pace import load_pace
        self.env_mode = env_mode
        if env_mode:
            self.kb = None
        else:
            self.kb = kb if kb is not None else cards.load_kb()
        self.scorer = scorer or get_scorer()
        self.pace = pace if pace is not None else load_pace()
        self.beam = beam
        self.depth = depth

    def __call__(self, obs: Dict, mask: List[bool], rng=None) -> int:
        # Play what you own first — always right here, and it keeps the
        # searched state (board ∪ shop) in sync with what search models.
        for i in range(N_PLAY):
            if mask[A_PLAY0 + i]:
                return A_PLAY0 + i
        plan = plan_turn_search(obs, kb=self.kb, scorer=self.scorer,
                                pace=self.pace, beam=self.beam,
                                depth=self.depth,
                                knowledge=not self.env_mode,
                                stat_tiebreak=1.0 if self.env_mode else 0.0)
        for kind, target in plan.actions:
            a = self._to_env_action(kind, target, obs, mask)
            if a is not None:
                return a
        return A_END

    @staticmethod
    def _to_env_action(kind: str, target: Optional[str], obs: Dict,
                       mask: List[bool]) -> Optional[int]:
        if kind == "buy":
            for i, m in enumerate(obs["shop"]):
                if m.get("name") == target and mask[A_BUY0 + i]:
                    return A_BUY0 + i
        elif kind == "sell":
            for i, m in enumerate(obs["board"]):
                if m.get("name") == target and mask[A_SELL0 + i]:
                    return A_SELL0 + i
        elif kind == "level" and mask[A_LEVEL]:
            return A_LEVEL
        elif kind == "roll" and mask[A_ROLL]:
            return A_ROLL
        return None
