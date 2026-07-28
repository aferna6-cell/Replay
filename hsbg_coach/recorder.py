"""Append (state, action, outcome) records to a JSONL dataset.

This is the training-data foundation. Every recorded line is one decision point:
the full observable state, the action taken, and (filled in at game end) the
final placement that the action contributed to. Population priors are *not*
stored here — they enter the model later as features. This file holds only your
own raw trajectories, which are the only raw trajectories obtainable.

JSONL = one JSON object per line, append-only, trivially streamable into a
training pipeline later.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bg import Snapshot, ActionType


@dataclass
class Decision:
    state: Dict                       # Snapshot.to_dict()
    action_type: str                  # ActionType value
    action_detail: Dict = field(default_factory=dict)  # e.g. {"card_id": "BGS_039"}
    wall_clock: float = field(default_factory=time.time)
    # Filled at game end via backfill_outcome():
    placement: Optional[int] = None   # 1..8 (1 = won the lobby)


class TrajectoryRecorder:
    """Buffers a game's decisions, then writes them with the final placement.

    Outcome is only known at game end, so we hold the game in memory and flush
    once placement is known (or on a forced flush). This makes credit assignment
    possible downstream: every decision carries the result it led to.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._current: List[Decision] = []
        self._game_id: Optional[str] = None

    def start_game(self, game_id: Optional[str] = None) -> None:
        # Flush any prior unfinished game (placement unknown) so we never lose data.
        if self._current:
            self._flush(placement_known=False)
        self._game_id = game_id or time.strftime("%Y%m%d-%H%M%S")
        self._current = []

    def record(
        self,
        snapshot: Snapshot,
        action_type: ActionType,
        action_detail: Optional[Dict] = None,
    ) -> None:
        self._current.append(
            Decision(
                state=snapshot.to_dict(),
                action_type=action_type.value,
                action_detail=action_detail or {},
            )
        )

    def finish_game(self, placement: Optional[int]) -> str:
        for d in self._current:
            d.placement = placement
        return self._flush(placement_known=placement is not None)

    def close(self) -> str:
        """Flush any buffered (unfinished) game so a process exit never drops data."""
        if self._current:
            return self._flush(placement_known=False)
        return ""

    # --- internals --------------------------------------------------------
    def _flush(self, placement_known: bool) -> str:
        if not self._current:
            return ""
        suffix = "" if placement_known else ".partial"
        path = os.path.join(self.data_dir, f"game-{self._game_id}.jsonl{suffix}")
        with open(path, "w", encoding="utf-8") as fh:
            for d in self._current:
                fh.write(json.dumps(_as_jsonable(d), separators=(",", ":")) + "\n")
        self._current = []
        return path


def _as_jsonable(d: Decision) -> Dict:
    return {
        "state": d.state,
        "action_type": d.action_type,
        "action_detail": d.action_detail,
        "placement": d.placement,
        "wall_clock": d.wall_clock,
    }
