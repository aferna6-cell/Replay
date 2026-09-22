"""Post-session Power.log → data/*.jsonl without requiring the coach overlay.

Aidan can play normally (no watch/overlay). After the session:

    python -m hsbg_coach setup          # once
    # … play Battlegrounds as usual …
    python -m hsbg_coach ingest         # scan default / recent Power.log(s)
    ./scripts/retrain.sh

Records END_TURN decision points (end of each recruit) with final placement when
the log exposes PLAYER_LEADERBOARD_PLACE — same labels as parse-file recording.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from . import config
from .bg import ActionType, BGTracker, Phase
from .parser import parse_line
from .recorder import TrajectoryRecorder


@dataclass
class IngestResult:
    path: str
    games: int
    recorded: List[str] = field(default_factory=list)
    placements: List[Optional[int]] = field(default_factory=list)


def list_power_logs(limit: Optional[int] = None) -> List[str]:
    """Newest-first Power.log paths under configured Hearthstone log dirs."""
    import glob

    matches: List[str] = []
    for d in config.log_dir_candidates():
        matches += glob.glob(os.path.join(d, "Power.log"))
        matches += glob.glob(os.path.join(d, "Hearthstone_*", "Power.log"))
    matches = [m for m in matches if os.path.isfile(m)]
    matches.sort(key=os.path.getmtime, reverse=True)
    if limit is not None:
        matches = matches[: max(0, int(limit))]
    return matches


def resolve_ingest_paths(path: Optional[str] = None, recent: int = 1) -> List[str]:
    """Explicit path, else the N newest detected Power.log files."""
    if path:
        return [path]
    logs = list_power_logs(limit=max(1, int(recent or 1)))
    if logs:
        return logs
    detected = config.Paths.detect().power_log
    return [detected] if detected else []


def _stable_prefix(log_path: str) -> str:
    try:
        mtime = int(os.path.getmtime(log_path))
    except OSError:
        mtime = 0
    digest = hashlib.sha1(f"{log_path}|{mtime}".encode()).hexdigest()[:10]
    return f"ingest-{digest}"


def _drive_quiet(tracker: BGTracker, recorder: TrajectoryRecorder, lines) -> None:
    """Like cli._drive, but silent (no combat board prints) and placement-tolerant.

    Real Power.logs sometimes emit PLAYER_LEADERBOARD_PLACE *after* the STEP that
    flips us to GAME_OVER. We keep the game buffer open until placement appears
    (or the file ends → recorder.close() writes a .partial).
    """
    prev_phase = tracker.phase
    prev_game = tracker.state.game_counter
    pending_finish = False

    for line in lines:
        ev = parse_line(line)
        if ev is None:
            continue
        tracker.feed(ev)

        if tracker.state.game_counter != prev_game:
            # New game starting — flush any unfinished prior game first.
            if pending_finish:
                path = recorder.finish_game(placement=tracker.placement())
                pending_finish = False
                if path:
                    recorder.written.append(path) if hasattr(recorder, "written") else None
            recorder.start_game()
            prev_game = tracker.state.game_counter

        if tracker.phase != prev_phase:
            if tracker.phase == Phase.COMBAT:
                # End-of-recruit decision point (same as watch/parse-file).
                recorder.record(tracker.snapshot(), ActionType.END_TURN)
            elif tracker.phase == Phase.GAME_OVER:
                place = tracker.placement()
                if place is not None:
                    recorder.finish_game(placement=place)
                    pending_finish = False
                else:
                    pending_finish = True  # wait for PLAYER_LEADERBOARD_PLACE
            prev_phase = tracker.phase

        # Placement tag arrived after GAME_OVER — finish now.
        if pending_finish and tracker.placement() is not None:
            recorder.finish_game(placement=tracker.placement())
            pending_finish = False


def ingest_power_log(
    log_path: str,
    data_dir: Optional[str] = None,
    quiet: bool = False,
) -> IngestResult:
    """Parse one Power.log and write trajectory jsonl under data_dir."""
    if not os.path.isfile(log_path):
        raise FileNotFoundError(log_path)

    data_dir = data_dir or config.DATA_DIR
    tracker = BGTracker()
    prefix = _stable_prefix(log_path)
    n = {"i": 0}
    written: List[str] = []
    placements: List[Optional[int]] = []

    class _Rec(TrajectoryRecorder):
        def start_game(self, game_id: Optional[str] = None) -> None:
            n["i"] += 1
            super().start_game(game_id or f"{prefix}-g{n['i']}")

        def finish_game(self, placement: Optional[int]) -> str:
            path = super().finish_game(placement=placement)
            if path:
                written.append(path)
                placements.append(placement)
                if not quiet:
                    place_s = placement if placement is not None else "?"
                    print(f"  recorded {path}  placement={place_s}")
            return path

        def close(self) -> str:
            path = super().close()
            if path:
                written.append(path)
                placements.append(None)
                if not quiet:
                    print(f"  recorded {path}  placement=? (partial)")
            return path

    recorder = _Rec(data_dir)
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        _drive_quiet(tracker, recorder, fh)
    recorder.close()

    return IngestResult(
        path=log_path,
        games=tracker.state.game_counter,
        recorded=list(written),
        placements=list(placements),
    )


def ingest(
    paths: Optional[Iterable[str]] = None,
    recent: int = 1,
    data_dir: Optional[str] = None,
    quiet: bool = False,
) -> List[IngestResult]:
    """Ingest one or more Power.log files. Returns per-file results."""
    targets = list(paths) if paths is not None else resolve_ingest_paths(recent=recent)
    if not targets:
        raise FileNotFoundError(
            "No Power.log found. Run `python -m hsbg_coach setup`, play a game, "
            "or pass --path /path/to/Power.log"
        )
    out: List[IngestResult] = []
    for p in targets:
        if not quiet:
            print(f"Ingesting {p} …")
        out.append(ingest_power_log(p, data_dir=data_dir, quiet=quiet))
    return out
