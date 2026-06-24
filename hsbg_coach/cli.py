"""Command-line entrypoint.

Subcommands:
  detect       find Hearthstone log locations on this machine
  setup        write log.config so Hearthstone emits the logs we parse
  watch        follow the live Power.log: print board on combat + record
  parse-file   parse a previously captured log (offline; great for dev/calibration)
"""

import argparse
import os
import sys
from typing import Iterable, Optional

from . import config, logfix
from .bg import BGTracker, Phase, ActionType
from .parser import parse_line
from .recorder import TrajectoryRecorder
from .tail import tail_lines


def cmd_detect(_args) -> int:
    paths = config.Paths.detect()
    print("Platform:", sys.platform)
    print("Log dir:    ", paths.log_dir or "NOT FOUND (searched candidates)")
    print("Power.log:  ", paths.power_log or "NOT FOUND")
    print("log.config: ", paths.log_config,
          "(exists)" if os.path.isfile(paths.log_config) else "(will be created)")
    if not paths.log_dir:
        print("\nSearched these log dirs:")
        for d in config.log_dir_candidates():
            print("  -", d, "[exists]" if os.path.isdir(d) else "")
        print("\nIf none exist, launch Hearthstone once after `setup`.")
    return 0


def cmd_setup(_args) -> int:
    paths = config.Paths.detect()
    changed = logfix.ensure_log_config(paths.log_config)
    if changed:
        print(f"Wrote logger config to {paths.log_config}")
        print("RESTART Hearthstone for it to take effect.")
    else:
        print(f"log.config already has the loggers we need: {paths.log_config}")
    return 0


def _drive(tracker: BGTracker, recorder: Optional[TrajectoryRecorder],
           lines: Iterable[str]) -> None:
    """Shared pipeline: feed lines -> tracker, react to phase changes."""
    prev_phase = tracker.phase
    prev_game = tracker.state.game_counter
    for line in lines:
        ev = parse_line(line)
        if ev is None:
            continue
        tracker.feed(ev)

        if recorder is not None and tracker.state.game_counter != prev_game:
            recorder.start_game()
            prev_game = tracker.state.game_counter

        if tracker.phase != prev_phase:
            _on_phase_change(tracker, recorder, prev_phase, tracker.phase)
            prev_phase = tracker.phase


def _on_phase_change(tracker, recorder, old: Phase, new: Phase) -> None:
    if new == Phase.COMBAT:
        snap = tracker.snapshot()
        _print_board(snap)
        # The end of recruit is a real decision point ("is my board ready to
        # fight?"). Per-action labeling (buy/sell/roll) is the next calibration
        # step; END_TURN is recordable today.
        if recorder is not None:
            recorder.record(snap, ActionType.END_TURN)
    elif new == Phase.GAME_OVER and recorder is not None:
        # Backfill the final placement onto every decision in the game so the
        # trajectory carries its outcome label (None if not yet reported).
        recorder.finish_game(placement=tracker.placement())


def _print_board(snap) -> None:
    print(f"\n=== COMBAT  (turn {snap.turn}, tier {snap.tavern_tier}, "
          f"gold {snap.gold}, hp {snap.hero_health}) ===")
    print("Your board:")
    for m in snap.board:
        print(f"  [{m.position}] {m.name or m.card_id} "
              f"{m.attack}/{m.health}")
    if snap.shop:
        print("Shop (last seen):")
        for m in snap.shop:
            print(f"  - {m.name or m.card_id} {m.attack}/{m.health}")
    for note in snap.notes:
        print("  note:", note)


def cmd_watch(args) -> int:
    paths = config.Paths.detect()
    power = args.path or paths.power_log
    if not power:
        print("No Power.log found. Run `setup`, launch Hearthstone, then retry.")
        print("Or pass --path to a captured log.")
        return 1
    print(f"Watching {power} (Ctrl-C to stop)")
    tracker = BGTracker()
    recorder = None if args.no_record else TrajectoryRecorder(config.DATA_DIR)
    try:
        _drive(tracker, recorder, tail_lines(power, from_start=args.from_start))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if recorder is not None:
            path = recorder.close()
            if path:
                print("Flushed in-progress trajectory to", path)
    return 0


def cmd_parse_file(args) -> int:
    if not os.path.isfile(args.path):
        print("No such file:", args.path)
        return 1
    tracker = BGTracker()
    recorder = None if args.no_record else TrajectoryRecorder(config.DATA_DIR)
    with open(args.path, "r", encoding="utf-8", errors="replace") as fh:
        _drive(tracker, recorder, fh)
    if recorder is not None:
        path = recorder.close()
        if path:
            print("Recorded trajectory ->", path)
    snap = tracker.snapshot()
    print(f"\nParsed. games={tracker.state.game_counter} "
          f"entities={len(tracker.state.entities)} "
          f"bg={tracker.in_bg} phase={tracker.phase.value} "
          f"local_player={tracker.local_player}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hsbg_coach",
                                description="Battlegrounds log parser + recorder")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("detect", help="find Hearthstone log locations").set_defaults(
        func=cmd_detect)
    sub.add_parser("setup", help="write log.config").set_defaults(func=cmd_setup)

    w = sub.add_parser("watch", help="follow live Power.log")
    w.add_argument("--path", help="override Power.log path")
    w.add_argument("--from-start", action="store_true",
                   help="read existing log content before tailing")
    w.add_argument("--no-record", action="store_true", help="don't write dataset")
    w.set_defaults(func=cmd_watch)

    f = sub.add_parser("parse-file", help="parse a captured log offline")
    f.add_argument("path")
    f.add_argument("--no-record", action="store_true", help="don't write dataset")
    f.set_defaults(func=cmd_parse_file)

    sub.add_parser("overlay", help="show the overlay with sample data (needs a display)"
                   ).set_defaults(func=cmd_overlay)
    return p


def cmd_overlay(_args) -> int:
    try:
        from .overlay import demo
    except Exception as exc:  # pragma: no cover - display-dependent
        print("Could not load overlay:", exc)
        return 1
    try:
        demo()
    except Exception as exc:  # pragma: no cover - needs a display
        print("Overlay needs a graphical display:", exc)
        return 1
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
