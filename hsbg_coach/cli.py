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
    if sys.platform.startswith("linux"):
        installs = config.hearthstone_installs()
        print("HS installs: ", "; ".join(installs) or
              "NOT FOUND in any Wine prefix")
        prefixes = config.wine_drive_cs()
        if prefixes:
            print("Wine prefixes searched:")
            for dc in prefixes:
                mark = "  [has Hearthstone]" if config._has_hearthstone(dc) else ""
                print("  -", dc, mark)
        if not installs:
            print("\nNo Hearthstone install found. If it lives somewhere "
                  "unusual, point at it directly:\n"
                  "  HSBG_HS_DIR=/path/to/Hearthstone python3 -m hsbg_coach detect")
    if not paths.log_dir:
        print("\nSearched these log dirs:")
        for d in config.log_dir_candidates():
            print("  -", d, "[exists]" if os.path.isdir(d) else "")
        print("\nIf none exist, run `setup`, then launch Hearthstone and "
              "start a game — the Logs folder is created on launch.")
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
    # Overlay/terminal modes can start with no log yet — they wait for Hearthstone.
    # --director needs a live panel to render into; plain watch routes to terminal.
    if getattr(args, "director", False) and not args.overlay:
        args.terminal = True
    if getattr(args, "terminal", False):
        return _watch_terminal(args.path, args)
    if args.overlay:
        return _watch_overlay(args.path, args)
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


def _watch_terminal(power, args) -> int:
    """In-place terminal panel — the most reliable 'overlay' on any machine.

    No GUI toolkit involved: it repaints a tidy panel in place each time your
    state changes (like htop). Float/pin your terminal window in a corner and it
    behaves exactly like an HDT overlay, with none of the macOS Tk breakage."""
    import time
    from .live import LiveCoach
    from .overlay import format_next

    recorder = None if args.no_record else TrajectoryRecorder(config.DATA_DIR)
    coach = LiveCoach(power, recorder=recorder, from_start=True)
    coach.start()
    director = _maybe_director(args, coach)
    print("HSBG Coach (terminal panel) — launch a Battlegrounds game. Ctrl-C to stop.")
    last = None
    try:
        while True:
            snap, odds, lines = coach.frame()
            if director is not None:
                lines = director.augment(snap, lines)
            text = format_next(snap, odds, lines)
            if text != last:
                # Home cursor + clear screen, then repaint the panel in place.
                print("\033[H\033[J" + text, flush=True)
                last = text
            time.sleep(0.05)            # 20 Hz — repaint near-instantly when you act
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        coach.stop()
        if director is not None:
            director.stop()
        if recorder is not None:
            recorder.close()
    return 0


def _maybe_director(args, coach):
    """Build + start a DirectorLoop when --director was asked for; explain and
    fall back to plain engine advice when its prerequisites are missing."""
    if not getattr(args, "director", False):
        return None
    from .director_live import DirectorLoop
    d = DirectorLoop(kb=coach.kb, hero_ctx_fn=lambda: coach.hero_ctx)
    if d.meta_pack is None:
        print("Director: no meta pack — run `python3 -m hsbg_coach refresh-meta` "
              "first. Continuing with engine advice only.")
    if d.client_error:
        print(f"Director: LLM unavailable ({d.client_error}) — engine advice only.")
    d.start()
    return d


def _watch_overlay_windows(power, args) -> int:
    """WSL: drive a NATIVE Windows always-on-top panel (scripts/
    windows_overlay.ps1) — the HDT-grade overlay. The terminal keeps
    echoing the panel too, but the floating window is the real UI."""
    import time
    from .live import LiveCoach
    from .overlay import format_next
    from .win_overlay import WindowsOverlay

    recorder = None if args.no_record else TrajectoryRecorder(config.DATA_DIR)
    coach = LiveCoach(power, recorder=recorder, from_start=True)
    coach.start()
    director = _maybe_director(args, coach)
    ov = WindowsOverlay()
    ov.start()
    print("Native Windows overlay opened (drag it anywhere over the game; "
          "Hearthstone must be windowed/borderless, not exclusive "
          "fullscreen). Ctrl-C here to stop.")
    last = None
    try:
        while ov.alive():
            snap, odds, lines = coach.frame()
            if director is not None:
                lines = director.augment(snap, lines)
            text = format_next(snap, odds, lines)
            ov.update(text)
            if text != last:
                print("\033[H\033[J" + text, flush=True)
                last = text
            time.sleep(0.2)
        print("Overlay window closed — stopping.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ov.stop()
        coach.stop()
        if director is not None:
            director.stop()
        if recorder is not None:
            recorder.close()
    return 0


def _watch_overlay(power, args) -> int:
    """Live overlay: background log thread feeds the coach; the overlay polls it."""
    if config.is_wsl():
        from .win_overlay import powershell_exe
        if powershell_exe():
            return _watch_overlay_windows(power, args)
        print("WSL detected but powershell.exe is unreachable — trying the "
              "Tk overlay instead.")
    from .live import LiveCoach
    recorder = None if args.no_record else TrajectoryRecorder(config.DATA_DIR)
    # Read the session from the start so we catch hero-select + early turns that
    # are written before/just-as the overlay attaches.
    coach = LiveCoach(power, recorder=recorder, from_start=True)
    coach.start()
    director = _maybe_director(args, coach)

    # Repaint a tidy panel in the terminal too (in place, like htop). The Tk
    # window is unreliable on Apple's deprecated system Tk, so this is always a
    # working readout — and confirms the parser is reading your game live.
    from .overlay import format_next
    last_text = [None]

    def frame_and_echo():
        snap, odds, lines = coach.frame()
        if director is not None:
            lines = director.augment(snap, lines)
        result = (snap, odds, lines)
        try:
            text = format_next(*result)
            if text != last_text[0]:
                print("\033[H\033[J" + text, flush=True)
                last_text[0] = text
        except Exception:
            pass
        return result

    try:
        from .overlay import Overlay
        ov = Overlay()
    except Exception as exc:  # pragma: no cover - needs a display
        print("Overlay window could not open:", exc)
        if "tkinter" in str(exc).lower() or "no module named" in str(exc).lower():
            print("  Fix: install Tk for this Python — on WSL/Debian/Ubuntu:\n"
                  "    sudo apt install python3-tk\n"
                  "  (WSL on Windows 11 shows the window via WSLg.)")
        print("Falling back to the terminal panel. Tip: keep it visible over "
              "the game with Windows Terminal's always-on-top (Settings -> "
              "Appearance -> Always on top, or Ctrl+Shift+P -> 'Toggle "
              "always on top').")
        coach.stop()
        if recorder is not None:
            recorder.close()
        return _watch_terminal(power, args)
    print("Overlay open — waiting for Hearthstone. Launch a Battlegrounds game; "
          "the panel updates each turn (and prints here too). Close the window to stop.")
    ov.poll(frame_and_echo, interval_ms=120)
    try:
        ov.run()
    finally:
        coach.stop()
        if director is not None:
            director.stop()
        if recorder is not None:
            recorder.close()
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
    w.add_argument("--overlay", action="store_true",
                   help="show the on-screen overlay with live recommendations")
    w.add_argument("--terminal", action="store_true",
                   help="live recommendations as an in-place terminal panel "
                        "(no GUI; reliable on any macOS — float your terminal window)")
    w.add_argument("--director", action="store_true",
                   help="LLM Turn Director: one move + why on top of the panel "
                        "(needs `refresh-meta` once and an LLM backend; falls "
                        "back to engine advice if the LLM is unavailable)")
    w.set_defaults(func=cmd_watch)

    f = sub.add_parser("parse-file", help="parse a captured log offline")
    f.add_argument("path")
    f.add_argument("--no-record", action="store_true", help="don't write dataset")
    f.set_defaults(func=cmd_parse_file)

    sub.add_parser("overlay", help="show the overlay with sample data (needs a display)"
                   ).set_defaults(func=cmd_overlay)

    s = sub.add_parser("stats", help="show hero/comp advice from population stats")
    s.add_argument("--hero", required=True, help="hero name (e.g. 'Old Murk-Eye')")
    s.add_argument("--tribes", help="comma-separated tribes available this lobby")
    s.add_argument("--hero-source", help="hero stats file/URL (default: Firestone snapshot)")
    s.add_argument("--comp-source", help="comp stats file/URL (default: Firestone snapshot)")
    s.set_defaults(func=cmd_stats)

    r = sub.add_parser("refresh-stats",
                       help="download the latest real stats from Firestone")
    r.add_argument("--mmr", type=int, default=10,
                   help="MMR percentile cutoff: 100(all) 50 25 10(default,top 10%%) 1")
    r.add_argument("--period", default="past-seven",
                   help="past-seven(default) | past-three | last-patch")
    r.set_defaults(func=cmd_refresh_stats)

    sub.add_parser("refresh-cards",
                   help="rebuild the BG card knowledge base from HearthstoneJSON"
                   ).set_defaults(func=cmd_refresh_cards)

    sub.add_parser("pace", help="show the top-10% leveling/scaling pace benchmark"
                   ).set_defaults(func=cmd_pace)

    sim = sub.add_parser("similar",
                         help="cards most synergistic with X (learned from winning boards)")
    sim.add_argument("--card", required=True, help="card name, e.g. 'Brann Bronzebeard'")
    sim.add_argument("-k", type=int, default=8)
    sim.set_defaults(func=cmd_similar)

    adv = sub.add_parser("advise",
                         help="rank every possible action for a snapshot (deep brain if trained)")
    adv.add_argument("--snapshot", help="path to a snapshot JSON (else a demo board)")
    adv.add_argument("--tribe", help="comp you're building toward, e.g. Murloc")
    adv.set_defaults(func=cmd_advise)

    pk = sub.add_parser("pick", help="rank an offered choice: hero / trinket / discover")
    pk.add_argument("kind", choices=["hero", "trinket", "discover"])
    pk.add_argument("options", nargs="+", help="the offered names")
    pk.add_argument("--board", help="discover only: comma-separated current board names")
    pk.add_argument("--tribe", help="comp you're building toward (discover synergy)")
    pk.set_defaults(func=cmd_pick)

    pl = sub.add_parser("plan",
                        help="multi-turn strategy lookahead (tempo vs level/greed)")
    pl.add_argument("--snapshot", help="path to a snapshot JSON (else a demo board)")
    pl.add_argument("--horizon", type=int, default=3, help="turns to look ahead")
    pl.add_argument("--tribe", help="comp you're building toward")
    pl.set_defaults(func=cmd_plan)

    t7 = sub.add_parser("tier7",
                        help="lobby-conditioned stats from your HSReplay Tier7 sub")
    t7.add_argument("kind", choices=["hero", "comps", "trinket"])
    t7.add_argument("options", nargs="*",
                    help="hero/trinket: the offered names; comps: none")
    t7.add_argument("--tribes", required=True,
                    help="comma-separated tribes in this lobby (e.g. Beast,Mech,Naga)")
    t7.add_argument("--rating", type=int, help="your BG rating (MMR)")
    t7.add_argument("--hero", help="trinket only: your hero")
    t7.add_argument("--turn", type=int, default=8, help="trinket only: current turn")
    t7.add_argument("--duos", action="store_true", help="hero only: duos stats")
    t7.set_defaults(func=cmd_tier7)

    rm = sub.add_parser("refresh-meta",
                        help="one motion: card KB + dbf map + last-patch stats "
                             "+ playbooks + meta pack (spec req 13)")
    rm.add_argument("--mmr", type=int, default=10)
    rm.add_argument("--skip-stats", action="store_true",
                    help="offline: rebuild playbooks + meta pack from committed data")
    rm.set_defaults(func=cmd_refresh_meta)

    lb = sub.add_parser("llm-bench",
                        help="measure your LLM backend's real latency (decides "
                             "local vs hosted)")
    lb.set_defaults(func=cmd_llm_bench)

    rv = sub.add_parser("review",
                        help="grade the latest recorded game and bank the lessons")
    rv.set_defaults(func=cmd_review)
    return p


def cmd_refresh_meta(args) -> int:
    from . import cards, firestone_stats, tier7
    from .meta_pack import build_meta_pack, save_meta_pack
    from .playbooks import generate_playbooks
    from .stats import _STATS_DIR
    if not args.skip_stats:
        print("1/4 Refreshing card KB + hero powers (HearthstoneJSON latest)…")
        try:
            kb_raw = cards.build_card_kb()
            cards.save_kb(kb_raw)
            cards.save_hero_power_kb(cards.build_hero_power_kb())
            tier7.refresh_dbf_map()
        except Exception as exc:
            print("   card refresh failed (offline?):", exc)
        print(f"2/4 Refreshing Firestone stats (mmr-{args.mmr}, last-patch)…")
        try:
            firestone_stats.refresh(_STATS_DIR, mmr=args.mmr, period="last-patch")
        except Exception as exc:
            print("   stats refresh failed (offline?):", exc)
    else:
        print("1-2/4 skipped (--skip-stats)")
    print("3/4 Regenerating comp playbooks…")
    paths = generate_playbooks()
    print(f"   wrote {len(paths)} playbooks -> data/playbooks/")
    print("4/4 Building the meta pack…")
    pack = build_meta_pack()
    out = save_meta_pack(pack)
    print(f"   wrote {out} (patch_build={pack.get('patch_build') or 'unknown'})")
    print("Done. The Director reads this pack on next launch.")
    return 0


def cmd_llm_bench(_args) -> int:
    from .llm_client import LLMClient, LLMError, bench
    try:
        client = LLMClient()
    except LLMError as exc:
        print("LLM not configured:", exc)
        return 1
    cfg = client.config
    print(f"Benchmarking {cfg.backend} / {cfg.model} at {cfg.url} …")
    r = bench(client)
    if not r.get("ok"):
        print("FAILED:", r.get("error"))
        print("Local: install Ollama and `ollama pull qwen2.5:3b-instruct`, or set "
              "HSBG_LLM_BACKEND=openai + HSBG_LLM_URL/HSBG_LLM_KEY for a hosted "
              "open-weights endpoint.")
        return 1
    lat = r["latency_s"]
    print(f"OK — {lat:.2f}s for a realistic Director prompt.")
    if lat <= 2.5:
        print("Verdict: fast enough for live move+why. Use this backend.")
    else:
        print("Verdict: too slow for live turns (>2.5s). Try a smaller local model "
              "(qwen2.5:1.5b-instruct) or a hosted open-weights endpoint "
              "(HSBG_LLM_BACKEND=openai + Groq/Together URL) — picks like "
              "hero/trinket can tolerate this latency, turns can't.")
    return 0


def cmd_review(_args) -> int:
    from .llm_client import LLMClient, LLMError
    from .reviewer import (append_lessons, append_training_examples,
                           promote_experiments, review_latest)
    client = None
    try:
        client = LLMClient()
    except LLMError:
        pass                                   # deterministic-only review is fine
    review = review_latest(config.DATA_DIR, client=client)
    if review is None:
        print("No completed recorded game found in", config.DATA_DIR)
        return 1
    print(f"Reviewed {os.path.basename(review.game_path)} "
          f"(placement: {review.placement or '?'}):")
    for g in review.grades:
        print(f"  T{g.turn}: [{g.verdict}] {g.suggested}"
              + (f" — better: {g.better}" if g.better else ""))
    append_lessons(review, os.path.join(config.DATA_DIR, "lessons.jsonl"))
    promote_experiments(review)
    append_training_examples(review, review.game_path,
                             os.path.join(config.DATA_DIR, "train_corpus.jsonl"))
    print(f"Banked {len(review.lessons)} lessons; corpus updated.")
    return 0


def cmd_tier7(args) -> int:
    from . import tier7
    token = tier7.find_token()
    if not token:
        print(tier7.TOKEN_HELP)
        return 1
    tribes = [t.strip() for t in args.tribes.split(",") if t.strip()]
    try:
        if args.kind == "hero":
            if not args.options:
                print("Give the offered hero names.")
                return 1
            rows = tier7.query_hero_pick(args.options, tribes, token=token,
                                         rating=args.rating, duos=args.duos)
            print("Tier7 hero pick — best first (this lobby's tribes):")
            for i, r in enumerate(rows, 1):
                avg = f"{r['avg_placement']:.2f}" if r["avg_placement"] is not None else "?"
                pick = f" · picked {r['pick_rate']:.0%}" if r.get("pick_rate") else ""
                comps = f" · comps: {', '.join(r['top_comps'])}" if r.get("top_comps") else ""
                mark = "  ◀ PICK" if i == 1 else ""
                print(f"  {i}. {r['name']} — tier {r.get('tier') or '?'} · "
                      f"avg {avg}{pick}{comps}{mark}")
        elif args.kind == "comps":
            rows = tier7.query_comps(tribes, token=token)
            print("Tier7 first-place comps for this lobby:")
            for r in rows[:10]:
                print(f"  {r['avg_final_placement']:.2f}  {r['name'] or r['id']} "
                      f"({r['popularity']:.0%}) — {', '.join(r['key_minions'])}")
        else:                                          # trinket (raw, calibrating)
            import json
            if not args.hero or not args.options:
                print("Trinket needs --hero and the offered trinket names.")
                return 1
            resp = tier7.query_trinket_pick(args.hero, args.options, tribes,
                                            args.turn, token=token,
                                            rating=args.rating)
            print(json.dumps(resp, indent=1)[:2000])
    except (RuntimeError, ValueError) as exc:
        print(f"Tier7 query failed: {exc}")
        return 1
    return 0


def cmd_plan(args) -> int:
    import json
    from . import cards
    from .multiturn import plan_multiturn
    from .pace import load_pace
    from .economy import HeroContext
    kb = cards.load_kb()
    if args.snapshot:
        with open(args.snapshot, encoding="utf-8") as fh:
            snap = json.load(fh)
    else:
        snap = _demo_snapshot(kb)
        print("(no --snapshot — using a demo board)\n")
    plans = plan_multiturn(snap, load_pace(), horizon=args.horizon)
    if not plans:
        print("No plan (missing pace data).")
        return 1
    print(f"Strategy lookahead ({args.horizon} turns) — best first:")
    for i, p in enumerate(plans, 1):
        print(f"  {i}. {p.name}: value {p.value:.1f}{'  ⚠ DIES' if p.died else ''}")
    best = plans[0]
    print(f"\nBest strategy: {best.name} — THIS TURN: {best.this_turn.upper()}")
    for tp in best.projection:
        print(tp.line())
    _print_policy_intent(snap)
    hero_ctx = HeroContext(target_tribe=args.tribe) if args.tribe else None
    from .turn_search import plan_turn_search
    print("\nThis turn, concretely (beam-searched):")
    for i, s in enumerate(plan_turn_search(snap, kb=kb).steps, 1):
        print(f"  {i}. {s}")
    return 0


def cmd_pick(args) -> int:
    from .draft import recommend_choice
    from .economy import HeroContext
    kwargs = {}
    if args.kind in ("discover", "trinket"):
        from . import cards
        kwargs["kb"] = cards.load_kb()
        board_names = [n.strip() for n in (args.board or "").split(",") if n.strip()]
        kwargs["board"] = [{"name": n} for n in board_names]
        if args.tribe:
            kwargs["hero_ctx"] = HeroContext(target_tribe=args.tribe)
    choices = recommend_choice(args.kind, args.options, **kwargs)
    if not choices:
        print("No options given.")
        return 1
    print(f"Pick ({args.kind}) — best first:")
    for i, c in enumerate(choices, 1):
        mark = "  ◀ PICK" if i == 1 else ""
        print(f"  {i}. {c.name} — {c.reason}{mark}")
    return 0


def cmd_advise(args) -> int:
    import json
    from . import cards
    from .advisor import advise_actions
    from .economy import HeroContext
    kb = cards.load_kb()
    if args.snapshot:
        with open(args.snapshot, encoding="utf-8") as fh:
            snap = json.load(fh)
    else:
        snap = _demo_snapshot(kb)
        print("(no --snapshot given — using a demo board built from real card data)\n")
    hero_ctx = HeroContext(target_tribe=args.tribe) if args.tribe else None
    pace = None
    try:
        from .pace import load_pace
        pace = load_pace()
    except Exception:
        pass
    from .game_value import rank_actions
    recs, base = rank_actions(snap, kb=kb, hero_ctx=hero_ctx, pace=pace)
    print(f"Whole-game ranking — expected final placement (now: {base:.1f}):")
    for r in recs:
        print(r.line())
    from .turn_search import plan_turn_search
    plan = plan_turn_search(snap, kb=kb, pace=pace)
    steps = plan.steps
    print(f"\nFull-turn plan (beam-searched, expected finish "
          f"{plan.expected:.1f}, {plan.gain:+.2f} vs doing nothing):")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    return 0


def _print_policy_intent(snap):
    """If the self-play RL policy is trained, show its this-turn intent."""
    import os
    pt = os.path.join(os.path.dirname(__file__), "..", "ml", "econ_policy.pt")
    if not os.path.isfile(pt):
        return
    try:
        from ml.econ_policy import load, recommend_intent
        from ml.econ_env import alive_at
        from .pace import load_pace, _at as _curve_at
        turn = (snap.get("turn") or 8)
        tier = (snap.get("tavern_tier") or 1)
        hp = snap.get("hero_health") or 30
        strength = sum((m.get("attack") or 0) + (m.get("health") or 0)
                       for m in snap.get("board", []))
        curve = _curve_at(load_pace().get("scaling", {}), turn) or max(1.0, strength)
        intent, prob = recommend_intent(turn, tier, strength, strength / curve,
                                        hp, alive_at(turn), load(pt))
        print(f"\nSelf-play RL agent says: {intent.upper()} ({prob:.0%} confidence)")
    except Exception:
        pass


def _demo_snapshot(kb):
    """A plausible recruit-phase board built from real card2vec vocab so the
    synergy + eval scorers light up."""
    from collections import defaultdict
    from .synergy import load_embeddings
    from .cards import by_name
    emb = load_embeddings()
    idx = by_name(kb)
    by_tribe = defaultdict(list)
    for n in emb:
        ck = idx.get(n)
        if ck and ck.tribes:
            by_tribe[ck.tribes[0]].append(n)
    if not by_tribe:
        return {"turn": 6, "tavern_tier": 3, "gold": 7, "hero_health": 25,
                "board": [], "shop": [], "hand": []}
    tribe = max(by_tribe, key=lambda t: len(by_tribe[t]))
    pool = by_tribe[tribe]
    board = [{"name": pool[i], "attack": 3 + i, "health": 3 + i, "position": i + 1}
             for i in range(min(4, len(pool)))]
    buy = pool[4] if len(pool) > 4 else pool[0]
    off = next((ns[0] for t, ns in by_tribe.items() if t != tribe and ns), buy)
    shop = [{"name": buy, "attack": 4, "health": 4},
            {"name": off, "attack": 3, "health": 2}]
    return {"turn": 6, "tavern_tier": 3, "gold": 7, "hero_health": 25,
            "board": board, "shop": shop, "hand": [], "_tribe": tribe}


def cmd_similar(args) -> int:
    import math
    from .synergy import load_embeddings, _cosine
    emb = load_embeddings()
    if not emb:
        print("No card2vec embeddings. Train with `python3 -m ml.train_card2vec`.")
        return 1
    if args.card not in emb:
        print(f"'{args.card}' not in the embedding vocab.")
        return 1
    q = emb[args.card]
    sims = sorted(((n, _cosine(q, v)) for n, v in emb.items() if n != args.card),
                  key=lambda x: x[1], reverse=True)
    print(f"Cards that win alongside {args.card}:")
    for name, s in sims[:args.k]:
        print(f"  {s:.3f}  {name}")
    return 0


def cmd_pace(_args) -> int:
    from .pace import load_pace
    pace = load_pace()
    if not pace:
        print("No pace benchmark. Run `refresh-stats` first.")
        return 1
    print("High-level pace. 'tavern' = tavern tier you should be on; "
          "'stats' = board total.")
    print(f"{'turn':>4} | {'tavern':>6} | {'board-stats':>11}")
    tv, sc = pace["tavern_tier"], pace["scaling"]
    for t in range(1, 14):
        if t in tv or t in sc:
            print(f"{t:>4} | {tv.get(t, '-'):>6} | {sc.get(t, '-'):>11}")
    return 0


def cmd_refresh_cards(_args) -> int:
    from . import cards
    print("Building BG card knowledge from HearthstoneJSON…")
    try:
        kb = cards.build_card_kb()
        path = cards.save_kb(kb)
    except Exception as exc:
        print("Refresh failed:", exc)
        return 1
    print(f"Wrote {len(kb)} BG minions -> {path}")
    return 0


def cmd_refresh_stats(args) -> int:
    from . import firestone_stats
    from .stats import _STATS_DIR
    print(f"Fetching Firestone stats (mmr-{args.mmr}, {args.period})…")
    try:
        result = firestone_stats.refresh(_STATS_DIR, mmr=args.mmr, period=args.period)
    except Exception as exc:
        print("Refresh failed:", exc)
        return 1
    print(f"Wrote {result['num_heroes']} heroes -> {result['heroes']}")
    print(f"Wrote {result['num_comps']} comps  -> {result['comps']}")
    return 0


def cmd_stats(args) -> int:
    from .stats import StatsDB, build_hero_context
    db = StatsDB.load(args.hero_source, args.comp_source)  # defaults to Firestone snapshot
    tribes = [t.strip() for t in args.tribes.split(",")] if args.tribes else None
    ctx = build_hero_context(args.hero, db, available_tribes=tribes)
    comp = db.best_comp_for_hero(args.hero, available_tribes=tribes)
    print(f"Hero: {ctx.hero}")
    print(f"Target comp: {comp.name if comp else '?'} "
          f"(tribe {ctx.target_tribe}, avg place "
          f"{comp.average_position if comp else '?'}, tier {comp.tier if comp else '?'})")
    print(f"Core minions: {', '.join(ctx.recommended_minions) or '—'}")
    if comp and comp.power_turns:
        print(f"Spikes on turns: {comp.power_turns}")
    print(f"Leveling bias: {ctx.level_aggression:+.2f} "
          f"({'greedier' if ctx.level_aggression > 0 else 'more tempo' if ctx.level_aggression < 0 else 'neutral'})")
    trinkets = db.best_trinkets(5)
    if trinkets:
        print("Top trinkets: "
              + ", ".join(f"{t.name} ({t.tier}, {t.average_position:.2f})" for t in trinkets))
    return 0


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
