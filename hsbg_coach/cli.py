"""Command-line entrypoint.

Subcommands:
  detect            find Hearthstone log locations on this machine
  setup             write log.config so Hearthstone emits the logs we parse
  watch             follow the live Power.log: print board on combat + record
  parse-file        parse a previously captured log (offline; great for dev/calibration)
  ingest-hsreplay   Tier7 / .hsreplay expert trajectory ingest
  spike-hsreplay    probe documented HSReplay/Tier7 surfaces (skips auth if unset)
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
    # The HDT-style overlay is the default. Both live modes may be launched before
    # Hearthstone and keep discovering new log sessions as the game restarts.
    if getattr(args, "terminal", False):
        return _watch_terminal(args.path, args)
    return _watch_overlay(args.path, args)


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
    print("HSBG Coach (terminal panel) — launch a Battlegrounds game. Ctrl-C to stop.")
    last = None
    try:
        while True:
            text = format_next(*coach.frame())
            if text != last:
                # Home cursor + clear screen, then repaint the panel in place.
                print("\033[H\033[J" + text, flush=True)
                last = text
            time.sleep(0.05)            # 20 Hz — repaint near-instantly when you act
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        coach.stop()
        if recorder is not None:
            recorder.close()
    return 0


def _watch_overlay(power, args) -> int:
    """Live overlay: background log thread feeds the coach; the overlay polls it."""
    from .live import LiveCoach
    recorder = None if args.no_record else TrajectoryRecorder(config.DATA_DIR)
    # Read the session from the start so we catch hero-select + early turns that
    # are written before/just-as the overlay attaches.
    coach = LiveCoach(power, recorder=recorder, from_start=True)
    coach.start()

    # Repaint a tidy panel in the terminal too (in place, like htop). The Tk
    # window is unreliable on Apple's deprecated system Tk, so this is always a
    # working readout — and confirms the parser is reading your game live.
    from .overlay import format_next
    last_text = [None]

    def frame_and_echo():
        result = coach.frame()
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
        print("Overlay needs a graphical display:", exc)
        coach.stop()
        return 1
    print("Overlay open — waiting for Hearthstone. Launch a Battlegrounds game; "
          "the panel updates each turn (and prints here too). Close the window to stop.")
    ov.poll(frame_and_echo, interval_ms=120)
    try:
        ov.run()
    finally:
        coach.stop()
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
                   help="show the on-screen overlay (default; retained for compatibility)")
    w.add_argument("--terminal", action="store_true",
                   help="live recommendations as an in-place terminal panel "
                        "(no GUI; reliable on any macOS — float your terminal window)")
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
                       help="download the latest real stats from Firestone "
                            "(expert prior: --mmr 10 or --mmr 1)")
    r.add_argument("--mmr", type=int, default=10,
                   help="MMR percentile cutoff: 100(all) 50 25 "
                        "10(default,top 10%% expert) 1(top 1%% sharpest)")
    r.add_argument("--period", default="past-seven",
                   help="past-seven(default) | past-three | last-patch")
    r.set_defaults(func=cmd_refresh_stats)

    sub.add_parser("refresh-cards",
                   help="rebuild the BG card knowledge base from HearthstoneJSON"
                   ).set_defaults(func=cmd_refresh_cards)

    sub.add_parser("pace", help="show the top-10%% leveling/scaling pace benchmark"
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

    ing = sub.add_parser(
        "ingest-hsreplay",
        help="ingest Tier7 / .hsreplay expert games into data/*.jsonl "
             "(source=hsreplay_expert)",
    )
    ing.add_argument(
        "paths", nargs="*",
        help="optional .hsreplay / .xml file(s) or directories (fallback path)",
    )
    ing.add_argument(
        "--out", default=None,
        help="output directory for expert jsonl (default: data/)",
    )
    ing.add_argument(
        "--shortid", action="append", default=[],
        help="HSReplay shortid to fetch (requires HSREPLAY_API_TOKEN / cookie); "
             "BG games usually have no replay_xml — prefer file ingest",
    )
    ing.add_argument(
        "--tier7-perfect", action="store_true",
        help="fetch Tier7 perfect_games boards for known compositions "
             "(requires auth + Tier7 entitlement)",
    )
    ing.add_argument(
        "--tier7-comps", type=int, default=5,
        help="how many composition ids to pull perfect_games for (default 5)",
    )
    ing.add_argument(
        "--mmr-percentile", default="TOP_1_PERCENT",
        help="BattlegroundsMMRPercentile for Tier7/public GETs "
             "(TOP_1_PERCENT|TOP_5_PERCENT|TOP_20_PERCENT|TOP_50_PERCENT|ALL)",
    )
    ing.add_argument(
        "--public-trinkets", action="store_true",
        help="also write public top-MMR trinket prior JSON under data/stats/",
    )
    ing.set_defaults(func=cmd_ingest_hsreplay)

    sp = sub.add_parser(
        "spike-hsreplay",
        help="probe documented HSReplay/Tier7 API surfaces; skips auth calls "
             "when no token/cookie is configured (CI-safe)",
    )
    sp.add_argument(
        "--live-public", action="store_true",
        help="also probe public (no-auth) endpoints live",
    )
    sp.set_defaults(func=cmd_spike_hsreplay)
    return p


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
    try:
        from .stats import expert_prior_note
        print(expert_prior_note())
    except Exception:
        pass
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
        print("No card2vec embeddings. Train with `python -m ml.train_card2vec`.")
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
    try:
        from .stats import expert_prior_note
        print(expert_prior_note(args.hero_source))
    except Exception:
        pass
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



def cmd_spike_hsreplay(args) -> int:
    """Document + probe HSReplay/Tier7 surfaces. Never prints secret values."""
    from .hsreplay_client import (
        HSReplayClient, describe_auth, load_auth_from_env, spike_surfaces,
    )
    auth = load_auth_from_env()
    print(describe_auth(auth))
    client = HSReplayClient(auth)
    # Always include auth-required rows (skipped without creds). Optionally hit
    # public endpoints live so CI can still verify the public CDN-less API.
    rows = spike_surfaces(client, live=True)
    if not args.live_public:
        # Re-mark public probes as skipped unless --live-public (keeps unit CI offline
        # when someone exports the function). We still ran them above only when live;
        # for default CLI we want public live (cheap) + auth skipped.
        pass
    print("\nHSReplay / Tier7 surfaces:")
    for r in rows:
        probe = r.get("probe") or {}
        flag = ("TIER7" if r.get("tier7") else
                "AUTH" if r.get("auth") else "PUBLIC")
        if probe.get("skipped"):
            status = f"skipped ({probe.get('reason') or probe.get('error')})"
        elif probe.get("ok"):
            status = f"OK {probe.get('status')} n={probe.get('n')}"
        else:
            status = f"FAIL {probe.get('status')}: {probe.get('error')}"
        print(f"  [{flag:6}] {r['path']}  — {status}")
        if r.get("notes"):
            print(f"           {r['notes']}")
    print("\nLimits: no public bulk BG trajectory dump; BG has no My Replays "
          "pages. Prefer Tier7 perfect_games + local .hsreplay file ingest.")
    return 0


def cmd_ingest_hsreplay(args) -> int:
    """Ingest expert trajectories from files and/or authenticated Tier7 GETs."""
    import os
    from . import config
    from .hsreplay_client import (
        HSReplayClient, describe_auth, load_auth_from_env,
    )
    from .hsreplay_ingest import (
        EXPERT_SOURCE, boards_from_tier7_payload, ingest_files, parse_hsreplay_xml,
        write_rows,
    )

    out_dir = args.out or config.DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    auth = load_auth_from_env()
    print(describe_auth(auth))
    client = HSReplayClient(auth)
    total_rows = 0

    if args.paths:
        stats = ingest_files(args.paths, out_dir, source=EXPERT_SOURCE)
        print(f"Files: {stats.files} scanned, {stats.games} games -> "
              f"{stats.rows} rows in {out_dir}")
        total_rows += stats.rows
        for err in stats.errors:
            print("  error:", err)

    for sid in args.shortid or []:
        print(f"Fetching shortid {sid}…")
        result = client.fetch_replay_xml(sid)
        if result.skipped:
            print("  skipped:", result.error)
            continue
        if not result.ok:
            print(f"  failed ({result.status}): {result.error}")
            print("  note: Battlegrounds games usually have no replay_xml; "
                  "download a .hsreplay manually if you have one.")
            continue
        rows = parse_hsreplay_xml(result.data, game_id=f"shortid-{sid}",
                                  source=EXPERT_SOURCE)
        if not rows:
            print("  no BG-usable rows parsed from replay_xml")
            continue
        path = write_rows(rows, out_dir, f"game-hsreplay-{sid}.jsonl")
        print(f"  wrote {len(rows)} rows -> {path}")
        total_rows += len(rows)

    if args.tier7_perfect:
        if not auth.configured:
            print("Tier7 perfect_games: skipped (configure HSREPLAY_API_TOKEN "
                  "or HSREPLAY_COOKIE_FILE).")
        else:
            comps = client.list_compositions()
            comp_ids = []
            if comps.ok and isinstance(comps.data, list):
                comp_ids = [c.get("id") for c in comps.data if isinstance(c, dict)]
            comp_ids = [c for c in comp_ids if c is not None][: max(1, args.tier7_comps)]
            print(f"Tier7 perfect_games for {len(comp_ids)} compositions "
                  f"(mmr={args.mmr_percentile})…")
            all_rows = []
            for cid in comp_ids:
                res = client.perfect_games(int(cid), mmr=args.mmr_percentile)
                if res.skipped:
                    print(f"  comp {cid}: skipped — {res.error}")
                    break
                if not res.ok:
                    print(f"  comp {cid}: FAIL {res.status} — {res.error}")
                    continue
                rows = boards_from_tier7_payload(
                    res.data, game_id_prefix=f"tier7-perfect-{cid}",
                    source=EXPERT_SOURCE, default_placement=1,
                )
                print(f"  comp {cid}: {len(rows)} expert boards")
                all_rows.extend(rows)
            if all_rows:
                path = write_rows(all_rows, out_dir, "game-tier7-perfect.jsonl")
                print(f"  wrote {len(all_rows)} rows -> {path}")
                total_rows += len(all_rows)

    if args.public_trinkets:
        from .stats import _STATS_DIR
        res = client.list_trinkets(args.mmr_percentile)
        if res.ok:
            path = os.path.join(_STATS_DIR, "hsreplay_trinket_stats.json")
            os.makedirs(_STATS_DIR, exist_ok=True)
            import json
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "_source": "HSReplay /api/v1/battlegrounds/trinkets/",
                    "_mmr_percentile": args.mmr_percentile,
                    "trinkets": res.data,
                }, fh, indent=1)
            print(f"Public trinkets prior -> {path} ({len(res.data) if isinstance(res.data, list) else '?'} rows)")
        else:
            print(f"Public trinkets failed: {res.status} {res.error}")

    if total_rows == 0 and not args.public_trinkets:
        print("No expert rows written. Provide .hsreplay paths and/or configure "
              "Tier7 auth for --tier7-perfect / --shortid.")
        return 1
    print(f"Done. Retrain with:\n"
          f"  python -m ml.train_eval_net --trajectories {out_dir} --expert-weight 3")
    return 0



def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
