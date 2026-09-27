"""Build ``states.v1`` decision-point rows from Firestone HSReplay XML replays.

One row per ``Options`` block offered to the local player. Each row carries the
``BGTracker.snapshot()`` of that moment (``Snapshot.to_dict()``, rebuildable
with ``Snapshot.from_dict``) so training and the live overlay share one
encoder, plus the legal options. See ``docs/replay_states_v1.md``.

    python -m hsbg_coach.replay_states --replays data/firestone/raw \\
        --manifest data/firestone/raw/pilot_manifest.json \\
        --out data/firestone/states/v1

Writes ``<out>/<reviewId>.jsonl.gz`` for games that pass every fidelity check,
``<out>/quarantine/<reviewId>.json`` (reasons + examples) for games that do
not, and ``<out>/report.json``. Every manifest entry and every replay in the
folder is accounted for in the report; nothing is dropped silently.
"""

import argparse
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .bg import BGTracker, DARK_GIFT_PREFIX
from .hsreplay_xml import iter_events

SCHEMA_VERSION = "states.v1"
DARK_DISCOVERY_BUTTON = "BG36_Button_DarkGift"
DARK_DISCOVERY_EFFECT = "BG36_MidGameEffect_010"   # CREATOR of the offered minions
MAX_EXAMPLES = 5
DEFAULT_CARDS = os.path.join("data", "firestone", "cards.json")


# --- card names -----------------------------------------------------------
def load_card_names(path: str = DEFAULT_CARDS) -> Dict[str, str]:
    """cardId -> English name from a HearthstoneJSON cards.json (downloaded
    once to ``path`` if missing). XML replays carry no entity names."""
    from .firestone_stats import CARDS_URL, _card_meta, _fetch_url
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(_fetch_url(CARDS_URL))
    return {cid: m["name"] for cid, m in _card_meta(path).items()}


# --- per-game build -------------------------------------------------------
class _Fail:
    """Failure reasons with a count and a few examples each."""

    def __init__(self):
        self.count: Counter = Counter()
        self.examples: Dict[str, list] = defaultdict(list)

    def add(self, reason: str, example=None) -> None:
        self.count[reason] += 1
        if example is not None and len(self.examples[reason]) < MAX_EXAMPLES:
            self.examples[reason].append(example)


def _legal(options: List[Dict]) -> List[Dict]:
    return [o for o in options if o["error"] == -1]


def _hero_power(tracker: BGTracker, legal_ids: set) -> Optional[Dict]:
    """Our hero power in PLAY, including passive ones (Snapshot.hero_power is
    None for passives). ``activatable`` = it is a legal option right now."""
    for ent in tracker.state.in_zone("PLAY", tracker.local_player):
        if ent.tags.get("CARDTYPE") == "HERO_POWER":
            return {"card_id": ent.card_id,
                    "name": tracker._display_name(ent.card_id, ent.name),
                    "cost": ent.tag_int("COST") or 0,
                    "used": ent.tags.get("EXHAUSTED") == "1",
                    "activatable": ent.id in legal_ids}
    return None


def _options_rows(tracker: BGTracker, legal: List[Dict]) -> List[Dict]:
    out = []
    for o in legal:
        ent = tracker.state.entities.get(o["entity"])
        out.append({"index": o["index"], "type": o["type"], "entity_id": o["entity"],
                    "card_id": ent.card_id if ent else None,
                    "zone": ent.zone if ent else None,
                    "targets": o["targets"]})
    return out


def build_game(path: str, meta: Dict, card_names: Optional[Dict[str, str]] = None,
               track_opponents: bool = True) -> Dict:
    """Replay one game; return {rows, failures, warnings, stats}."""
    tracker = BGTracker()
    tracker.track_opponents = track_opponents
    if card_names:
        tracker.state.card_names.update(card_names)
    fail = _Fail()
    rows: List[Dict] = []
    last_options_id = None
    game_entity_id = None
    stats = Counter()
    pending_picks: List[Dict] = []
    combat_board, await_combat = None, False

    for ev in iter_events(path):
        tracker.feed(ev)
        if ev.kind == "FULL_ENTITY" and ev.entity and ev.entity.name == "GameEntity":
            game_entity_id = ev.entity.id
        elif await_combat and ev.kind == "RAW" and "BlockType=ATTACK" in ev.text:
            # Board at the first attack after a decision point (diagnostic only:
            # explains atk/health gaps vs the manifest's final comp).
            await_combat = False
            combat_board = [(m.card_id, m.attack, m.health) for m in tracker.snapshot().board]
        elif ev.kind == "CHOSEN":
            _record_dark_discovery_picks(tracker, ev.items, pending_picks, fail, stats)
        elif ev.kind == "OPTIONS":
            stats["options_blocks"] += 1
            if ev.fields.get("id") == last_options_id:
                # Same Options id re-sent right after a re-dump: one decision
                # point. Keep the post-re-dump state (authoritative).
                stats["options_resent"] += 1
                rows.pop()
            last_options_id = ev.fields.get("id")
            row = _row(tracker, ev.items, meta, len(rows), game_entity_id)
            if rows and (row["raw_turn"] or 0) < (rows[-1]["raw_turn"] or 0):
                fail.add("invariant_turn_regressed", {"dp": row["dp_index"],
                                                      "raw_turn": row["raw_turn"],
                                                      "prev": rows[-1]["raw_turn"]})
            _check_row(tracker, row, fail, stats)
            _check_picks(row, pending_picks, fail, stats)
            rows.append(row)
            await_combat, combat_board = True, None

    stats["dark_discovery_picks_unverifiable"] += len(pending_picks)
    warnings = _final_checks(tracker, rows, meta, fail, combat_board)
    stats["rows"] = len(rows)
    if stats["options_blocks"] - stats["options_resent"] != len(rows):
        fail.add("row_count_mismatch", {"options": stats["options_blocks"],
                                        "resent": stats["options_resent"], "rows": len(rows)})
    return {"rows": rows, "failures": fail, "warnings": warnings, "stats": stats}


def _row(tracker, options, meta, dp_index, game_entity_id) -> Dict:
    snap = tracker.snapshot()
    ge = tracker.state.entities.get(game_entity_id)
    raw_turn = ge.tag_int("TURN") if ge is not None else snap.turn
    legal = _legal(options)
    legal_ids = {o["entity"] for o in legal}
    dd_available = any(
        (tracker.state.entities.get(i) is not None
         and tracker.state.entities[i].card_id == DARK_DISCOVERY_BUTTON)
        for i in legal_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": meta["reviewId"],
        "build": _int(meta.get("buildNumber")),
        "mmr": _int(meta.get("mmr")),
        "lobby_tribes": [t["name"] for t in (meta.get("tribes") or {}).get("available", [])],
        "dp_index": dp_index,
        "turn": (raw_turn + 1) // 2 if raw_turn is not None else None,
        "raw_turn": raw_turn,
        "snapshot": snap.to_dict(),
        "hero_power": _hero_power(tracker, legal_ids),
        "dark_discovery": {"available": dd_available},
        "options": _options_rows(tracker, legal),
    }


def _check_row(tracker, row, fail, stats) -> None:
    snap, dp = row["snapshot"], row["dp_index"]
    if snap["gold"] is None or snap["gold"] < 0:
        fail.add("invariant_gold", {"dp": dp, "gold": snap["gold"]})
    if len(snap["board"]) > 7:
        fail.add("invariant_board_gt_7", {"dp": dp, "board": len(snap["board"])})
    if len(snap["hand"]) > 10:
        fail.add("invariant_hand_gt_10", {"dp": dp, "hand": len(snap["hand"])})
    entities = tracker.state.entities
    for o in row["options"]:
        missing = [i for i in [o["entity_id"]] + o["targets"] if i and i not in entities]
        if missing:
            fail.add("invariant_option_entity_missing", {"dp": dp, "option": o["index"],
                                                         "missing": missing})
    for zone in ("board", "hand", "shop"):
        for m in snap[zone]:
            if m["tags"].get("HAS_DARK_GIFT") != "1":
                continue
            stats["gifted_minion_snapshots"] += 1
            stats[f"gifted_{zone}"] += 1
            if m["dark_gift"] is None:
                fail.add("dark_gift_unresolved", {"dp": dp, "zone": zone,
                                                  "entity_id": m["entity_id"],
                                                  "card_id": m["card_id"]})


def _record_dark_discovery_picks(tracker, chosen, pending, fail, stats) -> None:
    entities = tracker.state.entities
    for eid in chosen:
        ent = entities.get(eid)
        if ent is None:
            continue
        creator = entities.get(ent.tag_int("CREATOR") or -1)
        if creator is None or creator.card_id != DARK_DISCOVERY_EFFECT:
            continue
        stats["dark_discovery_picks"] += 1
        gift = entities.get(ent.tag_int("DARK_GIFT_ENTITY") or -1)
        if gift is None or not (gift.card_id or "").startswith(DARK_GIFT_PREFIX):
            fail.add("dark_discovery_pick_without_gift", {"entity_id": eid,
                                                         "card_id": ent.card_id})
            continue
        pending.append({"entity_id": eid, "card_id": ent.card_id, "gift": gift.card_id,
                        "persistent": ent.tags.get("HAS_DARK_GIFT") == "1"})


def _check_picks(row, pending, fail, stats) -> None:
    """A Dark Discovery pick's gift must be on a friendly minion at the next
    decision point. One-shot gifts (Double Vision: "get an extra copy of
    this") are spent on pick: the offered minion never gets HAS_DARK_GIFT, so
    there is nothing to find on the board; they are counted, not checked."""
    if not pending:
        return
    snap = row["snapshot"]
    gifts = {m["dark_gift"]["card_id"] for z in ("board", "hand")
             for m in snap[z] if m.get("dark_gift")}
    for p in pending:
        if not p["persistent"]:
            stats["dark_discovery_picks_one_shot"] += 1
        elif p["gift"] in gifts:
            stats["dark_discovery_picks_verified"] += 1
        else:
            fail.add("dark_discovery_pick_missing", dict(p, dp=row["dp_index"]))
    pending.clear()


def _final_checks(tracker, rows, meta, fail, combat_board=None) -> List[Dict]:
    """Final board vs manifest finalComp (card ids + golden, in order; atk/health
    only warns, with the board at the next combat's first attack alongside) and
    final placement vs manifest placement."""
    warnings: List[Dict] = []
    if not rows:
        fail.add("no_decision_points")
        return warnings
    ours = [(m["card_id"], m["tags"].get("PREMIUM") == "1") for m in rows[-1]["snapshot"]["board"]]
    fc = meta.get("finalComp") or {}
    fc = fc.get("board", []) if isinstance(fc, dict) else fc
    theirs = [(m["cardId"], bool(m.get("golden"))) for m in fc]
    if ours != theirs:
        reason = ("final_board_order" if Counter(ours) == Counter(theirs)
                  else "final_board_mismatch")
        fail.add(reason, {"ours": ours, "manifest": theirs})
    else:
        aligned = (combat_board is not None
                   and [c for c, _, _ in combat_board] == [c for c, _ in ours])
        for slot, (m, f) in enumerate(zip(rows[-1]["snapshot"]["board"], fc)):
            if (m["attack"], m["health"]) != (f.get("atk"), f.get("health")):
                warnings.append({"slot": slot, "card_id": m["card_id"],
                                 "ours": [m["attack"], m["health"]],
                                 "manifest": [f.get("atk"), f.get("health")],
                                 "first_attack": (list(combat_board[slot][1:])
                                                  if aligned else None)})
    place = _placement(tracker)
    if place != _int(meta.get("placement")):
        fail.add("placement_mismatch", {"ours": place, "manifest": meta.get("placement")})
    return warnings


def _placement(tracker) -> Optional[int]:
    hero = tracker._hero_entity()
    if hero is not None and hero.tag_int("PLAYER_LEADERBOARD_PLACE") is not None:
        return hero.tag_int("PLAYER_LEADERBOARD_PLACE")
    return tracker.placement()


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --- corpus run -----------------------------------------------------------
def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def run(replays: str, manifest: str, out: str, cards: str = DEFAULT_CARDS,
        limit: Optional[int] = None, track_opponents: bool = True) -> Dict:
    t_start = time.time()
    games = json.load(open(manifest, encoding="utf-8"))
    games = games.get("games", games) if isinstance(games, dict) else games
    by_id = {g["reviewId"]: g for g in games}
    on_disk = {f[:-len(".xml.gz")] for f in os.listdir(replays) if f.endswith(".xml.gz")}
    ids = sorted(set(by_id) | on_disk)[:limit]
    qdir = os.path.join(out, "quarantine")
    os.makedirs(qdir, exist_ok=True)
    card_names = load_card_names(cards)

    per_game, by_reason = [], defaultdict(list)
    totals = Counter()
    for gid in ids:
        t0 = time.time()
        rec = {"game_id": gid}
        failures, warnings, stats, rows = _Fail(), [], Counter(), []
        path = os.path.join(replays, gid + ".xml.gz")
        if gid not in by_id:
            failures.add("not_in_manifest")
        elif gid not in on_disk:
            failures.add("missing_replay")
        else:
            try:
                res = build_game(path, by_id[gid], card_names, track_opponents)
                failures, warnings, stats, rows = (res["failures"], res["warnings"],
                                                  res["stats"], res["rows"])
            except Exception as exc:     # a crash quarantines the game, never drops it
                failures.add("exception", f"{type(exc).__name__}: {exc}")
        states_path = os.path.join(out, gid + ".jsonl.gz")
        q_path = os.path.join(qdir, gid + ".json")
        rec.update(passed=not failures.count, reasons=sorted(failures.count),
                   rows=len(rows), atk_health_warnings=len(warnings), bytes=0,
                   stats=dict(stats))
        if rec["passed"]:
            with gzip.open(states_path + ".tmp", "wt", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            os.replace(states_path + ".tmp", states_path)
            rec["bytes"] = os.path.getsize(states_path)
            if os.path.exists(q_path):
                os.remove(q_path)
        else:
            _write_json(q_path, {"game_id": gid, "reasons": dict(failures.count),
                                 "examples": dict(failures.examples),
                                 "atk_health_warnings": warnings, "stats": dict(stats)})
            if os.path.exists(states_path):
                os.remove(states_path)
            for r in failures.count:
                by_reason[r].append(gid)
        rec["warnings"] = warnings
        rec["runtime_s"] = round(time.time() - t0, 2)
        totals.update(stats)
        per_game.append(rec)
        print(f"{gid} {'PASS' if rec['passed'] else 'FAIL ' + ','.join(rec['reasons'])} "
              f"rows={rec['rows']} bytes={rec['bytes']} {rec['runtime_s']}s", flush=True)

    passed = sum(g["passed"] for g in per_game)
    report = {
        "schema_version": SCHEMA_VERSION,
        "games_processed": len(per_game),
        "passed": passed,
        "pass_rate": round(passed / len(per_game), 4) if per_game else None,
        "failures_by_reason": dict(by_reason),
        "rows_written": sum(g["rows"] for g in per_game if g["passed"]),
        "bytes_written": sum(g["bytes"] for g in per_game),
        "runtime_s": round(time.time() - t_start, 2),
        "track_opponents": track_opponents,
        "totals": dict(totals),
        "games": per_game,
    }
    _write_json(os.path.join(out, "report.json"), report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replays", required=True, help="folder of <reviewId>.xml.gz")
    ap.add_argument("--manifest", required=True, help="manifest JSON ({games:[...]})")
    ap.add_argument("--out", default=os.path.join("data", "firestone", "states", "v1"))
    ap.add_argument("--cards", default=DEFAULT_CARDS,
                    help="HearthstoneJSON cards.json (downloaded here if missing)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-opponents", action="store_true",
                    help="skip per-event opponent capture (faster; opponents_seen/"
                         "opponent_profiles stay empty)")
    args = ap.parse_args(argv)
    rep = run(args.replays, args.manifest, args.out, args.cards, args.limit,
              track_opponents=not args.no_opponents)
    print(f"passed {rep['passed']}/{rep['games_processed']} rows={rep['rows_written']} "
          f"bytes={rep['bytes_written']} runtime={rep['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
