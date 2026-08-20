"""The between-game grader (spec `llm-turn-director_spec.md` req 7).

When a game ends, this module replays the Director's own suggestions
(`data/suggestions/<date>.jsonl`) against the recorded trajectory
(`hsbg_coach/recorder.py` output) and the known final placement, and grades
each suggested move with hindsight signals:

  * **board equity delta** — did the board get stronger or weaker over the
    turn, via an injectable ``scorer`` (default `board_value.get_scorer()`,
    same interface the live advisor uses: ``.equity(board, hero_id)``).
  * **placement** — a bad finish scrutinizes greedy calls harder; a good
    finish gives borderline calls the benefit of the doubt.
  * **leveling vs. the top-10% pace curve** (`pace.py`, spec req 9).
  * **gold float** — ending a turn with spendable gold and no stated
    justification is a bug per spec req 2, independent of equity.

This is the deterministic core the spec asks for: it must finish inside the
queue/hero-pick window and must work with **no LLM available**. An optional
``client`` (duck-typed ``.chat(system, user) -> str``, e.g.
`hsbg_coach.llm_client.LLMClient`) may be given to turn the deterministic
grades into sharper one-line lessons; absent a client, lessons are
templated from the grades directly. Every entry point degrades to
verdict "unknown" (or a templated lesson) on missing/malformed data —
this module never raises on a partial game.
"""

import bisect
import glob
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .board_value import get_scorer
from .pace import load_pace, pace_advice

# --- tunables ---------------------------------------------------------------
# No ground-truth "was this actually a good grade" dataset exists yet (that's
# what this module bootstraps). These thresholds are a reasonable starting
# point; revisit once real games + owner feedback accumulate. # CALIBRATE
EQUITY_EPS = 0.01          # |delta| below this is noise, not signal
PLACEMENT_GOOD = 3         # finished top-3 of 8 -> give borderline calls credit
PLACEMENT_BAD = 5          # finished 5th+ of 8 -> scrutinize borderline calls
PLACEMENT_BIAS = 0.05      # how much the game result nudges a borderline delta
MAX_LESSONS_PER_GAME = 5   # cap templated bad-turn lessons per game (noise control)
LESSONS_CAP = 200          # data/lessons.jsonl newest-N kept


@dataclass
class TurnGrade:
    turn: int
    suggested: str            # the Director's chosen move label, "" if none logged
    verdict: str               # "good" | "questionable" | "bad" | "unknown"
    better: str                 # what hindsight preferred; "" if nothing better found
    note: str
    experiment: bool


@dataclass
class GameReview:
    game_path: str
    placement: Optional[int]
    grades: List[TurnGrade] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    experiments: List[Dict] = field(default_factory=list)


# --- trajectory / suggestions loading ---------------------------------------

def _load_jsonl(path: Optional[str]) -> List[Dict]:
    """Read a JSONL file into a list of dicts. Missing file / bad lines never
    raise — malformed lines are skipped, a missing file yields []."""
    out: List[Dict] = []
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return out
    return out


def _group_by_turn(decisions: List[Dict]) -> Dict[int, List[Dict]]:
    by_turn: Dict[int, List[Dict]] = {}
    for d in decisions:
        t = (d.get("state") or {}).get("turn")
        if t is None:
            continue
        by_turn.setdefault(t, []).append(d)
    return by_turn


def _group_suggestions_by_turn(suggestions: List[Dict]) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for s in suggestions:
        t = s.get("turn")
        if t is None:
            continue
        out[t] = s   # last logged suggestion for a turn wins (re-suggest case)
    return out


def _extract_placement(decisions: List[Dict]) -> Optional[int]:
    for d in decisions:
        p = d.get("placement")
        if p is not None:
            return p
    return None


_GAME_ID_RE = re.compile(r"(\d{8})-(\d{6})")


def _default_suggestions_path(trajectory_path: str) -> Optional[str]:
    """Auto-discover `data/suggestions/<date>.jsonl` next to a trajectory file
    named the way `recorder.TrajectoryRecorder` names it by default
    (`game-YYYYMMDD-HHMMSS.jsonl`). Returns None (not an error) when the
    filename doesn't carry that pattern (custom game_id) or no matching
    suggestions file exists — callers then grade from the trajectory alone.
    # CALIBRATE: suggestions filename date format (assumed YYYY-MM-DD) —
    confirm against whatever writes data/suggestions/<date>.jsonl."""
    m = _GAME_ID_RE.search(os.path.basename(trajectory_path))
    if not m:
        return None
    ymd = m.group(1)
    date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
    data_dir = os.path.dirname(os.path.abspath(trajectory_path))
    path = os.path.join(data_dir, "suggestions", f"{date_str}.jsonl")
    return path if os.path.isfile(path) else None


def _next_turn_board(traj_turns: List[int], by_turn: Dict[int, List[Dict]],
                     turn: int) -> Optional[List]:
    idx = bisect.bisect_right(traj_turns, turn)
    if idx < len(traj_turns):
        nxt = traj_turns[idx]
        return (by_turn[nxt][0].get("state") or {}).get("board")
    return None


def _hero_of(decisions: List[Dict]) -> str:
    for d in decisions:
        h = (d.get("state") or {}).get("hero")
        if h:
            return h
    return "UNKNOWN"


def _equity(scorer, board, hero: str) -> Optional[float]:
    if scorer is None or not board:
        return None
    try:
        return float(scorer.equity(board, hero))
    except Exception:
        return None


def _gold_float(turn_decisions: List[Dict]):
    """(floated, gold) — True when an end_turn decision this turn left
    spendable gold on the table (spec req 2: a bug unless justified)."""
    for d in turn_decisions:
        if d.get("action_type") == "end_turn":
            gold = (d.get("state") or {}).get("gold")
            if isinstance(gold, (int, float)) and gold > 0:
                return True, int(gold)
    return False, 0


def _justified_float(chosen: Dict) -> bool:
    """Heuristic: does the Director's own words justify holding gold this
    turn (infinite economy, deliberate freeze/save — spec req 2)?
    # CALIBRATE: keyword list; tighten once real `why`/`hypothesis` text
    from the live Director is available."""
    if not chosen:
        return False
    if chosen.get("experiment"):
        return True
    text = f"{chosen.get('why', '')} {chosen.get('hypothesis', '')}".lower()
    return any(k in text for k in ("freeze", "save gold", "save for", "bank",
                                    "infinite econ", "infinite gold"))


def _mentions_leveling(chosen: Dict, decisions: List[Dict]) -> bool:
    text = f"{(chosen or {}).get('move', '')} {(chosen or {}).get('why', '')}".lower()
    if any(k in text for k in ("level", "tier up", "tier-up")):
        return True
    return any(d.get("action_type") == "tier_up" for d in decisions)


def _pick_better(candidates: List[Dict], chosen_move: str) -> str:
    """What hindsight preferred, from the candidates actually offered this
    turn — the first candidate that isn't the move that was chosen."""
    for c in candidates or []:
        action = c.get("action") or ""
        if not action or action == chosen_move:
            continue
        detail = c.get("detail") or {}
        return f"{action} {detail}".strip() if detail else action
    return ""


def _grade_turn(turn: int, decisions: List[Dict], suggestion: Optional[Dict],
                scorer, pace_data, next_board, placement: Optional[int]) -> TurnGrade:
    decisions = decisions or []
    chosen = (suggestion or {}).get("chosen") or {}
    candidates = (suggestion or {}).get("candidates") or []
    suggested = chosen.get("move", "") if suggestion else ""
    experiment = bool(chosen.get("experiment"))
    notes: List[str] = []

    if not decisions:
        # A suggestion was logged but the game trajectory has no record of
        # that turn (e.g. the game ended before it was reached).
        note = "no trajectory data for this turn"
        return TurnGrade(turn, suggested, "unknown", "", note, experiment)

    hero = _hero_of(decisions)
    board_before = decisions[0].get("state", {}).get("board") or []
    board_after = next_board if next_board is not None else (
        decisions[-1].get("state", {}).get("board") or [])
    eq_before = _equity(scorer, board_before, hero)
    eq_after = _equity(scorer, board_after, hero)
    delta: Optional[float] = None
    if eq_before is not None and eq_after is not None:
        delta = eq_after - eq_before
        notes.append(f"board equity {eq_before:.2f} -> {eq_after:.2f} ({delta:+.2f})")

    floated, gold = _gold_float(decisions)
    unjustified_float = floated and not _justified_float(chosen)
    if floated:
        notes.append(f"{'floated' if unjustified_float else 'held'} {gold} gold "
                     f"at end of turn"
                     + ("" if unjustified_float else " (justified)"))

    if not suggestion:
        # Trajectory-only grading: no suggestion was logged for this turn.
        if delta is None and not floated:
            verdict = "unknown"
        elif unjustified_float or (delta is not None and delta < -EQUITY_EPS):
            verdict = "bad"
        elif delta is not None and delta > EQUITY_EPS:
            verdict = "good"
        else:
            verdict = "questionable"
        return TurnGrade(turn, "", verdict, "", "; ".join(notes) or
                         "no suggestion logged for this turn", False)

    if delta is None and not unjustified_float:
        verdict = "unknown"
    else:
        bias = 0.0
        if placement is not None:
            if placement <= PLACEMENT_GOOD:
                bias = PLACEMENT_BIAS
            elif placement >= PLACEMENT_BAD:
                bias = -PLACEMENT_BIAS
        adjusted = (delta if delta is not None else 0.0) + bias
        if unjustified_float:
            verdict = "bad"
        elif adjusted > EQUITY_EPS:
            verdict = "good"
        elif adjusted < -EQUITY_EPS:
            verdict = "bad"
        else:
            verdict = "questionable"

    # Leveling-vs-pace check (spec req 9) — only meaningful for turns where
    # leveling was actually in play; nudges a borderline ("questionable")
    # verdict, never overrides a clear equity-driven good/bad.
    if pace_data is not None and _mentions_leveling(chosen, decisions):
        try:
            pv = pace_advice(decisions[0]["state"], pace_data)
            if pv.notes:
                notes.append("pace: " + "; ".join(pv.notes))
            if verdict == "questionable":
                low = suggested.lower()
                if pv.behind_leveling and "level" not in low and "tier" not in low:
                    verdict = "bad"
                elif pv.ahead_leveling and ("level" in low or "tier" in low):
                    verdict = "questionable"
        except Exception:
            pass

    better = ""
    if verdict == "bad":
        better = _pick_better(candidates, suggested)
        if not better and delta is not None and delta < -EQUITY_EPS:
            better = "hold — that buy/sell/roll lost board equity this turn"

    return TurnGrade(turn=turn, suggested=suggested, verdict=verdict, better=better,
                     note="; ".join(notes), experiment=experiment)


def review_game(trajectory_path: str, suggestions_path: Optional[str] = None,
                scorer=None, client=None) -> GameReview:
    """Grade one game's suggestions against its recorded trajectory.

    Deterministic core (spec req 7): pairs `data/suggestions/<date>.jsonl`
    entries to trajectory turns by turn number, tolerating either side being
    missing, and grades with hindsight board equity + placement + leveling
    pace + gold-float signals. `scorer` defaults to
    `board_value.get_scorer()` (torch-free heuristic when no eval net is
    trained) but accepts a stub so callers/tests never need torch. `client`
    is optional and only sharpens the lesson text — grading itself never
    depends on an LLM being reachable."""
    decisions = _load_jsonl(trajectory_path)
    placement = _extract_placement(decisions)

    by_turn = _group_by_turn(decisions)
    traj_turns = sorted(by_turn.keys())

    if suggestions_path is None:
        suggestions_path = _default_suggestions_path(trajectory_path)
    suggestions = _load_jsonl(suggestions_path)
    sugg_by_turn = _group_suggestions_by_turn(suggestions)

    if scorer is None:
        try:
            scorer = get_scorer()
        except Exception:
            scorer = None

    try:
        pace_data = load_pace()
    except Exception:
        pace_data = None

    all_turns = sorted(set(traj_turns) | set(sugg_by_turn.keys()))
    grades: List[TurnGrade] = []
    for t in all_turns:
        decs = by_turn.get(t, [])
        sugg = sugg_by_turn.get(t)
        next_board = _next_turn_board(traj_turns, by_turn, t)
        try:
            g = _grade_turn(t, decs, sugg, scorer, pace_data, next_board, placement)
        except Exception:
            suggested = (sugg or {}).get("chosen", {}).get("move", "") if sugg else ""
            g = TurnGrade(turn=t, suggested=suggested, verdict="unknown", better="",
                         note="grading error on this turn", experiment=False)
        grades.append(g)

    experiments: List[Dict] = []
    for t in all_turns:
        sugg = sugg_by_turn.get(t)
        chosen = (sugg or {}).get("chosen") or {}
        if sugg and chosen.get("experiment"):
            g = next((gr for gr in grades if gr.turn == t), None)
            experiments.append({
                "hypothesis": chosen.get("hypothesis", ""),
                "suggested": chosen.get("move", ""),
                "verdict": g.verdict if g else "unknown",
            })

    review = GameReview(game_path=trajectory_path, placement=placement,
                        grades=grades, lessons=[], experiments=experiments)
    review.lessons = _build_lessons(review, client)
    return review


# --- lessons ------------------------------------------------------------

def _templated_lessons(review: GameReview) -> List[str]:
    bad = [g for g in review.grades if g.verdict == "bad"]
    good = [g for g in review.grades if g.verdict == "good"]
    lessons: List[str] = []
    for g in bad[:MAX_LESSONS_PER_GAME]:
        msg = f"Turn {g.turn}: '{g.suggested or '(no suggestion logged)'}' graded bad"
        if g.better:
            msg += f" — hindsight preferred '{g.better}'"
        if g.note:
            msg += f" ({g.note})"
        lessons.append(msg)
    if review.placement is not None:
        lessons.append(f"Game placed {review.placement}/8 — "
                       f"{len(good)} good call(s), {len(bad)} bad call(s) graded.")
    if not lessons:
        lessons.append("No gradeable turns this game "
                       "(insufficient suggestion/trajectory overlap).")
    return lessons


def _parse_llm_lessons(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("lessons"), list):
            return [str(x).strip() for x in obj["lessons"] if str(x).strip()]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Non-JSON reply (a plain-text stub client, or json_mode wasn't honored):
    # treat each non-empty line as one lesson.
    return [ln.strip("-* ").strip() for ln in raw.splitlines() if ln.strip()]


def _build_lessons(review: GameReview, client) -> List[str]:
    templated = _templated_lessons(review)
    if client is None:
        return templated
    try:
        system = (
            "You are the between-game reviewer for a Hearthstone Battlegrounds "
            "coach. Turn deterministic turn-grades into sharp, one-line lessons "
            'the coach can reuse next game. Reply with strict JSON: '
            '{"lessons": ["...", "..."]}. No prose outside the JSON.'
        )
        user = json.dumps({
            "placement": review.placement,
            "grades": [
                {"turn": g.turn, "suggested": g.suggested, "verdict": g.verdict,
                 "better": g.better, "note": g.note} for g in review.grades
            ],
        }, separators=(",", ":"))
        raw = client.chat(system, user)
        lessons = _parse_llm_lessons(raw)
        if lessons:
            return lessons
    except Exception:
        pass
    return templated


def append_lessons(review: GameReview, path: str = "data/lessons.jsonl") -> None:
    """Append one JSON line per lesson, capped at the newest LESSONS_CAP."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing = _load_jsonl(path)
    ts = time.time()
    for lesson in review.lessons:
        existing.append({
            "ts": ts,
            "lesson": lesson,
            "evidence": _evidence_for(lesson, review),
            "game": review.game_path,
        })
    existing = existing[-LESSONS_CAP:]
    with open(path, "w", encoding="utf-8") as fh:
        for e in existing:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")


_TURN_LESSON_RE = re.compile(r"^Turn (\d+):")


def _evidence_for(lesson: str, review: GameReview) -> str:
    m = _TURN_LESSON_RE.match(lesson)
    if m:
        t = int(m.group(1))
        g = next((gr for gr in review.grades if gr.turn == t), None)
        if g and g.note:
            return g.note
        if g:
            return f"verdict={g.verdict}"
    return f"placement={review.placement}"


def load_lessons_for_prompt(path: str = "data/lessons.jsonl",
                            max_chars: int = 1500) -> str:
    """Newest-first compact text for the Director's context, capped at
    max_chars (truncates by whole lesson, never mid-line)."""
    entries = _load_jsonl(path)
    out: List[str] = []
    total = 0
    for e in reversed(entries):
        text = f"- {e.get('lesson', '')}"
        if total + len(text) + 1 > max_chars:
            break
        out.append(text)
        total += len(text) + 1
    return "\n".join(out)


# --- experiments (spec req 12) -------------------------------------------

_VALIDATED_HEADING = "## Validated"
_FAILED_HEADING = "## Do not repeat"


def _read_experiment_sections(path: str):
    if not os.path.isfile(path):
        return [], []
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return [], []
    validated: List[str] = []
    failed: List[str] = []
    section = None
    for line in text.splitlines():
        s = line.strip()
        if s == _VALIDATED_HEADING:
            section = "v"
            continue
        if s == _FAILED_HEADING:
            section = "f"
            continue
        if s.startswith("- "):
            if section == "v":
                validated.append(s)
            elif section == "f":
                failed.append(s)
    return validated, failed


def promote_experiments(review: GameReview,
                        path: str = "data/playbooks/_experiments.md") -> None:
    """Append validated (`verdict == "good"`) experiments as markdown bullets;
    failed (`verdict == "bad"`) ones go under "Do not repeat" so they aren't
    tried again. Experiments still "questionable"/"unknown" aren't promoted
    either way yet — not enough evidence."""
    game = os.path.basename(review.game_path or "")
    validated_new, failed_new = [], []
    for e in review.experiments:
        bullet = (f"- **{e.get('suggested', '')}** — {e.get('hypothesis', '')} "
                 f"(game: {game})")
        if e.get("verdict") == "good":
            validated_new.append(bullet)
        elif e.get("verdict") == "bad":
            failed_new.append(bullet)
    if not validated_new and not failed_new:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    validated, failed = _read_experiment_sections(path)
    validated += validated_new
    failed += failed_new
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Experiments\n\n")
        fh.write(_VALIDATED_HEADING + "\n")
        for line in validated:
            fh.write(line + "\n")
        fh.write("\n" + _FAILED_HEADING + "\n")
        for line in failed:
            fh.write(line + "\n")


# --- LoRA corpus (spec req 12 / phase 5) -----------------------------------

def _compact_state(state: Dict) -> Dict:
    board = [
        {"name": m.get("name") or m.get("card_id"),
         "attack": m.get("attack"), "health": m.get("health")}
        for m in (state.get("board") or [])
    ]
    return {
        "turn": state.get("turn"),
        "tavern_tier": state.get("tavern_tier"),
        "gold": state.get("gold"),
        "hero_health": state.get("hero_health"),
        "hero": state.get("hero"),
        "board": board,
    }


def append_training_examples(review: GameReview, trajectory_path: str,
                             path: str = "data/train_corpus.jsonl") -> int:
    """One JSON line per graded decision: {"state", "candidates", "chosen",
    "verdict", "placement", "better"} — the LoRA corpus `ml/lora_dataset.py`
    reads. Re-derives per-turn state + the original suggestion (candidates/
    chosen) from `trajectory_path` so `GameReview`'s fixed shape doesn't need
    to carry raw state. Returns the number of rows written."""
    decisions = _load_jsonl(trajectory_path)
    by_turn = _group_by_turn(decisions)
    sugg_by_turn = _group_suggestions_by_turn(
        _load_jsonl(_default_suggestions_path(trajectory_path)))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for g in review.grades:
        decs = by_turn.get(g.turn)
        if not decs:
            continue
        state = _compact_state(decs[0].get("state") or {})
        sugg = sugg_by_turn.get(g.turn)
        candidates = (sugg or {}).get("candidates", [])
        chosen = (sugg or {}).get("chosen") or {"move": g.suggested, "why": g.note}
        rows.append({
            "state": state,
            "candidates": candidates,
            "chosen": chosen,
            "verdict": g.verdict,
            "placement": review.placement,
            "better": g.better,
        })
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


# --- entry point the watch loop calls on game end ---------------------------

def review_latest(data_dir: str = "data", **kw) -> Optional[GameReview]:
    """Find the newest *completed* trajectory (`game-*.jsonl`, no `.partial`
    suffix — see `recorder.TrajectoryRecorder._flush`) and review it. Returns
    None when there's nothing to review yet."""
    paths = [p for p in glob.glob(os.path.join(data_dir, "game-*.jsonl"))
             if not p.endswith(".partial")]
    if not paths:
        return None
    latest = max(paths, key=lambda p: os.path.getmtime(p))
    return review_game(latest, **kw)
