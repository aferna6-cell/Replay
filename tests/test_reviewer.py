"""Between-game reviewer tests — offline, no torch, no network.

Fixtures are hand-built dicts matching the REAL shapes:
  * trajectory lines: hsbg_coach.recorder.TrajectoryRecorder._as_jsonable
    ({"state", "action_type", "action_detail", "placement", "wall_clock"})
  * suggestions lines: the fixed schema from the spec
    ({"ts", "turn", "snapshot", "candidates", "chosen"})
"""

import json
import os
import time

import pytest

from hsbg_coach.reviewer import (
    GameReview,
    TurnGrade,
    append_lessons,
    append_training_examples,
    load_lessons_for_prompt,
    promote_experiments,
    review_game,
    review_latest,
)


class StubScorer:
    """Deterministic, torch-free: equity = total board stats / 100."""

    def equity(self, board, hero_id="UNKNOWN", state=None):
        return sum((m.get("attack") or 0) + (m.get("health") or 0) for m in board) / 100.0


class BrokenScorer:
    """Simulates a scorer that can't score (e.g. a model load failure)."""

    def equity(self, board, hero_id="UNKNOWN", state=None):
        raise RuntimeError("scorer unavailable")


def _minion(name, attack, health):
    return {"entity_id": 1, "card_id": name, "name": name, "attack": attack,
            "health": health, "position": 1, "tags": {}}


def _state(turn, board, gold=2, tavern_tier=3, hero_health=30, hero="HERO_KAEL"):
    return {"game_counter": 1, "turn": turn, "phase": "recruit",
            "tavern_tier": tavern_tier, "gold": gold, "hero_health": hero_health,
            "board": board, "shop": [], "shop_spells": [], "hand_spells": [],
            "hero_power": None, "anomaly": None, "level_cost": None,
            "trinkets": [], "opponent_profiles": [], "hero": hero,
            "hero_name": "Kael'thas", "hand": [], "opponents_seen": [], "notes": []}


def _decision(state, action_type="buy", action_detail=None, placement=None):
    return {"state": state, "action_type": action_type,
            "action_detail": action_detail or {}, "placement": placement,
            "wall_clock": time.time()}


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _suggestion(turn, move, why="synergy", experiment=False, hypothesis="",
               source="director", candidates=None):
    return {
        "ts": time.time(), "turn": turn, "snapshot": {"turn": turn},
        "candidates": candidates if candidates is not None else [
            {"action": move, "detail": {}},
            {"action": "sell Deflect-o-Bot", "detail": {}},
        ],
        "chosen": {"move": move, "why": why, "experiment": experiment,
                  "hypothesis": hypothesis, "source": source},
    }


# --- pairing + grading verdicts ---------------------------------------------

def test_bad_suggestion_on_bad_game_graded_bad_with_better(tmp_path):
    traj = str(tmp_path / "game-20260101-000000.jsonl")
    board_t5 = [_minion("Alley Cat", 2, 2)]     # stats 4  -> equity 0.04
    board_t6 = [_minion("Micro Mummy", 1, 1)]   # stats 2  -> equity 0.02 (worse)
    _write_jsonl(traj, [
        _decision(_state(5, board_t5), "buy", placement=7),
        _decision(_state(6, board_t6), "end_turn", placement=7),
    ])
    sugg = str(tmp_path / "sugg.jsonl")
    _write_jsonl(sugg, [_suggestion(5, "buy Alley Cat")])

    review = review_game(traj, suggestions_path=sugg, scorer=StubScorer())

    assert review.placement == 7
    g5 = next(g for g in review.grades if g.turn == 5)
    assert g5.verdict == "bad"
    assert g5.suggested == "buy Alley Cat"
    assert g5.better != ""       # hindsight-preferred move was filled in
    assert "sell" in g5.better.lower()


def test_good_suggestion_graded_good(tmp_path):
    traj = str(tmp_path / "game-20260102-000000.jsonl")
    board_t5 = [_minion("Alley Cat", 2, 2)]       # stats 4  -> equity 0.04
    board_t6 = [_minion("Monstrous Macaw", 20, 20)]  # stats 40 -> equity 0.40
    _write_jsonl(traj, [
        _decision(_state(5, board_t5), "buy", placement=2),
        _decision(_state(6, board_t6), "end_turn", placement=2),
    ])
    sugg = str(tmp_path / "sugg.jsonl")
    _write_jsonl(sugg, [_suggestion(5, "buy Monstrous Macaw")])

    review = review_game(traj, suggestions_path=sugg, scorer=StubScorer())

    g5 = next(g for g in review.grades if g.turn == 5)
    assert g5.verdict == "good"
    assert g5.better == ""


def test_pairing_tolerates_suggestion_with_no_trajectory_turn(tmp_path):
    traj = str(tmp_path / "game-20260103-000000.jsonl")
    _write_jsonl(traj, [_decision(_state(3, [_minion("Alley Cat", 1, 1)]), placement=4)])
    sugg = str(tmp_path / "sugg.jsonl")
    # Turn 9 suggested but the game never reached it (trajectory has no turn 9).
    _write_jsonl(sugg, [_suggestion(9, "level up")])

    review = review_game(traj, suggestions_path=sugg, scorer=StubScorer())

    g9 = next(g for g in review.grades if g.turn == 9)
    assert g9.verdict == "unknown"
    assert g9.suggested == "level up"
    assert "no trajectory data" in g9.note


def test_pairing_tolerates_trajectory_turn_with_no_suggestion(tmp_path):
    traj = str(tmp_path / "game-20260104-000000.jsonl")
    _write_jsonl(traj, [
        _decision(_state(3, [_minion("Alley Cat", 1, 1)]), placement=4),
        _decision(_state(4, [_minion("Alley Cat", 5, 5)]), placement=4),
    ])
    review = review_game(traj, suggestions_path=None, scorer=StubScorer())

    g3 = next(g for g in review.grades if g.turn == 3)
    assert g3.suggested == ""     # nothing was logged as suggested this turn


def test_broken_scorer_degrades_to_unknown_not_raise(tmp_path):
    traj = str(tmp_path / "game-20260105-000000.jsonl")
    _write_jsonl(traj, [_decision(_state(5, [_minion("Alley Cat", 1, 1)]), placement=5)])
    sugg = str(tmp_path / "sugg.jsonl")
    _write_jsonl(sugg, [_suggestion(5, "buy Alley Cat")])

    review = review_game(traj, suggestions_path=sugg, scorer=BrokenScorer())

    g5 = next(g for g in review.grades if g.turn == 5)
    assert g5.verdict == "unknown"


# --- suggestions file missing -----------------------------------------------

def test_review_without_suggestions_file_grades_from_trajectory_only(tmp_path):
    traj = str(tmp_path / "game-20260106-000000.jsonl")
    board_t5 = [_minion("Alley Cat", 1, 1)]
    board_t6 = [_minion("Monstrous Macaw", 20, 20)]
    _write_jsonl(traj, [
        _decision(_state(5, board_t5), "buy", placement=1),
        _decision(_state(6, board_t6), "end_turn", placement=1),
    ])
    # Explicit nonexistent path (no auto-discovery fallback silently kicking in).
    review = review_game(traj, suggestions_path=str(tmp_path / "does-not-exist.jsonl"),
                         scorer=StubScorer())
    assert len(review.grades) == 2
    for g in review.grades:
        assert g.suggested == ""
        assert g.verdict in ("good", "bad", "questionable", "unknown")


def test_review_missing_trajectory_file_does_not_raise(tmp_path):
    review = review_game(str(tmp_path / "nope.jsonl"), scorer=StubScorer())
    assert review.placement is None
    assert review.grades == []
    assert review.lessons        # templated fallback lesson, never empty


# --- lessons -----------------------------------------------------------------

def test_lessons_written_and_capped(tmp_path):
    path = str(tmp_path / "lessons.jsonl")
    review = GameReview(game_path="game-x.jsonl", placement=5, grades=[],
                        lessons=["placeholder lesson"], experiments=[])
    for i in range(210):
        review.lessons = [f"lesson number {i}"]
        append_lessons(review, path=path)

    lines = open(path).read().splitlines()
    assert len(lines) == 200
    last = json.loads(lines[-1])
    assert last["lesson"] == "lesson number 209"
    assert "evidence" in last and "game" in last and "ts" in last


def test_load_lessons_for_prompt_is_newest_first_and_truncates(tmp_path):
    path = str(tmp_path / "lessons.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps({"ts": i, "lesson": f"lesson {i}",
                                 "evidence": "", "game": "g"}) + "\n")
    text = load_lessons_for_prompt(path, max_chars=10_000)
    lines = text.splitlines()
    assert lines[0] == "- lesson 4"     # newest (last appended) comes first
    assert lines[-1] == "- lesson 0"

    short = load_lessons_for_prompt(path, max_chars=15)
    assert short == "- lesson 4"        # only room for the newest one


# --- experiments (spec req 12) ----------------------------------------------

def test_promote_experiments_validated_and_failed(tmp_path):
    path = str(tmp_path / "playbooks" / "_experiments.md")
    review = GameReview(
        game_path="game-20260107-000000.jsonl", placement=3, grades=[],
        lessons=[],
        experiments=[
            {"hypothesis": "off-meta Elemental pivot", "suggested": "Try X", "verdict": "good"},
            {"hypothesis": "greedy triple chase", "suggested": "Try Y", "verdict": "bad"},
            {"hypothesis": "unresolved", "suggested": "Try Z", "verdict": "questionable"},
        ],
    )
    promote_experiments(review, path=path)
    text = open(path).read()
    assert "## Validated" in text and "## Do not repeat" in text
    validated_section, failed_section = text.split("## Do not repeat")
    assert "Try X" in validated_section
    assert "Try Y" in failed_section
    assert "Try Z" not in text     # questionable isn't promoted either way yet

    # Second game's experiments append without clobbering the first.
    review2 = GameReview(game_path="game-20260108-000000.jsonl", placement=1, grades=[],
                         lessons=[], experiments=[
                             {"hypothesis": "H2", "suggested": "Try W", "verdict": "good"}])
    promote_experiments(review2, path=path)
    text2 = open(path).read()
    assert "Try X" in text2 and "Try W" in text2   # both validated entries present


# --- LoRA corpus --------------------------------------------------------------

def test_corpus_lines_schema_exact(tmp_path):
    traj = str(tmp_path / "game-20260109-000000.jsonl")
    board_t5 = [_minion("Alley Cat", 2, 2)]
    board_t6 = [_minion("Micro Mummy", 1, 1)]
    _write_jsonl(traj, [
        _decision(_state(5, board_t5), "buy", placement=7),
        _decision(_state(6, board_t6), "end_turn", placement=7),
    ])
    sugg = str(tmp_path / "sugg.jsonl")
    _write_jsonl(sugg, [_suggestion(5, "buy Alley Cat")])
    review = review_game(traj, suggestions_path=sugg, scorer=StubScorer())

    corpus = str(tmp_path / "train_corpus.jsonl")
    n = append_training_examples(review, traj, path=corpus)
    assert n == len(review.grades)

    rows = [json.loads(l) for l in open(corpus)]
    assert len(rows) == n
    for row in rows:
        assert set(row.keys()) == {"state", "candidates", "chosen", "verdict",
                                   "placement", "better"}
        assert set(row["state"].keys()) == {"turn", "tavern_tier", "gold",
                                            "hero_health", "hero", "board"}
        assert isinstance(row["candidates"], list)
        assert isinstance(row["chosen"], dict)
        assert row["placement"] == 7

    bad_row = next(r for r in rows if r["verdict"] == "bad")
    assert bad_row["better"] != ""


def test_append_training_examples_appends_across_calls(tmp_path):
    traj = str(tmp_path / "game-20260110-000000.jsonl")
    _write_jsonl(traj, [_decision(_state(5, [_minion("Alley Cat", 1, 1)]), placement=3)])
    review = review_game(traj, suggestions_path=str(tmp_path / "none.jsonl"),
                         scorer=StubScorer())
    corpus = str(tmp_path / "train_corpus.jsonl")
    append_training_examples(review, traj, path=corpus)
    append_training_examples(review, traj, path=corpus)
    assert len(open(corpus).read().splitlines()) == 2 * len(review.grades)


# --- review_latest -------------------------------------------------------------

def test_review_latest_picks_newest_completed_game(tmp_path):
    data_dir = str(tmp_path)
    older = os.path.join(data_dir, "game-20260101-000000.jsonl")
    newer = os.path.join(data_dir, "game-20260102-000000.jsonl")
    partial = os.path.join(data_dir, "game-20260103-000000.jsonl.partial")
    _write_jsonl(older, [_decision(_state(1, []), placement=8)])
    _write_jsonl(newer, [_decision(_state(1, []), placement=1)])
    _write_jsonl(partial, [_decision(_state(1, []), placement=None)])

    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))
    os.utime(partial, (now + 100, now + 100))   # newest mtime but must be ignored

    review = review_latest(data_dir=data_dir, scorer=StubScorer())
    assert review is not None
    assert review.game_path == newer
    assert review.placement == 1


def test_review_latest_none_when_no_games(tmp_path):
    assert review_latest(data_dir=str(tmp_path)) is None
