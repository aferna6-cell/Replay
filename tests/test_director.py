"""director tests — no network. A tiny fake LLMClient stands in for the real
`llm_client.LLMClient`; only `.chat(system, user, json_mode=True)` is used by
`suggest_move`, so the fake only needs that one method."""

import json
from datetime import datetime, timezone

from hsbg_coach.actions import Action, BUY, END, ROLL
from hsbg_coach.director import (
    Suggestion, _extract_json, format_overlay_line, log_suggestion,
    suggest_move,
)
from hsbg_coach.game_value import WholeGameRec
from hsbg_coach.llm_client import LLMError


class FakeClient:
    def __init__(self, reply=None, raises: Exception = None):
        self.reply = reply
        self.raises = raises
        self.calls = []

    def chat(self, system, user, json_mode=True):
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if self.raises is not None:
            raise self.raises
        return self.reply


def _buy_candidate(name, placement=3.0, reason="a fine buy", gain=0.3):
    return WholeGameRec(Action(BUY, name, 3, {"minion": {"name": name}}),
                        placement=placement, reason=reason, gain=gain)


def _roll_candidate():
    return WholeGameRec(Action(ROLL, cost=1), placement=3.4, reason="roll for value",
                        gain=0.0)


def _end_candidate():
    return WholeGameRec(Action(END), placement=3.9, reason="pass the turn", gain=-0.2)


SNAPSHOT = {"turn": 6, "tavern_tier": 3, "gold": 4, "hero_health": 28,
           "board": [{"name": "Holo Rover", "attack": 3, "health": 3}],
           "shop": [{"name": "Monstrous Macaw", "attack": 3, "health": 2}]}


# --- _extract_json -------------------------------------------------------

def test_extract_json_plain():
    assert _extract_json('{"move": "Roll"}') == {"move": "Roll"}


def test_extract_json_strips_code_fence():
    text = '```json\n{"move": "Roll", "why": "ok"}\n```'
    assert _extract_json(text) == {"move": "Roll", "why": "ok"}


def test_extract_json_finds_first_balanced_object_amid_prose():
    text = 'Sure! Here is my answer:\n{"move": "End turn", "why": "safe"} — hope that helps'
    assert _extract_json(text) == {"move": "End turn", "why": "safe"}


def test_extract_json_returns_none_for_garbage():
    assert _extract_json("not json at all, sorry") is None
    assert _extract_json("") is None
    assert _extract_json(None) is None


# --- suggest_move: happy path --------------------------------------------

def test_suggest_move_happy_path_returns_llm_suggestion():
    candidates = [_buy_candidate("Monstrous Macaw"), _roll_candidate(), _end_candidate()]
    reply = json.dumps({"move": "Buy Monstrous Macaw",
                        "why": "on-tribe upgrade, cheap tempo"})
    client = FakeClient(reply=reply)

    suggestion = suggest_move(SNAPSHOT, candidates, meta_ctx="Beast S-tier this lobby",
                              plan_ctx="Plan A: Beast", lessons="", client=client)

    assert suggestion.source == "llm"
    assert suggestion.move == "Buy Monstrous Macaw"
    assert suggestion.why == "on-tribe upgrade, cheap tempo"
    assert suggestion.latency_s >= 0
    # prompts actually carried the context through
    assert client.calls[0]["json_mode"] is True
    assert "Beast S-tier this lobby" in client.calls[0]["user"]
    assert "Plan A: Beast" in client.calls[0]["user"]
    assert "Monstrous Macaw" in client.calls[0]["user"]


def test_suggest_move_parses_fenced_json_reply():
    candidates = [_roll_candidate(), _end_candidate()]
    reply = '```json\n{"move": "Roll the shop", "why": "no strong buy"}\n```'
    client = FakeClient(reply=reply)

    suggestion = suggest_move(SNAPSHOT, candidates, "", "", "", client)
    assert suggestion.source == "llm"
    assert suggestion.move == "Roll the shop"


# --- suggest_move: fallback paths ----------------------------------------

def test_suggest_move_falls_back_on_garbage_reply():
    candidates = [_buy_candidate("Monstrous Macaw"), _end_candidate()]
    client = FakeClient(reply="I refuse to answer in JSON today.")

    suggestion = suggest_move(SNAPSHOT, candidates, "", "", "", client)

    assert suggestion.source == "engine-fallback"
    assert suggestion.move == "Buy Monstrous Macaw"     # candidates[0]
    assert "unparseable LLM reply" in suggestion.why


def test_suggest_move_falls_back_on_llm_error():
    candidates = [_buy_candidate("Monstrous Macaw"), _end_candidate()]
    client = FakeClient(raises=LLMError("is Ollama running? `ollama serve`"))

    suggestion = suggest_move(SNAPSHOT, candidates, "", "", "", client)

    assert suggestion.source == "engine-fallback"
    assert suggestion.move == "Buy Monstrous Macaw"
    assert "Ollama running" in suggestion.why


def test_suggest_move_falls_back_on_validator_rejection():
    candidates = [_buy_candidate("Monstrous Macaw"), _end_candidate()]
    reply = json.dumps({"move": "Buy A Card That Was Never Offered", "why": "great"})
    client = FakeClient(reply=reply)

    suggestion = suggest_move(SNAPSHOT, candidates, "", "", "", client)

    assert suggestion.source == "engine-fallback"
    assert suggestion.move == "Buy Monstrous Macaw"
    assert "hallucinated buy" in suggestion.why


def test_suggest_move_fallback_with_no_candidates_never_returns_nothing():
    client = FakeClient(reply="garbage")
    suggestion = suggest_move(SNAPSHOT, [], "", "", "", client)
    assert suggestion.source == "engine-fallback"
    assert suggestion.move == "End turn"


# --- choice_kind framing --------------------------------------------------

def test_choice_kind_injects_framing_into_system_prompt():
    candidates = [_end_candidate()]
    client = FakeClient(reply=json.dumps({"move": "End turn", "why": "n/a"}))
    suggest_move(SNAPSHOT, candidates, "", "", "", client, choice_kind="dark_gift")
    assert "DARK GIFT" in client.calls[0]["system"]
    assert "CALIBRATE" in client.calls[0]["system"]


def test_no_choice_kind_omits_framing():
    candidates = [_end_candidate()]
    client = FakeClient(reply=json.dumps({"move": "End turn", "why": "n/a"}))
    suggest_move(SNAPSHOT, candidates, "", "", "", client)
    assert "DARK GIFT" not in client.calls[0]["system"]
    assert "HERO PICK" not in client.calls[0]["system"]


# --- log_suggestion: exact schema ----------------------------------------

def test_log_suggestion_schema_matches_reviewer_contract(tmp_path):
    candidates = [_buy_candidate("Monstrous Macaw"), _roll_candidate(),
                 _end_candidate(), _buy_candidate("Holo Rover"),
                 _buy_candidate("Scrap Scraper"), _buy_candidate("Extra Sixth")]
    suggestion = Suggestion(move="Buy Monstrous Macaw", why="tempo",
                            source="llm", experiment=False, hypothesis="",
                            plan_update="", latency_s=1.23)

    path = log_suggestion(suggestion, SNAPSHOT, candidates, dir=str(tmp_path))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert path == str(tmp_path / f"{today}.jsonl")

    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])

    assert set(entry.keys()) == {"ts", "turn", "snapshot", "candidates", "chosen"}
    assert entry["turn"] == 6
    assert entry["snapshot"] == SNAPSHOT
    assert len(entry["candidates"]) == 5              # first 5 only
    for row in entry["candidates"]:
        assert set(row.keys()) == {"action", "detail"}
    assert entry["candidates"][0] == {"action": "Buy Monstrous Macaw",
                                      "detail": "a fine buy"}
    assert entry["chosen"] == {
        "move": "Buy Monstrous Macaw", "why": "tempo", "experiment": False,
        "hypothesis": "", "source": "llm",
    }


def test_log_suggestion_appends_across_calls(tmp_path):
    suggestion = Suggestion(move="Roll the shop", why="no buy", source="llm")
    log_suggestion(suggestion, SNAPSHOT, [_roll_candidate()], dir=str(tmp_path))
    path = log_suggestion(suggestion, SNAPSHOT, [_roll_candidate()], dir=str(tmp_path))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 2


# --- format_overlay_line ---------------------------------------------------

def test_format_overlay_line_plain():
    s = Suggestion(move="Buy Monstrous Macaw", why="tempo upgrade", source="llm")
    assert format_overlay_line(s) == "Buy Monstrous Macaw — tempo upgrade"


def test_format_overlay_line_experiment_marker():
    s = Suggestion(move="Buy Off-Meta Tech", why="testing a read", source="llm",
                   experiment=True, hypothesis="this line is underrated")
    assert format_overlay_line(s) == "Buy Off-Meta Tech — testing a read [experiment]"
