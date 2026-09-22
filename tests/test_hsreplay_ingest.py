"""Fast unit tests for HSReplay / Tier7 expert ingest (offline fixtures)."""

import json
import os
from pathlib import Path

import pytest

from hsbg_coach.hsreplay_client import (
    AuthConfig, HSReplayClient, TIER7_SURFACES, describe_auth,
    load_auth_from_env, spike_surfaces,
)
from hsbg_coach.hsreplay_ingest import (
    EXPERT_SOURCE, boards_from_tier7_payload, ingest_files, parse_hsreplay_xml,
)

FIXTURES = Path(__file__).parent / "fixtures"
HSREPLAY_XML = FIXTURES / "minimal_bg.hsreplay"
TIER7_JSON = FIXTURES / "tier7_perfect_games.json"


def test_parse_minimal_hsreplay_emits_expert_rows():
    text = HSREPLAY_XML.read_text(encoding="utf-8")
    rows = parse_hsreplay_xml(text, game_id="fixture-min")
    assert rows, "expected at least one trajectory row"
    assert all(r["source"] == EXPERT_SOURCE for r in rows)
    assert all(r["game_id"] == "fixture-min" for r in rows)
    # Placement backfilled from PLAYER_LEADERBOARD_PLACE
    assert any(r["placement"] == 1 for r in rows)
    # At least one row carries a board with multiple minions
    boards = [r["state"]["board"] for r in rows]
    assert any(len(b) >= 2 for b in boards)
    # Action space uses the shared ActionType values
    assert any(r["action_type"] in ("end_turn", "buy", "play", "tier_up", "sell", "roll")
               for r in rows)


def test_ingest_files_writes_jsonl(tmp_path):
    stats = ingest_files([str(HSREPLAY_XML)], str(tmp_path))
    assert stats.files == 1
    assert stats.games == 1
    assert stats.rows >= 1
    out = list(tmp_path.glob("game-hsreplay-*.jsonl"))
    assert len(out) == 1
    lines = [json.loads(l) for l in out[0].read_text().splitlines() if l.strip()]
    assert lines[0]["source"] == EXPERT_SOURCE
    assert "state" in lines[0] and "action_type" in lines[0]


def test_tier7_perfect_games_fixture_to_rows():
    payload = json.loads(TIER7_JSON.read_text(encoding="utf-8"))
    rows = boards_from_tier7_payload(payload, game_id_prefix="tier7-test")
    assert len(rows) == 2
    assert all(r["source"] == EXPERT_SOURCE for r in rows)
    assert all(r["placement"] == 1 for r in rows)
    assert all(len(r["state"]["board"]) >= 2 for r in rows)
    assert rows[0]["action_type"] == "end_turn"


def test_auth_env_token_and_cookie_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HSREPLAY_API_TOKEN", raising=False)
    monkeypatch.delenv("HSREPLAY_BEARER", raising=False)
    monkeypatch.delenv("HSREPLAY_COOKIE_FILE", raising=False)
    assert load_auth_from_env({}).configured is False

    monkeypatch.setenv("HSREPLAY_API_TOKEN", "secret-token-value")
    auth = load_auth_from_env()
    assert auth.configured and auth.source == "HSREPLAY_API_TOKEN"
    headers = auth.headers()
    assert headers["Authorization"] == "Token secret-token-value"
    # describe_auth must never echo the secret
    note = describe_auth(auth)
    assert "secret-token-value" not in note
    assert "HSREPLAY_API_TOKEN" in note

    cookie = tmp_path / "cookies.txt"
    cookie.write_text("sessionid=abc; csrftoken=xyz\n", encoding="utf-8")
    monkeypatch.delenv("HSREPLAY_API_TOKEN", raising=False)
    monkeypatch.setenv("HSREPLAY_COOKIE_FILE", str(cookie))
    auth2 = load_auth_from_env()
    assert auth2.configured and auth2.source == "HSREPLAY_COOKIE_FILE"
    assert "sessionid=abc" in (auth2.cookie or "")


def test_spike_surfaces_skips_auth_without_token(monkeypatch):
    monkeypatch.delenv("HSREPLAY_API_TOKEN", raising=False)
    monkeypatch.delenv("HSREPLAY_BEARER", raising=False)
    monkeypatch.delenv("HSREPLAY_COOKIE_FILE", raising=False)
    client = HSReplayClient(AuthConfig())  # empty
    rows = spike_surfaces(client, live=False)
    assert len(rows) == len(TIER7_SURFACES)
    auth_rows = [r for r in rows if r.get("auth") or r.get("tier7")]
    assert auth_rows
    assert all(r["probe"].get("skipped") for r in auth_rows)


def test_client_get_json_skips_when_auth_required(monkeypatch):
    monkeypatch.delenv("HSREPLAY_API_TOKEN", raising=False)
    monkeypatch.delenv("HSREPLAY_BEARER", raising=False)
    monkeypatch.delenv("HSREPLAY_COOKIE_FILE", raising=False)
    client = HSReplayClient(AuthConfig())
    res = client.list_heroes()
    assert res.skipped is True
    assert res.ok is False


def test_trajectory_examples_upweights_expert(tmp_path, monkeypatch):
    """Expert jsonl rows are repeated by expert_weight in the training set builder."""
    pytest.importorskip("numpy")
    # Write a tiny expert + personal pair.
    expert = {
        "state": {
            "board": [
                {"card_id": "X", "name": "A", "attack": 1, "health": 1, "position": 1, "tags": {}},
                {"card_id": "Y", "name": "B", "attack": 2, "health": 2, "position": 2, "tags": {}},
            ],
            "hero": "UNKNOWN",
        },
        "action_type": "end_turn",
        "action_detail": {},
        "placement": 1,
        "source": "hsreplay_expert",
        "game_id": "e1",
    }
    personal = dict(expert)
    personal["source"] = "personal"
    personal["game_id"] = "p1"
    personal["placement"] = 4
    (tmp_path / "game-expert.jsonl").write_text(json.dumps(expert) + "\n", encoding="utf-8")
    (tmp_path / "game-personal.jsonl").write_text(json.dumps(personal) + "\n", encoding="utf-8")

    # Avoid depending on a full card KB: stub minion_from_snapshot via monkeypatch
    # if KB lacks these names — board_dataset filters unknown minions.
    from hsbg_coach import cards
    from ml import board_dataset as bd

    class _M:
        def __init__(self, name):
            self.name = name
            self.card_id = name
            self.attack = 1
            self.health = 1
            self.tribes = []
            self.keywords = []

    monkeypatch.setattr(bd, "minion_from_snapshot",
                        lambda x, byname: _M(x.get("name") or x.get("card_id")))
    monkeypatch.setattr(cards, "load_kb", lambda: {})
    monkeypatch.setattr(cards, "by_name", lambda kb: {})

    out = bd.trajectory_examples(str(tmp_path), expert_weight=3.0)
    n_expert = sum(1 for e in out if e.get("source") == "hsreplay_expert")
    n_personal = sum(1 for e in out if e.get("source") == "personal")
    assert n_expert == 3
    assert n_personal == 1
