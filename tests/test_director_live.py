"""DirectorLoop integration tests (offline — stub LLM, no threads needed for
the core paths; the worker thread is exercised once end-to-end)."""

import json
import time

from hsbg_coach.director_live import DirectorLoop
from hsbg_coach.llm_client import LLMClient, LLMConfig


def _client_returning(payload: dict) -> LLMClient:
    c = LLMClient(LLMConfig(backend="ollama", model="stub",
                            url="http://localhost:9"))
    c._post = lambda url, body, headers=None: {  # type: ignore[attr-defined]
        "message": {"content": json.dumps(payload)}}
    return c


SNAP = {"phase": "recruit", "turn": 6, "tavern_tier": 3, "gold": 7,
        "hero_health": 25, "hero_name": "Test Hero",
        "board": [{"name": "Alleycat", "attack": 1, "health": 1}],
        "shop": [{"name": "Kindly Grandmother", "attack": 1, "health": 1}]}


def _loop(tmp_path, client=None, pack=None):
    d = DirectorLoop(client=client or _client_returning(
        {"move": "Roll", "why": "hunting enablers"}), data_dir=str(tmp_path))
    d.meta_pack = pack if pack is not None else {
        "patch_build": "x", "tribes": [{"tribe": "Beast", "comps": []}],
        "heroes": [], "comps": [], "trinkets": [],
        "curve": {"standard": {}, "measured": {}}, "playbook_index": []}
    d.client_error = None
    d.drift = None
    return d


def test_augment_shows_engine_interim_then_llm_line(tmp_path):
    d = _loop(tmp_path)
    lines = d.augment(SNAP, ["Buy Kindly Grandmother — on-tribe"])
    assert lines[0].startswith("DIRECTOR (thinking…):")   # instant, non-blocking
    d.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        lines = d.augment(SNAP, ["Buy Kindly Grandmother — on-tribe"])
        if lines[0].startswith("DIRECTOR: "):
            break
        time.sleep(0.05)
    d.stop()
    assert lines[0].startswith("DIRECTOR: Roll — hunting enablers")


def test_augment_without_meta_pack_says_how_to_fix(tmp_path):
    d = _loop(tmp_path, pack=None)
    d.meta_pack = None
    lines = d.augment(SNAP, ["engine line"])
    assert "refresh-meta" in lines[0]
    assert lines[1] == "engine line"


def test_augment_passthrough_outside_recruit(tmp_path):
    d = _loop(tmp_path)
    assert d.augment({"phase": "combat"}, ["x"]) == ["x"]


def test_game_over_triggers_background_review_once(tmp_path, monkeypatch):
    d = _loop(tmp_path)
    calls = []
    monkeypatch.setattr("hsbg_coach.director_live.review_latest",
                        lambda *a, **k: calls.append(1) and None)
    d.augment({"phase": "recruit", "hero_name": "H"}, [])
    d.augment({"phase": "game_over"}, [])
    d.augment({"phase": "game_over"}, [])       # same transition — no double review
    deadline = time.time() + 2
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    assert len(calls) == 1


def test_new_hero_builds_and_saves_game_plan(tmp_path):
    pack = {"patch_build": "x", "heroes": [], "trinkets": [],
            "curve": {"standard": {}, "measured": {}}, "playbook_index": [],
            "tribes": [{"tribe": "Beast", "comps": []}],
            "comps": [{"name": "Beast Macaw", "tribe": "Beast", "tier": "S",
                       "averagePosition": 3.1, "coreCards": ["Monstrous Macaw"],
                       "enablers": []}]}
    d = _loop(tmp_path, pack=pack)
    d.augment(dict(SNAP, hero_name="Fresh Hero"), [])
    assert d.plan is not None
    assert (tmp_path / "game_plan.json").exists()
