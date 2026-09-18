"""Learn-as-you-play loop + Season 14 action coverage (fast, no network/torch)."""

from __future__ import annotations

import json

from hsbg_coach.actions import (
    legal_actions, BUY, BUY_SPELL, HERO_POWER, ACTIVATE, DARK_GIFT, PLAY, FREEZE,
    UNFREEZE,
)
from hsbg_coach.bg import ActionType, Snapshot, MinionView
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.config import personal_weight, population_weight, WEIGHTING
from hsbg_coach.live import advice_lines, _key, LiveCoach
from hsbg_coach.options import OptionsTracker
from hsbg_coach.overlay import format_next, format_overlay_text
from hsbg_coach.recorder import TrajectoryRecorder
from hsbg_coach import learn as learn_mod


EMB = {"b1": [1.0, 0.0], "b2": [1.0, 0.0], "good": [1.0, 0.0], "bad": [0.0, 1.0],
       "Courier": [1.0, 0.0]}


def test_personal_weight_grows_with_games():
    assert population_weight(0) == WEIGHTING["population_start"]
    assert personal_weight(0) == 1.0 - WEIGHTING["population_start"]
    assert personal_weight(WEIGHTING["personal_full_at_games"]) == (
        1.0 - WEIGHTING["population_floor"]
    )
    assert personal_weight(100) > personal_weight(0)


def test_recorder_jsonl_has_game_id_and_placement(tmp_path):
    rec = TrajectoryRecorder(str(tmp_path))
    rec.start_game("g1")
    snap = Snapshot(game_counter=1, turn=4, phase="recruit", tavern_tier=2,
                    gold=8, hero_health=30)
    rec.record(snap, ActionType.BUY, {"card_id": "BGS_039"})
    path = rec.finish_game(placement=2)
    rows = [json.loads(l) for l in open(path)]
    assert rows and rows[0]["game_id"] == "g1"
    assert rows[0]["placement"] == 2
    assert rows[0]["action_type"] == "buy"
    assert isinstance(rows[0]["state"], dict)


def test_learn_dry_run_validates_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(learn_mod.config, "DATA_DIR", str(tmp_path))
    # empty is ok
    assert learn_mod.run_retrain(data_dir=str(tmp_path), dry_run=True) == 0
    (tmp_path / "game-1.jsonl").write_text(json.dumps({
        "game_id": "1", "state": {"board": []}, "action_type": "end_turn",
        "placement": 3, "action_detail": {}, "wall_clock": 0,
    }) + "\n")
    st = learn_mod.validate_trajectories(str(tmp_path))
    assert st["ok"] and st["n_games"] == 1 and st["n_with_placement"] == 1
    assert learn_mod.run_retrain(data_dir=str(tmp_path), dry_run=True) == 0


def test_legal_actions_include_activate_dark_gift_hero_power_play():
    snap = {
        "gold": 6, "tavern_tier": 3, "hero_health": 30, "turn": 5,
        "board": [{"name": "b1", "attack": 3, "health": 3}],
        "shop": [{"name": "good", "attack": 5, "health": 5}],
        "shop_spells": [{"name": "Telescope", "card_id": "BG28_521", "cost": 2}],
        "hand": [{"name": "free", "card_id": "X", "attack": 1, "health": 1,
                  "tags": {"CARDTYPE": "MINION"}}],
        "hero_power": {"name": "Clickable HP", "cost": 1, "usable": True},
        "activatable": [{"name": "Courier", "card_id": "BG26_810", "cost": 3,
                         "usable": True, "entity_id": 1}],
        "dark_gift": {"name": "Dark Gift", "cost": 3, "usable": True},
        "shop_frozen": False,
    }
    kinds = {a.kind for a in legal_actions(snap)}
    assert HERO_POWER in kinds
    assert ACTIVATE in kinds
    assert DARK_GIFT in kinds
    assert PLAY in kinds
    assert BUY_SPELL in kinds
    assert FREEZE in kinds
    snap["shop_frozen"] = True
    kinds2 = {a.kind for a in legal_actions(snap)}
    assert UNFREEZE in kinds2 and FREEZE not in kinds2


def test_advice_returns_single_next_move_by_default():
    snap = {
        "phase": "recruit", "turn": 5, "tavern_tier": 2, "gold": 7,
        "hero_health": 30,
        "board": [{"name": "b1", "attack": 3, "health": 3},
                  {"name": "b2", "attack": 3, "health": 3}],
        "shop": [{"name": "good", "attack": 5, "health": 5},
                 {"name": "bad", "attack": 1, "health": 1}],
        "hero_power": {"name": "HP", "cost": 0, "usable": True},
        "activatable": [],
        "dark_gift": None,
    }
    lines = advice_lines(snap, kb=None, scorer=HeuristicScorer(EMB))
    assert len(lines) == 1
    text = format_next(snap, None, lines)
    assert text.startswith("→ ")
    assert "\n  or:" not in text
    full = format_overlay_text(snap, None, lines + ["extra should be ignored"])
    assert "NEXT → " in full
    assert "then:" not in full


def test_cache_key_changes_when_hero_power_or_activate_flips():
    base = {
        "phase": "recruit", "gold": 5, "tavern_tier": 2, "hero_health": 30,
        "board": [{"name": "b1", "attack": 2, "health": 2, "entity_id": 1}],
        "shop": [{"name": "s1", "attack": 2, "health": 2}],
        "hero_power": {"usable": False},
        "activatable": [{"entity_id": 9, "usable": False, "cost": 3}],
        "dark_gift": {"usable": False},
    }
    a = dict(base)
    b = dict(base, hero_power={"usable": True})
    c = dict(base, activatable=[{"entity_id": 9, "usable": True, "cost": 3}])
    assert _key(a) != _key(b)
    assert _key(a) != _key(c)


def test_options_tracker_detects_dark_gift_and_hero_power():
    tr = OptionsTracker()
    for line in [
        "D x GameState.DebugPrintOptions() - id=1",
        "D x GameState.DebugPrintOptions() -   option 0 type=END_TURN mainEntity= error=INVALID errorParam=",
        "D x GameState.DebugPrintOptions() -   option 1 type=POWER mainEntity="
        "[entityName=Dark Gift id=9 zone=PLAY zonePos=0 cardId=TB_BaconShop_DarkGift_Button "
        "player=1] error=NONE errorParam=",
        "D x GameState.DebugPrintOptions() -   option 2 type=POWER mainEntity="
        "[entityName=Fire Sale id=10 zone=PLAY zonePos=0 cardId=BG28_HERO_800p "
        "player=1] error=NONE errorParam=",
        "D x GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=STEP value=MAIN_END",
    ]:
        tr.feed(line)
    leg = tr.legal()
    assert any(o.is_dark_gift for o in leg)
    assert any(o.is_hero_power for o in leg)


def test_livecoach_reload_still_returns_ranked_move(tmp_path):
    coach = LiveCoach(power_log=None, top=1)
    coach._active = True
    coach.tracker.in_bg = True
    coach.scorer = HeuristicScorer(EMB)
    # Fake a recruit snapshot via cache inject
    snap = {
        "phase": "recruit", "turn": 4, "tavern_tier": 2, "gold": 6,
        "hero_health": 30, "board": [{"name": "b1", "attack": 4, "health": 4}],
        "shop": [{"name": "good", "attack": 6, "health": 6}],
        "notes": [],
    }
    coach._snap_cache = snap
    coach._snap_version = coach._version = 1
    coach._cache_key = None
    # Touch model path so reload path is exercised without a real .pt
    model = tmp_path / "eval_net.pt"
    model.write_bytes(b"fake")
    coach._model_path = str(model)
    coach._model_mtime = None
    s, odds, recs = coach.frame()
    assert s["phase"] == "recruit"
    assert recs and len(recs) == 1
