"""replay_labels on a hand-written replay + synthetic states.v1 rows.

Optional real-replay test: set HSBG_REPLAY_XML=<path to a Firestone .xml.gz>
(and optionally HSBG_REPLAY_MANIFEST=<manifest json>); states are built with
replay_states.build_game and then labeled.
"""

import gzip
import json
import os
import shutil

import pytest

from hsbg_coach import replay_labels as rl

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_labels_replay.xml")
META = {"reviewId": "synthetic", "mmr": 5000, "placement": 1, "buildNumber": 253216}


def _m(eid, cid, pos, ctype="MINION"):
    return {"entity_id": eid, "card_id": cid, "name": cid, "attack": 1, "health": 1,
            "position": pos, "tags": {"CARDTYPE": ctype}, "dark_gift": None}


def _o(index, eid, cid, targets=()):
    return {"index": index, "type": "POWER", "entity_id": eid, "card_id": cid,
            "zone": "PLAY", "targets": list(targets)}


END = {"index": 0, "type": "END_TURN", "entity_id": 0, "card_id": None, "zone": None,
       "targets": []}


def _row(dp, raw_turn, board, hand, shop, options, spells=(), frozen=False):
    return {"schema_version": "states.v1", "game_id": "synthetic", "build": 253216,
            "mmr": 5000, "lobby_tribes": [], "dp_index": dp, "turn": (raw_turn + 1) // 2,
            "raw_turn": raw_turn,
            "snapshot": {"board": board, "hand": hand, "shop": shop,
                         "shop_spells": list(spells), "shop_frozen": frozen,
                         "tavern_tier": 1, "gold": 3, "trinkets": []},
            "hero_power": {"card_id": "TEST_HP", "used": False, "activatable": True},
            "dark_discovery": {"available": False}, "options": [END] + options}


def _states():
    c, d, a = _m(30, "TEST_C", 1), _m(31, "TEST_D", 2), _m(20, "TEST_A", 1)
    after_move = [_m(31, "TEST_D", 1), _m(30, "TEST_C", 2), _m(20, "TEST_A", 3)]
    spell = {"name": "S", "card_id": "TEST_SPELL", "cost": 1, "entity_id": 22}
    return [
        _row(0, 1, [c, d], [], [a, _m(21, "TEST_B", 2)],
             [_o(1, 13, "TB_BaconShop_DragBuy", [20, 21, 90]), _o(2, 20, "TEST_A"),
              _o(3, 11, rl.REROLL), _o(4, 14, rl.DRAG_SELL, [30, 31, 90]),
              _o(5, 30, "TEST_C"), _o(6, 31, "TEST_D"), _o(7, 4, "TEST_HP", [20, 21])],
             spells=[spell]),
        _row(1, 1, [c, d], [_m(20, "TEST_A", 1)], [_m(21, "TEST_B", 1)],
             [_o(1, 20, "TEST_A"), _o(2, 30, "TEST_C"), _o(3, 31, "TEST_D")]),
        _row(2, 1, [c, _m(20, "TEST_A", 2), _m(31, "TEST_D", 3)], [], [_m(21, "TEST_B", 1)],
             [_o(1, 20, "TEST_A"), _o(2, 30, "TEST_C"), _o(3, 31, "TEST_D")]),
        _row(3, 1, after_move, [_m(40, "TEST_CHOOSE", 1, "SPELL")], [_m(21, "TEST_B", 1)],
             [_o(1, 31, "TEST_D"), _o(2, 30, "TEST_C"), _o(3, 20, "TEST_A"),
              _o(4, 40, "TEST_CHOOSE")]),
        _row(4, 1, after_move, [_m(51, "TEST_Y", 1)], [_m(21, "TEST_B", 1)],
             [_o(1, 31, "TEST_D")]),
        _row(5, 1, after_move, [_m(51, "TEST_Y", 1)], [_m(21, "TEST_B", 1)],
             [_o(1, 12, rl.FREEZE)]),
        _row(6, 3, after_move, [_m(51, "TEST_Y", 1)], [_m(23, "TEST_E", 1)],
             [_o(1, 12, rl.FREEZE)]),
        _row(7, 3, after_move, [_m(51, "TEST_Y", 1)], [_m(23, "TEST_E", 1)], [], frozen=True),
    ]


def test_scan_counts_resent_options_once_and_reads_actions():
    scan = rl.scan_replay(FIXTURE)
    dps = scan["dps"]
    assert len(dps) == 8 and scan["stats"]["options_resent"] == 1
    assert [len(d["acts"]) for d in dps] == [1, 1, 1, 1, 1, 0, 1, 0]   # ATTACK block ignored
    assert dps[0]["acts"][0]["target"] == 20
    assert dps[1]["acts"][0]["play_position"] == 2
    assert dps[2]["acts"][0]["block"] == "MOVE_MINION" and dps[2]["acts"][0]["last_position"] == 1
    assert dps[4]["acts"][0]["last_position"] is None                  # empty move
    assert [s["card_id"] for s in dps[3]["suboptions"][4]] == ["TEST_CHOOSE_a", "TEST_CHOOSE_b"]
    assert 8 not in dps[0]["suboptions"]


def test_enumerate_options_expands_targets_and_positions():
    scan = rl.scan_replay(FIXTURE)
    states = _states()
    opts, skipped = rl.enumerate_options(states[0], scan["dps"][0], scan["entities"])
    assert [o["type"] for o in opts] == ["end_turn", "buy", "buy", "reroll", "sell", "sell",
                                         "reposition", "reposition", "hero_power", "hero_power"]
    assert opts[1]["source"] == {"zone": "shop", "slot": 0} and opts[1]["card_id"] == "TEST_A"
    assert opts[5]["source"] == {"zone": "board", "slot": 1} and opts[5]["card_id"] == "TEST_D"
    assert opts[9]["target"] == {"zone": "shop", "slot": 1} and opts[9]["target_entity"] == 21
    assert skipped == {"buy_drop_target_bob": 1, "sell_drop_target_bob": 1,
                       "shop_card_drag_handle": 1}
    opts, _ = rl.enumerate_options(states[1], scan["dps"][1], scan["entities"])
    plays = [o for o in opts if o["type"] == "play"]
    assert [o["position"] for o in plays] == [0, 1, 2]                 # one per board slot
    opts, _ = rl.enumerate_options(states[3], scan["dps"][3], scan["entities"])
    assert [o["card_id"] for o in opts if o["type"] == "play"] == ["TEST_CHOOSE_a", "TEST_CHOOSE_b"]
    for o in opts:
        assert set(o) == {"type", "card_id", "source", "target", "position", "src_entity",
                          "target_entity", "choice_kind"}


def test_label_game_rows_quarantine_and_choices():
    res = rl.label_game(_states(), FIXTURE, META, 0.6)
    rows = res["rows"]
    chosen = [(r["dp_index"], r["options"][r["chosen"]]["type"]) for r in rows]
    assert chosen == [(0, "buy"), (1, "play"), (2, "reposition"), (3, "play"),
                      (5, "end_turn"), (6, "freeze"), (7, "end_turn")]
    by_dp = {r["dp_index"]: r for r in rows}
    assert by_dp[1]["options"][by_dp[1]["chosen"]]["position"] == 1
    assert by_dp[2]["options"][by_dp[2]["chosen"]]["position"] == 0
    assert by_dp[3]["options"][by_dp[3]["chosen"]]["card_id"] == "TEST_CHOOSE_b"
    assert by_dp[5]["inference_method"] == "no_action_turn_advanced"
    assert (by_dp[7]["inference_method"], by_dp[7]["confidence"]) == (
        "no_action_last_decision_point", "medium")
    assert [(q["dp_index"], q["reason"]) for q in res["quarantine"]] == [(4, "noop_move")]
    r = rows[0]
    assert set(r) == {"game_id", "build", "current_patch", "turn", "dp_index", "placement",
                      "mmr", "state", "options", "chosen", "inference_method", "confidence",
                      "weight"}
    assert r["current_patch"] is True and r["weight"] == 0.6 and r["placement"] == 1
    assert r["state"] == _states()[0]["snapshot"]
    assert {t: dict(c) for t, c in res["transitions"].items()} == {
        "buy": {"pass": 1}, "play": {"pass": 2}, "reposition": {"pass": 1},
        "end_turn": {"pass": 1, "na": 1}, "freeze": {"pass": 1}}
    hero, disc = res["choices"]
    assert (hero["choice_kind"], hero["after_dp_index"], hero["chosen"]) == ("hero", -1, 0)
    assert (disc["choice_kind"], disc["after_dp_index"], disc["chosen"],
            disc["inference_method"]) == ("discover", 3, 1, "chosen_entities")
    json.dumps(res["rows"])


def test_transition_failure_and_dp_mismatch_quarantine():
    states = _states()
    states[2]["snapshot"]["board"] = states[1]["snapshot"]["board"]   # play left no trace
    res = rl.label_game(states, FIXTURE, META, 1.0)
    assert ("transition_failed:play" in {q["reason"] for q in res["quarantine"]})
    res = rl.label_game(_states()[:5], FIXTURE, META, 1.0)
    assert res["rows"] == [] and {q["reason"] for q in res["quarantine"]} == {"dp_count_mismatch"}


def test_choice_moved_to_hand_is_medium():
    ch = {"after_dp": 2, "type": 2, "offered": [5, 6, 7], "chosen": [], "moved_to_hand": [6],
          "source_card": "X", "offered_cards": ["A", "B", "C"],
          "offered_types": ["BATTLEGROUND_TRINKET"] * 3}
    out = rl.label_choice(ch, "g")
    assert (out["choice_kind"], out["chosen"], out["confidence"]) == ("trinket", 1, "medium")
    out = rl.label_choice(dict(ch, moved_to_hand=[]), "g")
    assert out["chosen"] is None and out["inference_method"] == "unresolved"


def test_mmr_weight_curve():
    w = rl.mmr_weight_fn([1000, 5000, 9000])
    assert w(5000) == pytest.approx(0.6)
    assert w(1000) == pytest.approx(0.2 + 0.8 / 6, abs=1e-6)     # weights rounded to 1e-6
    assert w(9000) == pytest.approx(1 - 0.8 / 6, abs=1e-6)
    assert w(0) == pytest.approx(0.2) and w(10 ** 6) == pytest.approx(1.0)
    vals = [w(m) for m in range(0, 12000, 250)]
    assert vals == sorted(vals) and w(None) is None


def test_run_writes_outputs_and_drops_non_first(tmp_path):
    states_dir, replays = tmp_path / "states", tmp_path / "replays"
    states_dir.mkdir()
    replays.mkdir()
    with gzip.open(states_dir / "synthetic.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in _states():
            fh.write(json.dumps(r) + "\n")
    with open(FIXTURE, "rb") as src, gzip.open(replays / "synthetic.xml.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    manifest, corpus = tmp_path / "m.json", tmp_path / "c.json"
    manifest.write_text(json.dumps({"games": [META, {"reviewId": "other", "mmr": 9000,
                                                     "placement": 3}]}))
    corpus.write_text(json.dumps({"games": [{"mmr": m} for m in (1000, 5000, 9000)]}))
    out = tmp_path / "labels"
    s = rl.run(str(states_dir), str(replays), str(manifest), str(out), str(corpus))
    assert s["rows"] == 7 and s["decision_points"] == 8 and s["match_rate"] == 0.875
    assert s["games_dropped"] == [{"game_id": "other", "reason": "placement_not_1",
                                   "placement": 3}]
    assert s["quarantine"]["reasons"] == {"noop_move": 1}
    assert s["chosen_in_legal"]["rate"] == 1.0
    assert s["weight"]["per_game"] == {"synthetic": 0.6}
    assert s["rows_per_action_type"]["play"] == 2
    with gzip.open(out / "synthetic.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert len(fh.readlines()) == 7
    assert (out / "quarantine" / "synthetic.jsonl").read_text().count("\n") == 1
    assert (out / "choices" / "synthetic.jsonl").read_text().count("\n") == 2
    assert json.loads((out / "summary.json").read_text())["rows"] == 7


@pytest.mark.skipif(not os.environ.get("HSBG_REPLAY_XML"),
                    reason="set HSBG_REPLAY_XML to a local Firestone replay (.xml.gz)")
def test_real_replay():
    from hsbg_coach.replay_states import build_game
    path = os.environ["HSBG_REPLAY_XML"]
    gid = os.path.basename(path).split(".")[0]
    meta = {"reviewId": gid, "placement": 1}
    if os.environ.get("HSBG_REPLAY_MANIFEST"):
        games = json.load(open(os.environ["HSBG_REPLAY_MANIFEST"], encoding="utf-8"))
        meta = {g["reviewId"]: g for g in games.get("games", games)}[gid]
    states = build_game(path, meta, track_opponents=False)["rows"]
    res = rl.label_game(states, path, meta, 1.0)
    assert res["rows"]
    assert len(res["rows"]) + len(res["quarantine"]) == len(states)
    for r in res["rows"]:
        assert 0 <= r["chosen"] < len(r["options"])
