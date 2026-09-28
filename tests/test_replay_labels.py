"""replay_labels on a hand-written replay + synthetic states.v2 rows.

Optional real-replay test: set HSBG_REPLAY_XML=<path to a Firestone .xml.gz>
(and optionally HSBG_REPLAY_MANIFEST=<manifest json>); states are built with
replay_states.build_game and then labeled.
"""

import copy
import gzip
import json
import os
import shutil

import pytest

from hsbg_coach import replay_labels as rl

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_labels_replay.xml")
CREATED = "2026-09-27T20:59:53.982Z"
META = {"reviewId": "synthetic", "mmr": 5000, "placement": 1, "buildNumber": 253216,
        "creationDate": CREATED, "creationTimestamp": 1790542793000}
ROW_KEYS = {"game_id", "build", "current_patch", "turn", "dp_index", "placement", "mmr",
            "created_at", "created_ts", "state", "options", "chosen", "inference_method",
            "confidence", "weight", "flags"}
OPTION_KEYS = {"type", "card_id", "source", "target", "position", "src_entity",
               "target_entity", "choice_kind"}


def _m(eid, cid, pos, ctype="MINION", **tags):
    return {"entity_id": eid, "card_id": cid, "name": cid, "attack": 1, "health": 1,
            "position": pos, "tags": {"CARDTYPE": ctype, **tags}, "dark_gift": None}


def _o(index, eid, cid, targets=(), subs=()):
    return {"index": index, "type": "POWER", "entity_id": eid, "card_id": cid,
            "zone": "PLAY", "targets": list(targets), "sub_options": list(subs)}


END = {"index": 0, "type": "END_TURN", "entity_id": 0, "card_id": None, "zone": None,
       "targets": [], "sub_options": []}


def _snap(board, hand, shop, spells=(), frozen=False, hero="TEST_HERO_1"):
    return {"board": board, "hand": hand, "shop": shop, "shop_spells": list(spells),
            "shop_frozen": frozen, "tavern_tier": 1, "gold": 3, "trinkets": [],
            "hero": hero, "reroll_cost": 1}


def _row(dp, oi, raw_turn, board, hand, shop, options, spells=(), frozen=False):
    return {"schema_version": "states.v2", "game_id": "synthetic", "build": 253216,
            "mmr": 5000, "lobby_tribes": [], "kind": "options", "dp_index": dp,
            "options_index": oi, "turn": (raw_turn + 1) // 2, "raw_turn": raw_turn,
            "snapshot": _snap(board, hand, shop, spells, frozen),
            "hero_power": {"card_id": "TEST_HP", "used": False, "activatable": True},
            "dark_discovery": {"available": False}, "options": [END] + options}


def _choice(dp, raw_turn, kind, cards, hand=()):
    return {"schema_version": "states.v2", "game_id": "synthetic", "build": 253216,
            "mmr": 5000, "lobby_tribes": [], "kind": "choice", "dp_index": dp,
            "turn": (raw_turn + 1) // 2, "raw_turn": raw_turn,
            "snapshot": _snap([], list(hand), [], hero=None if kind == "hero" else "TEST_HERO_1"),
            "choice": {"choice_id": 0, "choice_type": "MULLIGAN" if kind == "hero" else "GENERAL",
                       "choice_kind": kind, "source_entity_id": 2, "source_card_id": None,
                       "min": 1, "max": 1,
                       "cards": [{"entity_id": e, "card_id": c, "name": c, "tags": {},
                                  "cardtype": "HERO" if kind == "hero" else "MINION",
                                  "dark_gift": None} for e, c in cards]}}


CHOOSE_SUBS = [{"index": 0, "entity_id": 41, "card_id": "TEST_CHOOSE_a", "targets": []},
               {"index": 1, "entity_id": 42, "card_id": "TEST_CHOOSE_b", "targets": []}]
DISCOVER = [(50, "TEST_X"), (51, "TEST_Y")]


def _states():
    c, d, a = _m(30, "TEST_C", 1), _m(31, "TEST_D", 2), _m(20, "TEST_A", 1)
    after_move = [_m(31, "TEST_D", 1), _m(30, "TEST_C", 2), _m(20, "TEST_A", 3)]
    spell = {"name": "S", "card_id": "TEST_SPELL", "cost": 1, "entity_id": 22, "buy_cost": 1}
    return [
        _choice(0, 1, "hero", [(60, "TEST_HERO_1"), (61, "TEST_HERO_2")]),
        _row(1, 0, 1, [c, d], [], [a, _m(21, "TEST_B", 2)],
             [_o(1, 13, "TB_BaconShop_DragBuy", [20, 21, 90]), _o(2, 20, "TEST_A"),
              _o(3, 11, rl.REROLL), _o(4, 14, rl.DRAG_SELL, [30, 31, 90]),
              _o(5, 30, "TEST_C"), _o(6, 31, "TEST_D"), _o(7, 4, "TEST_HP", [20, 21])],
             spells=[spell]),
        _row(2, 1, 1, [c, d], [_m(20, "TEST_A", 1)], [_m(21, "TEST_B", 1)],
             [_o(1, 20, "TEST_A"), _o(2, 30, "TEST_C"), _o(3, 31, "TEST_D")]),
        _row(3, 2, 1, [c, _m(20, "TEST_A", 2), _m(31, "TEST_D", 3)], [], [_m(21, "TEST_B", 1)],
             [_o(1, 20, "TEST_A"), _o(2, 30, "TEST_C"), _o(3, 31, "TEST_D")]),
        _row(4, 3, 1, after_move, [_m(40, "TEST_CHOOSE", 1, "SPELL")], [_m(21, "TEST_B", 1)],
             [_o(1, 31, "TEST_D"), _o(2, 30, "TEST_C"), _o(3, 20, "TEST_A"),
              _o(4, 40, "TEST_CHOOSE", subs=CHOOSE_SUBS)]),
        _choice(5, 1, "discover", DISCOVER),
        _choice(6, 1, "discover", DISCOVER),          # re-sent after a re-dump
        _row(7, 4, 1, after_move, [_m(51, "TEST_Y", 1)], [_m(21, "TEST_B", 1)],
             [_o(1, 31, "TEST_D")]),
        _row(8, 5, 1, after_move, [_m(51, "TEST_Y", 1)], [_m(21, "TEST_B", 1)],
             [_o(1, 12, rl.FREEZE)]),
        _row(9, 6, 3, after_move, [_m(51, "TEST_Y", 1)], [_m(23, "TEST_E", 1)],
             [_o(1, 12, rl.FREEZE)]),
        _row(10, 7, 3, after_move, [_m(51, "TEST_Y", 1)], [_m(23, "TEST_E", 1, FROZEN="1")], [],
             frozen=True),
    ]


def _options_rows():
    return [r for r in _states() if r["kind"] == "options"]


def test_scan_counts_resent_options_once_and_reads_actions_and_choices():
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
    assert [c["offered"] for c in scan["choices"]] == [[60, 61], [50, 51], [50, 51]]
    assert [c["chosen"] for c in scan["choices"]] == [[60], [], [51]]  # the re-sent offer gets the pick


def test_enumerate_options_expands_targets_and_positions():
    scan = rl.scan_replay(FIXTURE)
    states = _options_rows()
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
        assert set(o) == OPTION_KEYS
    v1 = copy.deepcopy(states[3])                    # states.v1 row: SubOptions from the XML
    for o in v1["options"]:
        del o["sub_options"]
    assert rl.enumerate_options(v1, scan["dps"][3], scan["entities"])[0] == opts


def test_shop_spell_slot_and_choice_options():
    opts, _ = rl.enumerate_options(_options_rows()[0])
    zones = rl.zone_map(_options_rows()[0]["snapshot"])
    assert zones[22][:2] == ("shop", 2)                                # len(shop) + j
    opts = rl.choice_options(_states()[5])
    assert [(o["type"], o["card_id"], o["source"], o["choice_kind"], o["src_entity"])
            for o in opts] == [("discover", "TEST_X", {"zone": "choice", "slot": 0}, "discover", 50),
                               ("discover", "TEST_Y", {"zone": "choice", "slot": 1}, "discover", 51)]
    assert all(set(o) == OPTION_KEYS for o in opts)


def test_label_game_options_and_choice_rows():
    res = rl.label_game(_states(), FIXTURE, META, 0.6)
    rows = res["rows"]
    chosen = [(r["dp_index"], r["options"][r["chosen"]]["type"]) for r in rows]
    assert chosen == [(0, "discover"), (1, "buy"), (2, "play"), (3, "reposition"), (4, "play"),
                      (6, "discover"), (8, "end_turn"), (9, "freeze"), (10, "end_turn")]
    by_dp = {r["dp_index"]: r for r in rows}
    assert by_dp[2]["options"][by_dp[2]["chosen"]]["position"] == 1
    assert by_dp[3]["options"][by_dp[3]["chosen"]]["position"] == 0
    assert by_dp[4]["options"][by_dp[4]["chosen"]]["card_id"] == "TEST_CHOOSE_b"
    hero = by_dp[0]["options"][by_dp[0]["chosen"]]
    assert (hero["card_id"], hero["choice_kind"], by_dp[0]["chosen"]) == ("TEST_HERO_1", "hero", 0)
    disc = by_dp[6]["options"][by_dp[6]["chosen"]]
    assert (by_dp[6]["chosen"], disc["card_id"], disc["choice_kind"], disc["source"]) == (
        1, "TEST_Y", "discover", {"zone": "choice", "slot": 1})
    assert (by_dp[6]["inference_method"], by_dp[6]["confidence"]) == ("chosen_entities", "high")
    assert by_dp[8]["inference_method"] == "no_action_turn_advanced"
    assert (by_dp[10]["inference_method"], by_dp[10]["confidence"]) == (
        "no_action_last_decision_point", "medium")
    assert [(q["dp_index"], q["kind"], q["reason"]) for q in res["quarantine"]] == [
        (5, "choice", "choice_offer_resent"), (7, "options", "noop_move")]
    r = rows[1]
    assert set(r) == ROW_KEYS
    assert r["current_patch"] is True and r["weight"] == 0.6 and r["placement"] == 1
    assert (r["created_at"], r["created_ts"]) == (CREATED, 1790542793982)
    assert r["flags"] == []
    assert r["state"] == _states()[1]["snapshot"]
    assert {t: dict(c) for t, c in res["transitions"].items()} == {
        "buy": {"pass": 1}, "play": {"pass": 2}, "reposition": {"pass": 1},
        "end_turn": {"pass": 1, "na": 1}, "freeze": {"pass": 1},
        "discover:hero": {"pass": 1}, "discover:discover": {"pass": 1}}
    json.dumps(res["rows"])


def test_transition_failure_and_dp_mismatch_quarantine():
    states = _states()
    states[3]["snapshot"]["board"] = states[2]["snapshot"]["board"]   # play left no trace
    res = rl.label_game(states, FIXTURE, META, 1.0)
    assert "transition_failed:play" in {q["reason"] for q in res["quarantine"]}
    res = rl.label_game(_states()[:9], FIXTURE, META, 1.0)
    assert res["rows"] == [] and {q["reason"] for q in res["quarantine"]} == {"dp_count_mismatch"}
    assert len(res["quarantine"]) == 9


def test_infer_pick_fallbacks():
    row = _choice(5, 1, "trinket", [(5, "A"), (6, "B"), (7, "C")])
    xch = {"offered": [5, 6, 7], "chosen": [], "moved_to_hand": [6]}
    res = rl.infer_pick(row, xch, None)
    assert (res["option"]["card_id"], res["method"], res["confidence"]) == ("B", "moved_to_hand", "medium")
    assert rl.infer_pick(row, dict(xch, moved_to_hand=[]), None)["reason"] == "choice_pick_unresolved"
    assert rl.infer_pick(row, xch, copy.deepcopy(row))["reason"] == "choice_offer_resent"
    assert rl.infer_pick(row, None, None)["reason"] == "choice_not_in_replay"
    assert rl.infer_pick(row, dict(xch, chosen=[9]), None)["reason"] == "chosen_entities_not_offered"
    res = rl.infer_pick(row, dict(xch, chosen=[7]), None)
    assert (res["option"]["source"]["slot"], res["method"]) == (2, "chosen_entities")


def test_pick_transition_pass_unverified_na():
    opt = rl.choice_options(_states()[6])[1]                          # TEST_Y, entity 51
    nxt = _options_rows()[4]
    assert rl.check_pick_transition(nxt, opt) == "pass"
    gone = copy.deepcopy(nxt)
    gone["snapshot"]["hand"] = []
    assert rl.check_pick_transition(gone, opt) == "unverified"
    gone["snapshot"]["board"].append(_m(99, "TEST_Y_G", 4))          # tripled: golden by card id
    assert rl.check_pick_transition(gone, opt) == "pass"
    assert rl.check_pick_transition(None, opt) == "na"


def test_magnetic_play_onto_mech_passes_transition():
    mech = _m(70, "TEST_MECH", 1)
    a = _row(1, 0, 1, [mech], [_m(71, "TEST_MAG", 1, MAGNETIC="1")], [], [_o(1, 71, "TEST_MAG")])
    b = _row(2, 1, 1, [mech], [], [], [])
    opt = rl.make_option("play", "TEST_MAG", source=("hand", 0), position=0, src_entity=71)
    assert rl.check_transition(a, b, opt) == "pass"
    a["snapshot"]["hand"][0]["tags"].pop("MAGNETIC")
    assert rl.check_transition(a, b, opt) == "fail"


def test_play_position_survives_destroy_and_freeze_per_minion():
    a_board = [_m(1, "U1", 1), _m(2, "U2", 2), _m(3, "U3", 3)]
    a = _row(1, 0, 1, a_board, [_m(9, "TEST_BC", 1)], [], [])
    b = _row(2, 1, 1, [a_board[0], _m(9, "TEST_BC", 2), a_board[2]], [], [], [])
    opt = rl.make_option("play", "TEST_BC", source=("hand", 0), position=2, src_entity=9)
    assert rl.check_transition(a, b, opt) == "pass"                   # U2 destroyed by battlecry
    opt["position"] = 3
    assert rl.check_transition(a, b, opt) == "fail"
    shop_a = [_m(20, "S1", 1, FROZEN="1"), _m(21, "S2", 2)]
    shop_b = [_m(20, "S1", 1, FROZEN="1"), _m(21, "S2", 2, FROZEN="1")]
    frz = rl.make_option("freeze", src_entity=12)
    fa, fb = _row(1, 0, 1, [], [], shop_a, []), _row(2, 1, 1, [], [], shop_b, [])
    fa["snapshot"]["shop_frozen"] = fb["snapshot"]["shop_frozen"] = True
    assert rl.check_transition(fa, fb, frz) == "pass"                 # re-freeze after a new minion
    assert rl.check_transition(fb, fb, frz) == "fail"
    assert rl.check_transition(_row(1, 0, 1, [], [], [], []), _row(2, 1, 1, [], [], [], []), frz) == "na"


def test_created_fields():
    assert rl.created_fields({"creationDate": CREATED}) == (CREATED, 1790542793982)
    assert rl.created_fields({"creationDate": "2026-09-24T02:28:50-04:00"}) == (
        "2026-09-24T06:28:50.000Z", 1790231330000)
    assert rl.created_fields({"creationTimestamp": 1790542793000}) == (
        "2026-09-27T20:59:53.000Z", 1790542793000)
    assert rl.created_fields({}) == (None, None)


def test_gift_flags_track_golden_from_gifted_copies():
    gift = {"card_id": "BG36_MidGameEffect_000t1", "name": "g"}
    x1, x2 = _m(1, "X", 1), _m(2, "X", 2)
    x1["dark_gift"] = x2["dark_gift"] = gift
    gold = _m(3, "X_G", 1)
    gold["dark_gift"] = gift
    rows = [_row(0, 0, 1, [x1, x2], [], [], []), _row(1, 1, 1, [], [gold], [], []),
            _row(2, 2, 3, [dict(gold, entity_id=9)], [], [], []), _row(3, 3, 3, [], [], [], [])]
    assert rl.gift_flags(rows) == {1: [rl.FLAG_GOLDEN_GIFT], 2: [rl.FLAG_GOLDEN_GIFT]}
    x2["dark_gift"] = None                                            # only one gifted copy
    assert rl.gift_flags(rows) == {}


def test_mmr_weight_curve():
    w = rl.mmr_weight_fn([1000, 5000, 9000])
    assert w(5000) == pytest.approx(0.6)
    assert w(1000) == pytest.approx(0.2 + 0.8 / 6, abs=1e-6)     # weights rounded to 1e-6
    assert w(9000) == pytest.approx(1 - 0.8 / 6, abs=1e-6)
    assert w(0) == pytest.approx(0.2) and w(10 ** 6) == pytest.approx(1.0)
    vals = [w(m) for m in range(0, 12000, 250)]
    assert vals == sorted(vals) and w(None) is None


def test_run_writes_outputs_and_drops_games(tmp_path):
    states_dir, replays = tmp_path / "states", tmp_path / "replays"
    (states_dir / "quarantine").mkdir(parents=True)
    replays.mkdir()
    with gzip.open(states_dir / "synthetic.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in _states():
            fh.write(json.dumps(r) + "\n")
    (states_dir / "quarantine" / "bad.json").write_text("{}")
    with open(FIXTURE, "rb") as src, gzip.open(replays / "synthetic.xml.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    manifest, corpus = tmp_path / "m.json", tmp_path / "c.json"
    manifest.write_text(json.dumps({"games": [
        META, {"reviewId": "other", "mmr": 9000, "placement": 3},
        {"reviewId": "bad", "mmr": 9000, "placement": 1}]}))
    corpus.write_text(json.dumps({"games": [{"mmr": m} for m in (1000, 5000, 9000)]}))
    out = tmp_path / "labels"
    s = rl.run(str(states_dir), str(replays), str(manifest), str(out), str(corpus))
    assert s["rows"] == 9 and s["decision_points"] == 11 and s["match_rate"] == 0.8182
    assert (s["rows_options"], s["rows_choice"]) == (7, 2)
    assert (s["non_decisions"], s["decision_match_rate"]) == (2, 1.0)
    assert s["games_dropped_by_reason"] == {"placement_not_1": 1, "states_quarantined": 1}
    assert s["quarantine"]["reasons"] == {"choice_offer_resent": 1, "noop_move": 1}
    assert s["chosen_in_legal"]["rate"] == 1.0
    assert s["games"][0]["weight"] == 0.6
    assert s["rows_per_action_type"]["play"] == 2 and s["rows_per_action_type"]["discover"] == 2
    assert s["rows_per_choice_kind"] == {"hero": 1, "discover": 1, "dark_discovery": 0,
                                         "trinket": 0, "quest": 0, "other": 0}
    assert s["created_at_range"] == [CREATED, CREATED]
    assert s["worst_games"][0]["game_id"] == "synthetic"
    with gzip.open(out / "synthetic.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert len(fh.readlines()) == 9
    assert (out / "quarantine" / "synthetic.jsonl").read_text().count("\n") == 2
    assert not (out / "choices").exists()
    assert json.loads((out / "summary.json").read_text())["rows"] == 9


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
    assert any(r["options"][r["chosen"]]["type"] == "discover" for r in res["rows"])
    for r in res["rows"]:
        assert 0 <= r["chosen"] < len(r["options"])