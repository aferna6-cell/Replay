"""bc-enc-v3 features (costs, hero power, Dark Gift, shop context, reposition,
Choose One variants) and live vs row parity of the shared encoder. The v3 blocks
are unchanged in bc-enc-v5 (tests/test_encoder_v4.py covers the v4 additions,
tests/test_encoder_v5.py the hero identity appended after them); only the hero
power id hash became a vocabulary one-hot."""

import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hsbg_coach import encode as enc  # noqa: E402
from hsbg_coach.bg import BGTracker, Snapshot  # noqa: E402
from hsbg_coach.parser import parse_line  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POWER_LOG = REPO / "Power.log"
XML_FIXTURE = REPO / "tests" / "fixtures" / "synthetic_bg_replay.xml"
N_STATE_V2 = 3 * enc._ZONE_DIM + 15            # bc-enc-v2 state scalars end here
V3 = {name: N_STATE_V2 + i for i, name in enumerate(enc.STATE_SCALARS[15:])}
OPT_V2 = (enc.OPTION_DIM - enc._V3_OPTION - enc._V4_OPTION
          - enc._V5_OPTION)   # bc-enc-v3 option block


def _m(name, i, cid=None, atk=2, hp=2, ctype="MINION", **tags):
    return {"entity_id": 100 + i, "card_id": cid or f"TEST_{i}", "name": name,
            "attack": atk, "health": hp, "position": i + 1,
            "tags": dict({"CARDTYPE": ctype}, **{k: str(v) for k, v in tags.items()})}


def _snap(**kw):
    d = {"game_counter": 1, "turn": 9, "phase": "recruit", "tavern_tier": 3, "gold": 5,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "level_cost": 7}
    d.update(kw)
    return d


def _state(snap, name):
    return float(enc.encode_state(snap)[V3[name]])


def _v3(snap, option):
    """bc-enc-v3 option block split into its parts."""
    v = enc.encode_option(snap, option)[OPT_V2:]
    parts, i = {}, 0
    for name, n in (("affordable", 1), ("buy", 5), ("shopctx", 5), ("pos", 6),
                    ("kw", len(enc.KEYWORDS)), ("variant", enc.HASH_BUCKETS)):
        parts[name] = v[i:i + n].tolist()
        i += n
    return parts


def _hp_block(state):
    """Hero-power one-hot. bc-enc-v5 appends the hero one-hot after it."""
    end = enc.STATE_DIM - enc.HERO_VOCAB_DIM
    return state[end - enc.HP_VOCAB_DIM:end]


def test_dims_and_version():
    assert enc.ENCODER_VERSION == "bc-enc-v5"
    assert enc.STATE_DIM == (N_STATE_V2 + len(V3) + enc.HP_VOCAB_DIM
                             + enc.HERO_VOCAB_DIM)
    assert enc.OPTION_DIM == 94 + enc._V3_OPTION + enc._V4_OPTION + enc._V5_OPTION


def test_buy_uses_slot_buy_cost_and_marks_pairs_triples_and_tribe():
    board = [_m("Scallywag", 0, cid="BG_PIR"), _m("Scallywag", 1, cid="BG_PIR"),
             _m("Other", 2, cid="BG_OTH")]
    shop = [dict(_m("Scallywag", 0, cid="BG_PIR"), buy_cost=1),
            dict(_m("Other", 1, cid="BG_OTH"), buy_cost=3), _m("New", 2, cid="BG_NEW")]
    s = _snap(board=board, shop=shop, gold=2)
    b0 = enc.make_option("buy", "BG_PIR", ("shop", 0))
    b1 = enc.make_option("buy", "BG_OTH", ("shop", 1))
    b2 = enc.make_option("buy", "BG_NEW", ("shop", 2))
    assert enc.option_cost(s, b0) == 1 and enc.option_cost(s, b1) == 3
    assert enc.option_cost(s, b2) == 3                     # no buy_cost -> 3
    p0, p1 = _v3(s, b0), _v3(s, b1)
    assert p0["affordable"] == [1.0] and p1["affordable"] == [0.0]
    assert p0["buy"][:3] == [pytest.approx(1 / 3), 1.0, 1.0]     # pair + triple maker
    assert p1["buy"][1:3] == [1.0, 0.0]                   # pair only
    assert _v3(s, b2)["buy"][1:3] == [0.0, 0.0]
    # live candidate generator: affordability from buy_cost
    buys = [o["source"]["slot"] for o in enc.legal_options(s) if o["type"] == "buy"]
    assert buys == [0]
    assert _state(s, "shop_pairs") == pytest.approx(2 / 7)
    assert _state(s, "shop_triples") == pytest.approx(1 / 7)
    assert _state(s, "min_buy_cost") == pytest.approx(1 / 3)
    assert _state(s, "n_buy_affordable") == pytest.approx(1 / 7)


def test_reroll_cost_free_and_affordability():
    free = _snap(gold=0, reroll_cost=0)
    assert _state(free, "reroll_free") == 1.0 and _state(free, "can_reroll") == 1.0
    assert "reroll" in [o["type"] for o in enc.legal_options(free)]
    assert enc.option_cost(free, enc.make_option("reroll")) == 0
    dear = _snap(gold=1, reroll_cost=2)
    assert _state(dear, "can_reroll") == 0.0
    assert "reroll" not in [o["type"] for o in enc.legal_options(dear)]
    assert _state(_snap(gold=3), "reroll_cost") == pytest.approx(1 / 5)   # default 1


def test_hero_power_identity_cost_and_used_this_turn():
    hp = {"name": "Hero Power", "card_id": "BG36_HERO_002p", "cost": 2, "usable": True}
    s = _snap(gold=5, hero_power=hp)
    used = _snap(gold=5, hero_power=dict(hp, usable=False))
    poor = _snap(gold=1, hero_power=dict(hp, usable=False))
    assert _state(s, "hp_present") == 1.0 and _state(s, "hp_used") == 0.0
    assert _state(used, "hp_used") == 1.0                 # affordable but not usable
    assert _state(poor, "hp_used") == 0.0 and _state(poor, "hp_affordable") == 0.0
    # bc-enc-v4: which power it is = one-hot over HERO_POWER_VOCAB (+ "other")
    h = _hp_block(enc.encode_state(s))
    assert h.sum() == 1.0 and np.array_equal(h, enc.hero_power_onehot("BG36_HERO_002p"))
    assert h[enc.HERO_POWER_VOCAB.index("BG36_HERO_002p")] == 1.0
    other = enc.encode_state(_snap(gold=5, hero_power=dict(hp, card_id="BG20_HERO_280p5")))
    assert not np.array_equal(_hp_block(other), h)
    assert _hp_block(enc.encode_state(_snap())).sum() == 0.0
    # crc32 buckets are stable across processes (python's hash() is salted);
    # still used for Choose One variant ids
    assert enc.id_hash("BG36_HERO_002p").index(1.0) == 1


def test_dark_gift_state_and_option():
    dg = {"name": "Dark Discovery", "card_id": "BG36_Button_DarkGift", "cost": 3,
          "usable": True, "entity_id": 280}
    s = _snap(gold=4, dark_gift=dg)
    assert [_state(s, k) for k in ("dg_present", "dg_usable", "dg_affordable")] == [1, 1, 1]
    assert _state(s, "dg_cost") == pytest.approx(0.3)
    opt = enc.make_option("play", "BG36_Button_DarkGift")
    assert enc.option_cost(s, opt) == 3 and _v3(s, opt)["affordable"] == [1.0]
    assert _v3(_snap(gold=2, dark_gift=dg), opt)["affordable"] == [0.0]


def test_freeze_and_reroll_see_shop_context():
    board = [_m("Scallywag", 0, cid="BG_PIR")]
    shop = [_m("Scallywag", 0, cid="BG_PIR", TECH_LEVEL=5), _m("x", 1, TECH_LEVEL=2)]
    s = _snap(board=board, shop=shop, tavern_tier=3, shop_frozen=True)
    f = _v3(s, enc.make_option("freeze"))["shopctx"]
    assert f[0] == 1.0 and f[2] == pytest.approx(1 / 7)   # frozen now; one pair in shop
    assert f[4] == pytest.approx(2 / 6)                   # best shop tier 5 vs tavern 3
    assert _v3(s, enc.make_option("reroll"))["shopctx"] == f
    assert _v3(s, enc.make_option("end_turn"))["shopctx"] == [0.0] * 5
    assert _state(s, "shop_max_tier_rel") == pytest.approx(2 / 6)


def test_reposition_and_play_position_block():
    board = [_m("a", 0, atk=1, hp=9), _m("b", 1, atk=5, hp=1), _m("c", 2, atk=3, hp=3)]
    s = _snap(board=board, hand=[_m("h", 0, cid="H", atk=9, hp=9)])
    left = _v3(s, enc.make_option("reposition", "TEST_1", ("board", 1), position=0))["pos"]
    assert left[:4] == [pytest.approx(-1 / 6), 1.0, 0.0, pytest.approx(3 / 7)]
    assert left[4:] == [1.0, 0.0]                         # strongest attack, lowest health
    right = _v3(s, enc.make_option("reposition", "TEST_0", ("board", 0), position=2))["pos"]
    assert right[:3] == [pytest.approx(2 / 6), 0.0, 1.0]
    play = _v3(s, enc.make_option("play", "H", ("hand", 0), position=3))["pos"]
    assert play[:4] == [0.0, 0.0, 1.0, pytest.approx(3 / 7)] and play[4:] == [1.0, pytest.approx(2 / 3)]
    assert _v3(s, enc.make_option("sell", "TEST_0", ("board", 0)))["pos"] == [0.0] * 6


def test_keywords_and_choose_one_variant_identity():
    board = [_m("dr", 0, DEATHRATTLE=1, AVENGE=3), _m("x", 1)]
    s = _snap(board=board, hand=[_m("Parent", 0, cid="BG31_880", CHOOSE_ONE=1)])
    kw = _v3(s, enc.make_option("sell", "TEST_0", ("board", 0)))["kw"]
    assert kw == [1.0, 1.0, 0.0, 0.0, 0.0]
    a = enc.make_option("play", "BG31_880t", ("hand", 0))
    b = enc.make_option("play", "BG31_880t2", ("hand", 0))
    parent = enc.make_option("play", "BG31_880", ("hand", 0))
    assert _v3(s, a)["variant"] == enc.id_hash("BG31_880t")
    assert _v3(s, b)["variant"] == enc.id_hash("BG31_880t2")
    assert _v3(s, parent)["variant"] == [0.0] * enc.HASH_BUCKETS


def test_context_cache_is_per_snapshot_object():
    s1 = _snap(shop=[_m("x", 0, TECH_LEVEL=6)], tavern_tier=1)
    s2 = dict(s1, shop=[_m("x", 0, TECH_LEVEL=1)])
    a, b = enc.encode_state(s1), enc.encode_state(s2)
    assert not np.array_equal(a, b)
    assert np.array_equal(enc.encode_state(s1), a)


# --- live vs row parity ------------------------------------------------------
def _forms(snap):
    return [snap, snap.to_dict(), Snapshot.from_dict(snap.to_dict()),
            json.loads(json.dumps(snap.to_dict()))]          # a jsonl training row


def _assert_same_encoding(snap):
    ref_s = enc.encode_state(snap)
    ref_opts = enc.legal_options(snap)
    ref_o = [enc.encode_option(snap, o) for o in ref_opts]
    for form in _forms(snap)[1:]:
        assert np.array_equal(enc.encode_state(form), ref_s)
        assert enc.legal_options(form) == ref_opts
        for o, v in zip(ref_opts, ref_o):
            assert np.array_equal(enc.encode_option(form, o), v)
    return ref_s, ref_opts


def test_live_power_log_snapshots_encode_identically_as_rows():
    """Every sampled recruit-phase snapshot of the sample Power.log (the live
    path: parse_line -> BGTracker) encodes identically as a JSON training row,
    and the v3 cost fields come from the tracker, not from Options."""
    if not POWER_LOG.is_file():
        pytest.skip("repo Power.log sample not present")
    t, checked, with_costs = BGTracker(), 0, 0
    with POWER_LOG.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            ev = parse_line(line)
            if ev:
                t.feed(ev)
            if i % 400 or not t.in_bg or t.snapshot().phase != "recruit":
                continue
            snap = t.snapshot()
            _, opts = _assert_same_encoding(snap)
            checked += 1
            if snap.reroll_cost is not None and any(m.buy_cost is not None for m in snap.shop):
                with_costs += 1
                for o in opts:
                    if o["type"] == "buy" and o["source"]["slot"] < len(snap.shop):
                        m = snap.shop[o["source"]["slot"]]
                        want = 3 if m.buy_cost is None else m.buy_cost
                        assert enc.option_cost(snap, o) == want
                assert enc.reroll_cost(snap) == snap.reroll_cost
    assert checked >= 20 and with_costs >= 10


def test_xml_rows_encode_like_the_tracker_snapshot():
    """State Builder rows (hsreplay_xml -> BGTracker -> Snapshot.to_dict ->
    JSON) encode exactly like the tracker's Snapshot object at that Options
    block, including hero power and Dark Gift features."""
    from hsbg_coach.hsreplay_xml import iter_events
    from hsbg_coach.replay_states import build_game
    meta = {"reviewId": "synthetic", "buildNumber": 1, "mmr": 5000, "placement": 1,
            "tribes": {"available": [{"id": 14, "name": "MURLOC"}]},
            "finalComp": {"board": [{"cardId": "BG_TEST_GIFTED", "golden": False,
                                     "atk": 5, "health": 5}]}}
    rows = [r for r in build_game(str(XML_FIXTURE), meta)["rows"] if r["kind"] == "options"]
    tracker, snaps = BGTracker(), []
    for ev in iter_events(str(XML_FIXTURE)):
        tracker.feed(ev)
        if ev.kind == "OPTIONS":
            snaps.append(tracker.snapshot())
    pairs = [(rows[0], snaps[0]), (rows[-1], snaps[-1])]  # the re-sent block keeps the later
    for row, snap in pairs:
        state = json.loads(json.dumps(row["snapshot"]))
        assert np.array_equal(enc.encode_state(state), enc.encode_state(snap))
        _assert_same_encoding(snap)
    hp_state = json.loads(json.dumps(rows[0]["snapshot"]))
    assert hp_state["hero_power"]["card_id"] == "BG36_HERO_002p"
    hp_ids = _hp_block(enc.encode_state(hp_state))
    assert hp_ids.sum() == 1.0 and hp_ids[enc.HERO_POWER_VOCAB.index("BG36_HERO_002p")] == 1.0
