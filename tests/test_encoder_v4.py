"""bc-enc-v4: hero power identity (vocabulary one-hot in state and on the
hero_power option), passive hero powers, turn progress, last-gold flags."""

import pytest

np = pytest.importorskip("numpy")

from hsbg_coach import encode as enc  # noqa: E402

Z = 3 * enc._ZONE_DIM
STATE_IX = {name: Z + i for i, name in enumerate(enc.STATE_SCALARS)}
V4_OPT = enc.OPTION_DIM - enc._V4_OPTION - enc._V5_OPTION  # bc-enc-v4 option block
HP = {"name": "Hero Power", "card_id": "BG36_HERO_002p", "cost": 2, "usable": True}


def _snap(**kw):
    d = {"game_counter": 1, "turn": 6, "phase": "recruit", "tavern_tier": 3, "gold": 5,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "level_cost": 7}
    d.update(kw)
    return d


def _st(snap, name):
    return float(enc.encode_state(snap)[STATE_IX[name]])


def _v4(snap, option):
    v = enc.encode_option(snap, option)[V4_OPT:V4_OPT + enc._V4_OPTION]
    return v[:enc.HP_VOCAB_DIM], v[enc.HP_VOCAB_DIM:].tolist()


def test_version_dims_and_vocab():
    assert enc.ENCODER_VERSION == "bc-enc-v5"
    assert enc.HP_VOCAB_DIM == len(enc.HERO_POWER_VOCAB) + 1
    assert len(set(enc.HERO_POWER_VOCAB)) == len(enc.HERO_POWER_VOCAB)
    assert enc.HERO_POWER_VOCAB[:2] == ("BG36_HERO_000p", "BG36_HERO_002p")
    assert enc.PASSIVE_HERO_POWERS <= set(enc.HERO_POWER_VOCAB)
    assert enc._V4_OPTION == enc.HP_VOCAB_DIM + 2
    assert enc.STATE_SCALARS[-5:] == ("rolls_affordable", "buys_affordable", "hp_passive",
                                      "gold_spent_turn", "gold_left_share")


def test_hero_power_onehot_known_unknown_and_none():
    a = enc.hero_power_onehot("BG36_HERO_000p")
    assert sum(a) == 1.0 and a[0] == 1.0
    other = enc.hero_power_onehot("NOT_A_REAL_HP")
    assert sum(other) == 1.0 and other[-1] == 1.0
    assert sum(enc.hero_power_onehot(None)) == 0.0


def test_hero_power_option_carries_which_power_it_is():
    """v3 encoded every hero power option identically apart from cost; v4 tells them apart."""
    s1, s2 = _snap(hero_power=HP), _snap(hero_power=dict(HP, card_id="BG21_HERO_000p"))
    o1 = enc.make_option("hero_power", "BG36_HERO_002p")
    o2 = enc.make_option("hero_power", "BG21_HERO_000p")
    id1, _ = _v4(s1, o1)
    id2, _ = _v4(s2, o2)
    assert id1.sum() == 1.0 and id1[enc.HERO_POWER_VOCAB.index("BG36_HERO_002p")] == 1.0
    assert id2[enc.HERO_POWER_VOCAB.index("BG21_HERO_000p")] == 1.0
    assert not np.array_equal(enc.encode_option(s1, o1), enc.encode_option(s2, o2))
    # the option's own card id wins (a second hero power, e.g. one granted mid-game)
    idx, _ = _v4(s1, o2)
    assert idx[enc.HERO_POWER_VOCAB.index("BG21_HERO_000p")] == 1.0
    # no id on the option -> the snapshot's power; non hero-power options get zeros
    idn, _ = _v4(s1, enc.make_option("hero_power"))
    assert np.array_equal(idn, id1)
    for opt in (enc.make_option("reroll"), enc.make_option("end_turn"), enc.make_option("level")):
        assert _v4(s1, opt)[0].sum() == 0.0


def test_passive_hero_power_is_not_usable_and_not_offered():
    passive = _snap(hero_power={"name": "Hero Power", "card_id": "TB_BaconShop_HP_042",
                                "cost": 0, "usable": True})
    assert _st(passive, "hp_passive") == 1.0
    assert _st(passive, "hp_usable") == 0.0 and _st(passive, "hp_affordable") == 0.0
    assert _st(passive, "hp_used") == 0.0 and _st(passive, "hp_present") == 1.0
    assert "hero_power" not in [o["type"] for o in enc.legal_options(passive)]
    active = _snap(hero_power=HP)
    assert _st(active, "hp_passive") == 0.0 and _st(active, "hp_usable") == 1.0
    assert "hero_power" in [o["type"] for o in enc.legal_options(active)]


def test_turn_progress_features():
    s = _snap(turn=6, gold=3)                     # turn-start gold min(6 + 2, 10) = 8
    assert enc.turn_start_gold(s) == 8
    assert _st(s, "gold_spent_turn") == pytest.approx(0.5)
    assert _st(s, "gold_left_share") == pytest.approx(3 / 8)
    rich = _snap(turn=12, gold=14)                # extra gold: nothing spent, share capped at 2
    assert enc.turn_start_gold(rich) == 10
    assert _st(rich, "gold_spent_turn") == 0.0
    assert _st(rich, "gold_left_share") == pytest.approx(1.4)


def test_last_gold_and_can_buy_after_flags():
    shop = [{"entity_id": 1, "card_id": "X", "name": "x", "attack": 1, "health": 1,
             "position": 1, "buy_cost": 3, "tags": {"CARDTYPE": "MINION"}}]
    hp = enc.make_option("hero_power", "BG36_HERO_002p")
    _, flags = _v4(_snap(gold=2, hero_power=HP, shop=shop), hp)
    assert flags == [1.0, 0.0]                    # spends the last gold, no buy after
    _, flags = _v4(_snap(gold=5, hero_power=HP, shop=shop), hp)
    assert flags == [0.0, 1.0]                    # 3 left: can still buy
    free = _snap(gold=0, hero_power=dict(HP, cost=0), shop=shop)
    assert _v4(free, hp)[1] == [0.0, 0.0]         # free power never "spends the last gold"


def test_hero_power_targets_ranked_as_one_decision():
    torch = pytest.importorskip("torch")
    from ml.bc_policy import BCPolicy, OptionScorer, group_hero_power
    s = _snap(gold=5, hero_power=HP, hand=[
        {"entity_id": 7, "card_id": "A", "name": "a", "attack": 1, "health": 1,
         "position": 1, "tags": {"CARDTYPE": "MINION"}},
        {"entity_id": 8, "card_id": "B", "name": "b", "attack": 1, "health": 1,
         "position": 2, "tags": {"CARDTYPE": "MINION"}}])
    opts = [enc.make_option("reroll"),
            enc.make_option("hero_power", "BG36_HERO_002p", target=("hand", 0)),
            enc.make_option("hero_power", "BG36_HERO_002p", target=("hand", 1)),
            enc.make_option("end_turn")]
    O = np.stack([enc.encode_option(s, o) for o in opts])
    raw = np.array([1.0, 0.7, 0.6, -2.0], np.float32)
    g = group_hero_power(raw, O)
    assert g[1] == pytest.approx(np.logaddexp(0.7, 0.6)) and g[1] > g[0]   # HP now wins
    assert g[2] == raw[2] and g[0] == raw[0] and g[3] == raw[3]            # others unchanged
    assert int(np.argmax(g)) == 1                                          # best target kept
    # one hero power option (live emits one) or none: unchanged
    assert np.array_equal(group_hero_power(raw[[0, 1, 3]], O[[0, 1, 3]]), raw[[0, 1, 3]])
    assert np.array_equal(group_hero_power(raw[[0, 3]], O[[0, 3]]), raw[[0, 3]])
    # BCPolicy applies it (decode recorded in the checkpoint meta)
    torch.manual_seed(0)
    pol = BCPolicy(OptionScorer())
    assert pol.meta["decode"] == "hp_group"
    st = enc.encode_state(s)
    assert np.allclose(pol.score_encoded(st, O), group_hero_power(pol.score_raw(st, O), O))
    pol.meta["decode"] = None
    assert np.array_equal(pol.score_encoded(st, O), pol.score_raw(st, O))
