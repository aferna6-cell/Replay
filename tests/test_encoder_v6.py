"""bc-enc-v6: E.T.C., King of Rock, Queen Azshara, and Lady Vashj each have
their own hero index, and their skins share that index rather than the unknown
slot. A bc-enc-v5 checkpoint is refused instead of loaded into the wider net,
and the live overlay falls back to the eval-net advisor the same way a v4
checkpoint did under v5."""

import pytest

np = pytest.importorskip("numpy")

from hsbg_coach import encode as enc  # noqa: E402

# HearthstoneJSON: portrait BG26_HERO_100_SKIN_A, hero power BG25_HERO_105p
# (Sign a New Artist), already in the v2.1 prefix of HERO_POWER_VOCAB.
_ETC = "BG26_HERO_100"
_ETC_HP = "BG25_HERO_105p"
_SKINS = (
    "BG26_HERO_100_SKIN_A",
    "BG26_HERO_100_SKIN_B",
    "BG26_HERO_100_SKIN_D",
    "BG26_HERO_100_SKIN_E",
)
# bc-enc-v5 layout. v6 appends three heroes and three hero powers.
_V5_STATE_DIM = 445
_V5_OPTION_DIM = 330
_AZSHARA = "BG22_HERO_007"
_VASHJ = "BG23_HERO_304"
_AZSHARA_SKINS = (
    "BG22_HERO_007_SKIN_A", "BG22_HERO_007_SKIN_B",
    "BG22_HERO_007_SKIN_C", "BG22_HERO_007_SKIN_D",
    "BG22_HERO_007_SKIN_At", "BG22_HERO_007_SKIN_Bt",
    "BG22_HERO_007_SKIN_Ct", "BG22_HERO_007_SKIN_Dt",
    "BG22_HERO_007t",
)
_VASHJ_SKINS = (
    "BG23_HERO_304_SKIN_A", "BG23_HERO_304_SKIN_B", "BG23_HERO_304_SKIN_C",
    "BG23_HERO_304_SKIN_D", "BG23_HERO_304_SKIN_E",
)


def _snap(**kw):
    d = {"game_counter": 1, "turn": 6, "phase": "recruit", "tavern_tier": 3, "gold": 10,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "hero": None, "level_cost": 7}
    d.update(kw)
    return d


def _hero_block_state(state):
    return state[-enc.HERO_VOCAB_DIM:]


def _hero_block_option(option_vec):
    return option_vec[-enc.HERO_VOCAB_DIM:]


def test_v6_appended_and_dims_grew_with_no_reserved_slot():
    """New ids are appended. Earlier indices, including Sylvanas, stay put."""
    assert enc.ENCODER_VERSION == "bc-enc-v6"
    assert enc.HERO_VOCAB[-4] == "TB_BaconShop_HERO_59t"
    assert enc.HERO_VOCAB[-3:] == (_ETC, _AZSHARA, _VASHJ)
    assert enc.HERO_VOCAB[0] == "BG36_HERO_002"
    assert enc.HERO_VOCAB.index("BG23_HERO_306") < enc.HERO_VOCAB.index(_ETC)
    assert enc.HERO_VOCAB_DIM == len(enc.HERO_VOCAB) + 1 == 121
    assert enc.HERO_VOCAB_DIM == _hero_block_state(enc.encode_state(_snap())).shape[0]
    # v5 was 445 / 330 with HP dim 78 and hero dim 118. Three heroes (+3) and
    # three hero powers (+3) and no spare slot: state +6, option +6.
    assert enc.HP_VOCAB_DIM == 81
    assert enc.HERO_POWER_VOCAB[-4] == "BG34_HERO_000p"
    assert enc.HERO_POWER_VOCAB[-3:] == (
        "BG22_HERO_007p", "BG22_HERO_007p2", "BG23_HERO_304p")
    assert enc.STATE_DIM == _V5_STATE_DIM + 6 == 451
    assert enc.OPTION_DIM == _V5_OPTION_DIM + 6 == 336
    assert enc._V5_OPTION == enc.HERO_VOCAB_DIM


def test_etc_and_skins_map_to_own_index_not_other():
    idx = enc.HERO_VOCAB.index(_ETC)
    assert idx == len(enc.HERO_VOCAB) - 3
    base = enc.hero_onehot(_ETC)
    assert base[idx] == 1.0 and base[-1] == 0.0 and sum(base) == 1.0
    # Distinct from the unknown slot and from E.T.C., Band Manager.
    assert enc.hero_onehot("BG99_HERO_999")[-1] == 1.0
    assert not np.array_equal(base, enc.hero_onehot("BG25_HERO_105"))
    for skin in _SKINS:
        assert enc.canonical_hero_id(skin) == _ETC
        oh = enc.hero_onehot(skin)
        assert oh == base
        assert oh[idx] == 1.0 and oh[-1] == 0.0
    skin_state = enc.encode_state(_snap(hero="BG26_HERO_100_SKIN_A"))
    base_state = enc.encode_state(_snap(hero=_ETC))
    assert np.array_equal(skin_state, base_state)
    block = _hero_block_state(skin_state)
    assert block.sum() == 1.0 and block[idx] == 1.0 and block[-1] == 0.0
    # A hero-pick offer of the skin uses the same slot; a non-hero option does not.
    snap = _snap(hero="TB_BaconShop_HERO_PH")
    offered = enc.make_option("discover", "BG26_HERO_100_SKIN_A",
                              source=("choice", 0), choice_kind="hero")
    offer_block = _hero_block_option(enc.encode_option(snap, offered))
    assert np.array_equal(offer_block, base)
    plain = enc.make_option("discover", "BG26_HERO_100_SKIN_A", choice_kind="discover")
    assert _hero_block_option(enc.encode_option(snap, plain)).sum() == 0.0


def test_etc_hero_power_is_already_in_vocab():
    """Sign a New Artist (BG25_HERO_105p) was in the v2.1 prefix. Not appended."""
    assert _ETC_HP in enc.HERO_POWER_VOCAB
    hp_idx = enc.HERO_POWER_VOCAB.index(_ETC_HP)
    assert hp_idx < 72
    oh = enc.hero_power_onehot(_ETC_HP)
    assert sum(oh) == 1.0 and oh[hp_idx] == 1.0 and oh[-1] == 0.0
    snap = _snap(hero=_ETC, hero_power={"name": "Sign a New Artist",
                                        "card_id": _ETC_HP, "cost": 3, "usable": True})
    state = enc.encode_state(snap)
    end = enc.STATE_DIM - enc.HERO_VOCAB_DIM
    hp_block = state[end - enc.HP_VOCAB_DIM:end]
    assert hp_block[hp_idx] == 1.0 and hp_block[-1] == 0.0
    assert _hero_block_state(state)[enc.HERO_VOCAB.index(_ETC)] == 1.0


def _assert_hero_and_skins(hero_id, skins):
    idx = enc.HERO_VOCAB.index(hero_id)
    base = enc.hero_onehot(hero_id)
    assert base[idx] == 1.0 and base[-1] == 0.0 and sum(base) == 1.0
    assert enc.hero_onehot("BG99_HERO_999")[-1] == 1.0
    for skin in skins:
        assert enc.canonical_hero_id(skin) == hero_id
        oh = enc.hero_onehot(skin)
        assert oh == base and oh[-1] == 0.0
        skin_state = enc.encode_state(_snap(hero=skin))
        assert np.array_equal(skin_state, enc.encode_state(_snap(hero=hero_id)))
        block = _hero_block_state(skin_state)
        assert block.sum() == 1.0 and block[idx] == 1.0 and block[-1] == 0.0
    snap = _snap(hero="TB_BaconShop_HERO_PH")
    offered = enc.make_option("discover", skins[0], source=("choice", 0), choice_kind="hero")
    assert np.array_equal(_hero_block_option(enc.encode_option(snap, offered)), base)


def test_azshara_and_vashj_skins_map_to_own_index_not_other():
    _assert_hero_and_skins(_AZSHARA, _AZSHARA_SKINS)
    _assert_hero_and_skins(_VASHJ, _VASHJ_SKINS)
    # The two new heroes do not share a slot with each other.
    assert enc.hero_onehot(_AZSHARA) != enc.hero_onehot(_VASHJ)


def test_azshara_and_vashj_hero_powers_are_in_vocab():
    """Ambition is the hidden-cost passive. Naga Conquest and Relics are not."""
    ambition, conquest, relics = "BG22_HERO_007p", "BG22_HERO_007p2", "BG23_HERO_304p"
    assert enc.HERO_POWER_VOCAB.index(ambition) == 77
    assert enc.HERO_POWER_VOCAB.index(conquest) == 78
    assert enc.HERO_POWER_VOCAB.index(relics) == 79
    for cid in (ambition, conquest, relics):
        oh = enc.hero_power_onehot(cid)
        assert sum(oh) == 1.0
        assert oh[enc.HERO_POWER_VOCAB.index(cid)] == 1.0
        assert oh[-1] == 0.0
    assert enc.is_passive_hero_power({"card_id": ambition, "cost": 0, "usable": True})
    assert not enc.is_passive_hero_power({"card_id": conquest, "cost": 1, "usable": True})
    assert not enc.is_passive_hero_power({"card_id": relics, "cost": 0, "usable": True})
    hidden = _snap(hero=_AZSHARA, gold=10,
                   hero_power={"name": "Azshara's Ambition", "card_id": ambition,
                               "cost": 0, "usable": True})
    assert "hero_power" not in [o["type"] for o in enc.legal_options(hidden)]
    offered = _snap(hero=_AZSHARA, gold=10,
                    hero_power={"name": "Naga Conquest", "card_id": conquest,
                                "cost": 1, "usable": True})
    assert [o["card_id"] for o in enc.legal_options(offered)
            if o["type"] == "hero_power"] == [conquest]
    end = enc.STATE_DIM - enc.HERO_VOCAB_DIM
    hp_block = enc.encode_state(offered)[end - enc.HP_VOCAB_DIM:end]
    assert hp_block[enc.HERO_POWER_VOCAB.index(conquest)] == 1.0
    assert hp_block[-1] == 0.0


def test_v5_checkpoint_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.bc_policy import KIND, CheckpointMismatch, OptionScorer, load_bc_policy
    old = OptionScorer(state_dim=_V5_STATE_DIM, option_dim=_V5_OPTION_DIM)
    path = tmp_path / "policy_net_v5.pt"
    torch.save({"meta": {"kind": KIND, "encoder_version": "bc-enc-v5",
                         "state_dim": _V5_STATE_DIM, "option_dim": _V5_OPTION_DIM,
                         "hidden": 128, "decode": "hp_group"},
                "state_dict": old.state_dict()}, path)
    with pytest.raises(CheckpointMismatch, match="encoder_version='bc-enc-v5'"):
        load_bc_policy(str(path))


def test_live_overlay_falls_back_on_v5(monkeypatch, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    from hsbg_coach import live
    from ml.bc_policy import KIND, OptionScorer
    monkeypatch.setenv("HSBG_NEXT_POLICY", "1")
    old = OptionScorer(state_dim=_V5_STATE_DIM, option_dim=_V5_OPTION_DIM)
    path = tmp_path / "policy_net_v5.pt"
    torch.save({"meta": {"kind": KIND, "encoder_version": "bc-enc-v5",
                         "state_dim": _V5_STATE_DIM, "option_dim": _V5_OPTION_DIM,
                         "hidden": 128},
                "state_dict": old.state_dict()}, path)
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(path))
    assert live._load_next_policy() is None
    err = capsys.readouterr().err
    assert "falling back to the eval-net advisor" in err
    assert "bc-enc-v5" in err
    assert "bc-enc-v6" in err
