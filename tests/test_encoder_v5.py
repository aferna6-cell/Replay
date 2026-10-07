"""bc-enc-v5: every 36.6.3 hero power has its own slot, skins map to the base
hero, and a hero pick encodes which hero it is. A bc-enc-v4 checkpoint is
refused instead of loaded into the wider net."""

import pytest

np = pytest.importorskip("numpy")

from hsbg_coach import encode as enc  # noqa: E402

# v2.1 prefix. Indices 0-71 are frozen; bc-enc-v5 only appends.
_V21_HERO_POWERS = (
    'BG36_HERO_000p', 'BG36_HERO_002p', 'BG21_HERO_000p', 'TB_BaconShop_HP_020',
    'TB_BaconShop_HP_010', 'TB_BaconShop_HP_024', 'BG25_HERO_103p', 'TB_BaconShop_HP_046',
    'TB_BaconShop_HP_075', 'BG25_HERO_105p', 'BG26_HERO_102p', 'BG20_HERO_301p',
    'BG21_HERO_010p', 'BG34_HERO_001p', 'BG28_HERO_801p', 'BG31_HERO_003p',
    'BG31_HERO_005p', 'BG20_HERO_101p', 'BG20_HERO_201p', 'TB_BaconShop_HP_011',
    'TB_BaconShop_HP_103', 'BG26_HERO_102p2', 'BG26_HERO_104p', 'TB_BaconShop_HP_053',
    'BG32_HERO_001p', 'TB_BaconShop_HP_047', 'BG21_HERO_020p', 'TB_BaconShop_HP_077',
    'TB_BaconShop_HP_081', 'TB_BaconShop_HP_072', 'TB_BaconShop_HP_022', 'BG22_HERO_000p_Alt',
    'BG23_HERO_303p2', 'BG31_HERO_006p', 'TB_BaconShop_HP_052', 'TB_BaconShop_HP_042',
    'TB_BaconShop_HP_028', 'BG22_HERO_001p', 'TB_BaconShop_HP_074', 'TB_BaconShop_HP_084',
    'BG20_HERO_280p5', 'BG23_HERO_305p', 'TB_BaconShop_HP_702t', 'TB_BaconShop_HP_068',
    'TB_BaconShop_HP_040', 'TB_BaconShop_HP_041k', 'TB_BaconShop_HP_064', 'TB_BaconShop_HP_102',
    'TB_BaconShop_HP_076', 'TB_BaconShop_HP_041l', 'BG24_HERO_100p', 'BG20_HERO_282p',
    'BG20_HERO_103p', 'BG28_HERO_400p', 'TB_BaconShop_HP_038', 'TB_BaconShop_HP_041d',
    'BG31_HERO_811p2', 'TB_BaconShop_HP_041b', 'TB_BaconShop_HP_049', 'BG20_HERO_283p',
    'TB_BaconShop_HP_041i', 'TB_BaconShop_HP_041h', 'TB_BaconShop_HP_041a', 'TB_BaconShop_HP_041g',
    'TB_BaconShop_HP_036', 'BG31_HERO_811p', 'TB_BaconShop_HP_041c', 'BG20_HERO_201p2',
    'TB_BaconShop_HP_015', 'TB_BaconShop_HP_041f', 'BG25_HERO_100p', 'TB_BaconShop_HP_057',
)
_NEW_HERO_POWERS = (
    'BG23_HERO_306p', 'TB_BaconShop_HP_104', 'TB_BaconShop_HP_001',
    'BG34_HERO_004p', 'BG34_HERO_000p',
)
# Sylvanas, Morchie, Drest'agath, Kith'ix.
_HERO_PICK = ('BG23_HERO_306', 'BG34_HERO_004', 'BG36_HERO_000', 'BG36_HERO_002')
_SRC_SLOT = len(enc.OPTION_TYPES) + enc._CARD_DIM + len(enc.ZONES)


def _snap(**kw):
    d = {"game_counter": 1, "turn": 6, "phase": "recruit", "tavern_tier": 3, "gold": 10,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "hero": None, "level_cost": 7}
    d.update(kw)
    return d


def _hp_block(state):
    end = enc.STATE_DIM - enc.HERO_VOCAB_DIM
    return state[end - enc.HP_VOCAB_DIM:end]


def _hero_block_state(state):
    return state[-enc.HERO_VOCAB_DIM:]


def _hero_block_option(option_vec):
    return option_vec[-enc.HERO_VOCAB_DIM:]


def test_version_dims_and_frozen_prefix():
    assert enc.ENCODER_VERSION == "bc-enc-v5"
    assert enc.HERO_POWER_VOCAB[:72] == _V21_HERO_POWERS
    assert enc.HERO_POWER_VOCAB[72:] == _NEW_HERO_POWERS
    assert enc.HP_VOCAB_DIM == 78 and enc.HERO_VOCAB_DIM == 118
    assert len(enc.HERO_VOCAB) == 117 and len(set(enc.HERO_VOCAB)) == 117
    assert enc.STATE_DIM == 445 and enc.OPTION_DIM == 330
    assert enc._V5_OPTION == enc.HERO_VOCAB_DIM
    snap = _snap()
    assert enc.encode_state(snap).shape == (enc.STATE_DIM,)
    assert enc.encode_option(snap, enc.make_option("end_turn")).shape == (enc.OPTION_DIM,)
    for hid in ("BG23_HERO_306", "BG34_HERO_004", "BG34_HERO_000",
                "BG36_HERO_000", "BG36_HERO_002"):
        assert hid in enc.HERO_VOCAB
    assert set(enc.HERO_SKIN_ALIASES.values()) <= set(enc.HERO_VOCAB)


def test_new_hero_powers_own_slots_not_other():
    """Each 36.6.3 id is its own index. The trailing slot stays 'other'."""
    assert enc.hero_power_onehot("BG36_HERO_000p")[0] == 1.0
    assert enc.hero_power_onehot("BG36_HERO_002p")[1] == 1.0
    for i, cid in enumerate(_NEW_HERO_POWERS):
        oh = enc.hero_power_onehot(cid)
        assert sum(oh) == 1.0
        assert oh[72 + i] == 1.0
        assert oh[-1] == 0.0
        assert oh[enc.HERO_POWER_VOCAB.index(cid)] == 1.0
    other = enc.hero_power_onehot("NOT_A_REAL_HP")
    assert other[-1] == 1.0 and sum(other) == 1.0
    assert sum(enc.hero_power_onehot(None)) == 0.0


def test_hp_alias_shares_an_existing_slot(monkeypatch):
    monkeypatch.setitem(enc.HP_ALIASES, "BG23_HERO_306p2", "BG23_HERO_306p")
    assert enc.hero_power_onehot("BG23_HERO_306p2") == enc.hero_power_onehot("BG23_HERO_306p")
    assert enc.hero_power_onehot("BG23_HERO_306p2")[-1] == 0.0


def test_passive_bg34_powers_are_never_offered():
    for cid in ("BG34_HERO_004p", "BG34_HERO_000p"):
        hp = {"name": "Hero Power", "card_id": cid, "cost": 0, "usable": True}
        assert enc.is_passive_hero_power(hp) is True
        snap = _snap(hero_power=hp, gold=10)
        assert "hero_power" not in [o["type"] for o in enc.legal_options(snap)]
        assert enc.encode_state(snap)[enc.STATE_SCALARS.index("hp_passive")
                                      + 3 * enc._ZONE_DIM] == 1.0
    active = {"name": "Reclaimed Souls", "card_id": "BG23_HERO_306p",
              "cost": 0, "usable": True}
    assert enc.is_passive_hero_power(active) is False
    offered = enc.legal_options(_snap(hero_power=active, gold=10))
    assert [o["card_id"] for o in offered if o["type"] == "hero_power"] == ["BG23_HERO_306p"]


def test_hero_power_option_without_a_state_power():
    """The option names the power; a snapshot with no hero power stays zeros."""
    snap = _snap(hero_power=None, hero=None)
    opt = enc.make_option("hero_power", "BG23_HERO_306p")
    vec = enc.encode_option(snap, opt)
    start = enc.OPTION_DIM - enc._V4_OPTION - enc._V5_OPTION
    hp = vec[start:start + enc.HP_VOCAB_DIM]
    assert hp.sum() == 1.0
    assert hp[enc.HERO_POWER_VOCAB.index("BG23_HERO_306p")] == 1.0
    assert hp[-1] == 0.0
    assert _hp_block(enc.encode_state(snap)).sum() == 0.0
    # a non-power option does not borrow that one-hot
    assert enc.encode_option(snap, enc.make_option("end_turn"))[start:start + enc.HP_VOCAB_DIM].sum() == 0.0


def test_skin_maps_to_base_hero():
    assert enc.canonical_hero_id("TB_BaconShop_HERO_44_SKIN_D") == "BG23_HERO_306"
    assert enc.canonical_hero_id("BG23_HERO_305_SKIN_B") == "BG23_HERO_305"
    assert enc.canonical_hero_id("TB_BaconShop_HERO_PH") is None
    assert enc.canonical_hero_id(None) is None
    for src, dst in enc.HERO_SKIN_ALIASES.items():
        assert enc.canonical_hero_id(src) == dst
        assert enc.canonical_hero_id(src + "_SKIN_A") == dst
        assert enc.hero_onehot(src + "_SKIN_D") == enc.hero_onehot(dst)
        assert enc.hero_onehot(dst)[enc.HERO_VOCAB.index(dst)] == 1.0
        assert enc.hero_onehot(dst)[-1] == 0.0
    skin = _snap(hero="TB_BaconShop_HERO_44_SKIN_D")
    base = _snap(hero="BG23_HERO_306")
    assert np.array_equal(enc.encode_state(skin), enc.encode_state(base))
    block = _hero_block_state(enc.encode_state(skin))
    assert block.sum() == 1.0 and block[enc.HERO_VOCAB.index("BG23_HERO_306")] == 1.0


def test_placeholder_hero_is_zeros_and_unknown_is_other():
    placeholder = enc.encode_state(_snap(hero="TB_BaconShop_HERO_PH"))
    missing = enc.encode_state(_snap())
    unknown = enc.encode_state(_snap(hero="BG99_HERO_999"))
    assert _hero_block_state(placeholder).sum() == 0.0
    assert np.array_equal(_hero_block_state(placeholder), _hero_block_state(missing))
    assert unknown[-1] == 1.0 and _hero_block_state(unknown).sum() == 1.0
    assert enc.hero_onehot("NOT_A_HERO")[-1] == 1.0
    # in-game hero and hero power occupy different slices
    both = enc.encode_state(_snap(
        hero="BG34_HERO_000",
        hero_power={"card_id": "BG34_HERO_000p", "cost": 0, "usable": True}))
    assert _hp_block(both)[enc.HERO_POWER_VOCAB.index("BG34_HERO_000p")] == 1.0
    assert _hero_block_state(both)[enc.HERO_VOCAB.index("BG34_HERO_000")] == 1.0


def test_hero_pick_options_differ_by_hero():
    """Four offered heroes differ outside the source-slot dim, each on its own index."""
    snap = _snap(hero="TB_BaconShop_HERO_PH")
    assert _hero_block_state(enc.encode_state(snap)).sum() == 0.0
    opts = [enc.make_option("discover", hid, source=("choice", i), choice_kind="hero")
            for i, hid in enumerate(_HERO_PICK)]
    vecs = [enc.encode_option(snap, o) for o in opts]
    assert len({float(v[_SRC_SLOT]) for v in vecs}) == 4
    stripped = []
    for v in vecs:
        w = np.array(v, copy=True)
        w[_SRC_SLOT] = 0.0
        stripped.append(w.tobytes())
    assert len(set(stripped)) == 4
    for hid, v in zip(_HERO_PICK, vecs):
        block = _hero_block_option(v)
        assert block.sum() == 1.0
        assert block[enc.HERO_VOCAB.index(hid)] == 1.0
        assert block[-1] == 0.0
    # a skin offered at hero select lands on the base hero's slot
    skin = enc.make_option("discover", "TB_BaconShop_HERO_44_SKIN_D",
                           source=("choice", 0), choice_kind="hero")
    skin_block = _hero_block_option(enc.encode_option(snap, skin))
    assert np.array_equal(skin_block, _hero_block_option(vecs[0]))
    # discovers and ordinary actions do not carry a hero one-hot
    for opt in (enc.make_option("discover", "BG23_HERO_306", choice_kind="discover"),
                enc.make_option("buy", "BG23_HERO_306", ("shop", 0)),
                enc.make_option("reroll"),
                enc.make_option("end_turn")):
        assert _hero_block_option(enc.encode_option(snap, opt)).sum() == 0.0


def test_v4_checkpoint_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.bc_policy import KIND, CheckpointMismatch, OptionScorer, load_bc_policy
    old = OptionScorer(state_dim=322, option_dim=207)
    path = tmp_path / "policy_net_v4.pt"
    torch.save({"meta": {"kind": KIND, "encoder_version": "bc-enc-v4",
                         "state_dim": 322, "option_dim": 207, "hidden": 128,
                         "decode": "hp_group"},
                "state_dict": old.state_dict()}, path)
    with pytest.raises(CheckpointMismatch, match="encoder_version='bc-enc-v4'"):
        load_bc_policy(str(path))


def test_live_overlay_falls_back_without_checkpoint_or_on_v4(monkeypatch, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    from hsbg_coach import live
    from ml.bc_policy import KIND, OptionScorer
    monkeypatch.setenv("HSBG_NEXT_POLICY", "1")
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(tmp_path / "missing.pt"))
    assert live._load_next_policy() is None
    err = capsys.readouterr().err
    assert "falling back to the eval-net advisor" in err
    assert "checkpoint not found" in err

    old = OptionScorer(state_dim=322, option_dim=207)
    path = tmp_path / "policy_net_v4.pt"
    torch.save({"meta": {"kind": KIND, "encoder_version": "bc-enc-v4",
                         "state_dim": 322, "option_dim": 207, "hidden": 128},
                "state_dict": old.state_dict()}, path)
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(path))
    assert live._load_next_policy() is None
    err = capsys.readouterr().err
    assert "falling back to the eval-net advisor" in err
    assert "bc-enc-v4" in err
    assert "bc-enc-v5" in err
