"""bc-enc-v6 float16 bounds: huge stats stay finite, small, and deterministic.

Option columns 58-59 are the acted-on card's attack and health. State columns
60-63 are the board zone's sum attack, sum health, mean attack, mean health.
Columns 200-203 are those four numbers for the hand, and 130-133 for the shop.
A linear /10 or /50 leaves a buffed minion in the tens or hundreds of thousands,
and float16 rounds to inf from 65520. bound_scalar is the shared map.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from hsbg_coach import encode as enc  # noqa: E402

# attack/10 of this value is 200_000, which float16 cannot store.
HUGE_ATK = 2_000_000
HUGE_HP = 3_000_000
HAND_ATK = 80_000
HAND_HP = 90_000
SHOP_ATK = 400_000
SHOP_HP = 500_000


def _minion(name, i, atk, hp, cid=None):
    return {"entity_id": 100 + i, "card_id": cid or f"H{i}", "name": name,
            "attack": atk, "health": hp, "position": i + 1,
            "tags": {"CARDTYPE": "MINION", "ATK": str(atk), "HEALTH": str(hp)}}


def huge_snapshot():
    """Board, shop, and hand each carry stats whose old /10 feature overflows float16."""
    board = [_minion("a", 0, HUGE_ATK, HUGE_HP, "A"),
             _minion("b", 1, HUGE_ATK, HUGE_HP, "B")]
    shop = [_minion("s", 0, SHOP_ATK, SHOP_HP, "S")]
    hand = [_minion("h", 0, HAND_ATK, HAND_HP, "H")]
    return {"game_counter": 1, "turn": 8, "phase": "recruit", "tavern_tier": 4,
            "gold": 100_000, "hero_health": 80, "hero_armor": 50_000,
            "board": board, "shop": shop, "hand": hand, "shop_spells": [],
            "hand_spells": [], "hero_power": None, "level_cost": 7,
            "hero": "BG36_HERO_000"}


def _zone_stat(zone_index, scalar_index):
    """Column of one board_vector scalar. scalar 1 is sum_atk."""
    return zone_index * enc._ZONE_DIM + enc.EMB_DIM + enc._N_TRIBES + scalar_index


def test_v5_stat_columns_are_where_the_overflow_was_measured():
    """v6 only appended one-hots, so these early columns match the v5 cache."""
    assert len(enc.OPTION_TYPES) + enc.EMB_DIM == 58
    assert _zone_stat(0, 1) == 60          # board sum attack
    assert _zone_stat(0, 4) == 63          # board mean health
    assert _zone_stat(1, 1) == 130         # shop sum attack
    assert _zone_stat(2, 1) == 200         # hand sum attack
    assert _zone_stat(2, 4) == 203         # hand mean health
    # The old stored feature, not the raw stat, is what the cache cast.
    old = np.array([HUGE_ATK / 10.0, (2 * HUGE_ATK) / 50.0], dtype=np.float32)
    with np.errstate(over="ignore"):
        assert not np.isfinite(old.astype(np.float16)).all()
    # A float32 value in the reported 36k-58k band does cast cleanly. The infs
    # were the tail past ~65520, not a product computed in float16.
    reported = np.array([36_000, 58_000], dtype=np.float32)
    assert np.isfinite(reported.astype(np.float16)).all()


def test_bound_scalar_is_odd_finite_and_small():
    assert enc.bound_scalar(0) == 0.0
    assert enc.bound_scalar(-8) == pytest.approx(-enc.bound_scalar(8))
    assert abs(enc.bound_scalar(1e18)) <= 10
    assert abs(enc.bound_scalar(-1e18)) <= 10
    assert math.isfinite(enc.bound_scalar(10 ** 7))


def test_huge_stats_encode_to_finite_bounded_values():
    snap = huge_snapshot()
    state = enc.encode_state(snap)
    sell = enc.make_option("sell", "A", ("board", 0))
    option = enc.encode_option(snap, sell)
    assert state.shape == (enc.STATE_DIM,) and option.shape == (enc.OPTION_DIM,)
    assert np.isfinite(state).all() and np.isfinite(option).all()
    assert np.max(np.abs(state)) <= 10
    assert np.max(np.abs(option)) <= 10

    atk_col = len(enc.OPTION_TYPES) + enc.EMB_DIM
    assert option[atk_col] == pytest.approx(enc.bound_scalar(HUGE_ATK))
    assert option[atk_col + 1] == pytest.approx(enc.bound_scalar(HUGE_HP))

    assert state[_zone_stat(0, 1)] == pytest.approx(enc.bound_scalar(2 * HUGE_ATK))
    assert state[_zone_stat(0, 2)] == pytest.approx(enc.bound_scalar(2 * HUGE_HP))
    assert state[_zone_stat(0, 3)] == pytest.approx(enc.bound_scalar(HUGE_ATK))
    assert state[_zone_stat(0, 4)] == pytest.approx(enc.bound_scalar(HUGE_HP))
    assert state[_zone_stat(1, 1)] == pytest.approx(enc.bound_scalar(SHOP_ATK))
    assert state[_zone_stat(1, 2)] == pytest.approx(enc.bound_scalar(SHOP_HP))
    assert state[_zone_stat(2, 1)] == pytest.approx(enc.bound_scalar(HAND_ATK))
    assert state[_zone_stat(2, 2)] == pytest.approx(enc.bound_scalar(HAND_HP))

    base = 3 * enc._ZONE_DIM
    gold_i = base + enc.STATE_SCALARS.index("gold")
    hp_i = base + enc.STATE_SCALARS.index("health")
    armor_i = base + enc.STATE_SCALARS.index("armor")
    assert state[gold_i] == pytest.approx(enc.bound_scalar(100_000))
    assert state[hp_i] == pytest.approx(enc.bound_scalar(80))
    assert state[armor_i] == pytest.approx(enc.bound_scalar(50_000))
    assert enc.ENCODER_VERSION == "bc-enc-v6"


def test_encode_is_deterministic():
    snap = huge_snapshot()
    opt = enc.make_option("sell", "A", ("board", 0))
    assert np.array_equal(enc.encode_state(snap), enc.encode_state(snap))
    assert np.array_equal(enc.encode_option(snap, opt), enc.encode_option(snap, opt))
    # A second snapshot object with the same fields encodes the same vector.
    again = dict(snap)
    again["board"] = [dict(m) for m in snap["board"]]
    assert np.array_equal(enc.encode_state(snap), enc.encode_state(again))
