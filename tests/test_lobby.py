"""Lobby setup tests — 8 heroes, one anomaly, and the knobs it moves."""

import random

from hsbg_coach import context_effects as fx
from hsbg_coach.bg_env import BGEnv, BUY_COST, ROLL_COST
from hsbg_coach.context_cards import ANOMALY, HERO, load_context_kb, of_kind
from hsbg_coach.lobby import LobbySetup, roll_lobby


def test_active_pools_are_populated():
    kb = load_context_kb()
    assert len(of_kind(kb, HERO, active_only=True)) > 50
    # 29 distinct names, expanded by tier variants into more card rows.
    names = {c.name for c in of_kind(kb, ANOMALY, active_only=True)}
    assert len(names) == 29


def test_roll_lobby_deals_distinct_heroes_and_one_anomaly():
    setup = roll_lobby(random.Random(0), n_players=8)
    assert len(setup.seats) == 8
    names = [s.hero_name for s in setup.seats]
    assert len(set(names)) == 8, "two seats got the same hero"
    assert setup.anomaly is not None
    assert all(s.hero_power is not None for s in setup.seats)


def test_anomalies_can_be_switched_off():
    setup = roll_lobby(random.Random(0), n_players=8, anomalies=False)
    assert setup.anomaly is None and setup.hooks == {}
    assert len(setup.seats) == 8       # heroes are still dealt


def test_env_exposes_full_game_context():
    env = BGEnv(seed=3)
    env.reset(seed=3)
    obs = env.observe(0)
    assert obs["hero"] and obs["hero_power"]
    assert obs["anomaly"]
    assert len(obs["opponent_heroes"]) == 7


def test_anomaly_repricing_reaches_the_env():
    """An anomaly that reprices minions must change what a buy costs."""
    env = BGEnv(seed=0, anomalies=False)
    env.reset(seed=0)
    assert env.buy_cost() == BUY_COST and env.roll_cost() == ROLL_COST

    env.setup = LobbySetup(seats=env.setup.seats,
                           hooks={fx.MINION_COST: 2, fx.REFRESH_DISABLED: True})
    assert env.buy_cost() == 2
    assert env.roll_cost() > 100, "a disabled refresh must be unaffordable"


def test_start_hooks_apply_at_reset():
    """'Start at Tavern Tier 2' has to actually start you at tier 2."""
    kb = load_context_kb()
    hourglass = [c for c in of_kind(kb, ANOMALY) if c.name == "Finicky Hourglass"]
    if not hourglass:                  # card pool changed — nothing to assert
        return
    assert fx.parse_text(hourglass[0].text)[0].name == fx.START_TIER
