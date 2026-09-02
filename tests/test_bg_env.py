"""Phase 0 recruit-phase simulator tests — rules, accounting, and full games."""

import random

from hsbg_coach.bg_env import (
    BGEnv, EnvMinion, build_pool, greedy_policy, random_policy, gold_at,
    A_BUY0, A_PLAY0, A_SELL0, A_ROLL, A_LEVEL, A_FREEZE, A_END, N_ACTIONS,
    BUY_COST, SELL_VALUE, MAX_BOARD, POOL_COPIES, SHOP_SLOTS,
)


def make_env(seed=0):
    env = BGEnv(seed=seed)
    env.reset(seed=seed)
    return env


# --- pool / shop ---------------------------------------------------------------
def test_pool_is_curated_and_tiered():
    catalogue = build_pool()
    assert len(catalogue) >= 30
    tiers = {m.tier for m in catalogue}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_reset_deals_tier1_shop_and_gold():
    env = make_env()
    obs = env.observe(0)
    assert obs["gold"] == gold_at(1) == 3
    assert obs["tavern_tier"] == 1
    assert 1 <= len(obs["shop"]) <= 3
    assert all((m["tags"] or {}).get("TECH_LEVEL") == "1" for m in obs["shop"])


def test_buy_moves_shop_to_hand_and_spends_gold():
    env = make_env()
    p = env.players[0]
    gold0, shop0 = p.gold, len(p.shop)
    assert env.legal_mask(0)[A_BUY0]
    env.step(A_BUY0)
    assert p.gold == gold0 - BUY_COST
    assert len(p.hand) == 1 and len(p.shop) == shop0 - 1


def test_play_then_sell_returns_copy_to_pool():
    env = make_env()
    env.step(A_BUY0)
    name = env.players[0].hand[0].name
    before = env._pool[name]
    env.step(A_PLAY0)
    assert env.players[0].board[0].name == name
    env.step(A_SELL0)
    assert env.players[0].board == []
    assert env._pool[name] == before + 1
    assert env.players[0].gold == gold_at(1) - BUY_COST + SELL_VALUE


def test_level_uses_discounted_cost():
    env = make_env()
    p = env.players[0]
    p.gold = 10
    p.turns_since_level = 2
    assert p.level_cost() == 5 - 2
    env.step(A_LEVEL)
    assert p.tier == 2 and p.gold == 10 - 3


def test_freeze_keeps_shop_across_turn():
    env = make_env(seed=3)
    p = env.players[0]
    kept = [m.name for m in p.shop]
    env.step(A_FREEZE)
    env.step(A_END)
    if env.done:                     # rare: lobby collapsed turn 1
        return
    # Full frozen shop is preserved; Phase 2N top-up only adds missing slots.
    assert [m.name for m in p.shop][:len(kept)] == kept
    assert len(p.shop) == SHOP_SLOTS[p.tier]


def test_freeze_topup_incomplete_shop():
    """Phase 2N-B: incomplete frozen shop refills to SHOP_SLOTS[tier]."""
    from hsbg_coach.bg_env import PHASE_2N_FREEZE_TOPUP, SHOP_SLOTS
    assert PHASE_2N_FREEZE_TOPUP
    env = make_env(seed=5)
    p = env.players[0]
    assert len(p.shop) >= 1
    kept = [m.name for m in p.shop[:1]]
    p.shop = p.shop[:1]              # simulate buy leaving incomplete freeze
    p.frozen = True
    env._deal_reason = "start_turn"
    env._deal_shop(p)
    assert not p.frozen
    assert len(p.shop) == SHOP_SLOTS[p.tier]
    assert p.shop[0].name == kept[0]


def test_death_returns_holdings_to_pool():
    """Phase 2N-B: eliminated players return board/hand/shop to the shared pool."""
    from hsbg_coach.bg_env import PHASE_2N_DEATH_RETURN
    assert PHASE_2N_DEATH_RETURN
    env = make_env(seed=9)
    # Pair seat 0 vs seat 1 with seat 1 at 0 HP after forced combat damage path.
    a, b = env.players[0], env.players[1]
    name = next(iter(env._catalogue))
    base = env._catalogue[name]
    held = EnvMinion(base.card_id, base.name, base.tier, base.attack,
                     base.health, list(base.tribes), list(base.keywords))
    b.board = [held]
    b.hand = [
        EnvMinion(base.card_id, base.name, base.tier, base.attack,
                  base.health, list(base.tribes), list(base.keywords))
    ]
    b.shop = [
        EnvMinion(base.card_id, base.name, base.tier, base.attack,
                  base.health, list(base.tribes), list(base.keywords))
    ]
    # Take the three copies out of the pool accounting so return is visible.
    env._pool[name] = max(0, env._pool.get(name, 0) - 3)
    before = env._pool[name]
    b.hp = 0
    # Invoke the death handling block via _run_combat with no living opponents
    # by zeroing all other players' HP too carefully — call death path directly
    # through the same code as combat by simulating one dead among alive.
    alive_before = [p for p in env.players if p.alive]
    assert b in alive_before
    place = len(alive_before)
    b.alive = False
    b.placement = place
    b.last_board = [EnvMinion(m.card_id, m.name, m.tier, m.attack,
                              m.health, list(m.tribes), list(m.keywords),
                              m.golden) for m in b.board]
    env._return_player_holdings_to_pool(b)
    assert env._pool[name] == before + 3
    assert b.board == [] and b.hand == [] and b.shop == []
    assert len(b.last_board) == 1


def test_pool_copies_tier6_is_seven():
    """Phase 2N-C: T6 copy count matches current Battlegrounds (7)."""
    assert POOL_COPIES[6] == 7
    env = make_env(seed=1)
    t6 = [n for n, m in env._catalogue.items() if m.tier == 6]
    assert t6
    assert env._pool[t6[0]] == 7
    env = make_env()
    p = env.players[0]
    base = env._catalogue[sorted(env._catalogue)[0]]
    copies = [EnvMinion(base.card_id, base.name, base.tier, base.attack,
                        base.health, list(base.tribes), list(base.keywords))
              for _ in range(3)]
    p.board = copies[:2]
    p.hand = [copies[2]]
    env._check_triple(p, base.name)
    assert p.board == []
    goldens = [m for m in p.hand if m.golden]
    assert len(goldens) == 1
    assert goldens[0].attack == base.attack * 2
    assert len(p.hand) == 2          # golden + the discover reward


def test_triple_merges_to_golden_plus_discover():
    env = make_env()
    p = env.players[0]
    base = env._catalogue[sorted(env._catalogue)[0]]
    copies = [EnvMinion(base.card_id, base.name, base.tier, base.attack,
                        base.health, list(base.tribes), list(base.keywords))
              for _ in range(3)]
    p.board = copies[:2]
    p.hand = [copies[2]]
    env._check_triple(p, base.name)
    assert p.board == []
    goldens = [m for m in p.hand if m.golden]
    assert len(goldens) == 1
    assert goldens[0].attack == base.attack * 2
    assert len(p.hand) == 2          # golden + the discover reward


def test_legal_mask_shape_and_end_always_legal():
    env = make_env()
    mask = env.legal_mask(0)
    assert len(mask) == N_ACTIONS and mask[A_END]
    env.players[0].gold = 0
    mask = env.legal_mask(0)
    assert not mask[A_BUY0] and not mask[A_ROLL]


def test_roll_replaces_shop():
    env = make_env(seed=7)
    p = env.players[0]
    p.gold = 10
    total_before = sum(env._pool.values()) + len(p.shop)
    env.step(A_ROLL)
    assert p.gold == 9
    assert sum(env._pool.values()) + len(p.shop) == total_before  # conservation


# --- full games ------------------------------------------------------------------
def test_full_scripted_game_assigns_all_placements():
    env = BGEnv(seed=11)
    records = env.play_scripted([greedy_policy] * 8)
    placements = sorted(p.placement for p in env.players)
    assert placements == list(range(1, 9))
    assert all(r["placement"] in range(1, 9) for r in records)
    assert {r["turn"] for r in records}  # multiple turns recorded


def test_agent_episode_terminates_with_reward():
    env = BGEnv(seed=5)
    env.reset(seed=5)
    rng = random.Random(0)
    total, done = 0.0, False
    for _ in range(600):
        mask = env.legal_mask(0)
        obs, reward, done, info = env.step(
            random_policy(env.observe(0), mask, rng))
        if done:
            total = reward
            assert 1 <= info["placement"] <= 8
            break
    assert done
    assert -1.0 <= total <= 1.0


def test_greedy_beats_random_on_average():
    places_g, places_r = [], []
    for i in range(12):
        env = BGEnv(seed=100 + i)
        env.play_scripted([greedy_policy] * 4 + [random_policy] * 4)
        for p in env.players:
            (places_g if p.idx < 4 else places_r).append(p.placement)
    assert sum(places_g) / len(places_g) < sum(places_r) / len(places_r)


def test_board_cap_respected_all_game():
    env = BGEnv(seed=42)
    for rec in env.play_scripted([greedy_policy] * 8):
        assert len(rec["state"]["board"]) <= MAX_BOARD
