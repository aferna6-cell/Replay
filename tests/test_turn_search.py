"""Turn-level beam search tests — stdlib only (heuristic scorer)."""

from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.turn_search import plan_turn_search, TurnPlan

EMB = {}  # heuristic scorer degrades to raw stats without embeddings


def snap(board=None, shop=None, gold=10, tier=3, hp=25, turn=8, hand=None):
    return {"board": board or [], "shop": shop or [], "hand": hand or [],
            "gold": gold, "tavern_tier": tier, "hero_health": hp,
            "turn": turn, "notes": []}


def m(name, atk, hp):
    return {"name": name, "attack": atk, "health": hp}


def _plan(s, **kw):
    return plan_turn_search(s, kb=None, scorer=HeuristicScorer(EMB),
                            pace={}, **kw)


def test_returns_plan_ending_with_end_turn():
    p = _plan(snap(board=[m("A", 5, 5)], shop=[m("B", 4, 4)], gold=3))
    assert isinstance(p, TurnPlan)
    assert p.steps[-1] == "End turn"
    assert p.searched >= 1


def test_buys_multiple_minions_and_declines_filler():
    p = _plan(snap(shop=[m("Big", 8, 8), m("Mid", 5, 5), m("Small", 1, 1)],
                   gold=10))
    buys = [s for s in p.steps if s.startswith("Buy")]
    assert len(buys) >= 2                      # both real minions bought
    assert not any("Small" in s for s in buys)  # 1/1 filler declined
    assert any(s.startswith("Roll") for s in p.steps)  # leftover gold -> roll
    assert p.expected <= p.base + 1e-9         # never worse than doing nothing


def test_respects_gold_constraint():
    p = _plan(snap(shop=[m("Big", 9, 9), m("Also", 9, 9)], gold=4))
    buys = [s for s in p.steps if s.startswith("Buy")]
    assert len(buys) <= 1                      # 4 gold buys exactly one


def test_finds_sell_then_buy_on_full_board():
    board = [m(f"w{i}", 1, 1) for i in range(7)]
    p = _plan(snap(board=board, shop=[m("Giant", 10, 10)], gold=3))
    joined = " | ".join(p.steps)
    assert "Sell" in joined and "Buy Giant" in joined
    sell_ix = next(i for i, s in enumerate(p.steps) if s.startswith("Sell"))
    buy_ix = next(i for i, s in enumerate(p.steps) if s.startswith("Buy Giant"))
    assert sell_ix < buy_ix                    # the ordering greedy can't produce


def test_prefers_triple_completion():
    board = [m("Pair", 2, 2), m("Pair", 2, 2)]
    p = _plan(snap(board=board, shop=[m("Pair", 2, 2), m("Fat", 6, 6)], gold=3))
    first_buy = next(s for s in p.steps if s.startswith("Buy"))
    assert "Pair" in first_buy and "TRIPLE" in first_buy


def test_no_gold_no_buys():
    p = _plan(snap(board=[m("A", 3, 3)], shop=[m("B", 9, 9)], gold=0))
    assert not any(s.startswith("Buy") for s in p.steps)


def test_search_beats_greedy_on_sell_to_afford():
    """2 gold + a sellable body: only sell->buy reaches the big shop minion.
    Greedy plan_turn can't start with a sell that pays for the buy."""
    s = snap(board=[m("tiny", 1, 1), m("ok", 4, 4)],
             shop=[m("Huge", 12, 12)], gold=2)
    p = _plan(s)
    joined = " | ".join(p.steps)
    assert "Sell tiny" in joined and "Buy Huge" in joined
