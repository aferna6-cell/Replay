"""Move-recommender tests. All stdlib (forces the heuristic scorer) so they run
without torch and are deterministic."""

from hsbg_coach.actions import (
    legal_actions, BUY, SELL, ROLL, LEVEL, REPOSITION, FREEZE, END, tavern_up_cost,
)
from hsbg_coach.board_value import HeuristicScorer, get_scorer
from hsbg_coach.advisor import advise_actions


# --- action enumeration -----------------------------------------------------
def _snap(gold, tier=1, board=2, shop=2):
    return {
        "gold": gold, "tavern_tier": tier, "hero_health": 30, "turn": 5,
        "board": [{"name": f"b{i}", "attack": 3, "health": 3} for i in range(board)],
        "shop": [{"name": f"s{i}", "attack": 3, "health": 3} for i in range(shop)],
    }


def _kinds(actions):
    out = {}
    for a in actions:
        out[a.kind] = out.get(a.kind, 0) + 1
    return out


def test_legal_actions_full_menu_when_rich():
    k = _kinds(legal_actions(_snap(gold=7, tier=2)))   # tier2 up-cost 7, affordable
    assert k[BUY] == 2 and k[SELL] == 2
    assert k.get(ROLL) == 1 and k.get(LEVEL) == 1
    assert k.get(REPOSITION) == 1 and k.get(FREEZE) == 1 and k.get(END) == 1


def test_legal_actions_gold_gates_buy_roll_level():
    k = _kinds(legal_actions(_snap(gold=0)))
    assert BUY not in k and ROLL not in k and LEVEL not in k
    assert k[SELL] == 2 and k[END] == 1            # sells + end still legal
    k2 = _kinds(legal_actions(_snap(gold=2, tier=2)))
    assert BUY not in k2 and LEVEL not in k2 and k2.get(ROLL) == 1  # roll costs 1


def test_tavern_up_cost_table():
    assert tavern_up_cost(1) == 5 and tavern_up_cost(5) == 10 and tavern_up_cost(6) is None


# --- scorer -----------------------------------------------------------------
EMB = {"b1": [1.0, 0.0], "b2": [1.0, 0.0], "b3": [1.0, 0.0],
       "good": [1.0, 0.0], "bad": [0.0, 1.0]}


def test_heuristic_scorer_monotonic():
    sc = HeuristicScorer(EMB)
    strong = [{"name": "b1", "attack": 5, "health": 5},
              {"name": "b2", "attack": 5, "health": 5}]
    weak = [{"name": "b1", "attack": 1, "health": 1}]
    assert sc.equity(strong) > sc.equity(weak)


def test_get_scorer_returns_usable_scorer():
    sc = get_scorer()
    assert hasattr(sc, "equity") and sc.name in {"heuristic", "eval_net"}


# --- advisor ----------------------------------------------------------------
def _board(n, atk=3, hp=3):
    return [{"name": f"b{i+1}", "attack": atk, "health": hp} for i in range(n)]


def test_best_move_is_the_synergistic_buy():
    snap = {"gold": 7, "tavern_tier": 1, "hero_health": 30, "turn": 4,
            "board": _board(3),
            "shop": [{"name": "good", "attack": 5, "health": 5},
                     {"name": "bad", "attack": 2, "health": 1}]}
    plan = advise_actions(snap, kb=None, scorer=HeuristicScorer(EMB))
    assert plan.best.action.kind == BUY and plan.best.action.target == "good"
    # the off-tribe low-stat buy must rank below the on-tribe one
    buys = {a.action.target: a for a in plan.of_kind(BUY)}
    assert buys["good"].priority > buys["bad"].priority


def test_every_action_kind_is_ranked():
    snap = {"gold": 7, "tavern_tier": 2, "hero_health": 30, "turn": 6,
            "board": _board(3),
            "shop": [{"name": "good", "attack": 5, "health": 5}]}
    plan = advise_actions(snap, kb=None, scorer=HeuristicScorer(EMB))
    kinds = {a.action.kind for a in plan.ranked}
    assert {BUY, SELL, ROLL, LEVEL, REPOSITION, FREEZE, END} <= kinds


def test_buy_on_full_board_models_sell_for_room():
    snap = {"gold": 7, "tavern_tier": 1, "hero_health": 30, "turn": 8,
            "board": _board(7),
            "shop": [{"name": "good", "attack": 9, "health": 9}]}
    plan = advise_actions(snap, kb=None, scorer=HeuristicScorer(EMB))
    buy = plan.of_kind(BUY)[0]
    assert "sell" in buy.reason.lower()           # had to free a slot


def test_summary_renders():
    snap = {"gold": 5, "tavern_tier": 1, "hero_health": 30, "turn": 3,
            "board": _board(2), "shop": [{"name": "good", "attack": 4, "health": 4}]}
    text = advise_actions(snap, kb=None, scorer=HeuristicScorer(EMB)).summary()
    assert "Best move:" in text and "All actions, ranked:" in text
