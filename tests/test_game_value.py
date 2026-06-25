"""Whole-game value tests — stdlib (heuristic scorer + real pace curves)."""

from hsbg_coach.game_value import expected_placement, rank_actions
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.actions import BUY, LEVEL
from hsbg_coach.pace import load_pace

PACE = load_pace()
EMB = {"a": [1.0, 0.0], "b": [1.0, 0.0], "good": [1.0, 0.0], "bad": [0.0, 1.0]}
SC = HeuristicScorer(EMB)


def _snap(hp=30, stats=6, turn=6, tier=3, gold=7, shop=None):
    half = stats // 2
    return {"turn": turn, "tavern_tier": tier, "gold": gold, "hero_health": hp,
            "board": [{"name": "a", "attack": half, "health": stats - half},
                      {"name": "b", "attack": half, "health": stats - half}],
            "shop": shop if shop is not None else
                    [{"name": "good", "attack": 6, "health": 6},
                     {"name": "bad", "attack": 2, "health": 1}]}


def test_expected_placement_in_range():
    p = expected_placement(_snap(), SC, PACE)
    assert 1.0 <= p <= 8.0


def test_stronger_board_finishes_better():
    strong = expected_placement(_snap(stats=60), SC, PACE)
    weak = expected_placement(_snap(stats=10), SC, PACE)
    assert strong < weak


def test_low_hp_worsens_expected_finish():
    healthy = expected_placement(_snap(hp=30, stats=25), SC, PACE)
    dying = expected_placement(_snap(hp=4, stats=25), SC, PACE)
    assert dying >= healthy
    assert dying > healthy or healthy >= 7.9          # unless both clamp at the floor


def test_rank_actions_best_first_and_improves():
    recs, base = rank_actions(_snap(), kb=None, scorer=SC, pace=PACE)
    assert recs == sorted(recs, key=lambda r: r.placement)
    assert recs[0].action.kind == BUY                 # buying the strong minion
    assert recs[0].placement < base                   # it improves expected finish
    assert recs[0].gain > 0


def test_leveling_is_on_the_same_axis_when_affordable():
    # tier 1 -> up-cost 5, gold 7 affords it: leveling appears with a placement.
    recs, _ = rank_actions(_snap(tier=1, gold=7), kb=None, scorer=SC, pace=PACE)
    levels = [r for r in recs if r.action.kind == LEVEL]
    assert levels and 1.0 <= levels[0].placement <= 8.0
