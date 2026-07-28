"""Hero/comp stats loader + HeroContext builder + recommender integration."""

from hsbg_coach.stats import (
    StatsDB, load_hero_stats, load_comp_stats, build_hero_context,
    SAMPLE_HEROES, SAMPLE_COMPS,
)
from hsbg_coach.economy import HeroContext
from hsbg_coach.recommend import recommend
from hsbg_coach.bg import ActionType


def db():
    return StatsDB.load(SAMPLE_HEROES, SAMPLE_COMPS)


def test_loads_sample_files():
    assert len(load_hero_stats(SAMPLE_HEROES)) >= 4
    assert len(load_comp_stats(SAMPLE_COMPS)) >= 5


def test_hero_lookup_by_name_and_cardid():
    d = db()
    assert d.hero("Galakrond").playstyle == "economy"
    assert d.hero("TB_BaconShop_HERO_15").name == "Old Murk-Eye"
    assert d.hero("Nonexistent") is None


def test_best_comps_ranked_by_placement():
    comps = db().best_comps(limit=3)
    positions = [c.average_position for c in comps]
    assert positions == sorted(positions)            # best (lowest) first


def test_best_comps_filtered_by_available_tribes():
    comps = db().best_comps(tribes=["Mech"], limit=5)
    assert all(c.tribe in (None, "Mech") for c in comps)


def test_best_comp_for_hero_prefers_hero_tribes():
    # Old Murk-Eye's best tribe is Murloc -> should land on the Murloc comp.
    comp = db().best_comp_for_hero("Old Murk-Eye")
    assert comp.tribe == "Murloc"


def test_build_hero_context_from_stats():
    ctx = build_hero_context("Old Murk-Eye", db())
    assert isinstance(ctx, HeroContext)
    assert ctx.target_tribe == "Murloc"
    assert "Old Murk-Eye" in ctx.recommended_minions or ctx.recommended_minions
    assert ctx.level_aggression < 0                  # tempo hero -> less greedy


def test_economy_hero_gets_positive_aggression():
    ctx = build_hero_context("Galakrond", db())      # playstyle: economy
    assert ctx.level_aggression > 0


def test_available_tribes_restrict_target():
    # If Murloc isn't in the lobby, Old Murk-Eye should target another tribe.
    ctx = build_hero_context("Old Murk-Eye", db(), available_tribes=["Beast", "Mech"])
    assert ctx.target_tribe in ("Beast", "Mech")


def test_recommendation_uses_stats_context():
    ctx = build_hero_context("Old Murk-Eye", db())
    snap = {
        "turn": 6, "tavern_tier": 2, "gold": 6, "hero_health": 30,
        "board": [{"name": "filler", "attack": 1, "health": 1}],
        "shop": [
            {"name": "Big Stats", "attack": 5, "health": 5, "tribe": "Beast"},
            {"name": "Murloc Warleader", "attack": 3, "health": 3, "tribe": "Murloc"},
        ],
    }
    recs = recommend(snap, hero_ctx=ctx)
    buy = next(r for r in recs if r.action == ActionType.BUY.value)
    assert "Murloc Warleader" in buy.rationale        # on-comp pick over raw stats
