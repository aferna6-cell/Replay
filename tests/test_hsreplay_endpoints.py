"""Endpoint-specific HSReplay handlers, modeled on a real 2026-08-27 capture."""

import json

import pytest

from hsbg_coach import hsreplay_import


@pytest.fixture
def names(monkeypatch):
    monkeypatch.setattr(hsreplay_import, "_NAME_INDEX", {
        "100001": "Sellemental", "100002": "Bream Counter",
        "200001": "Ship in a Bottle", "300001": "Reno Jackson",
    })


def _wrap(url, body):
    return {"url": url, "method": "GET", "post_data": None, "body": body}


def _real_capture_wrappers():
    q = "?BattlegroundsMMRPercentile=TOP_20_PERCENT&BattlegroundsTimeRange=LAST_7_DAYS"
    return [
        _wrap("https://hsreplay.net/api/v1/battlegrounds/meta_periods/",
              [{"period_start": "x", "name": "Season 10"}]),          # skipped
        _wrap("https://hsreplay.net/api/v1/battlegrounds/compositions/?hl=en",
              [{"id": 92, "name": "Elementals"}, {"id": 55, "name": "Nagas"}]),
        _wrap(f"https://hsreplay.net/analytics/query/battlegrounds_minion_list/{q}",
              {"series": {"data": {"ALL": [
                  {"minion_dbf_id": 100001, "minion_tier": 2,
                   "composition_ids": [92],
                   "normal_aggregates": {"avg_final_placement": 3.31,
                                         "num_games": 5000},
                   "premium_aggregates": {"avg_final_placement": 2.9}},
                  {"minion_dbf_id": 100002, "minion_tier": 3,
                   "composition_ids": [55],
                   "normal_aggregates": {"avg_final_placement": 2.95}},
              ]}}}),
        _wrap(f"https://hsreplay.net/analytics/query/battlegrounds_comp_stats/{q}",
              {"series": {"data": {"ALL": [
                  {"friendly_composition": 92, "num_games": 9000,
                   "avg_final_placement": 3.2,
                   "final_placement_distribution": [0.1] * 8,
                   "popularity": 0.21},
              ]}}}),
        _wrap(f"https://hsreplay.net/api/v1/battlegrounds/trinkets/{q}",
              [{"trinket_dbf_id": 200001, "pick_rate": 0.4,
                "avg_final_placement": 3.5, "tier": 2, "group": "lesser",
                "final_placement_distribution": [0.1] * 8,
                "extra_data": None}]),
        _wrap(f"https://hsreplay.net/api/v1/battlegrounds/heroes/{q}",
              [{"hero_dbf_id": 300001, "pick_rate": 0.3,
                "avg_final_placement": 3.9,
                "adjusted_avg_final_placement": 3.8, "tier_v2": "A",
                "anomaly_adjusted": False, "best_composition": 92,
                "key_minions_top3": [100001, 100002],
                "final_placement_distribution": [0.1] * 8}]),
        _wrap(f"https://hsreplay.net/analytics/query/battlegrounds_purchase_rates_by_turn/{q}",
              {"series": {"data": {"ALL": [
                  {"minion_dbf_id": 100001, "recruitment_round": 3,
                   "times_picked_0_owned": 60, "times_picked_1_owned": 30,
                   "times_picked_2_owned": 10,
                   "total_times_offered": 200,
                   "times_discover_picked_0_owned": 5,
                   "times_discover_picked_1_owned": 0,
                   "times_discover_picked_2_owned": 0,
                   "total_times_discover_offered": 10},
              ]}}}),
        _wrap(f"https://hsreplay.net/analytics/query/battlegrounds_minion_stats/{q}",
              {"series": {"data": {"ALL": [
                  {"normal_dbf_id": 100001, "is_premium": 0,
                   "combat_round": 5, "median_attack": 4,
                   "median_health": 5},
              ]}}}),
        _wrap(f"https://hsreplay.net/analytics/query/battlegrounds_tavern_up_stats_all/{q}",
              {"series": {"data": {"ALL": [
                  {"recruit_round": 4, "end_of_recruit_round_tier": 2,
                   "occurrences": 800, "pct_at_tier": 61.5,
                   "num_games": 1300},
              ]}}}),
    ]


def test_real_endpoints_end_to_end(tmp_path, names):
    out = tmp_path / "hsreplay_card_stats.json"
    result = hsreplay_import.ingest_wrappers(
        _real_capture_wrappers(), "test", out_path=str(out),
        stats_dir=str(tmp_path))
    assert result["categories"] == {
        "minions": 2, "comps": 1, "trinkets": 1, "heroes": 1,
        "purchase_rates": 1, "minion_curves": 1, "tavern_up": 1}
    assert result["overall"] == 2

    cards = json.loads(out.read_text())["cards"]
    assert cards[0] == {"name": "Bream Counter", "cardId": "100002",
                        "averagePlacement": 2.95, "techLevel": 3}

    comps = json.loads((tmp_path / "hsreplay_comps_stats.json").read_text())
    assert comps["items"][0]["name"] == "Elementals"     # id 92 resolved
    assert comps["items"][0]["averagePlacement"] == 3.2

    heroes = json.loads((tmp_path / "hsreplay_heroes_stats.json").read_text())
    assert heroes["items"][0]["name"] == "Reno Jackson"
    assert heroes["items"][0]["bestComposition"] == "Elementals"
    assert heroes["items"][0]["keyMinions"] == ["Sellemental", "Bream Counter"]

    trinkets = json.loads(
        (tmp_path / "hsreplay_trinkets_stats.json").read_text())
    assert trinkets["items"][0]["name"] == "Ship in a Bottle"

    pr = json.loads(
        (tmp_path / "hsreplay_purchase_rates_by_turn_stats.json").read_text())
    assert pr["turns"]["3"][0] == {"name": "Sellemental", "cardId": "100001",
                                   "timesOffered": 200, "pickRate": 0.5,
                                   "discoverPickRate": 0.5}

    tav = json.loads((tmp_path / "hsreplay_tavern_up_stats.json").read_text())
    assert tav["rounds"]["4"]["2"]["pctAtTier"] == 61.5

    curves = json.loads(
        (tmp_path / "hsreplay_minion_curves_stats.json").read_text())
    assert curves["rounds"]["5"][0]["medianAttack"] == 4
    assert pr["_percentile"] == "TOP_20_PERCENT"


def test_percentile_preference(tmp_path, names):
    top50 = _wrap("https://hsreplay.net/api/v1/battlegrounds/heroes/"
                  "?BattlegroundsMMRPercentile=TOP_50_PERCENT",
                  [{"hero_dbf_id": 300001, "avg_final_placement": 4.2}])
    top20 = _wrap("https://hsreplay.net/api/v1/battlegrounds/heroes/"
                  "?BattlegroundsMMRPercentile=TOP_20_PERCENT",
                  [{"hero_dbf_id": 300001, "avg_final_placement": 3.9}])
    hsreplay_import.ingest_wrappers(
        [top50, top20], "test", out_path=str(tmp_path / "o.json"),
        stats_dir=str(tmp_path))
    heroes = json.loads((tmp_path / "hsreplay_heroes_stats.json").read_text())
    assert heroes["items"][0]["averagePlacement"] == 3.9   # TOP_20 preferred
    assert heroes["_percentile"] == "TOP_20_PERCENT"
