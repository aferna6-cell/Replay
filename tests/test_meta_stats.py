"""Population-prior blending (card_meta_stats), the eval-net feature block
built on it, and the HSReplay export importer."""

import json

import pytest

from hsbg_coach import card_meta_stats, hsreplay_import


@pytest.fixture
def stats_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(card_meta_stats, "_STATS_DIR", str(tmp_path))
    card_meta_stats.reload()
    yield tmp_path
    card_meta_stats.reload()


def _write(path, cards):
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")


def test_prior_single_source(stats_dir):
    _write(stats_dir / "firestone_card_stats.json",
           [{"name": "Upper Hand", "averagePlacement": 2.45, "impact": 1.9}])
    assert card_meta_stats.prior("upper hand") == (2.45, 1.9)
    assert card_meta_stats.prior("Nonexistent Card") is None


def test_prior_blends_both_sources_hsreplay_weighted_higher(stats_dir):
    _write(stats_dir / "firestone_card_stats.json",
           [{"name": "Sellemental", "averagePlacement": 3.0, "impact": 0.4}])
    _write(stats_dir / "hsreplay_card_stats.json",
           [{"name": "Sellemental", "averagePlacement": 3.4, "impact": 0.2}])
    ap, impact = card_meta_stats.prior("Sellemental")
    # weighted mean: firestone 1.0, hsreplay 2.0
    assert ap == pytest.approx((3.0 * 1 + 3.4 * 2) / 3)
    assert impact == pytest.approx((0.4 * 1 + 0.2 * 2) / 3)


def test_prior_golden_falls_back_to_base(stats_dir):
    _write(stats_dir / "firestone_card_stats.json",
           [{"name": "Sellemental", "averagePlacement": 3.0}])
    assert card_meta_stats.prior("Golden Sellemental") == (3.0, 0.0)


def test_meta_prior_vector_and_feature_dim(stats_dir):
    np = pytest.importorskip("numpy")
    from ml.board_features import (_META, board_vector, feature_dim,
                                   meta_prior_vector)
    _write(stats_dir / "firestone_card_stats.json",
           [{"name": "Good Card", "averagePlacement": 2.4, "impact": 1.0},
            {"name": "Bad Card", "averagePlacement": 4.4, "impact": -0.5}])
    card_meta_stats.reload()

    def mk(name):
        return dict(name=name, atk=1, health=1, tier=1, golden=False,
                    divine=False, reborn=False, taunt=False, tribes=[])

    v = meta_prior_vector([mk("Good Card"), mk("Bad Card"), mk("Unknown")])
    neutral = card_meta_stats.NEUTRAL_PLACEMENT
    assert v[0] == pytest.approx(((neutral - 2.4) + (neutral - 4.4)) / 2)
    assert v[1] == pytest.approx(neutral - 2.4)          # best card's edge
    assert v[2] == pytest.approx(0.25)                    # mean impact
    assert v[3] == pytest.approx(1.0)                     # best impact
    assert v[4] == pytest.approx(2 / 3)                   # coverage
    # no data at all -> neutral zeros, and dims stay consistent
    assert np.all(meta_prior_vector([mk("Unknown")]) == 0.0)
    emb = {"Good Card": [0.0, 1.0]}
    assert board_vector([mk("Good Card")], emb).shape[0] == feature_dim(emb)
    assert len(v) == len(_META)


def test_hsreplay_import_json_loose_keys(tmp_path):
    export = tmp_path / "export.json"
    export.write_text(json.dumps({"series": {"data": {"ALL": [
        {"card_name": "Sellemental", "avg_placement": "3.61", "games": 9000},
        {"card_name": "Bream Counter", "avg_placement": 3.05,
         "placement_delta": 0.8},
        {"card_name": "Header Junk", "avg_placement": "not a number"},
    ]}}}), encoding="utf-8")
    out = tmp_path / "hsreplay_card_stats.json"
    assert hsreplay_import.import_file(str(export), str(out)) == 2
    cards = json.loads(out.read_text())["cards"]
    assert cards[0]["name"] == "Bream Counter"        # sorted by placement
    assert cards[0]["impact"] == 0.8
    assert cards[1] == {"name": "Sellemental", "averagePlacement": 3.61,
                        "totalPlayed": 9000}


def test_hsreplay_import_csv(tmp_path):
    export = tmp_path / "table.csv"
    export.write_text("Minion,Avg Place,Pick Rate\n"
                      "Upper Hand,2.45,12%\n"
                      "Sellemental,3.40,30%\n", encoding="utf-8")
    out = tmp_path / "hsreplay_card_stats.json"
    assert hsreplay_import.import_file(str(export), str(out)) == 2
    names = [c["name"] for c in json.loads(out.read_text())["cards"]]
    assert names == ["Upper Hand", "Sellemental"]


def test_hsreplay_import_rejects_empty(tmp_path):
    export = tmp_path / "junk.json"
    export.write_text(json.dumps({"whatever": 1}), encoding="utf-8")
    assert hsreplay_import.import_file(
        str(export), str(tmp_path / "out.json")) == 0
    assert not (tmp_path / "out.json").exists()


def _cap(tmp_path, n, url, rows, post=None):
    (tmp_path / f"cap_{n:04d}.json").write_text(json.dumps(
        {"url": url, "method": "GET", "post_data": post,
         "body": {"series": {"data": {"ALL": rows}}}}), encoding="utf-8")


def test_import_captures_overall_and_by_turn(tmp_path):
    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _cap(cap_dir, 1, "https://hsreplay.net/analytics/query/bg_minions/?RankRange=TOP_10",
         [{"name": "Sellemental", "avg_placement": 3.4, "games": 100}])
    _cap(cap_dir, 2, "https://hsreplay.net/analytics/query/bg_minions/?RankRange=TOP_10&Turn=5",
         [{"name": "Sellemental", "avg_placement": 3.1}])
    _cap(cap_dir, 3, "https://hsreplay.net/analytics/query/bg_minions/?turn_range=8",
         [{"name": "Bream Counter", "avg_placement": 2.9}])
    # later capture of the same filter overrides (a re-capture is a refresh)
    _cap(cap_dir, 4, "https://hsreplay.net/analytics/query/bg_minions/?RankRange=TOP_10",
         [{"name": "Sellemental", "avg_placement": 3.5, "games": 200}])
    out = tmp_path / "overall.json"
    by_turn = tmp_path / "by_turn.json"
    result = hsreplay_import.import_captures(str(cap_dir), str(out), str(by_turn),
                                             stats_dir=str(tmp_path))
    assert result["overall"] == 1
    assert result["turns"] == [5, 8]
    assert result["categories"] == {"minions": 1}
    overall = json.loads(out.read_text())["cards"]
    assert overall == [{"name": "Sellemental", "averagePlacement": 3.5,
                        "totalPlayed": 200}]
    turns = json.loads(by_turn.read_text())["turns"]
    assert turns["5"][0]["averagePlacement"] == 3.1
    assert turns["8"][0]["name"] == "Bream Counter"


def test_import_captures_post_body_turn(tmp_path):
    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _cap(cap_dir, 1, "https://hsreplay.net/api/v1/battlegrounds/stats/",
         [{"name": "Upper Hand", "avg_placement": 2.5}],
         post='{"filters": {"turn": 6}}')
    result = hsreplay_import.import_captures(
        str(cap_dir), str(tmp_path / "o.json"), str(tmp_path / "t.json"),
        stats_dir=str(tmp_path))
    assert result["overall"] == 0
    assert result["turns"] == [6]


def test_import_captures_ignores_junk(tmp_path):
    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    (cap_dir / "cap_0001.json").write_text("not json{", encoding="utf-8")
    _cap(cap_dir, 2, "https://hsreplay.net/api/v1/account/", [])
    result = hsreplay_import.import_captures(
        str(cap_dir), str(tmp_path / "o.json"), str(tmp_path / "t.json"),
        stats_dir=str(tmp_path))
    assert result == {"overall": 0, "turns": [], "categories": {}}
    assert not (tmp_path / "o.json").exists()


def test_import_captures_categorizes_all_data_types(tmp_path):
    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    _cap(cap_dir, 1, "https://hsreplay.net/analytics/query/bg_comp_stats/",
         [{"name": "Murlocs", "avg_placement": 3.2}])
    _cap(cap_dir, 2, "https://hsreplay.net/analytics/query/bg_trinket_stats/",
         [{"name": "Ship in a Bottle", "avg_placement": 3.5}])
    _cap(cap_dir, 3, "https://hsreplay.net/analytics/query/bg_dark_gift_stats/",
         [{"name": "Echoes of Argus", "avg_placement": 3.1}])
    _cap(cap_dir, 4, "https://hsreplay.net/analytics/query/bg_hero_stats/",
         [{"name": "Reno Jackson", "avg_placement": 3.9}])
    result = hsreplay_import.import_captures(
        str(cap_dir), str(tmp_path / "o.json"), str(tmp_path / "t.json"),
        stats_dir=str(tmp_path))
    assert result["categories"] == {"comps": 1, "trinkets": 1,
                                    "dark_gifts": 1, "heroes": 1}
    comps = json.loads((tmp_path / "hsreplay_comps_stats.json").read_text())
    assert comps["items"][0]["name"] == "Murlocs"
    gifts = json.loads((tmp_path / "hsreplay_dark_gifts_stats.json").read_text())
    assert gifts["items"][0]["name"] == "Echoes of Argus"
    # minions files untouched when no minion payloads captured
    assert not (tmp_path / "o.json").exists()


def test_import_file_detects_userscript_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(hsreplay_import, "_STATS_DIR", str(tmp_path))
    bundle = tmp_path / "hsreplay_bundle.json"
    bundle.write_text(json.dumps({"captures": [
        {"url": "https://hsreplay.net/analytics/query/bg_minions/?Turn=4",
         "body": {"data": [{"name": "Sellemental", "avg_placement": 3.2}]}},
        {"url": "https://hsreplay.net/analytics/query/bg_trinket_stats/",
         "body": {"data": [{"name": "Ship in a Bottle",
                            "avg_placement": 3.5}]}},
    ]}), encoding="utf-8")
    out = tmp_path / "hsreplay_card_stats.json"
    count = hsreplay_import.import_file(str(bundle), str(out))
    assert count == 2
    trinkets = json.loads(
        (tmp_path / "hsreplay_trinkets_stats.json").read_text())
    assert trinkets["items"][0]["name"] == "Ship in a Bottle"
    minions = json.loads(
        (tmp_path / "hsreplay_minions_stats.json").read_text())
    assert minions["by_turn"]["4"][0]["name"] == "Sellemental"
