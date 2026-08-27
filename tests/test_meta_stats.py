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


def test_prior_blends_both_sources(stats_dir):
    _write(stats_dir / "firestone_card_stats.json",
           [{"name": "Sellemental", "averagePlacement": 3.0, "impact": 0.4}])
    _write(stats_dir / "hsreplay_card_stats.json",
           [{"name": "Sellemental", "averagePlacement": 3.4, "impact": 0.2}])
    ap, impact = card_meta_stats.prior("Sellemental")
    assert ap == pytest.approx(3.2)
    assert impact == pytest.approx(0.3)


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
    result = hsreplay_import.import_captures(str(cap_dir), str(out), str(by_turn))
    assert result == {"overall": 1, "turns": [5, 8]}
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
        str(cap_dir), str(tmp_path / "o.json"), str(tmp_path / "t.json"))
    assert result == {"overall": 0, "turns": [6]}


def test_import_captures_ignores_junk(tmp_path):
    cap_dir = tmp_path / "caps"
    cap_dir.mkdir()
    (cap_dir / "cap_0001.json").write_text("not json{", encoding="utf-8")
    _cap(cap_dir, 2, "https://hsreplay.net/api/v1/account/", [])
    result = hsreplay_import.import_captures(
        str(cap_dir), str(tmp_path / "o.json"), str(tmp_path / "t.json"))
    assert result == {"overall": 0, "turns": []}
    assert not (tmp_path / "o.json").exists()
