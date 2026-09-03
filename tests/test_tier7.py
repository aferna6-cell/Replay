"""Tier7 bridge tests (offline — fixtures mirror HSTracker's verified structs).

Request/response shapes come from HSTracker's open-source Swift client
(BattlegroundsHeroPickStats / BattlegroundsCompStats /
BattlegroundsTrinketPickParams / BattlegroundsQuestPickParams). See
``decisions/2026-08-20-tier7-bridge.md``.
"""

import json

import pytest

from hsbg_coach import tier7


DBF_MAP = {
    "by_name": {"a. f. kay": 61063, "forest warden omu": 71774,
                "monstrous macaw": 61055, "ironforge anvil": 105000},
    "by_id": {"61063": "A. F. Kay", "71774": "Forest Warden Omu",
              "61055": "Monstrous Macaw", "105000": "Ironforge Anvil"},
}


# --- token discovery --------------------------------------------------------

def test_token_from_hdt_oauth_json_nested_and_case_insensitive():
    # HDT (C#) shape: OAuthData with nested TokenData; casing varies by version.
    assert tier7._token_in(
        {"TokenData": {"AccessToken": "tok-pascal"}}) == "tok-pascal"
    assert tier7._token_in(
        {"token_data": {"access_token": "tok-snake"}}) == "tok-snake"
    assert tier7._token_in({"Account": {"Id": 1}, "Code": "x"}) is None


def test_find_token_prefers_env_then_file(tmp_path, monkeypatch):
    f = tmp_path / "hsreplay_oauth.json"
    f.write_text(json.dumps({"TokenData": {"access_token": "from-file"}}))
    monkeypatch.setenv("HSREPLAY_OAUTH_FILE", str(f))
    monkeypatch.delenv("HSREPLAY_TOKEN", raising=False)
    assert tier7.find_token() == "from-file"
    monkeypatch.setenv("HSREPLAY_TOKEN", "from-env")
    assert tier7.find_token() == "from-env"


# --- param builders ---------------------------------------------------------

def test_tribes_to_minion_types_maps_and_rejects_unknown():
    assert tier7.tribes_to_minion_types(["Beast", "mech", "Naga"]) == [20, 17, 92]
    with pytest.raises(ValueError):
        tier7.tribes_to_minion_types(["Beast", "Zerg"])


def test_resolve_dbf_ids_names_literals_and_misses():
    assert tier7.resolve_dbf_ids(["A. F. Kay", "61055"], DBF_MAP) == [61063, 61055]
    with pytest.raises(ValueError) as e:
        tier7.resolve_dbf_ids(["Not A Hero"], DBF_MAP)
    assert "Not A Hero" in str(e.value)


def test_hero_pick_params_shape():
    p = tier7.hero_pick_params(["A. F. Kay", "Forest Warden Omu"],
                               ["Beast", "Mech"], DBF_MAP, rating=6500)
    assert p["hero_dbf_ids"] == [61063, 71774]
    assert p["minion_types"] == [20, 17]
    assert p["battlegrounds_rating"] == 6500
    assert p["game_language"] == "enUS"


def test_trinket_pick_params_matches_verified_struct():
    p = tier7.trinket_pick_params("A. F. Kay", ["Ironforge Anvil"],
                                  ["Beast"], turn=8, dbf_map=DBF_MAP, rating=6000)
    # Exact field set from HSTracker's BattlegroundsTrinketPickParams.
    assert set(p) == {"hero_dbf_id", "hero_power_dbf_ids", "minion_types",
                      "anomaly_dbf_id", "turn", "source_dbf_id",
                      "offered_trinkets", "game_language", "game_type",
                      "battlegrounds_rating"}
    assert p["offered_trinkets"] == [{"trinket_dbf_id": 105000}]
    assert p["turn"] == 8


# --- normalization ----------------------------------------------------------

HERO_RESP = {
    "data": [
        {"hero_dbf_id": 71774, "tier_v2": "S", "pick_rate": 0.61,
         "avg_placement": 3.4, "placement_distribution": [0.2] * 8,
         "first_place_comp_popularity": [
             {"id": 1, "key_minions_top3": [61055], "name": "Beast Macaw",
              "popularity": 0.4, "is_valid": True}]},
        {"hero_dbf_id": 61063, "tier_v2": "B", "pick_rate": 0.39,
         "avg_placement": 4.2, "placement_distribution": None,
         "first_place_comp_popularity": None},
    ],
    "toast": {"min_mmr": 6000, "mmr_filter_value": "top", "parameters": {}},
}


def test_normalize_hero_pick_sorts_and_resolves_names():
    rows = tier7.normalize_hero_pick(HERO_RESP, DBF_MAP["by_id"])
    assert [r["name"] for r in rows] == ["Forest Warden Omu", "A. F. Kay"]
    assert rows[0]["tier"] == "S" and rows[0]["avg_placement"] == 3.4
    assert rows[0]["top_comps"] == ["Beast Macaw"]
    assert rows[0]["min_mmr"] == 6000


COMPS_RESP = {
    "data": {"first_place_comps_lobby_races": [
        {"id": 2, "popularity": 0.1, "name": "Mech Magnet",
         "key_minions_top3": [61055], "avg_final_placement": 3.9},
        {"id": 1, "popularity": 0.4, "name": "Beast Macaw",
         "key_minions_top3": [61055], "avg_final_placement": 3.1},
    ]}
}


def test_normalize_comps_sorts_by_placement():
    rows = tier7.normalize_comps(COMPS_RESP, DBF_MAP["by_id"])
    assert [r["name"] for r in rows] == ["Beast Macaw", "Mech Magnet"]
    assert rows[0]["key_minions"] == ["Monstrous Macaw"]


# --- high-level query + corpus log -----------------------------------------

def test_query_hero_pick_offline_and_logs_corpus(tmp_path):
    calls = {}

    def fake_fetch(url, token, payload):
        calls["url"], calls["token"], calls["payload"] = url, token, payload
        return HERO_RESP

    log = tmp_path / "tier7_log.jsonl"
    rows = tier7.query_hero_pick(
        ["A. F. Kay", "Forest Warden Omu"], ["Beast"], token="tok",
        rating=6500, dbf_map=DBF_MAP, fetch=fake_fetch, log_path=str(log))
    assert calls["url"] == tier7.HERO_PICK_URL and calls["token"] == "tok"
    assert calls["payload"]["hero_dbf_ids"] == [61063, 71774]
    assert rows[0]["name"] == "Forest Warden Omu"
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["kind"] == "hero_pick"
    assert entry["params"] == calls["payload"]
    assert entry["response"] == HERO_RESP


def test_query_comps_offline(tmp_path):
    rows = tier7.query_comps(
        ["Beast", "Mech"], token="tok", dbf_map=DBF_MAP,
        fetch=lambda url, token, payload: COMPS_RESP,
        log_path=str(tmp_path / "log.jsonl"))
    assert rows[0]["name"] == "Beast Macaw"


def test_refresh_dbf_map_from_hsjson_shape(tmp_path):
    cards = [{"dbfId": 61063, "name": "A. F. Kay", "id": "TB_X"},
             {"dbfId": 99, "name": "A. F. Kay", "id": "TB_X_SKIN"},  # dup name
             {"name": "No Dbf"}]
    out = tier7.refresh_dbf_map(cards, path=str(tmp_path / "dbf_map.json"))
    assert out["by_name"]["a. f. kay"] == 61063        # first writer wins
    assert out["by_id"]["99"] == "A. F. Kay"
    assert (tmp_path / "dbf_map.json").exists()
