"""Meta pack tests — source priority (Tier7 vs Firestone), patch drift, and
the compact lobby-prompt slice. Fixtures mirror the shapes of the real
committed ``data/stats/firestone_*.json`` files (see ``hsbg_coach/stats.py``
and ``hsbg_coach/firestone_stats.py`` docstrings)."""

import json
import os

import pytest

from hsbg_coach import meta_pack
from hsbg_coach.cards import CardKnowledge


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def make_stats_dir(tmp_path):
    d = tmp_path / "stats"
    d.mkdir()
    _write(d / "firestone_hero_stats.json", {
        "_fetched": "2026-08-01",
        "heroes": [
            {"cardId": "H1", "name": "Hero One", "averagePosition": 3.8,
             "pickRate": 0.5, "bestTribes": ["Murloc"], "playstyle": "tempo"},
        ],
    })
    _write(d / "firestone_comp_stats.json", {
        "_fetched": "2026-08-01",
        "comps": [
            {"name": "Murloc Swarm", "archetype": "murloc_swarm", "tribe": "Murloc",
             "averagePosition": 3.5, "popularity": 0.05,
             "coreCards": ["Murloc One"], "powerTurns": [], "tier": "A"},
            {"name": "Beast Brawl", "archetype": "beast_brawl", "tribe": "Beast",
             "averagePosition": 4.0, "popularity": 0.03,
             "coreCards": ["Beast One"], "powerTurns": [], "tier": "B"},
        ],
    })
    _write(d / "firestone_trinket_stats.json", {
        "_fetched": "2026-08-01",
        "trinkets": [
            {"cardId": "T1", "name": "Trinket One", "averagePosition": 3.9,
             "pickRate": 0.4, "tier": "S", "text": "Buffs Murlocs."},
        ],
    })
    _write(d / "firestone_pace.json", {
        "_fetched": "2026-08-01",
        "tavern_tier": {"1": 1.0, "2": 1.6, "5": 3.3, "7": 4.3, "9": 5.2},
        "leveling": {}, "scaling": {},
    })
    return str(d)


def make_kb():
    return {
        "M1": CardKnowledge(card_id="M1", name="Murloc One", tier=1, attack=1,
                            health=1, tribes=["Murloc"], keywords=[],
                            text="A murloc."),
        "M2": CardKnowledge(card_id="M2", name="Murloc Two", tier=2, attack=2,
                            health=2, tribes=["Murloc"], keywords=[],
                            text="Give your other Murlocs +1/+1."),
        "M3": CardKnowledge(card_id="M3", name="Murloc Enabler", tier=1,
                            attack=1, health=2, tribes=[], keywords=[],
                            text="Whenever you play a Murloc, gain +1/+1."),
        "B1": CardKnowledge(card_id="B1", name="Beast One", tier=3, attack=3,
                            health=3, tribes=["Beast"], keywords=[],
                            text="A beast."),
    }


TIER7_ROWS = {
    "heroes": [
        {"dbf_id": 1, "name": "Hero One", "tier": "S", "pick_rate": 0.6,
         "avg_placement": 3.0, "placement_distribution": None,
         "top_comps": ["Murloc Swarm"]},
    ],
    "comps": [
        {"id": 1, "name": "Murloc Swarm Tier7", "popularity": 0.3,
         "avg_final_placement": 2.8, "key_minions": ["Murloc One", "Murloc Two"]},
    ],
}


# --- source priority ---------------------------------------------------

def test_build_meta_pack_firestone_only_when_no_tier7(tmp_path):
    stats_dir = make_stats_dir(tmp_path)
    pack = meta_pack.build_meta_pack(stats_dir=stats_dir, kb=make_kb())

    assert pack["sources"] == {"heroes": "firestone", "comps": "firestone"}
    assert pack["fetched"] == "2026-08-01"
    names = [c["name"] for c in pack["comps"]]
    assert "Murloc Swarm" in names and "Beast Brawl" in names
    murloc = next(c for c in pack["comps"] if c["name"] == "Murloc Swarm")
    assert murloc["source"] == "firestone"
    assert murloc["core_cards"] == ["Murloc One"]
    # Enabler derivation reaches into the kb: Murloc Two/Enabler both carry
    # buffs:Murloc / cares:Murloc tags and aren't the (excluded) core card.
    assert "Murloc Two" in murloc["enabler_cards"]
    assert "Murloc Enabler" in murloc["enabler_cards"]
    assert "Murloc One" not in murloc["enabler_cards"]        # excluded (core)


def test_build_meta_pack_tier7_wins_when_present(tmp_path):
    stats_dir = make_stats_dir(tmp_path)
    pack = meta_pack.build_meta_pack(stats_dir=stats_dir, kb=make_kb(),
                                     tier7_rows=TIER7_ROWS)

    assert pack["sources"]["comps"] == "tier7"
    assert pack["sources"]["heroes"] == "tier7+firestone"
    names = [c["name"] for c in pack["comps"]]
    assert names == ["Murloc Swarm Tier7"]         # tier7 comps fully replace
    tier7_comp = pack["comps"][0]
    assert tier7_comp["avg_placement"] == 2.8
    assert tier7_comp["tribe"] == "Murloc"          # inferred from key_minions
    assert tier7_comp["source"] == "tier7"

    hero = next(h for h in pack["heroes"] if h["name"] == "Hero One")
    assert hero["source"] == "tier7"                # tier7 row overrides firestone's
    assert hero["tier"] == "S"
    assert hero["top_comps"] == ["Murloc Swarm"]


def test_tier7_heroes_merge_keeps_firestone_top_n(tmp_path):
    stats_dir = make_stats_dir(tmp_path)
    rows = {"heroes": [
        {"dbf_id": 2, "name": "Brand New Hero", "tier": "A", "pick_rate": 0.2,
         "avg_placement": 4.9, "top_comps": []},
    ]}
    pack = meta_pack.build_meta_pack(stats_dir=stats_dir, kb=make_kb(),
                                     tier7_rows=rows)
    names = [h["name"] for h in pack["heroes"]]
    assert "Hero One" in names                       # firestone base kept
    assert "Brand New Hero" in names                  # tier7-requested addition


# --- tribes / curve / trinkets ------------------------------------------

def test_tribes_ranked_by_avg_placement(tmp_path):
    pack = meta_pack.build_meta_pack(stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    tribes = [t["tribe"] for t in pack["tribes"]]
    assert tribes == ["Murloc", "Beast"]              # Murloc avg 3.5 < Beast 4.0
    assert pack["tribes"][0]["top_comps"] == ["Murloc Swarm"]


def test_curve_has_standard_and_measured(tmp_path):
    pack = meta_pack.build_meta_pack(stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    assert pack["curve"]["standard"][7] == 5           # folk wisdom, spec req 9
    assert pack["curve"]["measured"][7] == 4.3          # from firestone_pace.json


def test_trinkets_included_with_text(tmp_path):
    pack = meta_pack.build_meta_pack(stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    assert pack["trinkets"][0]["name"] == "Trinket One"
    assert "Buffs Murlocs" in pack["trinkets"][0]["text"]


# --- patch drift ---------------------------------------------------------

def test_check_patch_drift_mismatch():
    pack = {"patch_build": "31.2"}
    msg = meta_pack.check_patch_drift(pack, "31.4")
    assert msg is not None and "31.2" in msg and "31.4" in msg


@pytest.mark.parametrize("pack_build,live_build", [
    ("31.2", "31.2"),      # match
    (None, "31.4"),        # pack doesn't know its patch
    ("31.2", None),        # caller doesn't know the live patch
    (None, None),
])
def test_check_patch_drift_fine_or_unknown(pack_build, live_build):
    assert meta_pack.check_patch_drift({"patch_build": pack_build}, live_build) is None


# --- pack_for_prompt -------------------------------------------------------

def test_pack_for_prompt_only_lobby_tribes_and_under_budget(tmp_path):
    pack = meta_pack.build_meta_pack(stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    text = meta_pack.pack_for_prompt(pack, ["Murloc"], max_chars=6000)
    assert len(text) <= 6000
    assert "Murloc Swarm" in text
    assert "Beast Brawl" not in text                  # off-lobby tribe excluded


def test_pack_for_prompt_truncates_deterministically(tmp_path):
    pack = meta_pack.build_meta_pack(stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    text1 = meta_pack.pack_for_prompt(pack, ["Murloc", "Beast"], max_chars=120)
    text2 = meta_pack.pack_for_prompt(pack, ["Murloc", "Beast"], max_chars=120)
    assert len(text1) <= 120
    assert text1 == text2                              # deterministic


# --- save/load -------------------------------------------------------------

def test_save_and_load_meta_pack_round_trip(tmp_path):
    pack = meta_pack.build_meta_pack(stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    path = str(tmp_path / "meta_pack.json")
    meta_pack.save_meta_pack(pack, path)
    loaded = meta_pack.load_meta_pack(path)
    assert loaded["comps"] == pack["comps"]
    assert loaded["patch_build"] == pack["patch_build"]


# --- playbook_index --------------------------------------------------------

def test_playbook_index_reflects_existing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/playbooks", exist_ok=True)
    with open("data/playbooks/murloc-swarm.md", "w", encoding="utf-8") as fh:
        fh.write("# Murloc Swarm\n")
    pack = meta_pack.build_meta_pack(
        stats_dir=make_stats_dir(tmp_path), kb=make_kb())
    assert "Murloc Swarm" in pack["playbook_index"]
    assert "Beast Brawl" not in pack["playbook_index"]


# --- real committed data sanity check --------------------------------------

def test_build_meta_pack_against_real_committed_data():
    pack = meta_pack.build_meta_pack(patch_build="test-patch")
    assert pack["patch_build"] == "test-patch"
    assert len(pack["tribes"]) > 3
    assert len(pack["comps"]) > 3
    assert all("core_cards" in c for c in pack["comps"])
