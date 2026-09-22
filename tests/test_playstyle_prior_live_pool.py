"""Guardrails: playstyle prior comps ⊆ live 36.6.1 non-Naga pool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = json.loads((ROOT / "data/playstyle_prior.json").read_text())
POOL = set(json.loads((ROOT / "data/cards/bg_live_pool_36_6_1.json").read_text())["names"])


def test_pool_count_and_aberration_present():
    assert len(POOL) >= 240
    assert any("Aberration" in (PRIOR.get("tribe_tiers") or {}).get("B", []) for _ in [0])
    soft = PRIOR.get("aberration_soft_key_minions") or []
    assert soft, "expected soft Aberration keys"
    assert set(soft) <= POOL


def test_no_naga_comps_or_keys():
    for tier, comps in PRIOR["comp_tiers"].items():
        for c in comps:
            assert "naga" not in c["name"].lower()
            assert c.get("tribe") != "Naga"
            for k in c["key_minions"]:
                assert "naga" not in k.lower()


def test_all_key_minions_in_live_pool():
    for tier, comps in PRIOR["comp_tiers"].items():
        for c in comps:
            missing = [k for k in c["key_minions"] if k not in POOL]
            assert not missing, f"{tier} {c['name']}: {missing}"
            assert len(c["key_minions"]) >= 2


def test_known_oop_never_keyed():
    banned = {
        "Archlich Kel'Thuzad",
        "Monstrous Macaw",
        "Young Murk-Eye",
        "Darkgaze Elder",
        "Zapp Slywick",
        "Groundbreaker",
        "Fauna Whisperer",
    }
    keyed = {
        k
        for comps in PRIOR["comp_tiers"].values()
        for c in comps
        for k in c["key_minions"]
    }
    assert not (keyed & banned)


def test_module_oop_and_s_boost():
    from hsbg_coach.playstyle_prior import (
        is_out_of_pool,
        key_minion_boost,
        score_board_playstyle,
        tribe_weight,
    )

    assert is_out_of_pool("Archlich Kel'Thuzad")
    assert is_out_of_pool("Monstrous Macaw")
    assert tribe_weight("Naga") == 0.0
    assert tribe_weight("Beast") == 1.0
    boost, why = key_minion_boost("Banana Slamma")
    assert boost < 0 and why and "S-comp" in why
    # S board of beetle keys should score high
    s = score_board_playstyle(
        ["Banana Slamma", "Turquoise Skitterer", "Headhunter Gryphon"],
        tribe="Beast",
    )
    assert s >= 0.9
