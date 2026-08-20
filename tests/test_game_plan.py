"""Game plan tests — comp ranking (with/without Tier7 hero-conditioned
boost), pivot triggers, and plan revision. Uses a small meta-pack-shaped
fixture rather than a full ``build_meta_pack()`` call so the ranking logic
is tested in isolation."""

from hsbg_coach import game_plan


def make_meta_pack():
    return {
        "patch_build": "31.2",
        "fetched": "2026-08-01",
        "sources": {"heroes": "firestone", "comps": "firestone"},
        "tribes": [
            {"tribe": "Murloc", "avg_placement": 3.5, "top_comps": ["Murloc Swarm"]},
            {"tribe": "Beast", "avg_placement": 4.2, "top_comps": ["Beast Brawl"]},
        ],
        "heroes": [],
        "comps": [
            {"name": "Murloc Swarm", "tribe": "Murloc", "tier": "A",
             "avg_placement": 3.5, "core_cards": ["Murloc One", "Murloc Two"],
             "enabler_cards": ["Murloc Enabler"], "source": "firestone"},
            {"name": "Beast Brawl", "tribe": "Beast", "tier": "B",
             "avg_placement": 4.2, "core_cards": ["Beast One"],
             "enabler_cards": [], "source": "firestone"},
            {"name": "Mech Madness", "tribe": "Mech", "tier": "C",
             "avg_placement": 4.6, "core_cards": ["Mech One"],
             "enabler_cards": [], "source": "firestone"},
        ],
        "trinkets": [],
        "curve": {"standard": {7: 5}, "measured": {7: 4.3, 8: 4.8}},
        "playbook_index": [],
    }


def test_build_game_plan_picks_best_lobby_comp():
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack())
    assert plan.plan_a == "Murloc Swarm"        # lowest avg_placement in lobby
    assert plan.plan_b == "Beast Brawl"
    assert plan.plan_a_enablers == ["Murloc Enabler"]
    assert plan.lobby_tribes == ["Murloc", "Beast"]
    assert plan.revised == []


def test_build_game_plan_produces_pivot_triggers():
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack())
    assert len(plan.pivot_triggers) >= 3
    joined = " ".join(plan.pivot_triggers)
    assert "Murloc Swarm" in joined and "Beast Brawl" in joined
    assert any("HP" in t for t in plan.pivot_triggers)
    assert any("pace" in t.lower() for t in plan.pivot_triggers)  # measured-pace trigger


def test_build_game_plan_excludes_off_lobby_comps():
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack())
    assert plan.plan_a != "Mech Madness"
    assert plan.plan_b != "Mech Madness"


def test_notes_state_flexible_tribes_rule():
    plan = game_plan.build_game_plan(["Murloc"], make_meta_pack())
    assert "not locks" in plan.notes or "preference" in plan.notes.lower()


def test_tier7_hero_rows_boost_matching_comp_over_better_avg_placement():
    # Beast Brawl has a worse avg_placement than Murloc Swarm, but a Tier7
    # hero-pick row for the offered hero names it as a top comp — that
    # lobby+hero-conditioned signal should win the tie-break (req: "Tier7-
    # derived rows win when present").
    rows = [{"top_comps": ["Beast Brawl"]}]
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack(),
                                     hero_offer=["Some Hero"],
                                     tier7_hero_rows=rows)
    assert plan.plan_a == "Beast Brawl"
    assert "Some Hero" in plan.notes


def test_falls_back_to_best_overall_comp_when_lobby_has_no_match():
    plan = game_plan.build_game_plan(["Naga"], make_meta_pack())
    assert plan.plan_a == "Murloc Swarm"        # best comp overall, not empty


def test_placeholder_plan_when_meta_pack_has_no_comps():
    empty_pack = {"comps": [], "curve": {"measured": {}}}
    plan = game_plan.build_game_plan(["Murloc"], empty_pack)
    assert plan.plan_a == ""
    assert "placeholder" in plan.notes.lower()
    assert plan.pivot_triggers


# --- plan_to_prompt --------------------------------------------------------

def test_plan_to_prompt_contains_key_fields_and_respects_budget():
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack())
    text = game_plan.plan_to_prompt(plan, max_chars=1200)
    assert "Plan A" in text and "Pivot triggers" in text
    assert len(text) <= 1200

    tiny = game_plan.plan_to_prompt(plan, max_chars=20)
    assert len(tiny) <= 20


# --- save/load ---------------------------------------------------------

def test_save_and_load_plan_round_trip(tmp_path):
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack())
    path = str(tmp_path / "game_plan.json")
    game_plan.save_plan(plan, path)
    loaded = game_plan.load_plan(path)
    assert loaded.plan_a == plan.plan_a
    assert loaded.pivot_triggers == plan.pivot_triggers
    assert loaded.revised == []


# --- revision --------------------------------------------------------------

def test_revise_plan_appends_and_returns():
    plan = game_plan.build_game_plan(["Murloc", "Beast"], make_meta_pack())
    out = game_plan.revise_plan(plan, "Turn 6: pivoting to Beast, Murloc pieces dry.")
    assert out is plan
    assert plan.revised == ["Turn 6: pivoting to Beast, Murloc pieces dry."]
    game_plan.revise_plan(plan, "Turn 9: back on Murloc, board recovered.")
    assert len(plan.revised) == 2
