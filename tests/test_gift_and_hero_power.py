"""Dark-gift routing + hero-power suggestion chain (owner reqs, 2026-08-20).

The Director must be able to say "Use hero power on Brann Bronzebeard" and
"Use dark gift", with the validator accepting the real cases and still
blocking hallucinated targets.
"""

from hsbg_coach.choices import classify
from hsbg_coach.director import _compact_state
from hsbg_coach.draft import recommend_choice
from hsbg_coach.validator import classify_move, validate, DARK_GIFT

from hsbg_coach.actions import HERO_POWER


SNAP_HP = {"gold": 2, "board": [{"name": "Brann Bronzebeard"}], "shop": [],
           "hero_power": {"name": "Gonna Need That!", "cost": 1, "usable": True,
                          "card_id": "BG_RENO_HP"}}


def test_classify_dark_gift_ids_and_fallback():
    assert classify(["BG32_DarkGift_101", "BG32_DarkGift_204"]) == "dark_gift"
    assert classify(["BG30_MagicItem_403"]) == "trinket"
    assert classify(["BGS_random_minion"]) == "discover"


def test_recommend_choice_dark_gift_ranks_like_discover():
    out = recommend_choice("dark_gift", ["Some Gift", "Other Gift"],
                           board=[], kb={})
    assert len(out) == 2                       # no ValueError, ranked output


def test_classify_move_hero_power_and_gift():
    assert classify_move("Use hero power on Brann Bronzebeard") == \
        (HERO_POWER, "Brann Bronzebeard")
    assert classify_move("Use hero power")[0] == HERO_POWER
    assert classify_move("Use dark gift")[0] == DARK_GIFT
    assert classify_move("Press dark gift now")[0] == DARK_GIFT


def test_validator_accepts_targeted_hero_power_on_board_minion():
    ok, reason = validate({"move": "Use hero power on Brann Bronzebeard",
                           "why": "double battlecries"}, SNAP_HP, [])
    assert ok is not None, reason


def test_validator_rejects_hero_power_without_usable_power():
    snap = dict(SNAP_HP, hero_power={"name": "X", "usable": False})
    ok, reason = validate({"move": "Use hero power", "why": "y"}, snap, [])
    assert ok is None and "no usable" in reason


def test_validator_rejects_hallucinated_hero_power_target():
    ok, reason = validate({"move": "Use hero power on Fake Minion",
                           "why": "y"}, SNAP_HP, [], kb_names=set())
    assert ok is None and "Fake Minion" in reason


def test_validator_accepts_dark_gift_press():
    ok, _ = validate({"move": "Use dark gift", "why": "tempo is fine"},
                     {"gold": 0}, [])
    assert ok is not None


def test_compact_state_includes_hero_power_text():
    s = _compact_state(dict(SNAP_HP, hero_power=dict(SNAP_HP["hero_power"],
                                                     text="Discover a spell.")))
    assert "Hero power: Gonna Need That!" in s
    assert "Discover a spell." in s
    assert "usable now" in s


# --- Grounding (spec req 15): facts given, inference = pathing only ---------

def test_compact_minion_renders_kb_annotations():
    from hsbg_coach.director import _compact_minion
    s = _compact_minion({"name": "Monstrous Macaw", "attack": 5, "health": 3,
                         "tier": 3, "tribes": ["Beast"],
                         "text": "After this attacks, trigger a friendly minion's Deathrattle."})
    assert "T3" in s and "Beast" in s and "Deathrattle" in s


def test_game_plan_carries_real_target_board():
    from hsbg_coach.game_plan import _target_board, build_game_plan, plan_to_prompt
    comp = {"name": "Mech Magnet", "core_cards": ["Fallback Card"]}
    fb = [{"name": "Mech Magnet", "archetype": "mech_magnet",
           "coreCards": ["Core A"],
           "examples": [{"mmr": 9000, "minions": [
               {"name": "Ingenious Inventor", "pos": 2},
               {"name": "Ingenious Inventor", "pos": 1},
               {"name": "Holo Rover", "pos": 3}]}]}]
    assert _target_board(comp, fb) == ["Ingenious Inventor", "Holo Rover"]
    assert _target_board({"name": "Unknown", "core_cards": ["X"]}, fb) == ["X"]
    pack = {"tribes": [], "heroes": [], "trinkets": [],
            "curve": {"standard": {}, "measured": {}}, "playbook_index": [],
            "comps": [{"name": "Murloc Scam", "tribe": "Murloc", "tier": "S",
                       "avg_placement": 3.0, "core_cards": ["Bile Spitter"],
                       "enabler_cards": []}]}
    plan = build_game_plan(["Murloc"], pack)
    assert plan.plan_a_target                      # real committed data resolves
    assert "TARGET BOARD" in plan_to_prompt(plan)


def test_grounding_rule_in_system_prompt():
    from hsbg_coach.director import _GAME_RULES_ESSENTIALS
    assert "PATHING" in _GAME_RULES_ESSENTIALS
    assert "Never guess what a card does" in _GAME_RULES_ESSENTIALS
