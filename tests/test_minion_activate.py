"""Season 14 Battlegrounds minion Activate regressions.

Activate is not a hero power: it is a clickable minion interaction during the
Recruit Phase. The engine currently shares the generic activatable scoring path
with hero powers, but preserves semantic_kind=minion_activate so descriptions,
logs, validation and future effect-specific scoring remain distinct.
"""

from hsbg_coach.actions import (
    HERO_POWER, MINION_ACTIVATE, legal_actions,
)
from hsbg_coach.cards import CardKnowledge, build_card_kb
from hsbg_coach.validator import validate


def _kb():
    card = CardKnowledge(
        card_id="BG36_701",
        name="Kelp Keeper",
        tier=4,
        attack=5,
        health=5,
        tribes=["Murloc"],
        keywords=["INTERACTABLE_OBJECT"],
        text="<b>Activate (1):</b> Trigger a friendly minion's Battlecry.",
    )
    return {card.card_id: card}


def _snap(*, gold=1, phase="recruit", tags=None):
    return {
        "turn": 7,
        "phase": phase,
        "tavern_tier": 4,
        "gold": gold,
        "hero_health": 30,
        "board": [{
            "entity_id": 77,
            "card_id": "BG36_701",
            "name": "Kelp Keeper",
            "attack": 5,
            "health": 5,
            "tags": dict(tags or {
                "INTERACTABLE_OBJECT": "1",
                "INTERACTABLE_OBJECT_COST": "1",
                "BACON_ACTIVATE_TOOLTIP": "1",
            }),
        }],
        "shop": [],
    }


def _activate_actions(snap):
    return [a for a in legal_actions(snap, _kb())
            if a.semantic_kind == MINION_ACTIVATE]


def test_activate_is_a_live_recruit_action_with_real_cost_and_effect():
    acts = _activate_actions(_snap())
    assert len(acts) == 1
    act = acts[0]
    # Shared generic activatable scorer for now; semantic type remains distinct.
    assert act.kind == HERO_POWER
    assert act.semantic_kind == MINION_ACTIVATE
    assert act.cost == 1
    assert act.target == "Kelp Keeper"
    assert act.describe() == "Activate Kelp Keeper (1g)"
    assert "Trigger a friendly minion's Battlecry" in act.detail["activate"]["text"]
    assert act.detail["activate"]["entity_id"] == 77


def test_activate_not_offered_in_combat_or_when_unaffordable():
    assert not _activate_actions(_snap(phase="combat"))
    assert not _activate_actions(_snap(gold=0))


def test_activate_not_offered_when_live_interaction_is_disabled_or_exhausted():
    assert not _activate_actions(_snap(tags={
        "INTERACTABLE_OBJECT": "0",
        "INTERACTABLE_OBJECT_COST": "1",
        "BACON_ACTIVATE_TOOLTIP": "1",
    }))
    assert not _activate_actions(_snap(tags={
        "INTERACTABLE_OBJECT": "1",
        "INTERACTABLE_OBJECT_COST": "1",
        "BACON_ACTIVATE_TOOLTIP": "1",
        "EXHAUSTED": "1",
    }))


def test_activate_can_fall_back_to_grounded_card_text_and_parse_cost():
    acts = _activate_actions(_snap(tags={}))
    assert len(acts) == 1
    assert acts[0].cost == 1
    assert acts[0].describe() == "Activate Kelp Keeper (1g)"


def test_director_validator_accepts_only_the_exact_activate_candidate():
    snap = _snap()
    act = _activate_actions(snap)[0]
    ok, reason = validate(
        {"move": "Activate Kelp Keeper (1g)", "why": "trigger the Battlecry now"},
        snap,
        [act],
        kb_names={"Kelp Keeper"},
    )
    assert ok is not None, reason

    bad, reason = validate(
        {"move": "Activate Fake Minion (1g)", "why": "fake"},
        snap,
        [act],
        kb_names={"Kelp Keeper"},
    )
    assert bad is None
    assert "doesn't match any legal action" in reason


def test_card_refresh_keeps_interactable_object_keyword():
    kb = build_card_kb([{
        "id": "BG36_701",
        "name": "Kelp Keeper",
        "type": "MINION",
        "techLevel": 4,
        "attack": 5,
        "health": 5,
        "race": "MURLOC",
        "mechanics": ["INTERACTABLE_OBJECT", "BATTLECRY"],
        "text": "Activate (1): Trigger a friendly minion's Battlecry.",
    }])
    card = kb["BG36_701"]
    assert card.has("INTERACTABLE_OBJECT")
    assert "Activate (1)" in card.text
