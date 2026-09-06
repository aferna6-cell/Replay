import pytest

from ml.phase_3u_prereg import (
    CANDIDATE_RULES,
    assert_no_reserved_confirmation_use,
    assert_reused_dev_only,
    evaluate_phase_3u_design,
    reference_is_independent,
)


def test_confirmation_band_is_rejected():
    with pytest.raises(ValueError):
        assert_no_reserved_confirmation_use(11500, 1)
    with pytest.raises(ValueError):
        assert_no_reserved_confirmation_use(11499, 2)
    with pytest.raises(ValueError):
        assert_no_reserved_confirmation_use(11699, 2)


def test_offline_diagnostics_are_reused_dev_only():
    assert_reused_dev_only(14200, 500)
    with pytest.raises(ValueError):
        assert_reused_dev_only(14199, 1)
    with pytest.raises(ValueError):
        assert_reused_dev_only(14699, 2)


def test_endogenous_or_aggregate_reference_does_not_identify_rule():
    assert not reference_is_independent(
        {
            "kind": "external_body_level_survival_or_damage",
            "generated_by": "replay_simulator",
            "allocation_sensitive": True,
        }
    )
    assert not reference_is_independent(
        {
            "kind": "external_body_level_survival_or_damage",
            "generated_by": "external",
            "allocation_sensitive": False,
        }
    )


def test_independent_body_level_reference_can_clear_reference_gate():
    assert reference_is_independent(
        {
            "kind": "held_out_real_game_body_level_trace",
            "generated_by": "external",
            "allocation_sensitive": True,
        }
    )


def test_no_reference_routes_to_underidentified_even_with_candidate_scores():
    out = evaluate_phase_3u_design(
        candidate_scores={name: float(i) for i, name in enumerate(CANDIDATE_RULES)},
        reference=None,
        conservation_passed=True,
    )
    assert out["route"] == "hold_allocation_underidentified"
    assert out["selection_requires_independent_reference"] is True
    assert out["no_behavior_change"] is True


def test_macro_or_candidate_sim_outcome_cannot_be_selection_target():
    out = evaluate_phase_3u_design(
        candidate_scores={name: 0.0 for name in CANDIDATE_RULES},
        reference={
            "kind": "held_out_real_game_body_level_trace",
            "generated_by": "external",
            "allocation_sensitive": True,
        },
        conservation_passed=True,
        disallowed_targets_used=("mean_game_length",),
    )
    assert out["route"] == "invalid_macro_or_endogenous_selection"


def test_reference_gate_precedes_candidate_selection_threshold():
    out = evaluate_phase_3u_design(
        candidate_scores={name: 1.0 for name in CANDIDATE_RULES},
        reference={
            "kind": "external_body_level_combat_stats",
            "generated_by": "external",
            "allocation_sensitive": True,
        },
        conservation_passed=True,
    )
    assert out["route"] == "reference_available_define_effect_threshold_before_selection"
