"""Adversarial contract tests for Phase 3T event-stream alignment.

These tests intentionally exercise asymmetric event streams. A scientific
first-divergence classifier must align events by chronology/causal identity;
it must not attribute a divergence merely because one arm emitted an extra
snapshot before the matching event in the other arm.
"""

from ml.phase_3t_alignment_adapter import first_synth_component_aligned as first_synth_component


def _slot(card: str, synth: int, *, tier: int = 1, raw: int = 4, obj_id: int = 1):
    return {
        "slot": 0,
        "card_id": card,
        "name": card,
        "tier": tier,
        "recruit_raw": raw,
        "synthetic_share": synth,
        "obj_id": obj_id,
    }


def _event(kind: str, seq: int, slots, *, subtype=None):
    return {
        "kind": kind,
        "turn": 5,
        "seq": seq,
        "subtype": subtype,
        "slots": slots,
    }


def test_unmatched_treatment_paint_is_classified_as_paint_not_membership():
    """An extra treatment repaint is the first causal state divergence."""
    a10 = [_slot("a", 10)]
    a20 = [_slot("a", 20)]
    b20 = [_slot("b", 20, tier=2, raw=6, obj_id=2)]

    control = [
        _event("turn_start", -2, a10),
        _event("play", 2, b20, subtype="open_slot"),
    ]
    treatment = [
        _event("turn_start", -2, a10),
        _event("paint", 1, a20, subtype="2s_repaint"),
        _event("play", 2, b20, subtype="open_slot"),
    ]

    assert first_synth_component(control, treatment) == "paint_repaint"


def test_noop_extra_snapshot_does_not_create_false_membership_divergence():
    """An unmatched snapshot with unchanged state must not shift later pairing."""
    a10 = [_slot("a", 10)]
    b10 = [_slot("b", 10, tier=2, raw=6, obj_id=2)]

    control = [
        _event("turn_start", -2, a10),
        _event("play", 2, b10, subtype="open_slot"),
    ]
    treatment = [
        _event("turn_start", -2, a10),
        _event("paint", 1, a10, subtype="noop_repaint"),
        _event("play", 2, b10, subtype="open_slot"),
    ]

    assert first_synth_component(control, treatment) == "residual"


def test_unmatched_control_membership_event_keeps_membership_attribution():
    """A real unmatched membership state change remains a membership cause."""
    a10 = [_slot("a", 10)]
    ab10 = [
        _slot("a", 10),
        {**_slot("b", 0, tier=2, raw=6, obj_id=2), "slot": 1},
    ]

    control = [
        _event("turn_start", -2, a10),
        _event("play", 1, ab10, subtype="open_slot"),
    ]
    treatment = [_event("turn_start", -2, a10)]

    assert first_synth_component(control, treatment) == "earlier_t5_membership"


def test_same_logical_event_with_shifted_sequence_index_is_not_a_divergence():
    """Raw seq drift alone cannot become a scientific first-difference cause."""
    a10 = [_slot("a", 10)]
    ab10 = [
        _slot("a", 10),
        {**_slot("b", 0, tier=2, raw=6, obj_id=2), "slot": 1},
    ]

    control = [
        _event("turn_start", -2, a10),
        _event("play", 2, ab10, subtype="open_slot"),
    ]
    treatment = [
        _event("turn_start", -2, a10),
        _event("play", 7, ab10, subtype="open_slot"),
    ]

    assert first_synth_component(control, treatment) == "residual"


def test_common_prefix_then_noop_then_real_paint_attributes_real_paint():
    """A no-op insertion must not hide the later first real state divergence."""
    a10 = [_slot("a", 10)]
    a20 = [_slot("a", 20)]

    control = [
        _event("turn_start", -2, a10),
        _event("paint", 4, a20, subtype="real_repaint"),
    ]
    treatment = [
        _event("turn_start", -2, a10),
        _event("paint", 1, a10, subtype="noop_repaint"),
        _event("paint", 4, a20, subtype="real_repaint"),
    ]

    assert first_synth_component(control, treatment) == "residual"


def test_identical_state_changing_streams_reconcile_to_residual():
    """Identical causal streams must not manufacture an attribution class."""
    a10 = [_slot("a", 10)]
    ab10 = [
        _slot("a", 10),
        {**_slot("b", 0, tier=2, raw=6, obj_id=2), "slot": 1},
    ]
    ab20 = [
        _slot("a", 20),
        {**_slot("b", 0, tier=2, raw=6, obj_id=2), "slot": 1},
    ]

    stream = [
        _event("turn_start", -2, a10),
        _event("play", 2, ab10, subtype="open_slot"),
        _event("paint", 3, ab20, subtype="2s_repaint"),
    ]

    assert first_synth_component(stream, list(stream)) == "residual"
