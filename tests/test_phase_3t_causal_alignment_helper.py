from ml.t5_causal_alignment import align_first_state_divergence


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
    return {"kind": kind, "turn": 5, "seq": seq, "subtype": subtype, "slots": slots}


def test_extra_treatment_paint_is_first_real_divergence():
    a10 = [_slot("a", 10)]
    a20 = [_slot("a", 20)]
    b20 = [_slot("b", 20, tier=2, raw=6, obj_id=2)]
    c = [_event("turn_start", -2, a10), _event("play", 2, b20, subtype="open_slot")]
    t = [
        _event("turn_start", -2, a10),
        _event("paint", 1, a20, subtype="2s_repaint"),
        _event("play", 2, b20, subtype="open_slot"),
    ]
    r = align_first_state_divergence(c, t)
    assert r.component == "paint_repaint"
    assert r.first_event_kind == "paint"
    assert r.side == "treatment"


def test_noop_snapshot_is_ignored():
    a10 = [_slot("a", 10)]
    b10 = [_slot("b", 10, tier=2, raw=6, obj_id=2)]
    c = [_event("turn_start", -2, a10), _event("play", 2, b10, subtype="open_slot")]
    t = [
        _event("turn_start", -2, a10),
        _event("paint", 1, a10, subtype="noop_repaint"),
        _event("play", 2, b10, subtype="open_slot"),
    ]
    r = align_first_state_divergence(c, t)
    assert r.component == "residual"
    assert r.reason == "streams_reconcile"


def test_shifted_sequence_for_same_state_change_reconciles():
    a10 = [_slot("a", 10)]
    ab10 = [_slot("a", 10), {**_slot("b", 0, tier=2, raw=6, obj_id=2), "slot": 1}]
    c = [_event("turn_start", -2, a10), _event("play", 2, ab10, subtype="open_slot")]
    t = [_event("turn_start", -2, a10), _event("play", 7, ab10, subtype="open_slot")]
    r = align_first_state_divergence(c, t)
    assert r.component == "residual"
    assert r.first_event_kind is None


def test_unmatched_control_membership_is_membership():
    a10 = [_slot("a", 10)]
    ab10 = [_slot("a", 10), {**_slot("b", 0, tier=2, raw=6, obj_id=2), "slot": 1}]
    c = [_event("turn_start", -2, a10), _event("play", 1, ab10, subtype="open_slot")]
    t = [_event("turn_start", -2, a10)]
    r = align_first_state_divergence(c, t)
    assert r.component == "earlier_t5_membership"
    assert r.first_event_kind == "play"
    assert r.side == "control"


def test_baseline_difference_is_carry_in():
    c = [_event("turn_start", -2, [_slot("a", 10)])]
    t = [_event("turn_start", -2, [_slot("a", 20)])]
    r = align_first_state_divergence(c, t)
    assert r.component == "carry_in"
    assert r.reason == "baseline_state_differs"


def test_unalignable_distinct_transitions_are_conservative_residual():
    a10 = [_slot("a", 10)]
    b10 = [_slot("b", 10, tier=2, obj_id=2)]
    c20 = [_slot("c", 20, tier=3, obj_id=3)]
    c = [_event("turn_start", -2, a10), _event("play", 2, b10, subtype="open_slot")]
    t = [_event("turn_start", -2, a10), _event("paint", 2, c20, subtype="2s_repaint")]
    r = align_first_state_divergence(c, t)
    assert r.component == "residual"
    assert r.reason == "unalignable_state_transitions"
