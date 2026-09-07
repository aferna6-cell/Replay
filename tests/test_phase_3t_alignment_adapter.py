from ml.phase_3t_alignment_adapter import first_synth_component_aligned


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


def test_adapter_attributes_unmatched_repaint_to_paint():
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
    assert first_synth_component_aligned(control, treatment) == "paint_repaint"


def test_adapter_ignores_noop_insertion_and_raw_seq_drift():
    a10 = [_slot("a", 10)]
    b10 = [_slot("b", 10, tier=2, raw=6, obj_id=2)]
    control = [
        _event("turn_start", -2, a10),
        _event("play", 2, b10, subtype="open_slot"),
    ]
    treatment = [
        _event("turn_start", -2, a10),
        _event("paint", 1, a10, subtype="noop_repaint"),
        _event("play", 7, b10, subtype="open_slot"),
    ]
    assert first_synth_component_aligned(control, treatment) == "residual"
