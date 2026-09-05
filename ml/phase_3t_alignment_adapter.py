"""Phase 3T causal-alignment adapter for the incumbent-synth diagnostic.

Measurement-only. This adapter keeps the legacy diagnostic payload shape while
routing first-divergence classification and first-event metadata through the
single causal alignment result in :mod:`ml.t5_causal_alignment`.

The installer is intentionally explicit and local to Phase 3T execution; it
changes diagnostic attribution only, never simulator behavior, RNG, seeds,
alpha, scaling, 2Q, or hero damage.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ml.open_slot_formation_diagnostic import board_membership_key, board_state_key
from ml.phase_3t_prereg import DIVERGENCE_COMPONENTS
from ml.play_lifecycle_diagnostic import decompose_play_pair
from ml.scale_sync_diagnostic import decompose_scale_pair
from ml.t5_causal_alignment import AlignmentResult, align_first_state_divergence


def first_synth_component_aligned(
    control_events: Sequence[Dict],
    treatment_events: Sequence[Dict],
) -> str:
    """Return the exclusive component from the shared causal alignment result."""
    return align_first_state_divergence(control_events, treatment_events).component


def decompose_t5_synth_pair_aligned(
    control_start: Dict,
    treatment_start: Dict,
    control_play: Optional[Dict],
    treatment_play: Optional[Dict],
    control_events: Sequence[Dict],
    treatment_events: Sequence[Dict],
    control_syncs: Optional[Sequence[Dict]] = None,
    treatment_syncs: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Legacy-compatible five-way split driven by one causal alignment result."""
    q = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    scale = decompose_scale_pair(
        control_start, treatment_start, control_play, treatment_play,
        control_syncs or [], treatment_syncs or [],
    )
    lifecycle = float(q.get("replacement_lifecycle") or 0.0)
    membership_inc = float(scale.get("membership_allocation") or 0.0)
    parts = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
    memb_parts = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
    complete = bool(q.get("snapshots_complete"))

    if complete:
        alignment = align_first_state_divergence(control_events, treatment_events)
    else:
        alignment = AlignmentResult(
            component="residual",
            first_event_kind=None,
            first_event_subtype=None,
            side=None,
            reason="snapshots_incomplete",
        )

    component = alignment.component
    parts[component] = lifecycle
    memb_parts[component] = membership_inc
    explained = sum(parts[name] for name in DIVERGENCE_COMPONENTS)

    return {
        "replacement_lifecycle": lifecycle,
        "membership_allocation": membership_inc,
        "subsequent_scaling": float(q.get("subsequent_scaling") or 0.0),
        "same_state_repaint": float(q.get("same_state_repaint") or 0.0),
        "delta_synth": float(q.get("delta_synth") or 0.0),
        **parts,
        **{
            f"membership_{name}": memb_parts[name]
            for name in DIVERGENCE_COMPONENTS
        },
        "divergence_component": component,
        "first_event_kind": alignment.first_event_kind,
        "first_event_subtype": alignment.first_event_subtype,
        "alignment_side": alignment.side,
        "alignment_reason": alignment.reason,
        "n_control_events": len(list(control_events or [])),
        "n_treatment_events": len(list(treatment_events or [])),
        "snapshots_complete": complete,
        "explained_lifecycle": explained,
        "residual_vs_lifecycle": lifecycle - explained,
        "same_t5_start_identity": (
            False if not control_events or not treatment_events
            else board_membership_key((control_events[0] or {}).get("slots"))
            == board_membership_key((treatment_events[0] or {}).get("slots"))
        ),
        "same_t5_start_state": (
            False if not control_events or not treatment_events
            else board_state_key((control_events[0] or {}).get("slots"))
            == board_state_key((treatment_events[0] or {}).get("slots"))
        ),
        "s_c_sticky": q.get("s_c_sticky"),
        "s_t_sticky_cf": q.get("s_t_sticky_cf"),
        "s_c_start": q.get("s_c_start"),
        "s_t_start": q.get("s_t_start"),
        "flow_gap_control": scale.get("flow_gap_control"),
        "flow_gap_treatment": scale.get("flow_gap_treatment"),
    }


def install_into_legacy_module() -> None:
    """Install aligned attribution into the existing Phase 3T diagnostic module."""
    import ml.t5_incumbent_synth_diagnostic as legacy

    legacy.first_synth_component = first_synth_component_aligned
    legacy.decompose_t5_synth_pair = decompose_t5_synth_pair_aligned
