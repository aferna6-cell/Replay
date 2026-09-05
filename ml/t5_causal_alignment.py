"""Causal alignment primitives for Phase 3T T5 event streams.

Measurement-only helper. It deliberately ignores raw cross-arm sequence-index
position and state-neutral snapshots. The result is conservative: when two
arms make incompatible state-changing transitions that cannot be aligned by
lookahead or event semantics, attribution is residual rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ml.open_slot_formation_diagnostic import board_membership_key, board_state_key

_MEMBERSHIP_KINDS = {
    "buy",
    "sell",
    "play",
    "triple",
    "transform",
    "membership",
    "prior_sell",
    "death_cleanup",
}


def kind_to_component(kind: Optional[str]) -> str:
    k = str(kind or "")
    if k in ("turn_start", "carry_in"):
        return "carry_in"
    if k in _MEMBERSHIP_KINDS:
        return "earlier_t5_membership"
    if k in ("paint", "repaint", "paint_repaint"):
        return "paint_repaint"
    if k in ("scale_sync", "scale-sync"):
        return "scale_sync"
    return "residual"


def _same_state(a: Optional[Dict], b: Optional[Dict]) -> bool:
    if a is None or b is None:
        return a is b
    a_slots = a.get("slots") or []
    b_slots = b.get("slots") or []
    return (
        board_state_key(a_slots) == board_state_key(b_slots)
        and board_membership_key(a_slots) == board_membership_key(b_slots)
    )


def _state_changed(previous: Optional[Dict], current: Dict) -> bool:
    if previous is None:
        return True
    return not _same_state(previous, current)


def normalize_state_changes(events: Sequence[Dict]) -> List[Dict]:
    """Drop state-neutral snapshots while preserving causal stream order.

    The first event is retained as the arm's baseline even when it is not
    explicitly named ``turn_start``. Subsequent rows are retained only when
    board identity or synth/recruit state changes.
    """
    out: List[Dict] = []
    previous: Optional[Dict] = None
    for raw in events or []:
        ev = dict(raw)
        if previous is None or _state_changed(previous, ev):
            out.append(ev)
        previous = ev
    return out


@dataclass(frozen=True)
class AlignmentResult:
    component: str
    first_event_kind: Optional[str]
    first_event_subtype: Optional[str]
    side: Optional[str]
    reason: str


def _result(ev: Optional[Dict], side: Optional[str], reason: str) -> AlignmentResult:
    if ev is None:
        return AlignmentResult("residual", None, None, side, reason)
    return AlignmentResult(
        kind_to_component(ev.get("kind")),
        ev.get("kind"),
        ev.get("subtype"),
        side,
        reason,
    )


def _find_matching_state(target: Dict, stream: Sequence[Dict], start: int) -> Optional[int]:
    for idx in range(start, len(stream)):
        if _same_state(target, stream[idx]):
            return idx
    return None


def align_first_state_divergence(
    control_events: Sequence[Dict],
    treatment_events: Sequence[Dict],
) -> AlignmentResult:
    """Return the earliest defensible causal state divergence.

    Rules:
    * state-neutral snapshots never create a divergence;
    * equal resulting states align even when raw sequence indices differ;
    * if one arm has extra real transitions before reaching the other arm's
      next state, the first such unmatched transition is the cause;
    * simultaneous same-kind transitions to different states attribute to that
      shared event kind;
    * otherwise return residual instead of inventing cross-arm chronology.
    """
    c = normalize_state_changes(control_events)
    t = normalize_state_changes(treatment_events)

    if not c and not t:
        return _result(None, None, "empty_streams")
    if not c:
        return _result(t[0], "treatment", "control_stream_empty")
    if not t:
        return _result(c[0], "control", "treatment_stream_empty")

    # A differing baseline is carry-in regardless of later event alignment.
    if not _same_state(c[0], t[0]):
        baseline = c[0] if str(c[0].get("kind") or "") == "turn_start" else t[0]
        return AlignmentResult(
            "carry_in",
            baseline.get("kind") if baseline else "turn_start",
            baseline.get("subtype") if baseline else None,
            "both",
            "baseline_state_differs",
        )

    i = j = 1
    while i < len(c) or j < len(t):
        if i >= len(c):
            return _result(t[j], "treatment", "unmatched_treatment_transition")
        if j >= len(t):
            return _result(c[i], "control", "unmatched_control_transition")

        ce = c[i]
        te = t[j]
        if _same_state(ce, te):
            i += 1
            j += 1
            continue

        # Look ahead by resulting causal state, not raw sequence/index.
        t_match = _find_matching_state(ce, t, j + 1)
        c_match = _find_matching_state(te, c, i + 1)

        if t_match is not None and c_match is None:
            return _result(te, "treatment", "extra_treatment_transition_before_match")
        if c_match is not None and t_match is None:
            return _result(ce, "control", "extra_control_transition_before_match")
        if t_match is not None and c_match is not None:
            # Both streams contain extra transitions. If their immediate event
            # semantics agree, the shared checkpoint is still defensible;
            # otherwise cross-arm ordering is ambiguous.
            if (
                str(ce.get("kind") or "") == str(te.get("kind") or "")
                and str(ce.get("subtype") or "") == str(te.get("subtype") or "")
            ):
                return _result(ce, "both", "shared_event_semantics_state_differs")
            return _result(None, "both", "ambiguous_bidirectional_lookahead")

        if (
            str(ce.get("kind") or "") == str(te.get("kind") or "")
            and str(ce.get("subtype") or "") == str(te.get("subtype") or "")
        ):
            return _result(ce, "both", "shared_event_semantics_state_differs")

        return _result(None, "both", "unalignable_state_transitions")

    return _result(None, None, "streams_reconcile")
