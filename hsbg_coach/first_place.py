"""1st-place rate — the number hero and trinket picks are ranked by.

Aidan plays for 1sts, so picks rank by how often a hero / trinket finishes
1st on HSReplay (the first entry of ``final_placement_distribution``). When a
row has no distribution the rate is estimated from its average placement with
a least-squares fit over the ingested HSReplay heroes, and flagged as an
estimate so it is never shown as a real HSReplay number.
"""

from __future__ import annotations

from typing import Optional, Tuple

# 1st% ≈ a + b·avg, fit over the 118 ingested HSReplay heroes (36.6.1).
_FIRST_A, _FIRST_B = 48.39, -7.91
# 1st-place points one full placement is worth (converts placement deltas).
PTS_PER_PLACE = -_FIRST_B


def estimate_first(avg: float) -> float:
    """Estimated 1st-place % from an average placement (fallback only)."""
    return max(1.0, min(60.0, _FIRST_A + _FIRST_B * float(avg)))


def first_rate(stats: Optional[dict]) -> Tuple[Optional[float], Optional[float], bool]:
    """(1st %, avg placement, estimated?) from an HSReplay stats dict."""
    st = stats or {}
    avg = st.get("avg_final_placement")
    avg = float(avg) if avg is not None else None
    dist = st.get("final_placement_distribution") or []
    if len(dist) >= 4:
        return float(dist[0]), avg, False
    if avg is not None:
        return estimate_first(avg), avg, True
    return None, None, False


def first_label(first: float, estimated: bool) -> str:
    return f"1st ~{first:.0f}% (est.)" if estimated else f"1st {first:.0f}%"
