"""Hero select from HSReplay hero placement (+ a small lobby-fit nudge).

Primary signal: HSReplay's average final placement for the hero (ingested in
``data/hsreplay_guides/heroes.json`` — all MMR, last 7 days), shown with the
1st-place and top-4 rates from HSReplay's placement distribution.

Lobby fit (only when the lobby's tribes are known at hero select): the hero's
HSReplay guide / favorable_tribes (see hero_comps) either lines up with an
S-tier tribe in this lobby (small bonus) or needs tribes that aren't in the
lobby at all (small penalty). Placement stays the deciding number; the nudge
only separates heroes that HSReplay has within a few hundredths.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

_MISSING_TRIBES_PEN = 0.20     # guide needs tribes absent from this lobby
_S_TRIBE_BONUS = 0.10          # guide favors an S-tier tribe in this lobby


def hsreplay_row(name: Optional[str]) -> Optional[dict]:
    """The ingested HSReplay hero entry for an offered hero name / card id."""
    if not name:
        return None
    try:
        from .hsreplay_guides import lookup_hero
        return lookup_hero({"hero_name": name}) or lookup_hero({"hero": name})
    except Exception:
        return None


def placement_line(row: dict) -> Tuple[Optional[float], str]:
    """(avg placement, 'HSReplay avg 3.61 · 1st 14% · top-4 55% · tier S')."""
    st = row.get("stats") or {}
    avg = st.get("avg_final_placement")
    if avg is None:
        return None, ""
    bits = [f"HSReplay avg {float(avg):.2f}"]
    dist = st.get("final_placement_distribution") or []
    if len(dist) >= 4:
        bits.append(f"1st {dist[0]:.0f}%")
        bits.append(f"top-4 {sum(dist[:4]):.0f}%")
    tier = st.get("tier_v2")
    if tier:
        bits.append(f"tier {str(tier).upper()}")
    return float(avg), " · ".join(bits)


def lobby_fit(name: str, available: Optional[Sequence[str]]) -> Tuple[float, Optional[str]]:
    """Placement adjustment (negative = better) + note for this lobby."""
    try:
        from .tribe_policy import filter_lobby_tribes
        from .hero_comps import fit_for_name
        from .lobby_playbook import rank_lobby_tribes
    except Exception:
        return 0.0, None
    lobby = filter_lobby_tribes(available) if available else []
    row = hsreplay_row(name)
    if not lobby or not row:
        return 0.0, None
    fit = fit_for_name(row.get("name"))
    # HSReplay's favorable_tribes is the hero's stated requirement; otherwise
    # the tribes its guide text favors.
    wants = list(fit.hs_tribes or fit.tribes)
    if not wants:
        return 0.0, None
    in_lobby = [t for t in wants if t in lobby]
    if not in_lobby:
        return _MISSING_TRIBES_PEN, f"guide wants {'/'.join(wants)} — not in this lobby"
    labels = {r.tribe: r.label for r in rank_lobby_tribes(lobby)}
    s_hits = [t for t in in_lobby if labels.get(t) == "S"]
    if s_hits:
        return -_S_TRIBE_BONUS, f"guide favors {'/'.join(s_hits)} (S in this lobby)"
    return 0.0, f"guide favors {'/'.join(in_lobby)} (in lobby)"
