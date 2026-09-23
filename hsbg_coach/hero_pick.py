"""Hero select from HSReplay 1st-place rate (+ a small lobby-fit nudge).

Primary signal: how often the hero finishes 1st on HSReplay — the first entry
of ``final_placement_distribution`` in ``data/hsreplay_guides/heroes.json``
(all MMR, last 7 days). Aidan plays for 1sts, so a hero that wins more games
beats one that just top-4s more often. Average placement breaks ties and is
still shown next to top-4.

Heroes HSReplay has no distribution for fall back to an estimate from their
average placement (a least-squares fit over the ingested heroes), labelled
"est." so it is never mistaken for a real HSReplay number.

Lobby fit (only when the lobby's tribes are known at hero select): the hero's
HSReplay guide / favorable_tribes (see hero_comps) either lines up with an
S-tier tribe in this lobby (small bonus) or needs tribes that aren't in the
lobby at all (small penalty). Both are a couple of 1st-place percentage points,
so the 1st rate stays the deciding number.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

# 1st-place percentage points subtracted by lobby_fit (positive = worse).
_MISSING_TRIBES_PEN = 2.0      # guide needs tribes absent from this lobby
_S_TRIBE_BONUS = 1.0           # guide favors an S-tier tribe in this lobby

from .first_place import estimate_first, first_rate  # noqa: F401 (re-export)


def hsreplay_row(name: Optional[str]) -> Optional[dict]:
    """The ingested HSReplay hero entry for an offered hero name / card id."""
    if not name:
        return None
    try:
        from .hsreplay_guides import lookup_hero
        return lookup_hero({"hero_name": name}) or lookup_hero({"hero": name})
    except Exception:
        return None


def placement_line(row: dict) -> Tuple[Optional[float], Optional[float], str]:
    """(1st %, avg placement, 'HSReplay 1st 25% · top-4 66% · avg 3.61 · tier S')."""
    st = row.get("stats") or {}
    first, avg, est = first_rate(st)
    if first is None:
        return None, None, ""
    dist = st.get("final_placement_distribution") or []
    bits = [f"HSReplay 1st ~{first:.0f}% (est.)" if est else f"HSReplay 1st {first:.0f}%"]
    if not est:
        bits.append(f"top-4 {sum(dist[:4]):.0f}%")
    if avg is not None:
        bits.append(f"avg {avg:.2f}")
    tier = st.get("tier_v2")
    if tier:
        bits.append(f"tier {str(tier).upper()}")
    return first, avg, " · ".join(bits)


def lobby_fit(name: str, available: Optional[Sequence[str]]) -> Tuple[float, Optional[str]]:
    """1st-place points to subtract (positive = worse) + note for this lobby."""
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
