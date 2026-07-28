"""In-game recognition — recalibrate every frame by reading what's actually
happening, not just ranking the next click.

Two recognizers, refreshed on every state change:
  * ``self_synergy`` — what is FORMING on your board: a tribe stacking up, a strong
    effect combo you may have stumbled into, keyword payoffs lining up. Surfaces
    "lean into it" so you commit to a synergy you found rather than scatter.
  * ``opponent_standout`` — an opponent whose comp is pulling ahead (much stronger
    board, a heavy keyword/tribe lean), so you recognize the real threat and tech
    or out-scale it.

Both are derived from the live board/opponent state + card effect text, so they
generalize across the card pool.
"""

from typing import List, Optional

from .board_value import _name


def _idx(kb):
    from .cards import by_name
    return by_name(kb) if kb else {}


def self_synergy(snapshot, kb) -> List[str]:
    """Recognitions about YOUR board: a forming tribe, a strong stumbled-upon combo,
    keyword payoffs stacking. Empty when nothing notable is forming yet."""
    board = (snapshot.get("board") if isinstance(snapshot, dict) else None) or []
    if len(board) < 2 or not kb:
        return []
    idx = _idx(kb)
    cks = [(m, idx.get(_name(m))) for m in board]
    cks = [(m, c) for m, c in cks if c]
    out: List[str] = []

    # 1) A tribe stacking up — recognize the comp that's forming.
    tribes = {}
    for _, c in cks:
        for t in (c.tribes or []):
            tribes[t.lower()] = tribes.get(t.lower(), 0) + 1
    if tribes:
        dom = max(tribes, key=tribes.get)
        n = tribes[dom]
        if n >= 3 and dom != "all":
            out.append(f"{n} {dom.capitalize()}s on board — a {dom.capitalize()} comp "
                       f"is forming; lean into it (buy/keep {dom.capitalize()}s)")

    # 2) A strong effect combo you may have stumbled into — the board card that
    #    meshes hardest with the rest.
    try:
        from .effect_synergy import board_synergy
        best_score, best_reason = 0.0, None
        for m, c in cks:
            rest = [oc for om, oc in cks if om is not m]
            score, reasons = board_synergy(c, rest)
            if score > best_score and reasons:
                best_score, best_reason = score, reasons[0]
        if best_score >= 2.0 and best_reason:
            out.append(f"strong combo on board — {best_reason}; protect and scale it")
    except Exception:
        pass

    # 3) Keyword payoff stacking (e.g. several Divine Shields → buff/AOE-protect).
    kw = {}
    for m, _ in cks:
        tags = (m.get("tags") if isinstance(m, dict) else getattr(m, "tags", None)) or {}
        for k in ("DIVINE_SHIELD", "REBORN", "POISONOUS", "WINDFURY"):
            if str(tags.get(k, "")) not in ("", "0", "False"):
                kw[k] = kw.get(k, 0) + 1
    if kw.get("DIVINE_SHIELD", 0) >= 3:
        out.append(f"{kw['DIVINE_SHIELD']} Divine Shields — buffs and pop-protection "
                   f"scale hard here")
    return out[:2]


def opponent_standout(snapshot) -> Optional[str]:
    """Recognize an opponent whose comp is pulling ahead, so you respect the threat.
    Compares the strongest opponent board to your own strength."""
    profs = (snapshot.get("opponent_profiles") if isinstance(snapshot, dict) else None) or []
    if not profs:
        return None
    board = (snapshot.get("board") if isinstance(snapshot, dict) else None) or []
    from .board_value import _val
    my = sum(_val(m) for m in board) or 1.0
    top = max(profs, key=lambda p: p.get("strength") or 0)
    s = top.get("strength") or 0
    if s <= 0:
        return None
    ratio = s / my
    if ratio >= 1.4:
        who = top.get("hero") or "an opponent"
        tribe = (top.get("tribe") or "").capitalize()
        lean = f" {tribe}" if tribe else ""
        return (f"{who} is pulling ahead ({s} stats,{lean} — ~{ratio:.1f}x your board) "
                f"— tech for it or out-scale fast")
    return None
