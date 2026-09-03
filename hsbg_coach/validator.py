"""Deterministic gate between the LLM's suggestion and the overlay.

The LLM (`director.py`) is fast but not trusted: it can hallucinate a buy the
shop never offered, or float gold into "end turn" out of laziness. Per spec
requirements 2 and 12, neither of those is allowed to reach the player — a
plain, testable, non-LLM function stands between the model and the panel and
either accepts the suggestion (possibly lightly corrected — e.g. fixing a
card name's casing to match the shop's own) or rejects it with a reason, in
which case the caller (`director.suggest_move`) falls back to the engine's
own top-ranked candidate.

Deliberately dependency-light: no card-KB import here. Callers pass
``kb_names`` (a set of known card names) so this module stays a pure function
of its inputs — easy to unit test, easy to reason about.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from .actions import (
    BUY, BUY_SPELL, END, FREEZE, HERO_POWER, LEVEL, REPOSITION, ROLL, SELL,
)

# An "end turn while gold is unspent" suggestion is only OK when the LLM
# names one of these — the model must say *why*, not just do it.
_SAFE_JUSTIFICATIONS = {"infinite_economy", "deliberate_freeze", "save_for_spike"}

# These are legal regardless of the shop's contents, so they never need a
# candidate match — "roll/level/freeze/end" per spec req 2/12's wording.
_GENERIC_LEGAL_KINDS = {ROLL, LEVEL, FREEZE, END}

# Pressing the dark-gift button isn't in the engine's action space (no log
# signal yet — spec req 10 # CALIBRATE), but the Director is explicitly asked
# to call its timing; the player can see whether the button exists.
DARK_GIFT = "dark_gift"

# Kinds that actually spend gold — used by the no-float-gold check to ask
# "was there something affordable to do instead of ending the turn?"
_SPEND_KINDS = {BUY, ROLL, LEVEL, HERO_POWER, BUY_SPELL}
_DEFAULT_SPEND_COST = {ROLL: 1, BUY: 3}

_PAREN_TAIL_RE = re.compile(r"\(.*?\)\s*$")
_BUY_RE = re.compile(r"^buy\s*:?\s+(.+)$", re.IGNORECASE)
_SELL_RE = re.compile(r"^sell\s*:?\s+(.+)$", re.IGNORECASE)


def _get(obj, key: str, default=None):
    """dict-or-object field access, matching the `_get` helper convention
    used across `game_value.py` / `turn_search.py` / `advisor.py`."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cand_action(candidate):
    """The Action-like object inside one `game_value.rank_actions` /
    `WholeGameRec` candidate — or the candidate itself, so a bare Action (or
    an already-normalized dict) works too."""
    a = _get(candidate, "action", None)
    return a if a is not None else candidate


def _strip_paren_tail(text: str) -> str:
    return _PAREN_TAIL_RE.sub("", text).strip()


def classify_move(move: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(kind, target)`` for a free-text move string like
    ``"Buy Monstrous Macaw"`` or ``"End turn"``. ``target`` is the card name
    for buy/sell, else None. ``kind`` is None when the text doesn't match any
    known shape (still handled — see `validate`'s free-text fallback)."""
    text = _strip_paren_tail((move or "").strip())
    low = text.lower()
    m = _BUY_RE.match(text)
    if m:
        return BUY, m.group(1).strip()
    m = _SELL_RE.match(text)
    if m:
        return SELL, m.group(1).strip()
    if "hero power" in low:
        # Targeted uses carry the target after "on": "Use hero power on Brann".
        m = re.search(r"hero power(?:\s*:\s*| on )(.+)$", text, re.IGNORECASE)
        return HERO_POWER, (m.group(1).strip() if m else None)
    if "dark gift" in low or low.startswith(("use gift", "press gift", "open gift")):
        return DARK_GIFT, None
    if low.startswith("end") or "end turn" in low or low.startswith("pass"):
        return END, None
    if low.startswith("roll"):
        return ROLL, None
    if "freeze" in low:
        return FREEZE, None
    if ("tier up" in low or "level up" in low or "tavern up" in low
            or low.startswith("level") or low.startswith("upgrade")):
        return LEVEL, None
    if "reposition" in low or "reorder" in low:
        return REPOSITION, None
    return None, None


def _normalize_text(text: str) -> str:
    text = _PAREN_TAIL_RE.sub("", text or "")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _describe_candidate(candidate) -> Optional[str]:
    """A human-readable description of one candidate, matching what
    `Action.describe()` (`actions.py`) would produce — used both for the
    free-text match fallback and, if the caller already serialized
    candidates (e.g. from `director.log_suggestion`'s schema), the
    ``{"action": "..."}`` string form directly."""
    a = _cand_action(candidate)
    describe = getattr(a, "describe", None)
    if callable(describe):
        try:
            return describe()
        except Exception:
            pass
    if isinstance(a, str):
        return a
    kind = (_get(a, "kind") or "").lower()
    target = _get(a, "target")
    if kind == BUY:
        return f"Buy {target}"
    if kind == SELL:
        return f"Sell {target}"
    if kind == ROLL:
        return "Roll the shop"
    if kind == FREEZE:
        return "Freeze the shop"
    if kind == REPOSITION:
        return "Reposition the board"
    if kind == HERO_POWER:
        return f"Use hero power: {target}"
    if kind == END:
        return "End turn"
    return None


def _candidate_targets(candidates, kind: str) -> Dict[str, str]:
    """``{lowercased target -> the candidate's own-cased target}`` for every
    candidate of `kind` — lets us both match case-insensitively and correct
    the suggestion to the engine's exact spelling."""
    out: Dict[str, str] = {}
    for c in candidates or []:
        a = _cand_action(c)
        if (_get(a, "kind") or "").lower() != kind:
            continue
        target = _get(a, "target")
        if target:
            out[str(target).strip().lower()] = str(target).strip()
    return out


def _any_affordable_spend(candidates, gold: int) -> bool:
    """True when at least one gold-spending candidate (buy/roll/level/hero
    power/spell) is affordable right now — the "or any affordable action
    exists" half of the no-float-gold rule."""
    for c in candidates or []:
        a = _cand_action(c)
        kind = (_get(a, "kind") or "").lower()
        if kind not in _SPEND_KINDS:
            continue
        cost = _get(a, "cost")
        if cost is None:
            cost = _DEFAULT_SPEND_COST.get(kind)
        if cost is None:
            continue
        try:
            cost = int(cost)
        except (TypeError, ValueError):
            continue
        if 0 <= cost <= gold:
            return True
    return False


def _matches_any_candidate_text(move: str, candidates) -> bool:
    norm_move = _normalize_text(move)
    if not norm_move:
        return False
    for c in candidates or []:
        desc = _describe_candidate(c)
        if desc and _normalize_text(desc) == norm_move:
            return True
    return False


def validate(suggestion: dict, snapshot, candidates: List,
            kb_names: Optional[Set[str]] = None
            ) -> Tuple[Optional[dict], Optional[str]]:
    """Gate one LLM suggestion. Returns ``(accepted, None)`` — `accepted` is
    `suggestion`, possibly with `move`'s card-name casing corrected to match
    the engine's own — or ``(None, rejection_reason)``.

    Rules (spec reqs 2 + 12):
      (a) NO-FLOAT-GOLD — reject an end-turn move while gold is unspent and
          something affordable exists, unless `justification` names a real
          reason (infinite_economy / deliberate_freeze / save_for_spike).
      (b) the move must be one of the engine's own candidates (buy/sell
          match by target name; everything else by normalized description)
          OR a legal generic action (roll/level/freeze/end) — this is what
          catches a hallucinated buy of a card the shop never offered.
      (c) any card named in a buy/sell must exist in `kb_names` (when given).
      (d) `experiment: true` requires a non-empty `hypothesis`.
    """
    if not isinstance(suggestion, dict):
        return None, "suggestion is not a JSON object"

    move = (suggestion.get("move") or "").strip()
    if not move:
        return None, "suggestion has no non-empty 'move'"

    hypothesis = (suggestion.get("hypothesis") or "").strip()
    if suggestion.get("experiment") and not hypothesis:
        return None, "experiment suggestion is missing a non-empty hypothesis"

    try:
        gold = int(_get(snapshot, "gold") or 0)
    except (TypeError, ValueError):
        gold = 0

    kind, target = classify_move(move)

    if kind == END and gold >= 1:
        justification = (suggestion.get("justification") or "").strip()
        if justification not in _SAFE_JUSTIFICATIONS and _any_affordable_spend(candidates, gold):
            return None, (
                f"no-float-gold: suggested ending the turn with {gold} gold "
                "unspent while a roll/buy/level was affordable, and no "
                "justification (infinite_economy/deliberate_freeze/"
                "save_for_spike) was given"
            )

    accepted = dict(suggestion)

    if kind in (BUY, SELL):
        by_lower = _candidate_targets(candidates, kind)
        exact = by_lower.get((target or "").strip().lower())
        if exact is None:
            verb = "buy" if kind == BUY else "sell"
            return None, (
                f"hallucinated {verb}: {target!r} is not among the engine's "
                "legal candidates this turn (not in the shop / not on your "
                "board)"
            )
        if kb_names is not None:
            kb_lower = {n.lower() for n in kb_names}
            if exact.lower() not in kb_lower:
                return None, f"unknown card name: {exact!r} is not in the card KB"
        if exact != target:
            verb = "Buy" if kind == BUY else "Sell"
            accepted["move"] = f"{verb} {exact}"
        return accepted, None

    if kind in _GENERIC_LEGAL_KINDS:
        return accepted, None

    if kind == DARK_GIFT:
        return accepted, None                  # timing call; button visibility
                                               # is the player's, not the log's

    if kind == HERO_POWER:
        hp = _get(snapshot, "hero_power")
        usable = bool(hp.get("usable")) if isinstance(hp, dict) else bool(hp)
        for p in (_get(snapshot, "hero_powers") or []):   # any of the buttons
            if isinstance(p, dict) and p.get("usable"):
                usable = True
                break
        has_candidate = any(
            (_get(_cand_action(c), "kind") or "").lower() == HERO_POWER
            for c in candidates or [])
        if not (usable or has_candidate):
            return None, ("hero power suggested but the log shows no usable "
                          "hero power this turn")
        # A named target ("on Brann Bronzebeard") must be a real card we know
        # about — board, shop, or KB — so a hallucinated target can't slip by.
        if target:
            names = {str(_get(m, "name") or "").lower()
                     for m in (_get(snapshot, "board") or [])}
            names |= {str(_get(m, "name") or "").lower()
                      for m in (_get(snapshot, "shop") or [])}
            if kb_names is not None:
                names |= {n.lower() for n in kb_names}
            if target.lower() not in names:
                return None, (f"hero power target {target!r} is not on your "
                              "board/shop or in the card KB")
        return accepted, None

    # Anything else (reposition, hero power, or text we couldn't classify):
    # only accept if it matches a candidate's own description, so a
    # hallucinated free-form move can't slip through either.
    if _matches_any_candidate_text(move, candidates):
        return accepted, None
    return None, f"unrecognized move: {move!r} doesn't match any legal action this turn"
