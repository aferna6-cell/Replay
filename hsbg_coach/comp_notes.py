"""Aidan's own comp notes layered on top of the HSReplay guides.

``data/hsreplay_guides/aidan_notes.json`` holds free-text notes per comp plus
"also buy" support rules (a regex over card text, e.g. cards that generate
spells / Blood Gems for Shop Buff Demons, or Discovers for APM Pirates).
HSReplay's guide text stays untouched; these notes are shown separately and,
once a comp is locked, cards matching its support rules count as on-plan.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "hsreplay_guides", "aidan_notes.json"))
_POOL = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "cards", "bg_live_pool_36_6_1.json"))
_TAGS = re.compile(r"<[^>]+>")


@lru_cache(maxsize=1)
def load_notes() -> dict:
    path = os.environ.get("AIDAN_NOTES_PATH") or _PATH
    if not os.path.isfile(path):
        return {"support_rules": {}, "comps": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _card_text() -> Dict[str, str]:
    if not os.path.isfile(_POOL):
        return {}
    with open(_POOL, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {m.get("name"): _TAGS.sub("", m.get("text") or "").replace("[x]", "")
            for m in doc.get("minions") or [] if m.get("name")}


def notes_for(comp_name: str) -> List[str]:
    return list(((load_notes().get("comps") or {}).get(comp_name) or {}).get("notes") or [])


def _rules_for(comp_name: str) -> List[dict]:
    notes = load_notes()
    rules = notes.get("support_rules") or {}
    want = ((notes.get("comps") or {}).get(comp_name) or {}).get("also_buy") or []
    return [dict(rules[r], id=r) for r in want if r in rules]


def support_label(card_name: Optional[str], comp_name: str) -> Optional[str]:
    """Label of the first support rule this live-pool card matches, else None."""
    if not card_name:
        return None
    text = _card_text().get(card_name)
    if not text:
        return None
    for rule in _rules_for(comp_name):
        try:
            if re.search(rule.get("pattern") or "$^", text, re.I):
                return rule.get("label") or rule["id"]
        except re.error:
            continue
    return None


def support_cards(comp_name: str) -> List[str]:
    """Every live-pool card matching the comp's support rules (for the page)."""
    return sorted(n for n in _card_text() if support_label(n, comp_name))
