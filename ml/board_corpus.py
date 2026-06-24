"""Turn the 20k winning boards into a corpus for card2vec.

Each winning board is a "sentence" — the set of cards that won together. card2vec
learns embeddings where cards that co-occur in winning boards land near each
other (the board analogue of word2vec on sentences). Cards are keyed by *name*
so golden/triple copies merge with their base.

Boards come from Firestone's comp data (`compStats[].heroStats[].finalBoards[]`).
"""

from collections import Counter
from typing import Dict, List, Optional

from hsbg_coach.firestone_stats import (
    _fetch_json, _card_meta, _names_from_meta, COMP_URL, CARDS_URL,
)


def extract_boards(comp_source: Optional[str] = None, period: str = "past-seven",
                   names: Optional[Dict[str, str]] = None,
                   cards_source: str = CARDS_URL) -> List[List[str]]:
    """Return one card-name set per winning board (deduped within a board)."""
    raw = _fetch_json(comp_source or COMP_URL.format(period=period))
    if names is None:
        names = _names_from_meta(_card_meta(cards_source))
    boards: List[List[str]] = []
    for comp in raw.get("compStats", []):
        for hero in comp.get("heroStats", []):
            for fb in hero.get("finalBoards", []):
                board = (fb.get("finalComp") or {}).get("board") or []
                cards = set()
                for m in board:
                    cid = m.get("cardID") or m.get("cardId")
                    nm = names.get(cid, cid)
                    if nm:
                        cards.add(nm)
                if len(cards) >= 2:
                    boards.append(sorted(cards))
    return boards


def build_vocab(boards: List[List[str]], min_count: int = 20):
    """Return (itos, stoi, counts) keeping cards seen >= min_count times."""
    counts = Counter(c for board in boards for c in board)
    itos = [c for c, n in counts.most_common() if n >= min_count]
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi, [counts[c] for c in itos]
