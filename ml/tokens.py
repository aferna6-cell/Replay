"""Per-minion tokens — the input the attention models read.

`board_features.py` mean-pools card2vec over the board, which erases which
cards you actually have (Baron + six deathrattles and Baron + vanilla stats
average to nearly the same vector). The set models instead see one token per
minion — card2vec identity + stats + tier + keyword flags + tribes — and let
self-attention model the minion↔minion interactions (synergy) directly.

One token builder serves the eval net (board only) and the policy net (board ∪
shop ∪ hand with a zone id). numpy-only; shared normalization is bounded
transforms (log1p stats, tier/6) so tokens need no fitted standardization.
"""

import math
from typing import Dict, List, Optional

import numpy as np

from hsbg_coach import cards
from .board_features import TRIBES, _TRIBE_IX, emb_dim

# Keyword flags read from the card knowledge base (same source for population,
# live-snapshot and env minions, so the channel means the same thing everywhere).
KB_KEYWORDS = ["POISONOUS", "WINDFURY", "CLEAVE", "DEATHRATTLE", "BATTLECRY",
               "MAGNETIC", "DIVINE_SHIELD", "TAUNT"]
_SCALARS = ["log_atk", "log_hp", "tier", "golden", "divine", "reborn", "taunt"]
# Relative stats: this minion's atk/hp as a fraction of the strongest minion
# in view. log1p compresses absolute stats (an 8/8 vs a 4/4 differs by ~0.14),
# which made "buy the biggest" hard to learn sharply — and with the env's
# multiplicative scaling, buying the smaller body locks in a permanent ratio
# deficit. Relative channels keep full discrimination at every stat scale.
# Filled by the token-set builders (board_tokens / encode_obs), not per-minion.
_REL = ["rel_atk", "rel_hp"]

MAX_BOARD_TOKENS = 7


def token_dim(emb: Dict[str, List[float]]) -> int:
    return emb_dim(emb) + len(_SCALARS) + len(TRIBES) + len(KB_KEYWORDS) + len(_REL)


_BYNAME = None


def neutral_state(state):
    """Context state with gold zeroed. In end-of-recruit training snapshots,
    leftover gold is a BEHAVIORAL signal (bad/random players strand gold), not
    a state value — a net that reads it rates every gold-spend as an
    improvement, which sent the turn search into level/roll churn. Within-turn
    gold accounting is the search's job; the value function reads the board,
    the tier, survival and the lobby."""
    if not state:
        return state
    s = dict(state)
    s["gold"] = 0.0
    return s


def default_byname():
    global _BYNAME
    if _BYNAME is None:
        _BYNAME = cards.by_name(cards.load_kb())
    return _BYNAME


def minion_token(m: Dict, emb: Dict[str, List[float]],
                 byname: Optional[Dict] = None) -> np.ndarray:
    """One normalized minion dict (board_features `_minion` shape) -> token."""
    byname = byname if byname is not None else default_byname()
    dim = emb_dim(emb)
    vec = np.asarray(emb.get(m["name"], np.zeros(dim)), dtype=float)
    scalars = np.array([
        math.log1p(max(m["atk"], 0)) / 5.0,
        math.log1p(max(m["health"], 0)) / 5.0,
        (m["tier"] or 1) / 6.0,
        float(m["golden"]), float(m["divine"]),
        float(m["reborn"]), float(m["taunt"]),
    ], dtype=float)
    tribes = np.zeros(len(TRIBES))
    for t in m["tribes"]:
        if t in _TRIBE_IX:
            tribes[_TRIBE_IX[t]] = 1.0
    kws = np.zeros(len(KB_KEYWORDS))
    ck = byname.get(m["name"])
    if ck is not None:
        for i, kw in enumerate(KB_KEYWORDS):
            if kw in ck.keywords or (kw == "POISONOUS" and "VENOMOUS" in ck.keywords):
                kws[i] = 1.0
    return np.concatenate([vec, scalars, tribes, kws, np.zeros(len(_REL))])


def fill_relative_stats(toks: np.ndarray, mask: np.ndarray,
                        raw: List[Dict]) -> None:
    """Fill the trailing rel_atk/rel_hp channels across a token SET in place.
    `raw` holds the minion dicts (any shape with attack/health readable) in
    token order; normalization is by the max across the whole set."""
    def _stat(m, *keys):
        for k in keys:
            v = m.get(k) if isinstance(m, dict) else None
            if v is not None:
                return max(0.0, float(v))
        return 0.0
    atks = [_stat(m, "atk", "attack") for m in raw]
    hps = [_stat(m, "health") for m in raw]
    max_a, max_h = max(atks, default=0.0) or 1.0, max(hps, default=0.0) or 1.0
    for i, (a, h) in enumerate(zip(atks, hps)):
        if i < toks.shape[0] and mask[i] > 0:
            toks[i, -2] = a / max_a
            toks[i, -1] = h / max_h


def board_tokens(minions: List[Dict], emb: Dict[str, List[float]],
                 byname: Optional[Dict] = None, max_tokens: int = MAX_BOARD_TOKENS
                 ):
    """Normalized minion dicts -> (tokens[max_tokens, F], mask[max_tokens])."""
    dim = token_dim(emb)
    toks = np.zeros((max_tokens, dim), dtype=np.float32)
    mask = np.zeros(max_tokens, dtype=np.float32)
    kept = minions[:max_tokens]
    for i, m in enumerate(kept):
        toks[i] = minion_token(m, emb, byname)
        mask[i] = 1.0
    fill_relative_stats(toks, mask, kept)
    return toks, mask


def examples_to_arrays(examples: List[Dict], emb: Dict[str, List[float]],
                       hero_stoi: Dict[str, int],
                       ctx_stats=None, byname: Optional[Dict] = None):
    """Dataset rows ({minions, hero, label, state}) -> set-model arrays.

    Returns (tokens, mask, ctx, hero, y, ctx_stats). Fits the context
    standardization when ctx_stats is None (train split), reuses it otherwise.
    """
    from .board_features import context_vector
    byname = byname if byname is not None else default_byname()
    n = len(examples)
    toks = np.zeros((n, MAX_BOARD_TOKENS, token_dim(emb)), dtype=np.float32)
    mask = np.zeros((n, MAX_BOARD_TOKENS), dtype=np.float32)
    ctx = np.zeros((n, 8), dtype=np.float32)
    hero = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)
    unk = hero_stoi.get("UNKNOWN", 0)
    for i, e in enumerate(examples):
        toks[i], mask[i] = board_tokens(e["minions"], emb, byname)
        ctx[i] = context_vector(neutral_state(e.get("state")))
        hero[i] = hero_stoi.get(e["hero"], unk)
        y[i] = e["label"]
    if ctx_stats is None:
        mean, std = ctx.mean(axis=0), ctx.std(axis=0)
        std[std < 1e-6] = 1.0
        ctx_stats = (mean, std)
    ctx = (ctx - ctx_stats[0]) / ctx_stats[1]
    return toks, mask, ctx.astype(np.float32), hero, y, ctx_stats
