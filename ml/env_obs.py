"""Encode a bg_env observation for the policy net.

The policy sees the whole decision context as tokens: board (7) ∪ shop (7) ∪
hand (10), each minion a `tokens.py` token plus a zone id so attention knows
what's owned, offered, or in hand — the buy decision is literally attention
between shop tokens and board tokens. Scalar state (turn, gold, tier, HP,
lobby) rides in a context vector.
"""

from typing import Dict, Optional, Tuple

import math
import numpy as np

from hsbg_coach import cards
from hsbg_coach.bg_env import N_ACTIONS  # re-exported for the trainers
from .board_features import minion_from_snapshot
from .tokens import minion_token, token_dim, fill_relative_stats

ZONE_BOARD, ZONE_SHOP, ZONE_HAND = 0, 1, 2
N_ZONES = 3
MAX_BOARD_T, MAX_SHOP_T, MAX_HAND_T = 7, 7, 10
N_TOKENS = MAX_BOARD_T + MAX_SHOP_T + MAX_HAND_T
POLICY_CTX_DIM = 8

_BYNAME = None


def _byname():
    global _BYNAME
    if _BYNAME is None:
        _BYNAME = cards.by_name(cards.load_kb())
    return _BYNAME


def obs_context(obs: Dict) -> np.ndarray:
    return np.array([
        (obs.get("turn") or 0) / 20.0,
        (obs.get("tavern_tier") or 1) / 6.0,
        (obs.get("gold") or 0) / 10.0,
        (obs.get("hero_health") or 0) / 40.0,
        (obs.get("players_alive") or 8) / 8.0,
        # `or` would turn a FREE tier-up (cost 0) into 10 — use an explicit
        # None check; 10 only stands in for "can't level" (tier 6).
        (10 if obs.get("level_cost") is None else obs["level_cost"]) / 10.0,
        1.0 if obs.get("frozen") else 0.0,
        math.log1p(max(obs.get("max_opp_strength") or 0, 0)) / 10.0,
    ], dtype=np.float32)


def encode_obs(obs: Dict, emb: Dict, byname: Optional[Dict] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """obs dict -> (tokens[24,F], mask[24], zones[24], ctx[8])."""
    byname = byname if byname is not None else _byname()
    dim = token_dim(emb)
    toks = np.zeros((N_TOKENS, dim), dtype=np.float32)
    mask = np.zeros(N_TOKENS, dtype=np.float32)
    zones = np.zeros(N_TOKENS, dtype=np.int64)
    layout = [("board", ZONE_BOARD, 0, MAX_BOARD_T),
              ("shop", ZONE_SHOP, MAX_BOARD_T, MAX_SHOP_T),
              ("hand", ZONE_HAND, MAX_BOARD_T + MAX_SHOP_T, MAX_HAND_T)]
    raw_by_slot: list = [{} for _ in range(N_TOKENS)]
    for key, zone, offset, cap in layout:
        for i, raw in enumerate((obs.get(key) or [])[:cap]):
            norm = minion_from_snapshot(raw, byname)
            if norm is None:
                continue
            toks[offset + i] = minion_token(norm, emb, byname)
            mask[offset + i] = 1.0
            zones[offset + i] = zone
            raw_by_slot[offset + i] = norm
    # Relative stats across the WHOLE view (board ∪ shop ∪ hand): the buy
    # decision is exactly "which of these is biggest relative to what I own."
    fill_relative_stats(toks, mask, raw_by_slot)
    return toks, mask, zones, obs_context(obs)
