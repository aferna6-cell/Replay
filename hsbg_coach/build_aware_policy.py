"""Build-aware greedy recruit scoring for Simulator Fidelity Phase 2D.

Frozen pre-specified buy valuation (not tuned on evaluation seeds 0–199):

    buy_score = raw_stats + path_bonus_stats
    raw_stats = attack + health
    path_bonus_stats = -path_adj / BUILD_PATH_BUY_DIVISOR

``path_adj`` is the first return value of ``path_value()`` (negative = advances
the inferred winning comp). ``BUILD_PATH_BUY_DIVISOR = 5.0`` matches the path
term scale in ``draft.rank_discover`` (``padj / 5.0``).

Commitment and tempo tradeoffs come from ``path_value()`` itself: early tiers
have near-zero commit weight, off-path penalties apply only once the board is
seeded and tavern tier is high enough.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .build_path import path_value

# Frozen Phase 2D constant — same divisor as draft discover ranking.
BUILD_PATH_BUY_DIVISOR = 5.0

POLICY_ID = "build_aware_greedy_policy"
POLICY_CONFIG_FINGERPRINT = {
    "policy_id": POLICY_ID,
    "buy_scoring": "raw_stats - path_adj / BUILD_PATH_BUY_DIVISOR",
    "build_path_buy_divisor": BUILD_PATH_BUY_DIVISOR,
    "path_value_module": "hsbg_coach.build_path.path_value",
    "control_policy_id": "greedy_policy",
    "control_buy_scoring": "attack + health",
}


def raw_stat_buy_score(obs: Dict, shop_idx: int) -> float:
    m = obs["shop"][shop_idx]
    return float((m.get("attack") or 0) + (m.get("health") or 0))


def build_aware_buy_score(obs: Dict, shop_idx: int) -> float:
    """Recruit buy score combining raw stats and build-path placement adjustment."""
    m = obs["shop"][shop_idx]
    raw = raw_stat_buy_score(obs, shop_idx)
    tribes = m.get("tribes") or []
    ctribe = tribes[0] if tribes else None
    padj, _ = path_value(
        obs.get("board") or [],
        m.get("name"),
        obs.get("tavern_tier"),
        candidate_tribe=ctribe,
    )
    return raw - padj / BUILD_PATH_BUY_DIVISOR


def explain_buy_scores(obs: Dict, shop_indices) -> Dict[int, Tuple[float, float, float]]:
    """Debug helper: shop_idx -> (raw, path_adj, buy_score)."""
    out = {}
    for i in shop_indices:
        m = obs["shop"][i]
        raw = raw_stat_buy_score(obs, i)
        tribes = m.get("tribes") or []
        ctribe = tribes[0] if tribes else None
        padj, _ = path_value(
            obs.get("board") or [], m.get("name"),
            obs.get("tavern_tier"), candidate_tribe=ctribe)
        out[i] = (raw, padj, raw - padj / BUILD_PATH_BUY_DIVISOR)
    return out
