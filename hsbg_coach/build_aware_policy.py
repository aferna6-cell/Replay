"""Build-aware greedy recruit scoring for Simulator Fidelity Phase 2D.

Frozen pre-specified buy valuation (not tuned on evaluation seeds 0–199):

    buy_score = raw_stats - path_adj / BUILD_PATH_BUY_DIVISOR
    raw_stats = attack + health
    path_adj  = path_value(...)[0]   # bounded to ~±1.3 placement units

``BUILD_PATH_BUY_DIVISOR = 5.0`` reuses the numeric divisor from
``draft.rank_discover`` (``padj / 5.0`` added to a lower-is-better equity
ranking). It is **not** a calibrated mapping from placement adjustment to raw
attack+health points. At most ``1.3 / 5 ≈ 0.26`` stat bonus — far below typical
+2–+5 raw-stat shop gaps. Phase 2D tests this exact frozen mapping as a
**negative control** for whether draft-scale path signal moves greedy buys.

Commitment/tempo still come from ``path_value()`` tier weighting internally.
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
    "divisor_note": (
        "Numeric reuse of draft.rank_discover divisor; NOT raw-stat calibration. "
        "Max path bonus ~0.26 stats (1.3/5)."),
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
