"""Phase 2E oracle/stress policy — seeded-core conversion diagnostic.

NOT a proposed production policy. When ``infer_target(board).core_have >= 1``
and a legally buyable **missing** core card for that target exists, buy the
highest-raw-stat matching core. Otherwise identical to raw-stat greedy.

Frozen evaluation seeds: ``1000–1199`` (200 lobbies, base seed 1000).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .build_aware_policy import raw_stat_buy_score
from .build_path import board_names, infer_target

POLICY_ID = "seeded_core_stress_greedy_policy"
PHASE_2E_EVAL_SEED_BASE = 1000

POLICY_CONFIG_FINGERPRINT = {
    "policy_id": POLICY_ID,
    "purpose": "Phase 2E oracle/stress test — not production policy",
    "trigger": "infer_target(board).core_have >= 1",
    "action": (
        "among legally buyable shop slots, if any missing core of current "
        "target exists, buy max(raw_stats) matching core (overrides roll when "
        "legal mask permits buy); else raw-stat greedy"),
    "control_policy_id": "greedy_policy",
    "evaluation_seed_base": PHASE_2E_EVAL_SEED_BASE,
    "evaluation_seed_range": "1000-1199",
}


def _held_names(obs: Dict) -> set:
    return set(board_names(obs.get("board") or [])) | set(
        board_names(obs.get("hand") or []))


def seeded_core_buy_override(obs: Dict, mask: List[bool],
                             legal_buy_indices: List[int]) -> Optional[int]:
    """Return shop index to buy, or None to fall back to normal greedy."""
    fit = infer_target(obs.get("board") or [])
    if fit is None or fit.have < 1:
        return None
    held = _held_names(obs)
    core = fit.arch.core
    matching = [
        i for i in legal_buy_indices
        if obs["shop"][i].get("name") in core
        and obs["shop"][i].get("name") not in held]
    if not matching:
        return None
    return max(matching, key=lambda i: raw_stat_buy_score(obs, i))
