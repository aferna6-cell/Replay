"""Phase 2G oracle — make board room for and deploy seeded cores.

NOT a production policy. Extends Phase 2E buy oracle with one deployment rule:
when a relevant core is stuck in hand on a full board, sell the weakest
**non-core** board minion (never the seed core) so play-first greedy can deploy it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .bg_env import A_SELL0, MAX_BOARD, N_SELL
from .build_path import board_names, infer_target
from .seeded_core_stress_policy import POLICY_CONFIG_FINGERPRINT as _P2E_FP

POLICY_ID = "seeded_core_deploy_stress_greedy_policy"
PHASE_2G_EVAL_SEED_BASE = 2000

POLICY_CONFIG_FINGERPRINT = {
    "policy_id": POLICY_ID,
    "purpose": "Phase 2G board-slot deployment stress test — not production policy",
    "base_oracle": _P2E_FP["policy_id"],
    "buy_oracle": _P2E_FP["action"],
    "deploy_rule": (
        "if infer_target(board).core_have >= 1 and board is full and hand holds "
        "a relevant core not on board and a non-core board minion exists: sell "
        "weakest non-core board minion; then normal play-first greedy with Phase 2E "
        "buy oracle. Never sell seed/core board pieces to make room."),
    "control_policy_id": "seeded_core_stress_greedy_policy",
    "evaluation_seed_base": PHASE_2G_EVAL_SEED_BASE,
    "evaluation_seed_range": "2000-2199",
}


def _board_stat(m: Dict) -> int:
    return (m.get("attack") or 0) + (m.get("health") or 0)


def seeded_core_deploy_sell_action(obs: Dict, mask: List[bool]) -> Optional[int]:
    """Return sell action to free a slot for a hand-held relevant core, or None."""
    board = obs.get("board") or []
    hand = obs.get("hand") or []
    if len(board) < MAX_BOARD:
        return None

    fit = infer_target(board)
    if fit is None or fit.have < 1:
        return None

    core_names = set(fit.arch.core.keys())
    on_board = set(board_names(board))

    hand_has_relevant_core = any(
        m.get("name") in core_names and m.get("name") not in on_board
        for m in hand)
    if not hand_has_relevant_core:
        return None

    non_core_board = [
        i for i, m in enumerate(board)
        if m.get("name") not in core_names]
    if not non_core_board:
        return None

    weakest = min(non_core_board, key=lambda i: _board_stat(board[i]))
    action = A_SELL0 + weakest
    if action < A_SELL0 + N_SELL and mask[action]:
        return action
    return None
