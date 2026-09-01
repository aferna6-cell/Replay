"""Phase 2H tempo-aware board-management recruit policy (methodology 2h_v2).

Realistic candidate policy (not an oracle): when ``infer_target(board).have >= 1``,
score buy and deploy transitions by raw stats plus build progress minus
replacement cost. Compound sell→play / sell→buy transitions are latched.

Frozen λ_build candidates for DEV calibration: {4, 8, 12}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bg_env import (
    A_BUY0,
    A_END,
    A_LEVEL,
    A_PLAY0,
    A_ROLL,
    A_SELL0,
    BUY_COST,
    MAX_BOARD,
    N_BUY,
    N_PLAY,
    N_SELL,
    ROLL_COST,
    STANDARD_TAVERN_TIER,
    _greedy,
)
from .build_path import _tier_commit, board_names, infer_target

METHODOLOGY_VERSION = "2h_v2"
POLICY_ID = "tempo_board_greedy_policy"
LAMBDA_BUILD_CANDIDATES = (4, 8, 12)

PHASE_2H_SCREEN_SEED = 3000
PHASE_2H_SCREEN_LOBBIES = 100
PHASE_2H_REPLICATION_SEED = 3100
PHASE_2H_REPLICATION_LOBBIES = 400
PHASE_2H_CONFIRM_SEED = 5000
PHASE_2H_CONFIRM_LOBBIES = 200

# Invalidated v1 confirmation (trace lobby-ID collapse + incomplete transitions).
PHASE_2H_INVALIDATED_V1_CONFIRM_SEED = 4000
PHASE_2H_INVALIDATED_V1_CONFIRM_LOBBIES = 200


def policy_config_fingerprint(lambda_build: float) -> Dict:
    return {
        "policy_id": POLICY_ID,
        "methodology_version": METHODOLOGY_VERSION,
        "purpose": "Phase 2H realistic tempo-aware board-management candidate",
        "lambda_build": lambda_build,
        "seeded_trigger": "infer_target(board).core_have >= 1",
        "transition_model": (
            "candidate_value = raw_stats + lambda_build * candidate_build_gain; "
            "replacement_cost = repl_raw + lambda_build * repl_build_value; "
            "net = candidate_value - replacement_cost; commit only if net > 0; "
            "compound sell→play/buy latched via pending intent"),
        "acquisition_gain": "missing core vs board+hand (shop buys)",
        "deployment_gain": "missing core vs board only (hand→board)",
        "tier_commitment": "hsbg_coach.build_path._TIER_COMMIT schedule",
        "control_policy_id": "greedy_policy",
        "oracle_upper_bound_policy_id": "seeded_core_deploy_stress_greedy_policy",
        "calibration_screen": f"seeds {PHASE_2H_SCREEN_SEED}–"
                              f"{PHASE_2H_SCREEN_SEED + PHASE_2H_SCREEN_LOBBIES - 1}",
        "calibration_replication": f"seeds {PHASE_2H_REPLICATION_SEED}–"
                                   f"{PHASE_2H_REPLICATION_SEED + PHASE_2H_REPLICATION_LOBBIES - 1}",
        "confirmation": f"seeds {PHASE_2H_CONFIRM_SEED}–"
                        f"{PHASE_2H_CONFIRM_SEED + PHASE_2H_CONFIRM_LOBBIES - 1}",
        "invalidated_v1_confirmation": (
            f"seeds {PHASE_2H_INVALIDATED_V1_CONFIRM_SEED}–"
            f"{PHASE_2H_INVALIDATED_V1_CONFIRM_SEED + PHASE_2H_INVALIDATED_V1_CONFIRM_LOBBIES - 1} "
            "(lobby-ID collapse + non-atomic transitions)"),
    }


def _raw_stats(m: Dict) -> float:
    return float((m.get("attack") or 0) + (m.get("health") or 0))


def _board_name_set(board: List[Dict]) -> set:
    return set(board_names(board))


def _held_names(obs: Dict) -> set:
    names = _board_name_set(obs.get("board") or [])
    for m in obs.get("hand") or []:
        n = m.get("name")
        if n:
            names.add(n)
    return names


def _shop_build_gain(name: Optional[str], fit, tier: int, held: set) -> float:
    """Acquisition: reward missing cores not already on board or in hand."""
    if not name or fit is None:
        return 0.0
    commit = _tier_commit(tier)
    if name in fit.arch.core and name not in held:
        return float(fit.arch.core[name]) * commit
    return 0.0


def _deploy_build_gain(name: Optional[str], fit, tier: int,
                       on_board: set) -> float:
    """Deployment: reward cores in hand that are not yet on the board."""
    if not name or fit is None:
        return 0.0
    commit = _tier_commit(tier)
    if name in fit.arch.core and name not in on_board:
        return float(fit.arch.core[name]) * commit
    return 0.0


def _replacement_build_value(name: Optional[str], fit, tier: int) -> float:
    if not name or fit is None:
        return 0.0
    if name in fit.arch.core:
        return float(fit.arch.core[name]) * _tier_commit(tier)
    return 0.0


def _candidate_value(raw: float, build_gain: float, lambda_build: float) -> float:
    return raw + lambda_build * build_gain


def _net_transition(*, cand_raw: float, cand_gain: float,
                    repl_raw: float, repl_gain: float,
                    lambda_build: float) -> float:
    return (_candidate_value(cand_raw, cand_gain, lambda_build)
            - _candidate_value(repl_raw, repl_gain, lambda_build))


@dataclass
class PendingTransition:
    source: str  # "hand" | "shop"
    candidate_slot: int
    candidate_name: str
    replacement_slot: Optional[int]
    net_value: float
    build_gain: float
    raw_sacrifice: float


@dataclass
class TempoBoardPolicyStats:
    transitions_planned: int = 0
    transitions_completed: int = 0
    core_buys_completed: int = 0
    core_deploys_completed: int = 0
    slot_sells_completed: int = 0
    raw_stat_sacrifice_completed_sum: float = 0.0
    build_gain_completed_sum: float = 0.0

    def record_planned_sell(self, pending: PendingTransition) -> None:
        self.transitions_planned += 1
        self.slot_sells_completed += 1

    def record_completed(self, pending: PendingTransition) -> None:
        self.transitions_completed += 1
        self.raw_stat_sacrifice_completed_sum += pending.raw_sacrifice
        self.build_gain_completed_sum += pending.build_gain
        if pending.source == "hand":
            self.core_deploys_completed += 1
        else:
            self.core_buys_completed += 1

    def record_direct_play(self, build_gain: float) -> None:
        self.transitions_completed += 1
        self.core_deploys_completed += 1
        self.build_gain_completed_sum += build_gain

    def record_direct_buy(self, build_gain: float) -> None:
        self.transitions_completed += 1
        self.core_buys_completed += 1
        self.build_gain_completed_sum += build_gain

    def summary(self) -> Dict:
        n = self.transitions_completed or 1
        return {
            "transitions_planned": self.transitions_planned,
            "transitions_completed": self.transitions_completed,
            "core_buys_completed": self.core_buys_completed,
            "core_deploys_completed": self.core_deploys_completed,
            "slot_sells_completed": self.slot_sells_completed,
            "mean_raw_stat_sacrifice_completed": (
                self.raw_stat_sacrifice_completed_sum / n),
            "mean_build_gain_completed": self.build_gain_completed_sum / n,
        }


class TempoBoardGreedyPolicy:
    """Per-seat policy — use ``policies_for_lobby(n)`` in play_scripted."""

    def __init__(self, lambda_build: float):
        self.lambda_build = float(lambda_build)
        self.stats = TempoBoardPolicyStats()
        self.pending: Optional[PendingTransition] = None

    def _try_complete_pending(self, obs: Dict, mask: List[bool]) -> Optional[int]:
        p = self.pending
        if p is None:
            return None
        if p.source == "hand":
            action = A_PLAY0 + p.candidate_slot
            hand = obs.get("hand") or []
            if (action < A_PLAY0 + N_PLAY and mask[action]
                    and p.candidate_slot < len(hand)
                    and hand[p.candidate_slot].get("name") == p.candidate_name):
                self.pending = None
                self.stats.record_completed(p)
                return action
        elif p.source == "shop":
            action = A_BUY0 + p.candidate_slot
            shop = obs.get("shop") or []
            if (action < A_BUY0 + N_BUY and mask[action]
                    and p.candidate_slot < len(shop)
                    and shop[p.candidate_slot].get("name") == p.candidate_name):
                self.pending = None
                self.stats.record_completed(p)
                return action
        self.pending = None
        return None

    def __call__(self, obs: Dict, mask: List[bool], rng) -> int:
        complete = self._try_complete_pending(obs, mask)
        if complete is not None:
            return complete

        fit = infer_target(obs.get("board") or [])
        if fit is None or fit.have < 1:
            return _greedy(obs, mask, rng, 0.0)

        tier = int(obs.get("tavern_tier") or 1)
        held = _held_names(obs)
        on_board = _board_name_set(obs.get("board") or [])
        board = obs.get("board") or []
        hand = obs.get("hand") or []
        shop = obs.get("shop") or []

        best_action: Optional[int] = None
        best_net = 0.0
        best_pending: Optional[PendingTransition] = None
        best_direct_gain = 0.0
        best_direct_kind: Optional[str] = None  # "play" | "buy"

        if len(board) < MAX_BOARD:
            for hi in range(min(len(hand), N_PLAY)):
                if not mask[A_PLAY0 + hi]:
                    continue
                m = hand[hi]
                name = m.get("name")
                gain = _deploy_build_gain(name, fit, tier, on_board)
                net = _candidate_value(_raw_stats(m), gain, self.lambda_build)
                if net > best_net:
                    best_net = net
                    best_action = A_PLAY0 + hi
                    best_pending = None
                    best_direct_gain = gain
                    best_direct_kind = "play"

        if len(board) >= MAX_BOARD and hand:
            for hi in range(min(len(hand), N_PLAY)):
                hm = hand[hi]
                hname = hm.get("name")
                cand_raw = _raw_stats(hm)
                cand_gain = _deploy_build_gain(hname, fit, tier, on_board)
                for bi in range(min(len(board), N_SELL)):
                    if not mask[A_SELL0 + bi]:
                        continue
                    bm = board[bi]
                    repl_raw = _raw_stats(bm)
                    repl_gain = _replacement_build_value(bm.get("name"), fit, tier)
                    net = _net_transition(
                        cand_raw=cand_raw, cand_gain=cand_gain,
                        repl_raw=repl_raw, repl_gain=repl_gain,
                        lambda_build=self.lambda_build)
                    if net > best_net:
                        best_net = net
                        best_action = A_SELL0 + bi
                        best_pending = PendingTransition(
                            source="hand", candidate_slot=hi,
                            candidate_name=hname, replacement_slot=bi,
                            net_value=net, build_gain=cand_gain,
                            raw_sacrifice=max(0.0, repl_raw - cand_raw))
                        best_direct_kind = None

        buy_slots = [i for i in range(min(len(shop), N_BUY)) if mask[A_BUY0 + i]]
        for si in buy_slots:
            sm = shop[si]
            sname = sm.get("name")
            cand_raw = _raw_stats(sm)
            cand_gain = _shop_build_gain(sname, fit, tier, held)
            if len(board) < MAX_BOARD:
                net = _candidate_value(cand_raw, cand_gain, self.lambda_build)
                if net > best_net:
                    best_net = net
                    best_action = A_BUY0 + si
                    best_pending = None
                    best_direct_gain = cand_gain
                    best_direct_kind = "buy"
            else:
                for bi in range(min(len(board), N_SELL)):
                    if not mask[A_SELL0 + bi]:
                        continue
                    bm = board[bi]
                    repl_raw = _raw_stats(bm)
                    repl_gain = _replacement_build_value(bm.get("name"), fit, tier)
                    net = _net_transition(
                        cand_raw=cand_raw, cand_gain=cand_gain,
                        repl_raw=repl_raw, repl_gain=repl_gain,
                        lambda_build=self.lambda_build)
                    if net > best_net:
                        best_net = net
                        best_action = A_BUY0 + si
                        best_pending = PendingTransition(
                            source="shop", candidate_slot=si,
                            candidate_name=sname, replacement_slot=bi,
                            net_value=net, build_gain=cand_gain,
                            raw_sacrifice=max(0.0, repl_raw - cand_raw))
                        best_direct_kind = None

        if best_action is not None and best_net > 0:
            if best_pending is not None:
                self.pending = best_pending
                self.stats.record_planned_sell(best_pending)
                return best_action
            if best_direct_kind == "play":
                self.stats.record_direct_play(best_direct_gain)
            elif best_direct_kind == "buy":
                self.stats.record_direct_buy(best_direct_gain)
            return best_action

        target = STANDARD_TAVERN_TIER.get(obs["turn"], 6.0)
        if mask[A_LEVEL] and tier < target - 0.45:
            return A_LEVEL
        if buy_slots and len(board) + len(hand) < MAX_BOARD + 1:
            return A_BUY0 + max(buy_slots, key=lambda i: _raw_stats(shop[i]))
        if mask[A_ROLL] and obs["gold"] >= BUY_COST + ROLL_COST:
            return A_ROLL
        return A_END


def policies_for_lobby(lambda_build: float, n: int = 8) -> List[TempoBoardGreedyPolicy]:
    return [TempoBoardGreedyPolicy(lambda_build) for _ in range(n)]


def aggregate_policy_stats(policies: List[TempoBoardGreedyPolicy]) -> Dict:
    agg = TempoBoardPolicyStats()
    for p in policies:
        s = p.stats
        agg.transitions_planned += s.transitions_planned
        agg.transitions_completed += s.transitions_completed
        agg.core_buys_completed += s.core_buys_completed
        agg.core_deploys_completed += s.core_deploys_completed
        agg.slot_sells_completed += s.slot_sells_completed
        agg.raw_stat_sacrifice_completed_sum += s.raw_stat_sacrifice_completed_sum
        agg.build_gain_completed_sum += s.build_gain_completed_sum
    return agg.summary()


def tempo_board_greedy_policy(obs: Dict, mask: List[bool], rng) -> int:
    """Default λ=8 singleton for imports; confirmation uses ``policies_for_lobby``."""
    return _DEFAULT_TEMPO_POLICY(obs, mask, rng)


_DEFAULT_TEMPO_POLICY = TempoBoardGreedyPolicy(8.0)
