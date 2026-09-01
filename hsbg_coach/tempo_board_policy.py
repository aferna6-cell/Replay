"""Phase 2H tempo-aware board-management recruit policy.

Realistic candidate policy (not an oracle): when ``infer_target(board).have >= 1``,
score buy and deploy transitions by raw stats plus build progress minus
replacement cost. Nothing is hard-forced.

Frozen λ_build candidates for DEV calibration: {4, 8, 12}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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

POLICY_ID = "tempo_board_greedy_policy"
LAMBDA_BUILD_CANDIDATES = (4, 8, 12)

PHASE_2H_SCREEN_SEED = 3000
PHASE_2H_SCREEN_LOBBIES = 100
PHASE_2H_REPLICATION_SEED = 3100
PHASE_2H_REPLICATION_LOBBIES = 400
PHASE_2H_CONFIRM_SEED = 4000
PHASE_2H_CONFIRM_LOBBIES = 200


def policy_config_fingerprint(lambda_build: float) -> Dict:
    return {
        "policy_id": POLICY_ID,
        "purpose": "Phase 2H realistic tempo-aware board-management candidate",
        "lambda_build": lambda_build,
        "seeded_trigger": "infer_target(board).core_have >= 1",
        "transition_model": (
            "candidate_value = raw_stats + lambda_build * candidate_build_gain; "
            "replacement_cost = repl_raw + lambda_build * repl_build_value; "
            "net = candidate_value - replacement_cost; commit only if net > 0"),
        "tier_commitment": "hsbg_coach.build_path._TIER_COMMIT schedule",
        "control_policy_id": "greedy_policy",
        "oracle_upper_bound_policy_id": "seeded_core_deploy_stress_greedy_policy",
        "calibration_screen": f"seeds {PHASE_2H_SCREEN_SEED}–"
                              f"{PHASE_2H_SCREEN_SEED + PHASE_2H_SCREEN_LOBBIES - 1}",
        "calibration_replication": f"seeds {PHASE_2H_REPLICATION_SEED}–"
                                   f"{PHASE_2H_REPLICATION_SEED + PHASE_2H_REPLICATION_LOBBIES - 1}",
        "confirmation": f"seeds {PHASE_2H_CONFIRM_SEED}–"
                        f"{PHASE_2H_CONFIRM_SEED + PHASE_2H_CONFIRM_LOBBIES - 1}",
    }


def _raw_stats(m: Dict) -> float:
    return float((m.get("attack") or 0) + (m.get("health") or 0))


def _held_names(obs: Dict) -> set:
    names = set(board_names(obs.get("board") or []))
    for m in obs.get("hand") or []:
        n = m.get("name")
        if n:
            names.add(n)
    return names


def _candidate_build_gain(name: Optional[str], fit, tier: int, held: set) -> float:
    if not name or fit is None:
        return 0.0
    commit = _tier_commit(tier)
    if name in fit.arch.core and name not in held:
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
class TransitionChoice:
    action: int
    net_value: float
    kind: str
    raw_sacrifice: float = 0.0
    build_gain: float = 0.0


@dataclass
class TempoBoardPolicyStats:
    build_aware_buys: int = 0
    build_aware_sells: int = 0
    raw_stat_sacrifice_sum: float = 0.0
    build_progress_gain_sum: float = 0.0
    transition_count: int = 0

    def record(self, choice: TransitionChoice) -> None:
        if choice.kind in ("buy", "buy_sell", "deploy_play", "deploy_sell"):
            self.transition_count += 1
            self.build_progress_gain_sum += choice.build_gain
            self.raw_stat_sacrifice_sum += choice.raw_sacrifice
        if choice.kind in ("buy", "buy_sell"):
            self.build_aware_buys += 1
        if choice.kind in ("deploy_sell", "buy_sell"):
            self.build_aware_sells += 1

    def summary(self) -> Dict:
        n = self.transition_count or 1
        return {
            "build_aware_buys": self.build_aware_buys,
            "build_aware_sells": self.build_aware_sells,
            "mean_raw_stat_sacrifice": self.raw_stat_sacrifice_sum / n,
            "mean_build_progress_gain": self.build_progress_gain_sum / n,
            "positive_transitions": self.transition_count,
        }


class TempoBoardGreedyPolicy:
    """Per-seat policy instance — use ``policies_for_lobby(n)`` in play_scripted."""

    def __init__(self, lambda_build: float):
        self.lambda_build = float(lambda_build)
        self.stats = TempoBoardPolicyStats()

    def __call__(self, obs: Dict, mask: List[bool], rng) -> int:
        fit = infer_target(obs.get("board") or [])
        if fit is None or fit.have < 1:
            return _greedy(obs, mask, rng, 0.0)

        tier = int(obs.get("tavern_tier") or 1)
        held = _held_names(obs)
        board = obs.get("board") or []
        hand = obs.get("hand") or []
        shop = obs.get("shop") or []

        best: Optional[TransitionChoice] = None

        def consider(choice: TransitionChoice) -> None:
            nonlocal best
            if choice.net_value <= 0:
                return
            if best is None or choice.net_value > best.net_value:
                best = choice

        if len(board) < MAX_BOARD:
            for hi in range(min(len(hand), N_PLAY)):
                if not mask[A_PLAY0 + hi]:
                    continue
                m = hand[hi]
                name = m.get("name")
                raw = _raw_stats(m)
                gain = _candidate_build_gain(name, fit, tier, held)
                net = _candidate_value(raw, gain, self.lambda_build)
                consider(TransitionChoice(
                    action=A_PLAY0 + hi, net_value=net, kind="deploy_play",
                    build_gain=gain))

        if len(board) >= MAX_BOARD and hand:
            for hi in range(min(len(hand), N_PLAY)):
                hm = hand[hi]
                hname = hm.get("name")
                cand_raw = _raw_stats(hm)
                cand_gain = _candidate_build_gain(hname, fit, tier, held)
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
                    sacrifice = max(0.0, repl_raw - cand_raw)
                    consider(TransitionChoice(
                        action=A_SELL0 + bi, net_value=net, kind="deploy_sell",
                        raw_sacrifice=sacrifice, build_gain=cand_gain))

        buy_slots = [i for i in range(min(len(shop), N_BUY)) if mask[A_BUY0 + i]]
        for si in buy_slots:
            sm = shop[si]
            sname = sm.get("name")
            cand_raw = _raw_stats(sm)
            cand_gain = _candidate_build_gain(sname, fit, tier, held)
            if len(board) < MAX_BOARD:
                net = _candidate_value(cand_raw, cand_gain, self.lambda_build)
                consider(TransitionChoice(
                    action=A_BUY0 + si, net_value=net, kind="buy",
                    build_gain=cand_gain))
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
                    sacrifice = max(0.0, repl_raw - cand_raw)
                    consider(TransitionChoice(
                        action=A_SELL0 + bi, net_value=net, kind="buy_sell",
                        raw_sacrifice=sacrifice, build_gain=cand_gain))

        if best is not None:
            self.stats.record(best)
            return best.action

        target = STANDARD_TAVERN_TIER.get(obs["turn"], 6.0)
        if mask[A_LEVEL] and tier < target - 0.45:
            return A_LEVEL
        if buy_slots and len(board) + len(hand) < MAX_BOARD + 1:
            best_raw = max(buy_slots, key=lambda i: _raw_stats(shop[i]))
            return A_BUY0 + best_raw
        if mask[A_ROLL] and obs["gold"] >= BUY_COST + ROLL_COST:
            return A_ROLL
        return A_END


def policies_for_lobby(lambda_build: float, n: int = 8) -> List[TempoBoardGreedyPolicy]:
    return [TempoBoardGreedyPolicy(lambda_build) for _ in range(n)]


def aggregate_policy_stats(policies: List[TempoBoardGreedyPolicy]) -> Dict:
    agg = TempoBoardPolicyStats()
    for p in policies:
        s = p.stats
        agg.build_aware_buys += s.build_aware_buys
        agg.build_aware_sells += s.build_aware_sells
        agg.raw_stat_sacrifice_sum += s.raw_stat_sacrifice_sum
        agg.build_progress_gain_sum += s.build_progress_gain_sum
        agg.transition_count += s.transition_count
    return agg.summary()


def tempo_board_greedy_policy(obs: Dict, mask: List[bool], rng) -> int:
    """Default λ=8 singleton for imports; confirmation uses frozen ``policies_for_lobby``."""
    return _DEFAULT_TEMPO_POLICY(obs, mask, rng)


_DEFAULT_TEMPO_POLICY = TempoBoardGreedyPolicy(8.0)
