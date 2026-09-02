"""Phase 2J board-relative multi-turn opportunity-cost recruit policy (2j_v1).

Changes only the transition-cost formulation vs Phase 2H:

    raw_loss = max(0, replacement_raw - candidate_raw)
    relative_tempo_loss = raw_loss / max(board_total_raw, 1)
    opportunity_cost = relative_tempo_loss * persistence_weight
    build_delta = candidate_build_gain - replacement_build_value
    transition_score = build_delta - alpha * opportunity_cost

Commit only when transition_score > 0. Free-slot opportunity_cost = 0.
No λ. Compound sell→play / sell→buy→play semantics unchanged.
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
from .build_path import infer_target
from .persistence_prior import (
    PersistencePrior,
    empty_prior,
    rank_band_for_index,
    raw_stats as _raw_stats,
)
from .tempo_board_policy import (
    PendingTransition,
    _board_name_set,
    _deploy_build_gain,
    _held_names,
    _replacement_build_value,
    _shop_build_gain,
)

METHODOLOGY_VERSION = "2j_v1"
POLICY_ID = "board_opportunity_cost_policy"
ALPHA_CANDIDATES = (0.5, 1.0, 2.0)

PHASE_2J_FIT_SEED = 7000
PHASE_2J_FIT_LOBBIES = 300
PHASE_2J_SCREEN_SEED = 7300
PHASE_2J_SCREEN_LOBBIES = 100
PHASE_2J_REPLICATION_SEED = 7400
PHASE_2J_REPLICATION_LOBBIES = 400
PHASE_2J_CONFIRM_SEED = 8000
PHASE_2J_CONFIRM_LOBBIES = 200


def board_total_raw(board: List[Dict]) -> float:
    return sum(_raw_stats(m) for m in board)


def relative_tempo_loss(*, cand_raw: float, repl_raw: float,
                        board_total: float) -> float:
    raw_loss = max(0.0, repl_raw - cand_raw)
    return raw_loss / max(board_total, 1.0)


def opportunity_cost(*, cand_raw: float, repl_raw: float, board_total: float,
                     persistence_weight: float) -> float:
    return relative_tempo_loss(
        cand_raw=cand_raw, repl_raw=repl_raw, board_total=board_total
    ) * persistence_weight


def transition_score(*, build_delta: float, opp_cost: float,
                     alpha: float) -> float:
    return build_delta - alpha * opp_cost


def policy_config_fingerprint(alpha: float, prior: PersistencePrior) -> Dict:
    return {
        "policy_id": POLICY_ID,
        "methodology_version": METHODOLOGY_VERSION,
        "purpose": "Phase 2J board-relative multi-turn opportunity-cost policy",
        "alpha": alpha,
        "lambda_build": None,
        "seeded_trigger": "infer_target(board).core_have >= 1",
        "transition_model": (
            "raw_loss=max(0,repl_raw-cand_raw); "
            "relative_tempo_loss=raw_loss/max(board_total_raw,1); "
            "opportunity_cost=relative_tempo_loss*persistence_weight; "
            "build_delta=cand_build-repl_build; "
            "score=build_delta-alpha*opportunity_cost; commit iff score>0; "
            "free-slot opportunity_cost=0; "
            "hand full-board: sell→play; shop full-board: sell→buy→play"),
        "persistence_prior": {
            "methodology_version": prior.methodology_version,
            "survival_horizon": prior.survival_horizon,
            "weight_1": prior.weight_1,
            "weight_2": prior.weight_2,
            "fit_seed_base": prior.fit_seed_base,
            "fit_lobbies": prior.fit_lobbies,
            "n_cells": len(prior.cells),
            "global_p_survive_1": prior.global_p_survive_1,
            "global_p_survive_2": prior.global_p_survive_2,
            "prior_hash_sha256": prior.content_hash_sha256(),
        },
        "acquisition_gain": "missing core vs board+hand (shop buys)",
        "deployment_gain": "missing core vs board only (hand→board)",
        "tier_commitment": "hsbg_coach.build_path._TIER_COMMIT schedule",
        "control_policy_id": "greedy_policy",
        "oracle_stress_policy_id": "seeded_core_deploy_stress_greedy_policy",
        "persistence_fit": f"seeds {PHASE_2J_FIT_SEED}–"
                           f"{PHASE_2J_FIT_SEED + PHASE_2J_FIT_LOBBIES - 1}",
        "calibration_screen": f"seeds {PHASE_2J_SCREEN_SEED}–"
                              f"{PHASE_2J_SCREEN_SEED + PHASE_2J_SCREEN_LOBBIES - 1}",
        "calibration_replication": (
            f"seeds {PHASE_2J_REPLICATION_SEED}–"
            f"{PHASE_2J_REPLICATION_SEED + PHASE_2J_REPLICATION_LOBBIES - 1}"),
        "confirmation": f"seeds {PHASE_2J_CONFIRM_SEED}–"
                        f"{PHASE_2J_CONFIRM_SEED + PHASE_2J_CONFIRM_LOBBIES - 1}",
        "alpha_candidates": list(ALPHA_CANDIDATES),
    }


@dataclass
class BoardOpportunityStats:
    compound_transitions_planned: int = 0
    compound_transitions_completed: int = 0
    compound_transitions_abandoned: int = 0
    replacement_sells: int = 0
    tempo_selected_buys: int = 0
    tempo_selected_deploys: int = 0
    target_core_buys: int = 0
    target_core_deploys: int = 0
    raw_stat_sacrifice_completed_sum: float = 0.0
    build_gain_completed_sum: float = 0.0
    relative_tempo_loss_sum: float = 0.0
    relative_tempo_loss_values: List[float] = field(default_factory=list)
    persistence_weight_sum: float = 0.0
    opportunity_cost_sum: float = 0.0
    replacement_transitions: int = 0

    def record_planned_sell(self, pending: PendingTransition,
                            *, rel_loss: float, p_weight: float,
                            opp_cost: float) -> None:
        self.compound_transitions_planned += 1
        self.replacement_sells += 1
        self.replacement_transitions += 1
        self.relative_tempo_loss_sum += rel_loss
        self.relative_tempo_loss_values.append(rel_loss)
        self.persistence_weight_sum += p_weight
        self.opportunity_cost_sum += opp_cost

    def record_buy_step(self, build_gain: float) -> None:
        self.tempo_selected_buys += 1
        if build_gain > 0:
            self.target_core_buys += 1

    def record_play_step(self, build_gain: float, *, compound: bool,
                         pending: Optional[PendingTransition] = None) -> None:
        self.tempo_selected_deploys += 1
        if build_gain > 0:
            self.target_core_deploys += 1
        if compound and pending is not None:
            self.compound_transitions_completed += 1
            self.raw_stat_sacrifice_completed_sum += pending.raw_sacrifice
            self.build_gain_completed_sum += pending.build_gain

    def record_abandoned(self) -> None:
        self.compound_transitions_abandoned += 1

    def summary(self) -> Dict:
        planned = self.compound_transitions_planned
        completed = self.compound_transitions_completed
        n_repl = self.replacement_transitions
        vals = sorted(self.relative_tempo_loss_values)
        p95 = None
        if vals:
            idx = min(len(vals) - 1, int(0.95 * (len(vals) - 1)))
            p95 = vals[idx]
        return {
            "compound_transitions_planned": planned,
            "compound_transitions_completed": completed,
            "compound_transitions_abandoned": self.compound_transitions_abandoned,
            "compound_transitions_completion_rate": (
                completed / planned if planned else None),
            "replacement_sells": self.replacement_sells,
            "replacement_transitions": n_repl,
            "tempo_selected_buys": self.tempo_selected_buys,
            "tempo_selected_deploys": self.tempo_selected_deploys,
            "target_core_buys": self.target_core_buys,
            "target_core_deploys": self.target_core_deploys,
            "mean_raw_stat_sacrifice_completed": (
                self.raw_stat_sacrifice_completed_sum / completed
                if completed else None),
            "mean_build_gain_completed": (
                self.build_gain_completed_sum / completed
                if completed else None),
            "mean_relative_tempo_loss": (
                self.relative_tempo_loss_sum / n_repl if n_repl else None),
            "p95_relative_tempo_loss": p95,
            "mean_persistence_weight": (
                self.persistence_weight_sum / n_repl if n_repl else None),
            "mean_opportunity_cost": (
                self.opportunity_cost_sum / n_repl if n_repl else None),
        }


class BoardOpportunityCostPolicy:
    """Per-seat Phase 2J policy — fresh instance per lobby."""

    def __init__(self, alpha: float, prior: PersistencePrior):
        self.alpha = float(alpha)
        self.prior = prior
        self.stats = BoardOpportunityStats()
        self.pending: Optional[PendingTransition] = None

    @staticmethod
    def _hand_slot_for_name(hand: List[Dict], name: str) -> Optional[int]:
        for i, m in enumerate(hand):
            if m.get("name") == name:
                return i
        return None

    def _abandon_pending(self) -> None:
        if self.pending is not None:
            self.stats.record_abandoned()
            self.pending = None

    def _try_complete_pending(self, obs: Dict, mask: List[bool]) -> Optional[int]:
        p = self.pending
        if p is None:
            return None
        hand = obs.get("hand") or []
        shop = obs.get("shop") or []

        if p.stage == "buy":
            action = A_BUY0 + p.candidate_slot
            if (action < A_BUY0 + N_BUY and mask[action]
                    and p.candidate_slot < len(shop)
                    and shop[p.candidate_slot].get("name") == p.candidate_name):
                self.stats.record_buy_step(p.build_gain)
                self.pending = PendingTransition(
                    source=p.source, stage="play", candidate_slot=-1,
                    candidate_name=p.candidate_name, replacement_slot=None,
                    net_value=p.net_value, build_gain=p.build_gain,
                    raw_sacrifice=p.raw_sacrifice)
                return action
            self._abandon_pending()
            return None

        if p.stage == "play":
            hi = (p.candidate_slot if p.candidate_slot >= 0
                  else self._hand_slot_for_name(hand, p.candidate_name))
            if hi is None:
                self._abandon_pending()
                return None
            action = A_PLAY0 + hi
            if (action < A_PLAY0 + N_PLAY and mask[action]
                    and hi < len(hand)
                    and hand[hi].get("name") == p.candidate_name):
                completed = self.pending
                self.pending = None
                self.stats.record_play_step(
                    p.build_gain, compound=True, pending=completed)
                return action
            self._abandon_pending()
            return None

        self._abandon_pending()
        return None

    def _score_replacement(self, *, cand_raw: float, cand_gain: float,
                           board: List[Dict], bi: int, fit, tier: int
                           ) -> tuple[float, float, float, float]:
        bm = board[bi]
        repl_raw = _raw_stats(bm)
        repl_gain = _replacement_build_value(bm.get("name"), fit, tier)
        total = board_total_raw(board)
        raws = [_raw_stats(m) for m in board]
        rb = rank_band_for_index(raws, bi)
        is_core = bm.get("name") in fit.arch.core
        p_weight = self.prior.persistence_weight(
            tier=tier, rank=rb, is_core=is_core)
        rel = relative_tempo_loss(
            cand_raw=cand_raw, repl_raw=repl_raw, board_total=total)
        opp = opportunity_cost(
            cand_raw=cand_raw, repl_raw=repl_raw, board_total=total,
            persistence_weight=p_weight)
        build_delta = cand_gain - repl_gain
        score = transition_score(
            build_delta=build_delta, opp_cost=opp, alpha=self.alpha)
        return score, rel, p_weight, opp

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
        best_score = 0.0
        best_pending: Optional[PendingTransition] = None
        best_direct_gain = 0.0
        best_direct_kind: Optional[str] = None
        best_meta: Optional[tuple] = None  # rel, p_weight, opp

        # Free-slot: opportunity_cost = 0 → score = build_delta = cand_gain
        if len(board) < MAX_BOARD:
            for hi in range(min(len(hand), N_PLAY)):
                if not mask[A_PLAY0 + hi]:
                    continue
                m = hand[hi]
                name = m.get("name")
                gain = _deploy_build_gain(name, fit, tier, on_board)
                score = gain  # build_delta, opp=0
                if score > best_score:
                    best_score = score
                    best_action = A_PLAY0 + hi
                    best_pending = None
                    best_direct_gain = gain
                    best_direct_kind = "play"
                    best_meta = None

        if len(board) >= MAX_BOARD and hand:
            for hi in range(min(len(hand), N_PLAY)):
                hm = hand[hi]
                hname = hm.get("name")
                cand_raw = _raw_stats(hm)
                cand_gain = _deploy_build_gain(hname, fit, tier, on_board)
                for bi in range(min(len(board), N_SELL)):
                    if not mask[A_SELL0 + bi]:
                        continue
                    score, rel, pw, opp = self._score_replacement(
                        cand_raw=cand_raw, cand_gain=cand_gain,
                        board=board, bi=bi, fit=fit, tier=tier)
                    if score > best_score:
                        best_score = score
                        best_action = A_SELL0 + bi
                        repl_raw = _raw_stats(board[bi])
                        best_pending = PendingTransition(
                            source="hand", stage="play", candidate_slot=hi,
                            candidate_name=hname, replacement_slot=bi,
                            net_value=score, build_gain=cand_gain,
                            raw_sacrifice=max(0.0, repl_raw - cand_raw))
                        best_direct_kind = None
                        best_meta = (rel, pw, opp)

        buy_slots = [i for i in range(min(len(shop), N_BUY)) if mask[A_BUY0 + i]]
        for si in buy_slots:
            sm = shop[si]
            sname = sm.get("name")
            cand_raw = _raw_stats(sm)
            cand_gain = _shop_build_gain(sname, fit, tier, held)
            if len(board) < MAX_BOARD:
                score = cand_gain
                if score > best_score:
                    best_score = score
                    best_action = A_BUY0 + si
                    best_pending = None
                    best_direct_gain = cand_gain
                    best_direct_kind = "buy"
                    best_meta = None
            else:
                for bi in range(min(len(board), N_SELL)):
                    if not mask[A_SELL0 + bi]:
                        continue
                    score, rel, pw, opp = self._score_replacement(
                        cand_raw=cand_raw, cand_gain=cand_gain,
                        board=board, bi=bi, fit=fit, tier=tier)
                    if score > best_score:
                        best_score = score
                        best_action = A_SELL0 + bi
                        repl_raw = _raw_stats(board[bi])
                        best_pending = PendingTransition(
                            source="shop", stage="buy", candidate_slot=si,
                            candidate_name=sname, replacement_slot=bi,
                            net_value=score, build_gain=cand_gain,
                            raw_sacrifice=max(0.0, repl_raw - cand_raw))
                        best_direct_kind = None
                        best_meta = (rel, pw, opp)

        if best_action is not None and best_score > 0:
            if best_pending is not None:
                self.pending = best_pending
                rel, pw, opp = best_meta or (0.0, 0.0, 0.0)
                self.stats.record_planned_sell(
                    best_pending, rel_loss=rel, p_weight=pw, opp_cost=opp)
                return best_action
            if best_direct_kind == "play":
                self.stats.record_play_step(best_direct_gain, compound=False)
            elif best_direct_kind == "buy":
                self.stats.record_buy_step(best_direct_gain)
            return best_action

        target = STANDARD_TAVERN_TIER.get(obs["turn"], 6.0)
        if mask[A_LEVEL] and tier < target - 0.45:
            return A_LEVEL
        if buy_slots and len(board) + len(hand) < MAX_BOARD + 1:
            return A_BUY0 + max(buy_slots, key=lambda i: _raw_stats(shop[i]))
        if mask[A_ROLL] and obs["gold"] >= BUY_COST + ROLL_COST:
            return A_ROLL
        return A_END


def policies_for_lobby(alpha: float, prior: PersistencePrior,
                       n: int = 8) -> List[BoardOpportunityCostPolicy]:
    return [BoardOpportunityCostPolicy(alpha, prior) for _ in range(n)]


def aggregate_policy_stats(
        policies: List[BoardOpportunityCostPolicy]) -> Dict:
    agg = BoardOpportunityStats()
    for p in policies:
        s = p.stats
        agg.compound_transitions_planned += s.compound_transitions_planned
        agg.compound_transitions_completed += s.compound_transitions_completed
        agg.compound_transitions_abandoned += s.compound_transitions_abandoned
        agg.replacement_sells += s.replacement_sells
        agg.tempo_selected_buys += s.tempo_selected_buys
        agg.tempo_selected_deploys += s.tempo_selected_deploys
        agg.target_core_buys += s.target_core_buys
        agg.target_core_deploys += s.target_core_deploys
        agg.raw_stat_sacrifice_completed_sum += s.raw_stat_sacrifice_completed_sum
        agg.build_gain_completed_sum += s.build_gain_completed_sum
        agg.relative_tempo_loss_sum += s.relative_tempo_loss_sum
        agg.relative_tempo_loss_values.extend(s.relative_tempo_loss_values)
        agg.persistence_weight_sum += s.persistence_weight_sum
        agg.opportunity_cost_sum += s.opportunity_cost_sum
        agg.replacement_transitions += s.replacement_transitions
    return agg.summary()
