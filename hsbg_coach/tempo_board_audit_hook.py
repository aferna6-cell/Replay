"""Build observational audit snapshots from tempo policy state (Phase 2I).

Scoring mirrors ``TempoBoardGreedyPolicy`` logic; used only when audit is enabled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .bg_env import A_BUY0, A_END, A_LEVEL, A_PLAY0, A_ROLL, A_SELL0, MAX_BOARD, N_BUY, N_PLAY, N_SELL
from .build_path import _tier_commit, infer_target
from .tempo_board_policy import (
    PendingTransition,
    TempoBoardGreedyPolicy,
    _board_name_set,
    _candidate_value,
    _deploy_build_gain,
    _held_names,
    _net_transition,
    _raw_stats,
    _replacement_build_value,
    _shop_build_gain,
)
from .tempo_margin_audit import CoreExposureScore, DecisionSnapshot, ScoredTransition


def _rank_transitions(transitions: List[ScoredTransition],
                      *, use_build: bool) -> Dict[Tuple[str, Optional[int], str], int]:
    """Rank by net (with or without build component)."""
    def key(t: ScoredTransition) -> float:
        if use_build:
            return t.net_value
        return t.raw_component - t.replacement_raw

    sorted_t = sorted(transitions, key=key, reverse=True)
    ranks: Dict[Tuple[str, Optional[int], str], int] = {}
    for i, t in enumerate(sorted_t, start=1):
        k = (t.action_type, t.candidate_slot, t.candidate_name or "")
        if k not in ranks:
            ranks[k] = i
    return ranks


def _core_frequency(fit, name: str) -> float:
    if fit is None or not name:
        return 0.0
    return float(fit.arch.core.get(name, 0.0))


def build_audit_snapshot(
        policy: TempoBoardGreedyPolicy,
        obs: Dict,
        mask: List[bool],
        action: int,
        *,
        ctx: Dict,
        pending: Optional[PendingTransition] = None,
        compound_stage: Optional[str] = None) -> DecisionSnapshot:
    board = obs.get("board") or []
    hand = obs.get("hand") or []
    shop = obs.get("shop") or []
    tier = int(obs.get("tavern_tier") or 1)
    fit = infer_target(board)
    seeded = fit is not None and fit.have >= 1
    target = (ctx.get("target_before") or {})
    if seeded and fit is not None:
        target_archetype = fit.arch.key
        core_have = fit.have
    else:
        target_archetype = target.get("archetype_key")
        core_have = target.get("core_have") or 0

    board_full = len(board) >= MAX_BOARD
    transitions: List[ScoredTransition] = []
    core_scores: Dict[str, CoreExposureScore] = {}

    if seeded and fit is not None:
        held = _held_names(obs)
        on_board = _board_name_set(board)
        commit = _tier_commit(tier)
        buy_slots = [i for i in range(min(len(shop), N_BUY)) if mask[A_BUY0 + i]]

        for hi in range(min(len(hand), N_PLAY)):
            if not mask[A_PLAY0 + hi]:
                continue
            m = hand[hi]
            name = m.get("name")
            gain = _deploy_build_gain(name, fit, tier, on_board)
            raw = _raw_stats(m)
            transitions.append(ScoredTransition(
                action_type="play", candidate_name=name, candidate_slot=hi,
                raw_component=raw, build_gain=gain,
                build_component=policy.lambda_build * gain,
                replacement_name=None, replacement_slot=None,
                replacement_raw=0.0, replacement_build_value=0.0,
                replacement_component=0.0,
                net_value=_candidate_value(raw, gain, policy.lambda_build),
                is_target_core=gain > 0, action_id=A_PLAY0 + hi))

        if board_full and hand:
            for hi in range(min(len(hand), N_PLAY)):
                hm = hand[hi]
                hname = hm.get("name")
                cand_raw = _raw_stats(hm)
                cand_gain = _deploy_build_gain(hname, fit, tier, on_board)
                for bi in range(min(len(board), N_SELL)):
                    if not mask[A_SELL0 + bi]:
                        continue
                    bm = board[bi]
                    repl_name = bm.get("name")
                    repl_raw = _raw_stats(bm)
                    repl_gain = _replacement_build_value(repl_name, fit, tier)
                    net = _net_transition(
                        cand_raw=cand_raw, cand_gain=cand_gain,
                        repl_raw=repl_raw, repl_gain=repl_gain,
                        lambda_build=policy.lambda_build)
                    transitions.append(ScoredTransition(
                        action_type="hand_sell_play", candidate_name=hname,
                        candidate_slot=hi, raw_component=cand_raw,
                        build_gain=cand_gain,
                        build_component=policy.lambda_build * cand_gain,
                        replacement_name=repl_name, replacement_slot=bi,
                        replacement_raw=repl_raw,
                        replacement_build_value=repl_gain,
                        replacement_component=policy.lambda_build * repl_gain,
                        net_value=net, is_target_core=cand_gain > 0,
                        action_id=A_SELL0 + bi))

        for si in buy_slots:
            sm = shop[si]
            sname = sm.get("name")
            cand_raw = _raw_stats(sm)
            cand_gain = _shop_build_gain(sname, fit, tier, held)
            freq = _core_frequency(fit, sname)
            if not board_full:
                net = _candidate_value(cand_raw, cand_gain, policy.lambda_build)
                transitions.append(ScoredTransition(
                    action_type="buy", candidate_name=sname, candidate_slot=si,
                    raw_component=cand_raw, build_gain=cand_gain,
                    build_component=policy.lambda_build * cand_gain,
                    replacement_name=None, replacement_slot=None,
                    replacement_raw=0.0, replacement_build_value=0.0,
                    replacement_component=0.0, net_value=net,
                    is_target_core=cand_gain > 0, action_id=A_BUY0 + si))
                if sname in fit.arch.core and cand_gain > 0:
                    core_scores[sname] = CoreExposureScore(
                        core_name=sname, shop_slot=si,
                        candidate_raw=cand_raw, core_frequency=freq,
                        tier_commitment=commit, build_gain=cand_gain,
                        build_component=policy.lambda_build * cand_gain,
                        board_full=False, replacement_name=None,
                        replacement_raw=0.0, replacement_build_value=0.0,
                        replacement_component=0.0,
                        core_transition_raw=cand_raw,
                        core_transition_build=cand_gain, core_net_value=net,
                        core_free_slot_value=net,
                        core_actual_replacement_value=net,
                        rank_with_build=0, rank_without_build=0, rank_total=len(transitions))
            else:
                best_repl_net = None
                best_repl: Tuple[Optional[str], float, float] = (None, 0.0, 0.0)
                for bi in range(min(len(board), N_SELL)):
                    if not mask[A_SELL0 + bi]:
                        continue
                    bm = board[bi]
                    repl_name = bm.get("name")
                    repl_raw = _raw_stats(bm)
                    repl_gain = _replacement_build_value(repl_name, fit, tier)
                    net = _net_transition(
                        cand_raw=cand_raw, cand_gain=cand_gain,
                        repl_raw=repl_raw, repl_gain=repl_gain,
                        lambda_build=policy.lambda_build)
                    transitions.append(ScoredTransition(
                        action_type="shop_sell_buy", candidate_name=sname,
                        candidate_slot=si, raw_component=cand_raw,
                        build_gain=cand_gain,
                        build_component=policy.lambda_build * cand_gain,
                        replacement_name=repl_name, replacement_slot=bi,
                        replacement_raw=repl_raw,
                        replacement_build_value=repl_gain,
                        replacement_component=policy.lambda_build * repl_gain,
                        net_value=net, is_target_core=cand_gain > 0,
                        action_id=A_SELL0 + bi))
                    if best_repl_net is None or net > best_repl_net:
                        best_repl_net = net
                        best_repl = (repl_name, repl_raw, repl_gain)
                free_net = _candidate_value(cand_raw, cand_gain, policy.lambda_build)
                if sname in fit.arch.core and cand_gain > 0:
                    repl_name, repl_raw, repl_gain = best_repl
                    core_scores[sname] = CoreExposureScore(
                        core_name=sname, shop_slot=si,
                        candidate_raw=cand_raw, core_frequency=freq,
                        tier_commitment=commit, build_gain=cand_gain,
                        build_component=policy.lambda_build * cand_gain,
                        board_full=True, replacement_name=repl_name,
                        replacement_raw=repl_raw,
                        replacement_build_value=repl_gain,
                        replacement_component=policy.lambda_build * repl_gain,
                        core_transition_raw=cand_raw,
                        core_transition_build=cand_gain,
                        core_net_value=best_repl_net or free_net,
                        core_free_slot_value=free_net,
                        core_actual_replacement_value=best_repl_net,
                        rank_with_build=0, rank_without_build=0,
                        rank_total=len(transitions))

        rank_build = _rank_transitions(transitions, use_build=True)
        rank_raw = _rank_transitions(transitions, use_build=False)
        for name, cs in core_scores.items():
            si = cs.shop_slot
            key_buy = ("buy", si, name)
            key_shop = ("shop_sell_buy", si, name)
            rk = rank_build.get(key_shop, rank_build.get(key_buy, cs.rank_total))
            rkr = rank_raw.get(key_shop, rank_raw.get(key_buy, cs.rank_total))
            cs.rank_with_build = rk
            cs.rank_without_build = rkr

    chosen = _decode_chosen(action, obs, mask, transitions, pending, compound_stage,
                            fit, tier, policy.lambda_build)

    return DecisionSnapshot(
        lobby=ctx["lobby"],
        seat=ctx["seat"],
        turn=ctx["turn"],
        shop_generation=ctx["shop_generation"],
        seeded=seeded,
        lambda_build=policy.lambda_build,
        tavern_tier=tier,
        core_have=int(core_have),
        target_archetype=target_archetype,
        gold=float(obs.get("gold") or 0),
        board_full=board_full,
        pending_stage=compound_stage or (pending.stage if pending else None),
        all_transitions=transitions,
        chosen=chosen,
        core_scores=core_scores,
        action_id=action,
    )


def _decode_chosen(action: int, obs: Dict, mask: List[bool],
                   transitions: List[ScoredTransition],
                   pending: Optional[PendingTransition],
                   compound_stage: Optional[str],
                   fit, tier: int, lambda_build: float) -> ScoredTransition:
    if compound_stage == "buy" and pending:
        return ScoredTransition(
            action_type="compound_buy", candidate_name=pending.candidate_name,
            candidate_slot=pending.candidate_slot,
            raw_component=0.0, build_gain=pending.build_gain,
            build_component=lambda_build * pending.build_gain,
            replacement_name=None, replacement_slot=pending.replacement_slot,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=pending.net_value,
            is_target_core=pending.build_gain > 0, action_id=action)
    if compound_stage == "play" and pending:
        return ScoredTransition(
            action_type="compound_play", candidate_name=pending.candidate_name,
            candidate_slot=pending.candidate_slot,
            raw_component=0.0, build_gain=pending.build_gain,
            build_component=lambda_build * pending.build_gain,
            replacement_name=None, replacement_slot=pending.replacement_slot,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=pending.net_value,
            is_target_core=pending.build_gain > 0, action_id=action)

    for t in transitions:
        if t.action_id == action:
            return t

    board = obs.get("board") or []
    hand = obs.get("hand") or []
    shop = obs.get("shop") or []
    held = _held_names(obs) if fit else set()
    on_board = _board_name_set(board)

    if action == A_LEVEL:
        return ScoredTransition(
            action_type="level", candidate_name=None, candidate_slot=None,
            raw_component=0.0, build_gain=0.0, build_component=0.0,
            replacement_name=None, replacement_slot=None,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=0.0, is_target_core=False,
            action_id=action)
    if action == A_ROLL:
        return ScoredTransition(
            action_type="roll", candidate_name=None, candidate_slot=None,
            raw_component=0.0, build_gain=0.0, build_component=0.0,
            replacement_name=None, replacement_slot=None,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=0.0, is_target_core=False,
            action_id=action)
    if action == A_END:
        return ScoredTransition(
            action_type="end", candidate_name=None, candidate_slot=None,
            raw_component=0.0, build_gain=0.0, build_component=0.0,
            replacement_name=None, replacement_slot=None,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=0.0, is_target_core=False,
            action_id=action)
    if A_BUY0 <= action < A_BUY0 + N_BUY:
        si = action - A_BUY0
        sm = shop[si] if si < len(shop) else {}
        name = sm.get("name")
        gain = _shop_build_gain(name, fit, tier, held) if fit else 0.0
        raw = _raw_stats(sm)
        return ScoredTransition(
            action_type="greedy_buy", candidate_name=name, candidate_slot=si,
            raw_component=raw, build_gain=gain,
            build_component=lambda_build * gain,
            replacement_name=None, replacement_slot=None,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0,
            net_value=_candidate_value(raw, gain, lambda_build),
            is_target_core=gain > 0, action_id=action)
    if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
        hi = action - A_PLAY0
        m = hand[hi] if hi < len(hand) else {}
        name = m.get("name")
        gain = _deploy_build_gain(name, fit, tier, on_board) if fit else 0.0
        raw = _raw_stats(m)
        return ScoredTransition(
            action_type="greedy_play", candidate_name=name, candidate_slot=hi,
            raw_component=raw, build_gain=gain,
            build_component=lambda_build * gain,
            replacement_name=None, replacement_slot=None,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0,
            net_value=_candidate_value(raw, gain, lambda_build),
            is_target_core=gain > 0, action_id=action)
    if A_SELL0 <= action < A_SELL0 + N_SELL:
        bi = action - A_SELL0
        bm = board[bi] if bi < len(board) else {}
        return ScoredTransition(
            action_type="sell", candidate_name=bm.get("name"),
            candidate_slot=bi, raw_component=_raw_stats(bm),
            build_gain=0.0, build_component=0.0,
            replacement_name=None, replacement_slot=bi,
            replacement_raw=0.0, replacement_build_value=0.0,
            replacement_component=0.0, net_value=0.0, is_target_core=False,
            action_id=action)
    return ScoredTransition(
        action_type="other", candidate_name=None, candidate_slot=None,
        raw_component=0.0, build_gain=0.0, build_component=0.0,
        replacement_name=None, replacement_slot=None,
        replacement_raw=0.0, replacement_build_value=0.0,
        replacement_component=0.0, net_value=0.0, is_target_core=False,
        action_id=action)
