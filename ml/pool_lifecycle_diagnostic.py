"""Phase 3E — observational board-pool lifecycle split of the 3D A1 term.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. Reproduces 3D board-pool magnitude A1 ≈ +0.422,
then splits that arm gap into inherited carry, current-turn scaling add,
replacement retention/loss, and lifecycle selection + leftover.

Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.
"""

from __future__ import annotations

import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

import hsbg_coach.bg_env as _bg_env
from hsbg_coach.bg_env import (
    A_BUY0,
    A_END,
    A_FREEZE,
    A_LEVEL,
    A_PLAY0,
    A_ROLL,
    A_SELL0,
    BGEnv,
    N_BUY,
    N_PLAY,
    N_SELL,
    board_level_abstract_scaling_enabled,
    board_synthetic_total,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.attack_source_diagnostic import AttackSourceTracer
from ml.attacker_punch_diagnostic import _arm_prefixes, _walk
from ml.phase_2z_prereg import (
    cursor_bin,
    gen_bin,
    target_bin,
    unsupported_bin,
)
from ml.phase_3a_prereg import (
    N_CLEAVE_BINS,
    N_CURSOR_BINS,
    N_DECILES,
    N_DS_BINS,
    N_GEN_BINS,
    N_ORDINARY_BINS,
    N_POISON_BINS,
    N_SOC_BINS,
    N_TARGET_BINS,
    N_TEAM_BINS,
    N_UNSUP_BINS,
    SLOT_BIN_CAP,
    cleave_bin,
    ds_bin,
    ordinary_bin,
    poison_bin,
    slot_bin,
    soc_bin,
)
from ml.phase_3b_prereg import N_HIT_BINS, hit_count_bin
from ml.phase_3c_prereg import N_ATK_BINS, N_PAIR_BINS, N_SYNTH_ATK_BINS
from ml.phase_3d_prereg import (
    N_CONC_BINS,
    N_DELTA_BINS,
    N_POOL_BINS,
    allocation_concentration_value,
    board_pool_value,
    combat_delta_value,
)
from ml.phase_3e_prereg import (
    FLOW_ABS_TOL,
    INSTRUMENT_TURNS,
    N_CARRY_BINS,
    N_REPLACE_BINS,
    N_SCALE_BINS,
    N_SELECT_BINS,
    PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2Y_UNEXPLAINED,
    PHASE_2Z_UNEXPLAINED,
    PHASE_3A_UNEXPLAINED,
    PHASE_3B_DAMAGE_PER_HIT,
    PHASE_3C_ATTACKER_ATTACK_STRENGTH,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3D_SHARE_BOARD_POOL,
    POOL_FLOW_IDENTITY,
    assert_seed_range_allowed,
    carry_pool_value,
    diagnose_phase_3e,
    replacement_loss_value,
    scaling_add_value,
    selection_state_value,
)
from ml.survivor_composition_diagnostic import TIERS
from ml.synthetic_allocation_diagnostic import (
    _cond_p,
    _hits,
    _kitagawa_two,
    _safe_div,
    bin_value,
    decile_edges,
)

METHODOLOGY_VERSION = "3e_v1"

_LIFE_KEYS = (
    "opp_carry_attack_pool",
    "opp_attack_pool_prior_combat_end",
    "opp_attack_pool_recruit_start",
    "opp_attack_pool_pre_scale",
    "opp_attack_pool_post_scale",
    "opp_attack_pool_combat_start",
    "opp_scale_add_attack",
    "opp_replace_loss_attack",
    "opp_replace_loss_attack_events",
    "opp_stats_pool_recruit_start",
    "opp_stats_pool_pre_scale",
    "opp_stats_pool_post_scale",
    "opp_scale_add_stats",
    "opp_replace_loss_stats",
    "opp_synthetic_carried",
    "opp_synthetic_preserved",
    "opp_synthetic_lost",
    "opp_n_replacements",
    "opp_n_sells",
    "opp_n_alive",
    "opp_alive",
    "opp_select_board_size",
    "opp_flow_residual",
    "opp_flow_ok",
    "opp_event_flow_residual",
    "opp_event_flow_ok",
    "opp_combat_matches_post_scale",
    "opp_residual_add",
    "opp_firestone_target",
    "opp_pace_target",
    "opp_growth_factor",
    "opp_ratio_g",
    "opp_tavern_tier",
    "opp_turns_since_level",
    "opp_just_leveled",
    "opp_end_of_recruit_pre_scaling_stats",
    "opp_mean_tier_recruit_start",
    "opp_board_size_recruit_start",
    "opp_board_size_pre_scale",
    "opp_board_size_post_scale",
)

_PART_NAMES = (
    "recruit_mix",
    "synthetic_allocation",
    "slot_opportunity",
    "teammate_protection",
    "targeting_taunt",
    "attack_cursor",
    "represented_generated",
    "unsupported_coverage",
    "divine_shield",
    "poison_venomous",
    "cleave",
    "start_of_combat",
    "ordinary_combat",
    "damaging_hits",
    "inherited_carry_pool",
    "current_turn_scaling_add",
    "replacement_churn",
    "lifecycle_selection",
    "board_pool_magnitude",
    "allocation_concentration",
    "combat_mutation",
    "attacker_attack_strength",
    "attacker_synth_composition",
    "pairing_order",
    "still_unexplained",
)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _decode_action(action: int) -> str:
    if A_BUY0 <= action < A_BUY0 + N_BUY:
        return "buy"
    if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
        return "play"
    if A_SELL0 <= action < A_SELL0 + N_SELL:
        return "sell"
    if action == A_ROLL:
        return "roll"
    if action == A_LEVEL:
        return "level"
    if action == A_FREEZE:
        return "freeze"
    if action == A_END:
        return "end"
    return "unknown"


def _view_attack_pool(m: Dict) -> int:
    atk = int(m.get("attack") or 0)
    rec = m.get("recruit_attack")
    if rec in (None, ""):
        rec = atk
    return int(atk) - int(rec or 0)


def _view_stats_pool(m: Dict) -> int:
    atk = int(m.get("attack") or 0)
    hp = int(m.get("health") or 0)
    ra = m.get("recruit_attack")
    rh = m.get("recruit_health")
    if ra in (None, ""):
        ra = atk
    if rh in (None, ""):
        rh = hp
    return (atk + hp) - (int(ra or 0) + int(rh or 0))


def _minion_attack_pool(m) -> int:
    atk = int(getattr(m, "attack", 0) or 0)
    rec = getattr(m, "recruit_attack", None)
    if rec is None:
        rec = atk
    return int(atk) - int(rec or 0)


def board_attack_pool(board) -> int:
    return int(sum(_minion_attack_pool(m) for m in (board or [])))


def views_attack_pool(board) -> int:
    return int(sum(_view_attack_pool(m) for m in (board or [])))


def views_stats_pool(board) -> int:
    return int(sum(_view_stats_pool(m) for m in (board or [])))


def _two_s_on() -> bool:
    return bool(_bg_env.PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING)


def player_stats_pool(player) -> float:
    board = list(getattr(player, "board", None) or [])
    on_board = float(board_synthetic_total(board)) if board else 0.0
    if _two_s_on():
        field = float(getattr(player, "abstract_pool", 0.0) or 0.0)
        if not board:
            return field
        return on_board
    return on_board


def _mean_tier(board) -> float:
    if not board:
        return 0.0
    tiers = []
    for m in board:
        try:
            tiers.append(int(getattr(m, "tier", 1) or 1))
        except (TypeError, ValueError):
            tiers.append(1)
    return float(sum(tiers)) / float(len(tiers))


def _tier_hist(board) -> Dict[str, int]:
    hist = {str(t): 0 for t in range(1, 7)}
    for m in board or []:
        try:
            t = int(getattr(m, "tier", 1) or 1)
        except (TypeError, ValueError):
            t = 1
        hist[str(min(6, max(1, t)))] += 1
    return hist


def _snapshot_player(player, *, n_alive: int) -> Dict:
    board = list(getattr(player, "board", None) or [])
    return {
        "alive": bool(getattr(player, "alive", False)),
        "tier": int(getattr(player, "tier", 1) or 1),
        "hp": int(getattr(player, "hp", 0) or 0),
        "board_size": len(board),
        "mean_tier": _mean_tier(board),
        "tier_hist": _tier_hist(board),
        "attack_pool": float(board_attack_pool(board)),
        "stats_pool": float(player_stats_pool(player)),
        "strength": float(player.strength()) if hasattr(player, "strength") else 0.0,
        "n_alive": int(n_alive),
    }


class PoolLifecycleTracer(AttackSourceTracer):
    """3D punch rows plus per-seat-turn attack-pool lifecycle snapshots."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.turn_rows: List[Dict] = []
        self.replacement_events: List[Dict] = []
        self._seat_turn: Dict[Tuple[int, int], Dict] = {}
        self._budget_by_seat: Dict[int, Dict] = {}
        self._prior_combat_end: Dict[int, Dict] = {}
        self._pending: Dict[int, Dict] = {}
        self._pre_obs: Optional[Dict] = None
        self._pre_attack_pool: Optional[float] = None
        self._pre_stats_pool: Optional[float] = None

    def attach_to_env(self, env: BGEnv) -> None:
        super().attach_to_env(env)
        env.scaling_audit_hook = self._on_scaling_audit

    def _on_scaling_audit(self, env, player, seat: int, budget: Dict) -> None:
        if env.turn not in INSTRUMENT_TURNS:
            return
        self._budget_by_seat[int(seat)] = dict(budget)

    def _acc(self, seat: int, turn: int) -> Dict:
        key = (int(seat), int(turn))
        if key not in self._seat_turn:
            self._seat_turn[key] = {
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": int(seat),
                "turn": int(turn),
                "n_replacements": 0,
                "n_sells": 0,
                "event_attack_delta": 0.0,
                "event_stats_delta": 0.0,
                "event_replace_loss_attack": 0.0,
                "synthetic_lost_events": 0.0,
                "synthetic_preserved_events": 0.0,
            }
        return self._seat_turn[key]

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._seat_turn.clear()
        self._budget_by_seat.clear()
        self._prior_combat_end.clear()
        self._pending.clear()

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        if turn not in INSTRUMENT_TURNS:
            return
        n_alive = 1
        acc = self._acc(seat, turn)
        snap = _snapshot_player(player, n_alive=n_alive)
        prior = self._prior_combat_end.get(int(seat))
        acc["attack_pool_prior_combat_end"] = (
            float(prior["attack_pool"]) if prior else snap["attack_pool"]
        )
        acc["attack_pool_recruit_start"] = snap["attack_pool"]
        acc["stats_pool_recruit_start"] = snap["stats_pool"]
        acc["board_size_recruit_start"] = snap["board_size"]
        acc["mean_tier_recruit_start"] = snap["mean_tier"]
        acc["tier_hist_recruit_start"] = dict(snap["tier_hist"])
        acc["alive_at_recruit"] = snap["alive"]
        acc["tier_at_recruit"] = snap["tier"]
        acc["hp_at_recruit"] = snap["hp"]
        acc["strength_recruit_start"] = snap["strength"]

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask,
    ) -> None:
        self._pre_obs = obs
        board = list(obs.get("board") or [])
        self._pre_attack_pool = float(views_attack_pool(board))
        self._pre_stats_pool = float(views_stats_pool(board))

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int,
        ended: bool, player=None,
    ) -> None:
        kind = _decode_action(action)
        if turn not in INSTRUMENT_TURNS or player is None:
            if ended or kind in ("roll", "end", "level"):
                self._pending.pop(seat, None)
            return
        acc = self._acc(seat, turn)
        post_atk = float(board_attack_pool(player.board))
        post_stats = float(player_stats_pool(player))
        pre_atk = self._pre_attack_pool
        pre_stats = self._pre_stats_pool
        if pre_atk is not None:
            d_atk = post_atk - float(pre_atk)
            acc["event_attack_delta"] += d_atk
        if pre_stats is not None:
            acc["event_stats_delta"] += post_stats - float(pre_stats)

        obs = self._pre_obs or {}
        board = list(obs.get("board") or [])
        shop = list(obs.get("shop") or [])
        hand = list(obs.get("hand") or [])

        if A_SELL0 <= action < A_SELL0 + N_SELL:
            acc["n_sells"] += 1
            slot = action - A_SELL0
            sold = board[slot] if 0 <= slot < len(board) else None
            lost_atk = float(_view_attack_pool(sold)) if sold else 0.0
            acc["event_replace_loss_attack"] += max(0.0, lost_atk)
            if _two_s_on():
                acc["synthetic_preserved_events"] += lost_atk
            else:
                acc["synthetic_lost_events"] += lost_atk
            if seat in self._pending:
                self._pending.pop(seat, None)
            if sold is not None:
                self._pending[seat] = {
                    "seat": seat,
                    "turn": turn,
                    "sold_attack_pool": lost_atk,
                    "sold_stats_pool": float(_view_stats_pool(sold)),
                    "sold_name": sold.get("name"),
                    "pre_attack_pool": pre_atk,
                    "pre_stats_pool": pre_stats,
                    "candidate": None,
                    "source": None,
                }
            return

        pending = self._pending.get(seat)
        if pending is not None:
            if A_BUY0 <= action < A_BUY0 + N_BUY and pending.get("candidate") is None:
                slot = action - A_BUY0
                if 0 <= slot < len(shop):
                    pending["candidate"] = shop[slot]
                    pending["source"] = "shop"
                return
            if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
                slot = action - A_PLAY0
                played = hand[slot] if 0 <= slot < len(hand) else None
                if pending.get("candidate") is None:
                    pending["candidate"] = played
                    pending["source"] = "hand"
                self._complete_replacement(seat, turn, player, pending, post_atk, post_stats)
                return
            if kind in ("roll", "end", "level") or ended:
                self._pending.pop(seat, None)

    def _complete_replacement(
        self, seat: int, turn: int, player, pending: Dict,
        post_atk: float, post_stats: float,
    ) -> None:
        acc = self._acc(seat, turn)
        acc["n_replacements"] += 1
        pre_atk = float(pending.get("pre_attack_pool") or 0.0)
        pre_stats = float(pending.get("pre_stats_pool") or 0.0)
        sold_atk = float(pending.get("sold_attack_pool") or 0.0)
        cand = pending.get("candidate") or {}
        self.replacement_events.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "seat": seat,
            "turn": turn,
            "source": pending.get("source"),
            "sold_name": pending.get("sold_name"),
            "sold_attack_pool": sold_atk,
            "sold_stats_pool": float(pending.get("sold_stats_pool") or 0.0),
            "candidate_name": cand.get("name") if isinstance(cand, dict) else None,
            "pre_attack_pool": pre_atk,
            "post_attack_pool": post_atk,
            "pre_stats_pool": pre_stats,
            "post_stats_pool": post_stats,
            "attack_pool_delta": float(post_atk) - pre_atk,
            "stats_pool_delta": float(post_stats) - pre_stats,
            "two_s_on": _two_s_on(),
            "synthetic_carried": (
                sold_atk if _two_s_on() else 0.0
            ),
            "synthetic_lost": (
                0.0 if _two_s_on() else sold_atk
            ),
            "synthetic_preserved": (
                sold_atk if _two_s_on() else 0.0
            ),
        })
        self._pending.pop(seat, None)

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        if turn not in INSTRUMENT_TURNS:
            self._pending.pop(seat, None)
            return
        acc = self._acc(seat, turn)
        snap = _snapshot_player(player, n_alive=1)
        acc["attack_pool_pre_scale"] = snap["attack_pool"]
        acc["stats_pool_pre_scale"] = snap["stats_pool"]
        acc["board_size_pre_scale"] = snap["board_size"]
        acc["mean_tier_pre_scale"] = snap["mean_tier"]
        acc["tier_hist_pre_scale"] = dict(snap["tier_hist"])
        acc["strength_pre_scale"] = snap["strength"]
        acc["tier_pre_scale"] = snap["tier"]
        self._pending.pop(seat, None)

    def after_scale_all(self, env: BGEnv) -> None:
        turn = env.turn
        n_alive = sum(1 for p in env.players if p.alive)
        for seat, player in enumerate(env.players):
            if turn not in INSTRUMENT_TURNS:
                continue
            if not player.alive and (seat, turn) not in self._seat_turn:
                continue
            acc = self._acc(seat, turn)
            snap = _snapshot_player(player, n_alive=n_alive)
            budget = self._budget_by_seat.pop(seat, None) or {}
            acc["n_alive"] = n_alive
            acc["alive_at_post_scale"] = snap["alive"]
            acc["attack_pool_post_scale"] = snap["attack_pool"]
            acc["stats_pool_post_scale"] = snap["stats_pool"]
            acc["board_size_post_scale"] = snap["board_size"]
            acc["mean_tier_post_scale"] = snap["mean_tier"]
            acc["tier_hist_post_scale"] = dict(snap["tier_hist"])
            acc["strength_post_scale"] = snap["strength"]
            acc["tier_post_scale"] = snap["tier"]
            pre_atk = float(acc.get("attack_pool_pre_scale") or snap["attack_pool"])
            carry = float(acc.get("attack_pool_recruit_start") or pre_atk)
            acc["scale_add_attack"] = float(snap["attack_pool"]) - pre_atk
            acc["replace_loss_attack"] = carry - pre_atk
            pre_stats = float(acc.get("stats_pool_pre_scale") or snap["stats_pool"])
            carry_stats = float(acc.get("stats_pool_recruit_start") or pre_stats)
            acc["scale_add_stats"] = float(snap["stats_pool"]) - pre_stats
            acc["replace_loss_stats"] = carry_stats - pre_stats
            acc["residual_add"] = float(budget["residual_add"]) if budget.get("residual_add") is not None else None
            acc["firestone_target"] = budget.get("firestone_target")
            acc["pace_target"] = budget.get("pace_target")
            acc["growth_factor"] = budget.get("growth_factor")
            acc["ratio_g"] = budget.get("ratio_g")
            acc["ratio_add"] = budget.get("ratio_add")
            acc["over"] = budget.get("over")
            acc["residual_clamp_active"] = budget.get("residual_clamp_active")
            acc["just_leveled"] = budget.get("just_leveled")
            acc["tavern_tier"] = budget.get("tavern_tier", snap["tier"])
            acc["turns_since_level"] = budget.get("turns_since_level")
            acc["end_of_recruit_pre_scaling_stats"] = budget.get(
                "end_of_recruit_pre_scaling_stats", acc.get("strength_pre_scale")
            )
            acc["curve_ratio"] = budget.get("curve_ratio")
            post = float(snap["attack_pool"])
            add = float(acc["scale_add_attack"])
            loss = float(acc["replace_loss_attack"])
            # post = carry + add - snapshot_loss  (snapshot_loss = carry - pre)
            acc["flow_residual"] = post - (carry + add - loss)
            acc["flow_ok"] = abs(float(acc["flow_residual"])) <= FLOW_ABS_TOL
            event_loss = float(acc.get("event_replace_loss_attack") or 0.0)
            acc["event_flow_residual"] = post - (carry + add - event_loss)
            acc["event_flow_ok"] = abs(float(acc["event_flow_residual"])) <= max(
                FLOW_ABS_TOL, 1.0
            )
            carried = carry
            lost = max(0.0, float(acc["replace_loss_attack"]))
            acc["synthetic_carried"] = carried
            acc["synthetic_lost"] = lost
            acc["synthetic_preserved"] = max(0.0, carried - lost)
            self.turn_rows.append(dict(acc))

    def after_combat(self, env: BGEnv) -> None:
        n_alive = sum(1 for p in env.players if p.alive)
        for seat, player in enumerate(env.players):
            self._prior_combat_end[seat] = _snapshot_player(
                player, n_alive=n_alive,
            )

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        turn = int(fight.get("turn") or env.turn)
        n_alive = sum(1 for p in env.players if p.alive)
        for seat, player in enumerate(env.players):
            if turn not in INSTRUMENT_TURNS:
                continue
            if (seat, turn) not in self._seat_turn and not player.alive:
                continue
            acc = self._acc(seat, turn)
            snap = _snapshot_player(player, n_alive=n_alive)
            acc["attack_pool_combat_start"] = snap["attack_pool"]
            acc["stats_pool_combat_start"] = snap["stats_pool"]
            acc["board_size_combat_start"] = snap["board_size"]
            acc["mean_tier_combat_start"] = snap["mean_tier"]
            acc["n_alive"] = n_alive
            acc["alive_at_combat"] = snap["alive"]
            post = float(acc.get("attack_pool_post_scale") or snap["attack_pool"])
            acc["combat_matches_post_scale"] = (
                abs(float(snap["attack_pool"]) - post) <= FLOW_ABS_TOL
            )
            for row in reversed(self.turn_rows):
                if int(row.get("seat") or -1) == seat and int(row.get("turn") or -1) == turn:
                    row["attack_pool_combat_start"] = acc["attack_pool_combat_start"]
                    row["stats_pool_combat_start"] = acc.get("stats_pool_combat_start")
                    row["board_size_combat_start"] = acc.get("board_size_combat_start")
                    row["mean_tier_combat_start"] = acc.get("mean_tier_combat_start")
                    row["alive_at_combat"] = acc.get("alive_at_combat")
                    row["combat_matches_post_scale"] = acc["combat_matches_post_scale"]
                    row["n_alive"] = n_alive
                    break
        life = None
        if fight.get("kind") == "live" and fight.get("loser_seat") is not None:
            key = (int(fight["loser_seat"]), turn)
            life = self._seat_turn.get(key)
        rec["opp_lifecycle"] = dict(life) if life else {}
        rec["loser_seat"] = fight.get("loser_seat")
        rec["winner_seat"] = fight.get("winner_seat")
        for r in rec.get("start_minions") or []:
            _stamp_lifecycle(r, life, n_alive=n_alive)


def _stamp_lifecycle(row: Dict, life: Optional[Dict], *, n_alive: int) -> None:
    src = life or {}
    row["opp_carry_attack_pool"] = float(src.get("attack_pool_recruit_start") or 0.0)
    row["opp_attack_pool_prior_combat_end"] = float(
        src.get("attack_pool_prior_combat_end") or 0.0
    )
    row["opp_attack_pool_recruit_start"] = float(
        src.get("attack_pool_recruit_start") or 0.0
    )
    row["opp_attack_pool_pre_scale"] = float(src.get("attack_pool_pre_scale") or 0.0)
    row["opp_attack_pool_post_scale"] = float(src.get("attack_pool_post_scale") or 0.0)
    combat = src.get("attack_pool_combat_start")
    if combat in (None, ""):
        combat = src.get("attack_pool_post_scale") or row.get("opp_board_pool_attack") or 0.0
    row["opp_attack_pool_combat_start"] = float(combat or 0.0)
    row["opp_scale_add_attack"] = float(src.get("scale_add_attack") or 0.0)
    row["opp_replace_loss_attack"] = float(src.get("replace_loss_attack") or 0.0)
    row["opp_replace_loss_attack_events"] = float(
        src.get("event_replace_loss_attack") or 0.0
    )
    row["opp_stats_pool_recruit_start"] = float(src.get("stats_pool_recruit_start") or 0.0)
    row["opp_stats_pool_pre_scale"] = float(src.get("stats_pool_pre_scale") or 0.0)
    row["opp_stats_pool_post_scale"] = float(src.get("stats_pool_post_scale") or 0.0)
    row["opp_scale_add_stats"] = float(src.get("scale_add_stats") or 0.0)
    row["opp_replace_loss_stats"] = float(src.get("replace_loss_stats") or 0.0)
    row["opp_synthetic_carried"] = float(src.get("synthetic_carried") or 0.0)
    row["opp_synthetic_preserved"] = float(src.get("synthetic_preserved") or 0.0)
    row["opp_synthetic_lost"] = float(src.get("synthetic_lost") or 0.0)
    row["opp_n_replacements"] = int(src.get("n_replacements") or 0)
    row["opp_n_sells"] = int(src.get("n_sells") or 0)
    row["opp_n_alive"] = int(src.get("n_alive") or n_alive)
    row["opp_alive"] = bool(src.get("alive_at_combat", src.get("alive_at_recruit", True)))
    row["opp_select_board_size"] = float(
        src.get("board_size_combat_start")
        or src.get("board_size_post_scale")
        or row.get("opp_board_size")
        or 0.0
    )
    row["opp_flow_residual"] = float(src.get("flow_residual") or 0.0)
    row["opp_flow_ok"] = bool(src.get("flow_ok", True))
    row["opp_event_flow_residual"] = float(src.get("event_flow_residual") or 0.0)
    row["opp_event_flow_ok"] = bool(src.get("event_flow_ok", True))
    row["opp_combat_matches_post_scale"] = bool(
        src.get("combat_matches_post_scale", True)
    )
    row["opp_residual_add"] = src.get("residual_add")
    row["opp_firestone_target"] = src.get("firestone_target")
    row["opp_pace_target"] = src.get("pace_target")
    row["opp_growth_factor"] = src.get("growth_factor")
    row["opp_ratio_g"] = src.get("ratio_g")
    row["opp_tavern_tier"] = src.get("tavern_tier")
    row["opp_turns_since_level"] = src.get("turns_since_level")
    row["opp_just_leveled"] = src.get("just_leveled")
    row["opp_end_of_recruit_pre_scaling_stats"] = src.get(
        "end_of_recruit_pre_scaling_stats"
    )
    row["opp_mean_tier_recruit_start"] = src.get("mean_tier_recruit_start")
    row["opp_board_size_recruit_start"] = src.get("board_size_recruit_start")
    row["opp_board_size_pre_scale"] = src.get("board_size_pre_scale")
    row["opp_board_size_post_scale"] = src.get("board_size_post_scale")


def run_lifecycle_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    fights: List[Dict] = []
    lengths: List[float] = []
    turn_rows: List[Dict] = []
    replacement_events: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = PoolLifecycleTracer(i, seed + i, arm)
                env = BGEnv(seed=seed + i)
                tracer.attach_to_env(env)
                recs = env.play_scripted(
                    [greedy_policy] * env.n_players, recruit_tracer=tracer
                )
                game_length = max((r["turn"] for r in recs), default=0)
                if tracer.game_length is None:
                    tracer.game_length = game_length
                lengths.append(float(game_length))
                fights.extend(tracer.fights)
                turn_rows.extend(tracer.turn_rows)
                replacement_events.extend(tracer.replacement_events)
                del env
                del tracer

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "board_level_abstract_scaling": bool(board_level_abstract_scaling),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "fights": fights,
        "game_lengths": lengths,
        "turn_rows": turn_rows,
        "replacement_events": replacement_events,
    }


def run_greedy_control_lifecycle(lobbies: int, seed: int) -> Dict:
    return run_lifecycle_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_lifecycle(lobbies: int, seed: int) -> Dict:
    return run_lifecycle_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def collect_lifecycle_minions(hits: Sequence[Dict]) -> List[Dict]:
    from ml.attack_source_diagnostic import collect_source_minions
    rows = collect_source_minions(hits)
    for row in rows:
        if row.get("opp_carry_attack_pool") in (None,):
            row["opp_carry_attack_pool"] = 0.0
        if row.get("opp_scale_add_attack") in (None,):
            row["opp_scale_add_attack"] = 0.0
        if row.get("opp_replace_loss_attack") in (None,):
            row["opp_replace_loss_attack"] = 0.0
        if row.get("opp_select_board_size") in (None,):
            row["opp_select_board_size"] = float(row.get("opp_board_size") or 0.0)
        if row.get("opp_flow_ok") in (None,):
            row["opp_flow_ok"] = True
    return rows


def _lifecycle_row_key(
    r: Dict,
    recruit_edges: Dict[int, List[float]],
    synth_edges: Dict[Tuple[int, int], List[float]],
    team_edges: Dict[int, List[float]],
    carry_edges: Dict[int, List[float]],
    scale_edges: Dict[int, List[float]],
    replace_edges: Dict[int, List[float]],
    select_edges: Dict[int, List[float]],
    pool_edges: Dict[int, List[float]],
    conc_edges: Dict[int, List[float]],
    delta_edges: Dict[int, List[float]],
    atk_edges: Dict[int, List[float]],
    synth_atk_edges: Dict[int, List[float]],
    pair_edges: Dict[int, List[float]],
) -> Tuple:
    from ml.phase_3c_prereg import (
        attacker_attack_value,
        attacker_synth_share_value,
        pairing_order_value,
    )
    t = int(r["tier"])
    rb = bin_value(float(r["recruit_raw"]), recruit_edges[t])
    sb = bin_value(float(r["synthetic_share"]), synth_edges[(t, rb)])
    kb = slot_bin(r.get("board_slot"))
    mb = bin_value(float(r.get("teammate_combat_raw") or 0), team_edges[t])
    hb = hit_count_bin(r)
    cy = bin_value(carry_pool_value(r), carry_edges[t])
    sc = bin_value(scaling_add_value(r), scale_edges[t])
    rp = bin_value(replacement_loss_value(r), replace_edges[t])
    sl = bin_value(selection_state_value(r), select_edges[t])
    pb_pool = bin_value(board_pool_value(r), pool_edges[t])
    cb = bin_value(allocation_concentration_value(r), conc_edges[t])
    db = bin_value(combat_delta_value(r), delta_edges[t])
    ab = bin_value(attacker_attack_value(r), atk_edges[t])
    yb = bin_value(attacker_synth_share_value(r), synth_atk_edges[t])
    pb = bin_value(pairing_order_value(r), pair_edges[t])
    return (
        t, rb, sb, kb, mb,
        target_bin(r), cursor_bin(r), gen_bin(r), unsupported_bin(r),
        ds_bin(r), poison_bin(r), cleave_bin(r), soc_bin(r), ordinary_bin(r),
        hb, cy, sc, rp, sl, pb_pool, cb, db, ab, yb, pb,
    )


def reweight_pool_lifecycle(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
    observed_residual: Optional[float] = None,
    observed_leftover_2y: Optional[float] = None,
    observed_leftover_2z: Optional[float] = None,
    observed_leftover_3a: Optional[float] = None,
    observed_damage_per_hit: Optional[float] = None,
    observed_attack_strength: Optional[float] = None,
    observed_board_pool: Optional[float] = None,
) -> Dict:
    """Hold 3B cells through hit-count, then split A1 into carry/scale/replace/leftover.

    Nested Kitagawa:

        hold P(recruit-raw | tier) … P(hit_count_bin | 3A cells)
            ↓
        hold P(opp carry-pool quintile | …)            →  (1) inherited carry
            ↓
        hold P(opp scaling-add quintile | …)           →  (2) current-turn add
            ↓
        hold P(opp replace-loss quintile | …)          →  (3) replacement churn
            ↓
        hold P(opp board-size quintile | …)            →  (4a) lifecycle selection
            ↓
        hold P(opp board-pool quintile | …)            →  remaining A1
            ↓
        hold P(concentration / combat-delta / attack)  →  3D A2–A4
    """
    from ml.phase_3c_prereg import (
        attacker_attack_value,
        attacker_synth_share_value,
        pairing_order_value,
    )
    pooled = list(control_rows) + list(treatment_rows)
    recruit_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        recruit_edges[t] = decile_edges(
            [float(r["recruit_raw"]) for r in pooled if int(r["tier"]) == t]
        )

    synth_edges: Dict[Tuple[int, int], List[float]] = {}
    for t in TIERS:
        n_r = len(recruit_edges[t]) + 1
        for rb in range(n_r):
            vs = [
                float(r["synthetic_share"])
                for r in pooled
                if int(r["tier"]) == t
                and bin_value(float(r["recruit_raw"]), recruit_edges[t]) == rb
            ]
            synth_edges[(t, rb)] = decile_edges(vs)

    team_edges: Dict[int, List[float]] = {}
    carry_edges: Dict[int, List[float]] = {}
    scale_edges: Dict[int, List[float]] = {}
    replace_edges: Dict[int, List[float]] = {}
    select_edges: Dict[int, List[float]] = {}
    pool_edges: Dict[int, List[float]] = {}
    conc_edges: Dict[int, List[float]] = {}
    delta_edges: Dict[int, List[float]] = {}
    atk_edges: Dict[int, List[float]] = {}
    synth_atk_edges: Dict[int, List[float]] = {}
    pair_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        cell = [r for r in pooled if int(r["tier"]) == t]
        team_edges[t] = decile_edges(
            [float(r.get("teammate_combat_raw") or 0) for r in cell],
            n=N_TEAM_BINS,
        )
        carry_edges[t] = decile_edges(
            [carry_pool_value(r) for r in cell], n=N_CARRY_BINS,
        )
        scale_edges[t] = decile_edges(
            [scaling_add_value(r) for r in cell], n=N_SCALE_BINS,
        )
        replace_edges[t] = decile_edges(
            [replacement_loss_value(r) for r in cell], n=N_REPLACE_BINS,
        )
        select_edges[t] = decile_edges(
            [selection_state_value(r) for r in cell], n=N_SELECT_BINS,
        )
        pool_edges[t] = decile_edges(
            [board_pool_value(r) for r in cell], n=N_POOL_BINS,
        )
        conc_edges[t] = decile_edges(
            [allocation_concentration_value(r) for r in cell], n=N_CONC_BINS,
        )
        delta_edges[t] = decile_edges(
            [combat_delta_value(r) for r in cell], n=N_DELTA_BINS,
        )
        atk_edges[t] = decile_edges(
            [attacker_attack_value(r) for r in cell], n=N_ATK_BINS,
        )
        synth_atk_edges[t] = decile_edges(
            [attacker_synth_share_value(r) for r in cell], n=N_SYNTH_ATK_BINS,
        )
        pair_edges[t] = decile_edges(
            [pairing_order_value(r) for r in cell], n=N_PAIR_BINS,
        )

    def _key(r: Dict) -> Tuple:
        return _lifecycle_row_key(
            r, recruit_edges, synth_edges, team_edges,
            carry_edges, scale_edges, replace_edges, select_edges,
            pool_edges, conc_edges, delta_edges,
            atk_edges, synth_atk_edges, pair_edges,
        )

    n_c_t, s_c_t, n_c, s_c = _arm_prefixes(control_rows, n_hits_c, _key)
    n_t_t, s_t_t, n_t, s_t = _arm_prefixes(treatment_rows, n_hits_t, _key)

    const_bins = {
        5: N_TARGET_BINS,
        6: N_CURSOR_BINS,
        7: N_GEN_BINS,
        8: N_UNSUP_BINS,
        9: N_DS_BINS,
        10: N_POISON_BINS,
        11: N_CLEAVE_BINS,
        12: N_SOC_BINS,
        13: N_ORDINARY_BINS,
        14: N_HIT_BINS,
    }
    n_depths = 25  # tier + 24 nested (recruit … pairing)

    def n_bins_at(depth: int, prefix: Tuple) -> int:
        if depth == 1:
            return len(recruit_edges[prefix[0]]) + 1
        if depth == 2:
            return len(synth_edges[(prefix[0], prefix[1])]) + 1
        if depth == 3:
            return SLOT_BIN_CAP + 1
        if depth == 4:
            return len(team_edges[prefix[0]]) + 1
        if depth == 15:
            return len(carry_edges[prefix[0]]) + 1
        if depth == 16:
            return len(scale_edges[prefix[0]]) + 1
        if depth == 17:
            return len(replace_edges[prefix[0]]) + 1
        if depth == 18:
            return len(select_edges[prefix[0]]) + 1
        if depth == 19:
            return len(pool_edges[prefix[0]]) + 1
        if depth == 20:
            return len(conc_edges[prefix[0]]) + 1
        if depth == 21:
            return len(delta_edges[prefix[0]]) + 1
        if depth == 22:
            return len(atk_edges[prefix[0]]) + 1
        if depth == 23:
            return len(synth_atk_edges[prefix[0]]) + 1
        if depth == 24:
            return len(pair_edges[prefix[0]]) + 1
        return int(const_bins[depth])

    totals = {name: 0.0 for name in _PART_NAMES}
    b_direct = 0.0
    per_tier = {}
    for tier in TIERS:
        nc, nt = n_c_t[tier], n_t_t[tier]
        pc = _cond_p(s_c_t[tier], nc)
        pt = _cond_p(s_t_t[tier], nt)
        _mix_t, rate_t, _gap_t, excl_t = _kitagawa_two(
            nc, nt, pc, pt, float(tier)
        )
        zero = {
            "exclusive_support": True,
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "within_tier_B": 0.0,
            **{name: 0.0 for name in _PART_NAMES},
            "phase_3d_board_pool_magnitude_hat": 0.0,
            "phase_3c_attacker_attack_strength_hat": 0.0,
        }
        if excl_t:
            per_tier[str(tier)] = zero
            continue
        n_bar = 0.5 * (nc + nt)
        b_direct += rate_t
        parts = _walk(
            (tier,), 1, n_bins_at, n_c, s_c, n_t, s_t, nc, nt, n_depths,
        )
        scale = float(tier) * n_bar
        scaled = [scale * v for v in parts]
        cell = {
            "exclusive_support": False,
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "within_tier_B": rate_t,
        }
        for name, val in zip(_PART_NAMES, scaled):
            cell[name] = val
            totals[name] += val
        a1_hat = (
            cell["inherited_carry_pool"] + cell["current_turn_scaling_add"]
            + cell["replacement_churn"] + cell["lifecycle_selection"]
            + cell["board_pool_magnitude"]
        )
        a_hat = (
            a1_hat + cell["allocation_concentration"]
            + cell["combat_mutation"] + cell["attacker_attack_strength"]
        )
        cell["phase_3d_board_pool_magnitude_hat"] = a1_hat
        cell["phase_3c_attacker_attack_strength_hat"] = a_hat
        per_tier[str(tier)] = cell

    a1_hat = (
        totals["inherited_carry_pool"] + totals["current_turn_scaling_add"]
        + totals["replacement_churn"] + totals["lifecycle_selection"]
        + totals["board_pool_magnitude"]
    )
    a_hat = (
        a1_hat + totals["allocation_concentration"]
        + totals["combat_mutation"] + totals["attacker_attack_strength"]
    )
    obs_b = float(observed_B) if observed_B is not None else b_direct
    obs_a = (
        float(observed_attack_strength)
        if observed_attack_strength is not None
        else PHASE_3C_ATTACKER_ATTACK_STRENGTH
    )
    obs_a1 = (
        float(observed_board_pool)
        if observed_board_pool is not None
        else PHASE_3D_BOARD_POOL_MAGNITUDE
    )
    obs_dmg = (
        float(observed_damage_per_hit)
        if observed_damage_per_hit is not None
        else PHASE_3B_DAMAGE_PER_HIT
    )

    def _share(part: float, denom: float) -> Optional[float]:
        if abs(denom) < 1e-12:
            return None
        return float(part) / denom

    sel_left = totals["lifecycle_selection"] + totals["board_pool_magnitude"]
    return {
        "method": (
            "nested_kitagawa_3b_hit_cells_then_carry_scale_replace_select_pool"
        ),
        "n_deciles": N_DECILES,
        "n_carry_bins": N_CARRY_BINS,
        "n_scale_bins": N_SCALE_BINS,
        "n_replace_bins": N_REPLACE_BINS,
        "n_select_bins": N_SELECT_BINS,
        "n_pool_bins": N_POOL_BINS,
        "within_tier_B": b_direct,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "phase_2z_unexplained": PHASE_2Z_UNEXPLAINED,
        "phase_3a_unexplained": PHASE_3A_UNEXPLAINED,
        "phase_3b_damage_per_hit": PHASE_3B_DAMAGE_PER_HIT,
        "phase_3c_attacker_attack_strength": PHASE_3C_ATTACKER_ATTACK_STRENGTH,
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "observed_B_used_for_shares": obs_b,
        "observed_attack_strength_used_for_shares": obs_a,
        "observed_board_pool_used_for_shares": obs_a1,
        "observed_damage_per_hit_used_for_shares": obs_dmg,
        **totals,
        "phase_3d_board_pool_magnitude_hat": a1_hat,
        "phase_3c_attacker_attack_strength_hat": a_hat,
        "lifecycle_selection_plus_leftover": sel_left,
        "share_of_a1_inherited_carry_pool": _share(
            totals["inherited_carry_pool"], obs_a1
        ),
        "share_of_a1_current_turn_scaling_add": _share(
            totals["current_turn_scaling_add"], obs_a1
        ),
        "share_of_a1_replacement_churn": _share(
            totals["replacement_churn"], obs_a1
        ),
        "share_of_a1_lifecycle_selection": _share(
            totals["lifecycle_selection"], obs_a1
        ),
        "share_of_a1_still_unexplained": _share(
            totals["board_pool_magnitude"], obs_a1
        ),
        "share_of_a1_lifecycle_selection_plus_leftover": _share(
            sel_left, obs_a1
        ),
        "share_of_a_inherited_carry_pool": _share(
            totals["inherited_carry_pool"], obs_a
        ),
        "share_of_a_current_turn_scaling_add": _share(
            totals["current_turn_scaling_add"], obs_a
        ),
        "share_of_a_replacement_churn": _share(
            totals["replacement_churn"], obs_a
        ),
        "per_tier": per_tier,
    }


def _additive_flow(rows: Sequence[Dict]) -> Dict:
    """Mean opposing-board attack-pool flow on punch-row sample."""
    if not rows:
        return {
            "n": 0,
            "mean_combat_start": None,
            "mean_carry": None,
            "mean_scale_add": None,
            "mean_replace_loss": None,
            "mean_pre_scale": None,
            "mean_post_scale": None,
            "mean_flow_residual": None,
            "p_flow_ok": None,
            "reconstructed": None,
        }
    combat = [board_pool_value(r) for r in rows]
    carry = [carry_pool_value(r) for r in rows]
    add = [scaling_add_value(r) for r in rows]
    loss = [replacement_loss_value(r) for r in rows]
    pre = [float(r.get("opp_attack_pool_pre_scale") or 0.0) for r in rows]
    post = [float(r.get("opp_attack_pool_post_scale") or 0.0) for r in rows]
    resid = [float(r.get("opp_flow_residual") or 0.0) for r in rows]
    ok = [1.0 if r.get("opp_flow_ok") is not False else 0.0 for r in rows]
    recon = [c + a - l for c, a, l in zip(carry, add, loss)]
    return {
        "n": len(rows),
        "mean_combat_start": _mean(combat),
        "mean_carry": _mean(carry),
        "mean_scale_add": _mean(add),
        "mean_replace_loss": _mean(loss),
        "mean_pre_scale": _mean(pre),
        "mean_post_scale": _mean(post),
        "mean_flow_residual": _mean(resid),
        "p_flow_ok": _mean(ok),
        "reconstructed": _mean(recon),
    }


def _delta_flow(c: Dict, t: Dict) -> Dict:
    keys = (
        "mean_combat_start", "mean_carry", "mean_scale_add",
        "mean_replace_loss", "mean_pre_scale", "mean_post_scale",
        "mean_flow_residual", "reconstructed",
    )
    out = {}
    for k in keys:
        a, b = c.get(k), t.get(k)
        out[k] = None if a is None or b is None else float(b) - float(a)
    # Treatment − control. A1 gap is control having more pool, so negative.
    # Additive: Δcombat ≈ Δcarry + Δadd − Δloss + Δresidual
    dc = out.get("mean_combat_start")
    dcarry = out.get("mean_carry")
    dadd = out.get("mean_scale_add")
    dloss = out.get("mean_replace_loss")
    dres = out.get("mean_flow_residual")
    parts = None
    if None not in (dcarry, dadd, dloss):
        parts = float(dcarry) + float(dadd) - float(dloss)
        if dres is not None:
            parts = parts + float(dres)
    out["additive_rhs"] = parts
    out["additive_gap"] = None if dc is None or parts is None else float(dc) - float(parts)
    if dc is not None and abs(float(dc)) > 1e-12:
        out["share_delta_carry"] = None if dcarry is None else float(dcarry) / float(dc)
        out["share_delta_scale_add"] = None if dadd is None else float(dadd) / float(dc)
        # loss *reduces* combat start, so contribution is −Δloss / Δcombat
        out["share_delta_replace"] = None if dloss is None else (-float(dloss)) / float(dc)
        leftover = None
        if parts is not None and dc is not None:
            leftover = float(dc) - (float(dcarry or 0) + float(dadd or 0) - float(dloss or 0))
        out["share_delta_leftover"] = (
            None if leftover is None else leftover / float(dc)
        )
    else:
        out["share_delta_carry"] = None
        out["share_delta_scale_add"] = None
        out["share_delta_replace"] = None
        out["share_delta_leftover"] = None
    return out


def _by_tier_lifecycle(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r["tier"]) == tier]
        n = len(cell)
        out[str(tier)] = {
            "n_start": n,
            "mean_opp_board_pool": _mean([board_pool_value(r) for r in cell]),
            "mean_opp_carry": _mean([carry_pool_value(r) for r in cell]),
            "mean_opp_scale_add": _mean([scaling_add_value(r) for r in cell]),
            "mean_opp_replace_loss": _mean([replacement_loss_value(r) for r in cell]),
            "mean_opp_board_size": _mean([selection_state_value(r) for r in cell]),
            "p_flow_ok": _safe_div(
                float(sum(1 for r in cell if r.get("opp_flow_ok") is not False)),
                float(n),
            ),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
        }
    return out


def _slim_turn(row: Dict) -> Dict:
    keep = {
        "lobby", "seed", "arm", "seat", "turn",
        "attack_pool_prior_combat_end", "attack_pool_recruit_start",
        "attack_pool_pre_scale", "attack_pool_post_scale",
        "attack_pool_combat_start", "scale_add_attack", "replace_loss_attack",
        "stats_pool_recruit_start", "stats_pool_pre_scale", "stats_pool_post_scale",
        "scale_add_stats", "replace_loss_stats",
        "synthetic_carried", "synthetic_preserved", "synthetic_lost",
        "n_replacements", "n_sells", "n_alive", "alive_at_recruit",
        "alive_at_combat", "board_size_recruit_start", "board_size_pre_scale",
        "board_size_post_scale", "board_size_combat_start",
        "mean_tier_recruit_start", "mean_tier_post_scale",
        "flow_ok", "flow_residual", "event_flow_ok", "event_flow_residual",
        "combat_matches_post_scale", "residual_add", "firestone_target",
        "pace_target", "growth_factor", "ratio_g", "tavern_tier",
        "turns_since_level", "just_leveled",
        "end_of_recruit_pre_scaling_stats",
    }
    return {k: row.get(k) for k in keep}


def _summarize_turns(turn_rows: Sequence[Dict]) -> Dict:
    window = [r for r in turn_rows if int(r.get("turn") or 0) in INSTRUMENT_TURNS]
    n_flow = sum(1 for r in window if r.get("flow_ok") is False)
    n_event = sum(1 for r in window if r.get("event_flow_ok") is False)
    n_match = sum(1 for r in window if r.get("combat_matches_post_scale") is False)
    by_turn: Dict[str, Dict] = {}
    for t in INSTRUMENT_TURNS:
        cell = [r for r in window if int(r.get("turn") or 0) == t]
        by_turn[str(t)] = {
            "n": len(cell),
            "mean_carry": _mean([float(r.get("attack_pool_recruit_start") or 0) for r in cell]),
            "mean_pre_scale": _mean([float(r.get("attack_pool_pre_scale") or 0) for r in cell]),
            "mean_scale_add": _mean([float(r.get("scale_add_attack") or 0) for r in cell]),
            "mean_replace_loss": _mean([float(r.get("replace_loss_attack") or 0) for r in cell]),
            "mean_post_scale": _mean([float(r.get("attack_pool_post_scale") or 0) for r in cell]),
            "mean_combat_start": _mean([
                float(r.get("attack_pool_combat_start") or r.get("attack_pool_post_scale") or 0)
                for r in cell
            ]),
            "mean_n_alive": _mean([float(r.get("n_alive") or 0) for r in cell]),
            "mean_board_size": _mean([float(r.get("board_size_post_scale") or 0) for r in cell]),
            "mean_n_replacements": _mean([float(r.get("n_replacements") or 0) for r in cell]),
            "p_flow_ok": _safe_div(
                float(sum(1 for r in cell if r.get("flow_ok") is not False)),
                float(len(cell)),
            ),
        }
    return {
        "n_seat_turns": len(window),
        "flow_mismatches": n_flow,
        "event_flow_mismatches": n_event,
        "combat_post_scale_mismatches": n_match,
        "mean_carry": _mean([float(r.get("attack_pool_recruit_start") or 0) for r in window]),
        "mean_scale_add": _mean([float(r.get("scale_add_attack") or 0) for r in window]),
        "mean_replace_loss": _mean([float(r.get("replace_loss_attack") or 0) for r in window]),
        "mean_combat_start": _mean([
            float(r.get("attack_pool_combat_start") or r.get("attack_pool_post_scale") or 0)
            for r in window
        ]),
        "mean_n_replacements": _mean([float(r.get("n_replacements") or 0) for r in window]),
        "by_turn": by_turn,
    }


def summarize_lifecycle_arm(raw: Dict) -> Dict:
    from ml.attack_source_diagnostic import summarize_source_arm
    summary = summarize_source_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_lifecycle_minions(hits)
    n_hits = len(hits)
    turn_sum = _summarize_turns(raw.get("turn_rows") or [])
    flow = _additive_flow(rows)
    n_flow_bad = sum(1 for r in rows if r.get("opp_flow_ok") is False)
    summary.update({
        "_rows": rows,
        "_n_hits": n_hits,
        "by_tier_lifecycle": _by_tier_lifecycle(rows, n_hits),
        "turn_summary": turn_sum,
        "punch_flow": flow,
        "n_turn_rows": len(raw.get("turn_rows") or []),
        "n_replacement_events": len(raw.get("replacement_events") or []),
        "flow_mismatches_turns": turn_sum["flow_mismatches"],
        "flow_mismatches_punch": n_flow_bad,
        "event_flow_mismatches": turn_sum["event_flow_mismatches"],
        "combat_post_scale_mismatches": turn_sum["combat_post_scale_mismatches"],
        "example_turns": [_slim_turn(r) for r in (raw.get("turn_rows") or [])[:8]],
        "example_replacements": list(raw.get("replacement_events") or [])[:8],
        "pool_flow_identity": POOL_FLOW_IDENTITY,
    })
    return summary


def compare_lifecycle(control: Dict, treatment: Dict) -> Dict:
    from ml.attack_source_diagnostic import compare_source
    base = compare_source(control, treatment)
    rows_c = list(control.get("_rows") or [])
    rows_t = list(treatment.get("_rows") or [])
    n_c = int(control.get("_n_hits") or control.get("n_hits") or 0)
    n_t = int(treatment.get("_n_hits") or treatment.get("n_hits") or 0)

    three_d = base.get("reweighting") or {}
    a1_3d = three_d.get("board_pool_magnitude")
    if a1_3d is None:
        a1_3d = PHASE_3D_BOARD_POOL_MAGNITUDE

    decomp = base.get("decomposition") or {}
    b_obs = decomp.get("within_tier_survival_B")
    if b_obs is None:
        b_obs = PHASE_2V_WITHIN_TIER_B

    reweight = reweight_pool_lifecycle(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=PHASE_2X_RESIDUAL_POSITION,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
        observed_leftover_3a=PHASE_3A_UNEXPLAINED,
        observed_damage_per_hit=PHASE_3B_DAMAGE_PER_HIT,
        observed_attack_strength=PHASE_3C_ATTACKER_ATTACK_STRENGTH,
        observed_board_pool=PHASE_3D_BOARD_POOL_MAGNITUDE,
    )
    reweight["reproduced_3d_board_pool_magnitude"] = a1_3d
    reweight["reproduced_3d_share_board_pool"] = three_d.get(
        "share_of_a_board_pool_magnitude"
    )

    flow_c = control.get("punch_flow") or _additive_flow(rows_c)
    flow_t = treatment.get("punch_flow") or _additive_flow(rows_t)
    flow_delta = _delta_flow(flow_c, flow_t)
    turns_c = control.get("turn_summary") or {}
    turns_t = treatment.get("turn_summary") or {}

    rec = dict(base.get("reconciliation") or {})
    rec.update({
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "phase_3d_share_board_pool": PHASE_3D_SHARE_BOARD_POOL,
        "reproduced_3d_board_pool_magnitude": a1_3d,
        "reproduced_3d_share_board_pool": three_d.get(
            "share_of_a_board_pool_magnitude"
        ),
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "flow_mismatches_control": int(control.get("flow_mismatches_turns") or 0),
        "flow_mismatches_treatment": int(treatment.get("flow_mismatches_turns") or 0),
        "flow_mismatches_punch_control": int(control.get("flow_mismatches_punch") or 0),
        "flow_mismatches_punch_treatment": int(treatment.get("flow_mismatches_punch") or 0),
        "event_flow_mismatches_control": int(control.get("event_flow_mismatches") or 0),
        "event_flow_mismatches_treatment": int(treatment.get("event_flow_mismatches") or 0),
        "combat_post_scale_mismatches_control": int(
            control.get("combat_post_scale_mismatches") or 0
        ),
        "combat_post_scale_mismatches_treatment": int(
            treatment.get("combat_post_scale_mismatches") or 0
        ),
        "a1_lifecycle_hat": reweight.get("phase_3d_board_pool_magnitude_hat"),
        "phase_2v_survivor_tier_sum_delta": PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    })
    base["reweighting_3d"] = {
        "board_pool_magnitude": three_d.get("board_pool_magnitude"),
        "share_of_a_board_pool_magnitude": three_d.get(
            "share_of_a_board_pool_magnitude"
        ),
        "allocation_concentration": three_d.get("allocation_concentration"),
        "combat_mutation": three_d.get("combat_mutation"),
        "attacker_attack_strength": three_d.get("attacker_attack_strength"),
    }
    base["reweighting"] = reweight
    base["reconciliation"] = rec
    base["additive_flow"] = {
        "control": flow_c,
        "treatment": flow_t,
        "delta_treatment_minus_control": flow_delta,
    }
    base["turn_summary"] = {"control": turns_c, "treatment": turns_t}
    src_c = control.get("by_tier_lifecycle") or {}
    src_t = treatment.get("by_tier_lifecycle") or {}

    def _tier_delta(table_c, table_t, key):
        out = {}
        for tier in TIERS:
            k = str(tier)
            a = (table_c or {}).get(k, {}).get(key)
            b = (table_t or {}).get(k, {}).get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    base["by_tier_lifecycle"] = {
        "control": src_c,
        "treatment": src_t,
        "delta_mean_opp_board_pool": _tier_delta(src_c, src_t, "mean_opp_board_pool"),
        "delta_mean_opp_carry": _tier_delta(src_c, src_t, "mean_opp_carry"),
        "delta_mean_opp_scale_add": _tier_delta(src_c, src_t, "mean_opp_scale_add"),
        "delta_mean_opp_replace_loss": _tier_delta(
            src_c, src_t, "mean_opp_replace_loss"
        ),
    }
    base["example_turns"] = {
        "control": control.get("example_turns") or [],
        "treatment": treatment.get("example_turns") or [],
    }
    base["example_replacements"] = {
        "control": control.get("example_replacements") or [],
        "treatment": treatment.get("example_replacements") or [],
    }
    return base


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_3e(comparison, non_evaluative=non_evaluative)

