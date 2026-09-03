"""Phase 2R — replacement churn / combat-loss collapse diagnostic (measurement only).

Isolates why Phase 2Q's recruit-value treatment unblocks replaces but collapses
post-scale macro. Instruments every completed full-board replacement on T8–T14:

  - incumbent combat stats vs incumbent recruit-value stats
  - candidate recruit / combat stats
  - combat-strength loss from sell→buy→play (or sell→play)
  - residual scaling added afterward
  - next-turn carried strength
  - replacement frequency / churn per turn
  - death / game-length impact

Control: ``PHASE_2Q_RECRUIT_VALUE_STATS=False``
Treatment: ``PHASE_2Q_RECRUIT_VALUE_STATS=True``

No scaling retune, no α retune, toggle default remains OFF.
Fresh DEV seeds 13700–14199. Confirm 11500–11699 reserved.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_BUY0,
    A_END,
    A_FREEZE,
    A_LEVEL,
    A_PLAY0,
    A_ROLL,
    A_SELL0,
    BGEnv,
    MAX_BOARD,
    N_BUY,
    N_PLAY,
    N_SELL,
    combat_raw,
    greedy_policy,
    recruit_raw,
    recruit_value_stats_enabled,
)
from hsbg_coach.board_opportunity_policy import policies_for_lobby
from hsbg_coach.pace import board_stats
from hsbg_coach.persistence_prior import PersistencePrior
from ml.scaling_budget_diagnostic import (
    ScalingBudgetTracer,
    aggregate_scaling_budget,
    directional_macro_policy_harm,
    symmetric_absolute_fidelity,
)

METHODOLOGY_VERSION = "2r_v1"

PHASE_2R_SEED = 13700
PHASE_2R_LOBBIES = 500
INSTRUMENT_TURNS = tuple(range(8, 15))  # T8–T14 inclusive

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),
    (11000, 11499),
    (11500, 11699),  # confirmation — reserved
    (11700, 12199),  # 2N
    (12200, 12699),  # 2O
    (12700, 13199),  # 2P
    (13200, 13699),  # 2Q
)

# Fraction of the control−treatment post-scale combat gap at T10 that must be
# explained by excess replacement combat loss (net of residual recovery) for the
# churn route.
CHURN_EXPLAINS_FRACTION = 0.55


def recompute_churn_explains_t10(per_turn_control: Dict, per_turn_treatment: Dict) -> Dict:
    """Recompute the published T8–T10 attribution from per-turn tables only.

    ``churn_explains_fraction_t10`` =
        sum_{t=8..10} (treatment − control) mean_net_after_residual
        / (control − treatment) mean_post_scaling_stats at T10

    ``mean_net_after_residual`` is combat_removed − residual_added (per seat-turn).
    Same-turn T10 replacements are *not* required; carry-forward from earlier
    cratering plus later residual undershoot is the intended identity.
    """
    cum = 0.0
    per_turn = {}
    for t in (8, 9, 10):
        key = str(t)
        c_pt = per_turn_control.get(key) or {}
        t_pt = per_turn_treatment.get(key) or {}
        c_net = c_pt.get("mean_net_after_residual")
        t_net = t_pt.get("mean_net_after_residual")
        if c_net is None or t_net is None:
            raise ValueError(f"missing mean_net_after_residual at T{t}")
        d_net = float(t_net) - float(c_net)
        per_turn[key] = {
            "excess_mean_net_after_residual": d_net,
            "control_n_replacements": c_pt.get("n_replacements"),
            "treatment_n_replacements": t_pt.get("n_replacements"),
            "control_mean_residual": c_pt.get("mean_residual_scaling_added"),
            "treatment_mean_residual": t_pt.get("mean_residual_scaling_added"),
        }
        cum += d_net
    c_post = (per_turn_control.get("10") or {}).get("mean_post_scaling_stats")
    t_post = (per_turn_treatment.get("10") or {}).get("mean_post_scaling_stats")
    if c_post is None or t_post is None:
        raise ValueError("missing T10 mean_post_scaling_stats")
    deficit = float(c_post) - float(t_post)
    same_turn = per_turn["10"]["excess_mean_net_after_residual"]
    return {
        "cumulative_excess_net_loss_t8_t10": cum,
        "treatment_post_stats_deficit_t10": deficit,
        "excess_mean_net_loss_t10": same_turn,
        "churn_explains_fraction_t10": (cum / deficit) if deficit > 1e-6 else None,
        "churn_explains_fraction_t10_same_turn": (
            same_turn / deficit if deficit > 1e-6 else None
        ),
        "per_turn": per_turn,
    }


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    lo, hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2R seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return float(st.median(xs)) if xs else None


def _pctl(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    idx = q * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


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


def _minion_snapshot(m: Dict) -> Dict:
    return {
        "name": m.get("name"),
        "card_id": m.get("card_id"),
        "attack": int(m.get("attack") or 0),
        "health": int(m.get("health") or 0),
        "recruit_attack": (
            int(m["recruit_attack"]) if m.get("recruit_attack") is not None
            else int(m.get("attack") or 0)
        ),
        "recruit_health": (
            int(m["recruit_health"]) if m.get("recruit_health") is not None
            else int(m.get("health") or 0)
        ),
        "combat_raw": float(combat_raw(m)),
        "recruit_raw": float(recruit_raw(m)),
        "golden": bool((m.get("tags") or {}).get("PREMIUM") == "1"),
    }


class ReplacementChurnTracer:
    """Track completed full-board replacements and per-turn strength accounting."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.scaling = ScalingBudgetTracer(lobby_id, seed, arm)
        self.replacement_events: List[Dict] = []
        self.turn_rows: List[Dict] = []
        self.incomplete_abandons = 0
        self.full_board_decisions = 0
        self.full_board_sells = 0

        self._pending: Dict[int, Dict] = {}
        self._board_full_at_action = False
        self._pre_obs: Optional[Dict] = None
        self._seat_turn_acc: Dict[Tuple[int, int], Dict] = {}
        self._post_scale: Dict[Tuple[int, int], Dict] = {}
        self._prev_post_scale: Dict[int, Dict] = {}  # seat → last post-scale info

    def attach_to_env(self, env: BGEnv) -> None:
        self.scaling.attach_to_env(env)

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id
        self.scaling.begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._pending.clear()
        self._prev_post_scale.clear()

    def _acc(self, seat: int, turn: int) -> Dict:
        key = (seat, turn)
        if key not in self._seat_turn_acc:
            self._seat_turn_acc[key] = {
                "replacements": 0,
                "combat_removed": 0.0,
                "recruit_gain": 0.0,
                "combat_loss_events": [],
                "sources": Counter(),
            }
        return self._seat_turn_acc[key]

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.scaling.begin_seat_recruit(seat, turn, player)
        # Link previous turn's post-scale row to this turn's carried strength.
        prev = self._prev_post_scale.get(seat)
        if prev is not None and int(prev.get("turn", -1)) == turn - 1:
            prev["next_turn_carried_strength"] = float(player.strength())
            prev["next_turn_alive"] = bool(player.alive)
        if turn in INSTRUMENT_TURNS:
            self._acc(seat, turn)  # ensure row exists even with zero replaces

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask: List[bool]
    ) -> None:
        board = obs.get("board") or []
        self._board_full_at_action = len(board) >= MAX_BOARD
        self._pre_obs = obs
        if self._board_full_at_action and turn in INSTRUMENT_TURNS:
            self.full_board_decisions += 1

    def _abandon_pending(self, seat: int) -> None:
        if seat in self._pending:
            self.incomplete_abandons += 1
            del self._pending[seat]

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int, ended: bool,
        player=None,
    ) -> None:
        kind = _decode_action(action)
        if turn not in INSTRUMENT_TURNS:
            if ended or kind in ("roll", "end", "level"):
                self._abandon_pending(seat)
            return

        obs = self._pre_obs or {}
        board = list(obs.get("board") or [])
        shop = list(obs.get("shop") or [])
        hand = list(obs.get("hand") or [])

        if (
            self._board_full_at_action
            and A_SELL0 <= action < A_SELL0 + N_SELL
        ):
            self.full_board_sells += 1
            slot = action - A_SELL0
            if 0 <= slot < len(board):
                # New sell abandons any incomplete prior pending for this seat.
                if seat in self._pending:
                    self.incomplete_abandons += 1
                inc = _minion_snapshot(board[slot])
                self._pending[seat] = {
                    "seat": seat,
                    "turn": turn,
                    "incumbent": inc,
                    "board_combat_before": float(
                        sum(combat_raw(m) for m in board)
                    ),
                    "board_recruit_before": float(
                        sum(recruit_raw(m) for m in board)
                    ),
                    "candidate": None,
                    "source": None,
                    "sell_slot": slot,
                }
            return

        pending = self._pending.get(seat)
        if pending is None:
            return

        if A_BUY0 <= action < A_BUY0 + N_BUY and pending.get("candidate") is None:
            slot = action - A_BUY0
            if 0 <= slot < len(shop):
                pending["candidate"] = _minion_snapshot(shop[slot])
                pending["source"] = "shop"
            return

        if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
            slot = action - A_PLAY0
            played = None
            if 0 <= slot < len(hand):
                played = _minion_snapshot(hand[slot])
            if pending.get("candidate") is None:
                # sell → play (hand-sourced replacement)
                pending["candidate"] = played
                pending["source"] = "hand"
            # Complete on play (shop path already set candidate on buy).
            self._complete_replacement(seat, turn, player, pending)
            return

        if kind in ("roll", "end", "level") or ended:
            self._abandon_pending(seat)

    def _complete_replacement(
        self, seat: int, turn: int, player, pending: Dict
    ) -> None:
        cand = pending.get("candidate")
        inc = pending["incumbent"]
        if cand is None:
            self._abandon_pending(seat)
            return
        board_combat_after = float(player.strength()) if player is not None else None
        board_recruit_after = None
        if player is not None:
            board_recruit_after = float(
                sum(
                    float(m.recruit_attack or 0) + float(m.recruit_health or 0)
                    for m in player.board
                )
            )
        combat_loss = float(inc["combat_raw"]) - float(cand["combat_raw"])
        recruit_gain = float(cand["recruit_raw"]) - float(inc["recruit_raw"])
        event = {
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "seat": seat,
            "turn": turn,
            "source": pending.get("source"),
            "incumbent_name": inc.get("name"),
            "incumbent_combat_raw": float(inc["combat_raw"]),
            "incumbent_recruit_raw": float(inc["recruit_raw"]),
            "incumbent_attack": inc.get("attack"),
            "incumbent_health": inc.get("health"),
            "incumbent_recruit_attack": inc.get("recruit_attack"),
            "incumbent_recruit_health": inc.get("recruit_health"),
            "incumbent_golden": bool(inc.get("golden")),
            "candidate_name": cand.get("name"),
            "candidate_combat_raw": float(cand["combat_raw"]),
            "candidate_recruit_raw": float(cand["recruit_raw"]),
            "candidate_attack": cand.get("attack"),
            "candidate_health": cand.get("health"),
            "combat_strength_loss": combat_loss,
            "recruit_value_gain": recruit_gain,
            "board_combat_before": pending.get("board_combat_before"),
            "board_combat_after_play": board_combat_after,
            "board_recruit_before": pending.get("board_recruit_before"),
            "board_recruit_after_play": board_recruit_after,
            "net_board_combat_delta": (
                (board_combat_after - float(pending["board_combat_before"]))
                if board_combat_after is not None
                and pending.get("board_combat_before") is not None
                else None
            ),
        }
        self.replacement_events.append(event)
        acc = self._acc(seat, turn)
        acc["replacements"] += 1
        acc["combat_removed"] += combat_loss
        acc["recruit_gain"] += recruit_gain
        acc["combat_loss_events"].append(combat_loss)
        acc["sources"][pending.get("source") or "unknown"] += 1
        del self._pending[seat]

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.scaling.end_seat_recruit(seat, turn, player)
        if seat in self._pending:
            self._abandon_pending(seat)

    def after_scale_all(self, env: BGEnv) -> None:
        self.scaling.after_scale_all(env)
        turn = env.turn
        if turn not in INSTRUMENT_TURNS:
            return
        # Index scaling records just written for this lobby/turn.
        by_seat = {}
        for rec in self.scaling.records:
            if (
                rec.get("lobby") == self.lobby_id
                and int(rec.get("turn") or -1) == turn
            ):
                by_seat[int(rec["seat"])] = rec

        for seat, player in enumerate(env.players):
            key = (seat, turn)
            acc = self._seat_turn_acc.get(key)
            if acc is None and seat not in by_seat:
                continue
            if acc is None:
                acc = self._acc(seat, turn)
            srec = by_seat.get(seat) or {}
            residual = srec.get("residual_add")
            scaling_delta = srec.get("scaling_delta")
            combat_removed = float(acc["combat_removed"])
            recruit_gain = float(acc["recruit_gain"])
            residual_f = float(residual) if residual is not None else 0.0
            row = {
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": seat,
                "turn": turn,
                "alive": bool(player.alive),
                "n_replacements": int(acc["replacements"]),
                "combat_strength_removed": combat_removed,
                "recruit_value_gain": recruit_gain,
                "residual_scaling_added": residual_f,
                "scaling_delta": scaling_delta,
                "net_after_residual": combat_removed - residual_f,
                # Positive net_after_residual ⇒ residual did not fully recover
                # the combat removed by replacements this turn.
                "residual_recovery_ratio": (
                    (residual_f / combat_removed) if combat_removed > 1e-9 else None
                ),
                "start_of_recruit_stats": srec.get("start_of_recruit_stats"),
                "end_of_recruit_pre_scaling_stats": srec.get(
                    "end_of_recruit_pre_scaling_stats"
                ),
                "post_scaling_stats": srec.get("post_scaling_stats"),
                "recruit_delta": srec.get("recruit_delta"),
                "firestone_target": srec.get("firestone_target"),
                "post_scale_over_firestone": srec.get("post_scale_over_firestone"),
                "pre_scale_over_firestone": srec.get("pre_scale_over_firestone"),
                "source_counts": dict(acc["sources"]),
                "next_turn_carried_strength": None,
                "next_turn_alive": None,
            }
            self.turn_rows.append(row)
            self._prev_post_scale[seat] = row
            # Free per-event list to limit memory.
            acc["combat_loss_events"].clear()

        # Drop accumulators for this turn.
        for seat in range(len(env.players)):
            self._seat_turn_acc.pop((seat, turn), None)

    def end_lobby(self, players) -> None:
        self.scaling.end_lobby(players)
        self._pending.clear()


def run_churn_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    policy_factory: Optional[Callable[[int], Sequence[Callable]]] = None,
    policy: Optional[Callable] = None,
    scaling_mode: str = "residual",
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    all_records: List[Dict] = []
    all_events: List[Dict] = []
    all_turn_rows: List[Dict] = []
    rows: List[Dict] = []
    full_board_decisions = 0
    full_board_sells = 0
    incomplete_abandons = 0
    policy_objects: List = []

    with recruit_value_stats_enabled(recruit_value_stats):
        for i in range(lobbies):
            if policy_factory is not None:
                policies = list(policy_factory(i))
            else:
                pol = policy or greedy_policy
                policies = [pol] * 8
            policy_objects.extend(p for p in policies if hasattr(p, "stats"))
            tracer = ReplacementChurnTracer(i, seed + i, arm)
            env = BGEnv(seed=seed + i, scaling_mode=scaling_mode)
            tracer.attach_to_env(env)
            recs = env.play_scripted(policies, recruit_tracer=tracer)
            game_length = max((r["turn"] for r in recs), default=0)
            for r in recs:
                s = r["state"]
                rows.append({
                    "lobby": i,
                    "seed": seed + i,
                    "seat": r["seat"],
                    "turn": r["turn"],
                    "game_length": game_length,
                    "tavern_tier": float(s["tavern_tier"]),
                    "gold": float(s.get("gold") or 0),
                    "board_size": float(len(s.get("board") or [])),
                    "board_stats": float(board_stats(s)),
                    "players_alive": float(s["players_alive"]),
                    "placement": r.get("placement"),
                    "arm": arm,
                })
            all_records.extend(tracer.scaling.records)
            all_events.extend(tracer.replacement_events)
            all_turn_rows.extend(tracer.turn_rows)
            full_board_decisions += tracer.full_board_decisions
            full_board_sells += tracer.full_board_sells
            incomplete_abandons += tracer.incomplete_abandons
            del env

    replace_rate = (
        full_board_sells / full_board_decisions if full_board_decisions else None
    )
    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "records": all_records,
        "rows": rows,
        "replacement_events": all_events,
        "turn_rows": all_turn_rows,
        "full_board_decisions": full_board_decisions,
        "full_board_sells": full_board_sells,
        "full_board_replace_rate": replace_rate,
        "incomplete_abandons": incomplete_abandons,
        "policy_objects": policy_objects,
    }


def run_greedy_control(lobbies: int, seed: int) -> Dict:
    return run_churn_arm(
        lobbies, seed, arm="greedy_control", recruit_value_stats=False,
        policy=greedy_policy,
    )


def run_greedy_treatment(lobbies: int, seed: int) -> Dict:
    return run_churn_arm(
        lobbies, seed, arm="greedy_treatment", recruit_value_stats=True,
        policy=greedy_policy,
    )


def run_phase_2j_control(
    lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_churn_arm(
        lobbies, seed, arm="phase_2j_control", recruit_value_stats=False,
        policy_factory=factory,
    )


def run_phase_2j_treatment(
    lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_churn_arm(
        lobbies, seed, arm="phase_2j_treatment", recruit_value_stats=True,
        policy_factory=factory,
    )


def _loss_distribution(losses: List[float]) -> Dict:
    return {
        "n": len(losses),
        "mean": _mean(losses),
        "median": _median(losses),
        "p10": _pctl(losses, 0.10),
        "p25": _pctl(losses, 0.25),
        "p75": _pctl(losses, 0.75),
        "p90": _pctl(losses, 0.90),
        "p95": _pctl(losses, 0.95),
        "min": float(min(losses)) if losses else None,
        "max": float(max(losses)) if losses else None,
        "share_positive_loss": (
            sum(1 for x in losses if x > 0) / len(losses) if losses else None
        ),
        "share_loss_ge_20": (
            sum(1 for x in losses if x >= 20) / len(losses) if losses else None
        ),
        "share_loss_ge_50": (
            sum(1 for x in losses if x >= 50) / len(losses) if losses else None
        ),
    }


def _per_turn_decomposition(turn_rows: List[Dict], events: List[Dict]) -> Dict:
    by_turn: Dict[str, Dict] = {}
    events_by_turn: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        events_by_turn[str(int(e["turn"]))].append(float(e["combat_strength_loss"]))

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in turn_rows:
        grouped[str(int(r["turn"]))].append(r)

    for t in INSTRUMENT_TURNS:
        key = str(t)
        rows = grouped.get(key) or []
        losses = events_by_turn.get(key) or []
        n_repl = sum(int(r["n_replacements"]) for r in rows)
        combat_removed = [float(r["combat_strength_removed"]) for r in rows]
        recruit_gain = [float(r["recruit_value_gain"]) for r in rows]
        residual = [float(r["residual_scaling_added"] or 0) for r in rows]
        net = [float(r["net_after_residual"]) for r in rows]
        carried = [
            float(r["next_turn_carried_strength"])
            for r in rows if r.get("next_turn_carried_strength") is not None
        ]
        post = [
            float(r["post_scaling_stats"])
            for r in rows if r.get("post_scaling_stats") is not None
        ]
        by_turn[key] = {
            "n_seat_turns": len(rows),
            "n_replacements": n_repl,
            "replacements_per_seat_turn": (
                n_repl / len(rows) if rows else None
            ),
            "mean_combat_strength_removed": _mean(combat_removed),
            "mean_recruit_value_gain": _mean(recruit_gain),
            "mean_residual_scaling_added": _mean(residual),
            "mean_net_after_residual": _mean(net),
            "sum_combat_strength_removed": float(sum(combat_removed)),
            "sum_recruit_value_gain": float(sum(recruit_gain)),
            "sum_residual_scaling_added": float(sum(residual)),
            "mean_post_scaling_stats": _mean(post),
            "mean_next_turn_carried_strength": _mean(carried),
            "replacement_loss_distribution": _loss_distribution(losses),
        }
    return by_turn


def summarize_churn_arm(raw: Dict) -> Dict:
    from ml.fidelity_metrics import (
        aggregate_lobby_dynamics,
        aggregate_turn_curves,
        summarize_divergence,
    )

    records = raw["records"]
    rows = raw["rows"]
    events = raw["replacement_events"]
    turn_rows = raw["turn_rows"]

    agg = aggregate_scaling_budget(records)
    fid = symmetric_absolute_fidelity(records)
    turn_curves = aggregate_turn_curves(rows)
    lobby = aggregate_lobby_dynamics(rows)
    per_turn = _per_turn_decomposition(turn_rows, events)

    losses = [float(e["combat_strength_loss"]) for e in events]
    recruit_gains = [float(e["recruit_value_gain"]) for e in events]
    sources = Counter(e.get("source") or "unknown" for e in events)

    # Aggregate T8–T14 totals for routing.
    sum_combat = sum(float(e["combat_strength_loss"]) for e in events)
    sum_recruit = sum(float(e["recruit_value_gain"]) for e in events)
    sum_residual = sum(
        float(r.get("residual_scaling_added") or 0) for r in turn_rows
    )
    n_seat_turns = len(turn_rows)
    n_repl = len(events)

    policy_stats = None
    pols = raw.get("policy_objects") or []
    if pols:
        from hsbg_coach.board_opportunity_policy import aggregate_policy_stats
        try:
            policy_stats = aggregate_policy_stats(pols)
        except Exception:
            policy_stats = {"n_policies": len(pols)}

    # Alive curve slice T8–T14 from turn_curves.
    alive_curve = {}
    for t in INSTRUMENT_TURNS:
        bucket = turn_curves.get(str(t)) or {}
        alive_curve[str(t)] = {
            "sim_players_alive": bucket.get("sim_players_alive"),
            "reference_alive_prior": bucket.get("reference_alive_prior"),
            "alive_error_vs_prior": bucket.get("alive_error_vs_prior"),
        }

    post_scale = {}
    for t in INSTRUMENT_TURNS:
        bucket = fid.get(str(t)) or {}
        post_scale[str(t)] = {
            "mean_post_scale_over_firestone": bucket.get(
                "mean_post_scale_over_firestone"
            ),
            "mean_pre_scale_over_firestone": bucket.get(
                "mean_pre_scale_over_firestone"
            ),
            "abs_distance_from_one_post": bucket.get("abs_distance_from_one_post"),
        }

    return {
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "n_lobbies": raw["n_lobbies"],
        "seed_base": raw["seed_base"],
        "full_board_replace_rate": raw["full_board_replace_rate"],
        "full_board_decisions": raw["full_board_decisions"],
        "full_board_sells": raw["full_board_sells"],
        "incomplete_abandons": raw["incomplete_abandons"],
        "n_completed_replacements": n_repl,
        "n_seat_turns_instrumented": n_seat_turns,
        "replacements_per_seat_turn": (
            n_repl / n_seat_turns if n_seat_turns else None
        ),
        "headline_t8_t14": {
            "sum_combat_strength_removed": float(sum_combat),
            "sum_recruit_value_gain": float(sum_recruit),
            "sum_residual_scaling_added": float(sum_residual),
            "net_combat_removed_after_residual": float(sum_combat - sum_residual),
            "mean_combat_loss_per_replacement": _mean(losses),
            "mean_recruit_gain_per_replacement": _mean(recruit_gains),
            "source_counts": dict(sources),
        },
        "replacement_loss_distribution": _loss_distribution(losses),
        "per_turn_decomposition": per_turn,
        "post_scale_firestone_ratios": post_scale,
        "alive_curve_t8_t14": alive_curve,
        "symmetric_absolute_fidelity_turns_8_14": fid,
        "lobby_dynamics": lobby,
        "headline_end_recruit": summarize_divergence(turn_curves),
        "aggregation_by_turn": {
            k: {
                "n": v.get("n"),
                "recruit_delta": v.get("recruit_delta"),
                "scaling_delta": v.get("scaling_delta"),
                "post_scale_over_firestone": v.get("post_scale_over_firestone"),
            }
            for k, v in (agg.get("by_turn") or {}).items()
        },
        "policy_stats": policy_stats,
        # Keep a small sample of events for inspection (not the full dump).
        "example_replacement_events": events[:40],
    }


def compare_control_treatment(control: Dict, treatment: Dict) -> Dict:
    def _delta(a, b):
        if a is None or b is None:
            return None
        return float(b) - float(a)

    def _post(arm: Dict, turn: str) -> Optional[float]:
        return (
            (arm.get("post_scale_firestone_ratios") or {}).get(turn) or {}
        ).get("mean_post_scale_over_firestone")

    def _alive(arm: Dict, turn: str) -> Optional[float]:
        return (
            (arm.get("alive_curve_t8_t14") or {}).get(turn) or {}
        ).get("sim_players_alive")

    c_h = control.get("headline_t8_t14") or {}
    t_h = treatment.get("headline_t8_t14") or {}
    c_len = (control.get("lobby_dynamics") or {}).get("avg_game_length")
    t_len = (treatment.get("lobby_dynamics") or {}).get("avg_game_length")

    paired_ratios = {}
    paired_alive = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        c_r = _post(control, key)
        t_r = _post(treatment, key)
        paired_ratios[key] = {
            "control": c_r,
            "treatment": t_r,
            "delta": _delta(c_r, t_r),
        }
        c_a = _alive(control, key)
        t_a = _alive(treatment, key)
        paired_alive[key] = {
            "control": c_a,
            "treatment": t_a,
            "delta": _delta(c_a, t_a),
        }

    harm = directional_macro_policy_harm(
        control.get("symmetric_absolute_fidelity_turns_8_14") or {},
        treatment.get("symmetric_absolute_fidelity_turns_8_14") or {},
    )

    # Excess combat removed by treatment replacements, net of residual recovery.
    c_net = c_h.get("net_combat_removed_after_residual")
    t_net = t_h.get("net_combat_removed_after_residual")
    excess_net_loss = _delta(c_net, t_net)

    # Approximate absolute post-scale combat gap at T10 via mean post stats
    # from per-turn decomposition (more direct than ratio alone).
    c_t10 = (control.get("per_turn_decomposition") or {}).get("10") or {}
    t_t10 = (treatment.get("per_turn_decomposition") or {}).get("10") or {}
    c_post_stats = c_t10.get("mean_post_scaling_stats")
    t_post_stats = t_t10.get("mean_post_scaling_stats")
    post_stats_gap = _delta(t_post_stats, c_post_stats)  # control − treatment > 0
    # Flip: how much lower treatment is vs control.
    treatment_post_deficit = (
        float(c_post_stats) - float(t_post_stats)
        if c_post_stats is not None and t_post_stats is not None
        else None
    )

    # Per-seat-turn mean excess net loss at T10.
    c_mean_net = c_t10.get("mean_net_after_residual")
    t_mean_net = t_t10.get("mean_net_after_residual")
    excess_mean_net_t10 = _delta(c_mean_net, t_mean_net)

    # Cumulative excess unrecovered replacement loss T8–T10 (carry-forward).
    # Same-turn T10 loss alone understates cratering from earlier replaces.
    cum_excess_net = 0.0
    cum_ok = True
    for t in (8, 9, 10):
        c_pt = (control.get("per_turn_decomposition") or {}).get(str(t)) or {}
        t_pt = (treatment.get("per_turn_decomposition") or {}).get(str(t)) or {}
        d_net = _delta(
            c_pt.get("mean_net_after_residual"),
            t_pt.get("mean_net_after_residual"),
        )
        if d_net is None:
            cum_ok = False
            break
        cum_excess_net += float(d_net)

    # Residual coupling signal: treatment residual mean at T10 vs control.
    residual_delta_t10 = _delta(
        c_t10.get("mean_residual_scaling_added"),
        t_t10.get("mean_residual_scaling_added"),
    )

    churn_fraction_same_turn = None
    if (
        treatment_post_deficit is not None
        and excess_mean_net_t10 is not None
        and treatment_post_deficit > 1e-6
    ):
        churn_fraction_same_turn = (
            float(excess_mean_net_t10) / float(treatment_post_deficit)
        )

    churn_fraction = None
    if (
        treatment_post_deficit is not None
        and cum_ok
        and treatment_post_deficit > 1e-6
    ):
        # Primary: cumulative T8–T10 unrecovered replacement loss vs T10 deficit.
        churn_fraction = float(cum_excess_net) / float(treatment_post_deficit)

    per_turn_delta = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        c_pt = (control.get("per_turn_decomposition") or {}).get(key) or {}
        t_pt = (treatment.get("per_turn_decomposition") or {}).get(key) or {}
        per_turn_delta[key] = {
            "replacements_per_seat_turn": _delta(
                c_pt.get("replacements_per_seat_turn"),
                t_pt.get("replacements_per_seat_turn"),
            ),
            "mean_combat_strength_removed": _delta(
                c_pt.get("mean_combat_strength_removed"),
                t_pt.get("mean_combat_strength_removed"),
            ),
            "mean_recruit_value_gain": _delta(
                c_pt.get("mean_recruit_value_gain"),
                t_pt.get("mean_recruit_value_gain"),
            ),
            "mean_residual_scaling_added": _delta(
                c_pt.get("mean_residual_scaling_added"),
                t_pt.get("mean_residual_scaling_added"),
            ),
            "mean_net_after_residual": _delta(
                c_pt.get("mean_net_after_residual"),
                t_pt.get("mean_net_after_residual"),
            ),
            "mean_post_scaling_stats": _delta(
                c_pt.get("mean_post_scaling_stats"),
                t_pt.get("mean_post_scaling_stats"),
            ),
        }

    return {
        "deltas": {
            "full_board_replace_rate": _delta(
                control.get("full_board_replace_rate"),
                treatment.get("full_board_replace_rate"),
            ),
            "n_completed_replacements": _delta(
                control.get("n_completed_replacements"),
                treatment.get("n_completed_replacements"),
            ),
            "sum_combat_strength_removed": _delta(
                c_h.get("sum_combat_strength_removed"),
                t_h.get("sum_combat_strength_removed"),
            ),
            "sum_recruit_value_gain": _delta(
                c_h.get("sum_recruit_value_gain"),
                t_h.get("sum_recruit_value_gain"),
            ),
            "sum_residual_scaling_added": _delta(
                c_h.get("sum_residual_scaling_added"),
                t_h.get("sum_residual_scaling_added"),
            ),
            "net_combat_removed_after_residual": excess_net_loss,
            "mean_combat_loss_per_replacement": _delta(
                c_h.get("mean_combat_loss_per_replacement"),
                t_h.get("mean_combat_loss_per_replacement"),
            ),
            "post_scale_over_firestone_t10": _delta(
                _post(control, "10"), _post(treatment, "10")
            ),
            "post_scale_over_firestone_t14": _delta(
                _post(control, "14"), _post(treatment, "14")
            ),
            "mean_game_length": _delta(c_len, t_len),
            "treatment_post_stats_deficit_t10": treatment_post_deficit,
            "excess_mean_net_loss_t10": excess_mean_net_t10,
            "cumulative_excess_net_loss_t8_t10": (
                float(cum_excess_net) if cum_ok else None
            ),
            "residual_scaling_delta_t10": residual_delta_t10,
            "churn_explains_fraction_t10_same_turn": churn_fraction_same_turn,
            "churn_explains_fraction_t10": churn_fraction,
        },
        "control": {
            "full_board_replace_rate": control.get("full_board_replace_rate"),
            "n_completed_replacements": control.get("n_completed_replacements"),
            "headline_t8_t14": c_h,
            "post_scale_over_firestone_t10": _post(control, "10"),
            "post_scale_over_firestone_t14": _post(control, "14"),
            "mean_game_length": c_len,
            "mean_combat_loss_per_replacement": c_h.get(
                "mean_combat_loss_per_replacement"
            ),
            "replacement_loss_distribution": control.get(
                "replacement_loss_distribution"
            ),
        },
        "treatment": {
            "full_board_replace_rate": treatment.get("full_board_replace_rate"),
            "n_completed_replacements": treatment.get("n_completed_replacements"),
            "headline_t8_t14": t_h,
            "post_scale_over_firestone_t10": _post(treatment, "10"),
            "post_scale_over_firestone_t14": _post(treatment, "14"),
            "mean_game_length": t_len,
            "mean_combat_loss_per_replacement": t_h.get(
                "mean_combat_loss_per_replacement"
            ),
            "replacement_loss_distribution": treatment.get(
                "replacement_loss_distribution"
            ),
        },
        "paired_post_scale_firestone_ratios": paired_ratios,
        "paired_alive_curve": paired_alive,
        "per_turn_decomposition_delta": per_turn_delta,
        "directional_macro_policy_harm": harm,
        "churn_explains_threshold": CHURN_EXPLAINS_FRACTION,
    }


def diagnose_phase_2r(
    greedy_cmp: Dict,
    phase_2j_cmp: Optional[Dict] = None,
) -> Dict:
    """Route: churn/loss explains collapse vs residual/pace coupling."""
    d = greedy_cmp.get("deltas") or {}
    frac = d.get("churn_explains_fraction_t10")
    excess_net = d.get("cumulative_excess_net_loss_t8_t10")
    if excess_net is None:
        excess_net = d.get("excess_mean_net_loss_t10")
    deficit = d.get("treatment_post_stats_deficit_t10")
    replace_up = (
        d.get("full_board_replace_rate") is not None
        and float(d["full_board_replace_rate"]) > 0
    )
    loss_up = (
        d.get("mean_combat_loss_per_replacement") is not None
        and float(d["mean_combat_loss_per_replacement"]) > 0
    ) or (
        d.get("sum_combat_strength_removed") is not None
        and float(d["sum_combat_strength_removed"]) > 0
    )
    post_collapse = (
        d.get("post_scale_over_firestone_t10") is not None
        and float(d["post_scale_over_firestone_t10"]) < -0.1
    )
    length_down = (
        d.get("mean_game_length") is not None
        and float(d["mean_game_length"]) < -0.5
    )

    churn_explains = (
        replace_up
        and loss_up
        and post_collapse
        and frac is not None
        and float(frac) >= CHURN_EXPLAINS_FRACTION
        and excess_net is not None
        and float(excess_net) > 0
        and deficit is not None
        and float(deficit) > 0
    )

    if churn_explains:
        primary = "replacement_churn_loss_explains_macro_collapse"
    elif replace_up and post_collapse and (
        frac is None or float(frac) < CHURN_EXPLAINS_FRACTION
    ):
        primary = "residual_or_pace_coupling_dominates"
    elif replace_up and not post_collapse:
        primary = "churn_up_without_macro_collapse"
    else:
        primary = "inconclusive"

    next_step = {
        "replacement_churn_loss_explains_macro_collapse": (
            "Replacement churn/loss explains most of the post-scale deficit. "
            "Next design should preserve legitimate accumulated combat value "
            "while using unscaled recruit value for selection (e.g. combat "
            "carry-over on replace, or residual budget that does not assume "
            "scaled incumbents persist). Do not retune α; do not burn confirm."
        ),
        "residual_or_pace_coupling_dominates": (
            "Replacement churn rises and macro collapses, but unrecovered "
            "replacement combat loss does not account for most of the T10 "
            "post-scale deficit. Inspect residual/pace coupling (target, "
            "clamp, apply timing) as the residual mechanism. No α retune; "
            "no confirm burn."
        ),
        "churn_up_without_macro_collapse": (
            "Unexpected — churn rose without material post-scale collapse. "
            "Re-check measurement wiring."
        ),
        "inconclusive": "Inspect failed deltas before advancing.",
    }.get(primary, "Inspect failed deltas before advancing.")

    return {
        "primary_finding": primary,
        "greedy_comparison": greedy_cmp,
        "phase_2j_comparison": phase_2j_cmp,
        "phase_2j_report_only": True,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_phase_2j_alpha": 0.5,
        "confirm_seeds_reserved": "11500–11699",
        "feature_toggle_default_off": True,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "churn_explains_fraction_t10": frac,
        "churn_explains_threshold": CHURN_EXPLAINS_FRACTION,
        "recommended_next_step": next_step,
        "game_length_impact": {
            "delta": d.get("mean_game_length"),
            "shortened": bool(length_down),
        },
    }
