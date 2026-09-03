"""Phase 2Q — combat vs recruit-value representation split diagnostic.

Control: ``PHASE_2Q_RECRUIT_VALUE_STATS=False`` (scaled combat stats for
replacement valuation — current contaminated behavior).

Treatment: ``PHASE_2Q_RECRUIT_VALUE_STATS=True`` (recruit-value stats exclude
synthetic residual/ratio scaling; combat unchanged).

Fresh DEV seeds 13200–13699. Confirm 11500–11699 reserved. No scaling retune,
no α retune, no card-effect work.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence

from hsbg_coach.bg_env import (
    A_SELL0,
    BGEnv,
    MAX_BOARD,
    N_SELL,
    greedy_policy,
    recruit_value_stats_enabled,
    valuation_raw,
)
from hsbg_coach.board_opportunity_policy import policies_for_lobby
from hsbg_coach.pace import board_stats
from hsbg_coach.persistence_prior import PersistencePrior
from ml.replacement_value_diagnostic import (
    ReplacementValueTracer,
    summarize_arm as summarize_replacement_arm,
)
from ml.scaling_budget_diagnostic import (
    ScalingBudgetTracer,
    aggregate_scaling_budget,
    directional_macro_policy_harm,
    symmetric_absolute_fidelity,
)

METHODOLOGY_VERSION = "2q_v1"

PHASE_2Q_SEED = 13200
PHASE_2Q_LOBBIES = 500
INSTRUMENT_TURNS = tuple(range(7, 15))
RECRUIT_DELTA_TURNS = (9, 10, 11, 12)

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
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    lo, hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2Q seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


class MultiplexTracer:
    """Compose scaling-budget + replacement tracers; track full-board sells."""

    def __init__(
        self,
        lobby_id: int,
        seed: int,
        arm: str,
        policies: Optional[Sequence] = None,
        *,
        include_composition: bool = False,
    ):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.policies = list(policies or [])
        self.scaling = ScalingBudgetTracer(lobby_id, seed, arm)
        self.replacement = ReplacementValueTracer(
            lobby_id, seed, arm, policies=self.policies
        )
        self.composition = None
        if include_composition:
            from ml.composition_trace import RecruitTracer
            self.composition = RecruitTracer(lobby_id, seed)
        self.full_board_decisions = 0
        self.full_board_sells = 0
        self.sell_by_turn: Dict[int, Counter] = defaultdict(Counter)
        self._board_full_at_action = False

    def attach_to_env(self, env: BGEnv) -> None:
        self.scaling.attach_to_env(env)

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id
        self.scaling.begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self.replacement.begin_lobby(lobby_id, rng_seed, lobby_tribes)
        if self.composition is not None:
            self.composition.begin_lobby(lobby_id, rng_seed, lobby_tribes)

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.scaling.begin_seat_recruit(seat, turn, player)
        if self.composition is not None:
            self.composition.begin_seat_recruit(seat, turn, player)

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask: List[bool]
    ) -> None:
        board = obs.get("board") or []
        self._board_full_at_action = len(board) >= MAX_BOARD
        if self._board_full_at_action and turn in INSTRUMENT_TURNS:
            self.full_board_decisions += 1
        if self.composition is not None:
            self.composition.before_action(seat, turn, shop_generation, obs, mask)
        self.replacement.before_action(seat, turn, shop_generation, obs, mask)
        if self.replacement._last_state_idx is None:
            return
        row = self.replacement.state_rows[self.replacement._last_state_idx]
        if not board:
            return
        wi = row.get("weakest_board_slot")
        if wi is not None and 0 <= int(wi) < len(board):
            m = board[int(wi)]
            ra, rh = m.get("recruit_attack"), m.get("recruit_health")
            if ra is not None and rh is not None:
                row["weakest_board_recruit_raw"] = float(ra) + float(rh)
            row["weakest_board_valuation_raw"] = float(valuation_raw(m))
        v_idx = min(range(len(board)), key=lambda i: valuation_raw(board[i]))
        vm = board[v_idx]
        row["valuation_weakest_slot"] = v_idx
        row["valuation_weakest_raw"] = float(valuation_raw(vm))
        best = float(row["best_shop_printed_raw"])
        natural = float(row["weakest_board_natural_printed_raw"])
        policy_accepts = best > float(valuation_raw(vm))
        base_accepts = best > natural
        row["policy_rule_accepts"] = bool(policy_accepts)
        row["scaling_blocked_under_policy"] = bool(
            base_accepts and not policy_accepts
        )

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int, ended: bool,
        player=None,
    ) -> None:
        if self.composition is not None:
            self.composition.after_action(
                seat, turn, shop_generation, action, ended, player
            )
        self.replacement.after_action(
            seat, turn, shop_generation, action, ended, player
        )
        if (
            self._board_full_at_action
            and turn in INSTRUMENT_TURNS
            and A_SELL0 <= action < A_SELL0 + N_SELL
        ):
            self.full_board_sells += 1
            self.sell_by_turn[turn]["sell"] += 1
        elif self._board_full_at_action and turn in INSTRUMENT_TURNS:
            self.sell_by_turn[turn]["non_sell"] += 1

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.scaling.end_seat_recruit(seat, turn, player)
        self.replacement.end_seat_recruit(seat, turn, player)
        if self.composition is not None:
            self.composition.end_seat_recruit(seat, turn, player)

    def after_scale_all(self, env: BGEnv) -> None:
        self.scaling.after_scale_all(env)

    def end_lobby(self, players) -> None:
        self.scaling.end_lobby(players)
        self.replacement.end_lobby(players)
        if self.composition is not None:
            self.composition.game_length = max(
                (ts.get("turn") or 0 for ts in self.composition.turn_summaries),
                default=0,
            )
            self.composition.end_lobby(players)


def run_split_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    policy_factory: Optional[Callable[[int], Sequence[Callable]]] = None,
    policy: Optional[Callable] = None,
    scaling_mode: str = "residual",
    include_composition: bool = False,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    all_records: List[Dict] = []
    state_rows: List[Dict] = []
    candidate_rows: List[Dict] = []
    rows: List[Dict] = []
    full_board_decisions = 0
    full_board_sells = 0
    sell_by_turn: Dict[str, Dict[str, int]] = {}
    policy_objects: List = []
    comp_events: List[Dict] = []
    comp_turn_summaries: List[Dict] = []
    comp_player_finals: List[Dict] = []
    lobby_meta: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        for i in range(lobbies):
            if policy_factory is not None:
                policies = list(policy_factory(i))
            else:
                pol = policy or greedy_policy
                policies = [pol] * 8
            policy_objects.extend(p for p in policies if hasattr(p, "stats"))
            tracer = MultiplexTracer(
                i, seed + i, arm, policies=policies,
                include_composition=include_composition,
            )
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
            state_rows.extend(tracer.replacement.state_rows)
            # Candidate reject dumps are large; 2Q gates need state rows only.
            tracer.replacement.candidate_rows.clear()
            full_board_decisions += tracer.full_board_decisions
            full_board_sells += tracer.full_board_sells
            for t, counts in tracer.sell_by_turn.items():
                bucket = sell_by_turn.setdefault(
                    str(t), {"sell": 0, "non_sell": 0}
                )
                bucket["sell"] += int(counts.get("sell", 0))
                bucket["non_sell"] += int(counts.get("non_sell", 0))
            if tracer.composition is not None:
                for pf in tracer.composition.player_finals:
                    pf["game_length"] = game_length
                comp_events.extend(tracer.composition.events)
                comp_turn_summaries.extend(tracer.composition.turn_summaries)
                comp_player_finals.extend(tracer.composition.player_finals)
                lobby_meta.append({
                    "lobby": i,
                    "seed": seed + i,
                    "lobby_tribes": list(env.lobby_tribes),
                    "game_length": game_length,
                })
                # Free per-lobby event buffer after copy.
                tracer.composition.events.clear()
            del env

    replace_rate = (
        full_board_sells / full_board_decisions if full_board_decisions else None
    )
    replace_by_turn = {}
    for t, counts in sorted(sell_by_turn.items(), key=lambda kv: int(kv[0])):
        total = counts["sell"] + counts["non_sell"]
        replace_by_turn[t] = {
            **counts,
            "full_board_replace_rate": (
                counts["sell"] / total if total else None
            ),
        }

    composition_traces = None
    if include_composition:
        from hsbg_coach.build_path import load_archetypes
        composition_traces = {
            "lobbies": lobbies,
            "seed": seed,
            "scaling_mode": scaling_mode,
            "events": comp_events,
            "turn_summaries": comp_turn_summaries,
            "player_finals": comp_player_finals,
            "lobby_meta": lobby_meta,
            "archetypes": [
                {"key": a.key, "name": a.name, "tribe": a.tribe,
                 "core_cards": list(a.core.keys()), "board_count": a.board_count}
                for a in load_archetypes()
            ],
        }

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "records": all_records,
        "rows": rows,
        "state_rows": state_rows,
        "candidate_rows": candidate_rows,
        "full_board_decisions": full_board_decisions,
        "full_board_sells": full_board_sells,
        "full_board_replace_rate": replace_rate,
        "full_board_replace_by_turn": replace_by_turn,
        "policy_objects": policy_objects,
        "composition_traces": composition_traces,
    }


def run_greedy_control(lobbies: int, seed: int) -> Dict:
    return run_split_arm(
        lobbies, seed, arm="greedy_control", recruit_value_stats=False,
        policy=greedy_policy,
    )


def run_greedy_treatment(lobbies: int, seed: int) -> Dict:
    return run_split_arm(
        lobbies, seed, arm="greedy_treatment", recruit_value_stats=True,
        policy=greedy_policy,
    )


def run_phase_2j_control(
    lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
    *, include_composition: bool = False,
) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_split_arm(
        lobbies, seed, arm="phase_2j_control", recruit_value_stats=False,
        policy_factory=factory, include_composition=include_composition,
    )


def run_phase_2j_treatment(
    lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
    *, include_composition: bool = False,
) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_split_arm(
        lobbies, seed, arm="phase_2j_treatment", recruit_value_stats=True,
        policy_factory=factory, include_composition=include_composition,
    )


def _recruit_delta_window(agg: Dict, turns=RECRUIT_DELTA_TURNS) -> Dict:
    by_turn = agg.get("by_turn") or {}
    out = {}
    vals = []
    for t in turns:
        bucket = by_turn.get(str(t)) or {}
        rd = bucket.get("recruit_delta")
        out[str(t)] = rd
        if rd is not None:
            vals.append(float(rd))
    out["mean_t9_t12"] = _mean(vals)
    return out


def _post_scale_window(fid: Dict, turns=(9, 10, 11, 12, 14)) -> Dict:
    """``symmetric_absolute_fidelity`` returns flat ``{turn: metrics}``."""
    out = {}
    for t in turns:
        bucket = fid.get(str(t)) or {}
        out[str(t)] = {
            "mean_post_scale_over_firestone": bucket.get(
                "mean_post_scale_over_firestone"
            ),
            "abs_distance_from_one_post": bucket.get("abs_distance_from_one_post"),
            "mean_pre_scale_over_firestone": bucket.get(
                "mean_pre_scale_over_firestone"
            ),
        }
    return out


def summarize_split_arm(raw: Dict) -> Dict:
    from ml.fidelity_metrics import (
        aggregate_lobby_dynamics,
        aggregate_turn_curves,
        summarize_divergence,
    )

    records = raw["records"]
    rows = raw["rows"]
    agg = aggregate_scaling_budget(records)
    fid = symmetric_absolute_fidelity(records)
    replacement = summarize_replacement_arm({
        "arm": raw["arm"],
        "n_lobbies": raw["n_lobbies"],
        "seed_base": raw["seed_base"],
        "state_rows": raw["state_rows"],
        "candidate_rows": raw["candidate_rows"],
    })

    state_rows = raw["state_rows"]
    n = len(state_rows)
    blocked = sum(1 for r in state_rows if r.get("scaling_blocked_under_policy"))
    by_turn: Dict[str, Dict] = {}
    for r in state_rows:
        t = str(int(r["turn"]))
        b = by_turn.setdefault(t, {"n": 0, "blocked": 0})
        b["n"] += 1
        b["blocked"] += int(bool(r.get("scaling_blocked_under_policy")))
    for t, b in by_turn.items():
        b["pct_valuation_blocked"] = b["blocked"] / b["n"] if b["n"] else None

    turn_curves = aggregate_turn_curves(rows)
    lobby = aggregate_lobby_dynamics(rows)

    policy_stats = None
    pols = raw.get("policy_objects") or []
    if pols:
        from hsbg_coach.board_opportunity_policy import aggregate_policy_stats
        try:
            policy_stats = aggregate_policy_stats(pols)
        except Exception:
            policy_stats = {"n_policies": len(pols)}

    mechanism = None
    lifecycle = None
    traces = raw.get("composition_traces")
    if traces is not None:
        from ml.composition_diagnostic import aggregate_diagnostics
        from ml.core_lifecycle_diagnostic import analyze_core_lifecycles
        from ml.phase_2d_acceptance import composition_mechanism_summary
        diagnostic = aggregate_diagnostics(traces)
        lifecycle = analyze_core_lifecycles(traces)
        mechanism = composition_mechanism_summary(diagnostic)
        # Drop heavy traces from returned summary path; caller keeps raw if needed.
        raw["composition_traces"] = None

    return {
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "n_lobbies": raw["n_lobbies"],
        "seed_base": raw["seed_base"],
        "aggregation": {
            "by_turn": {
                k: {
                    "n": v.get("n"),
                    "recruit_delta": v.get("recruit_delta"),
                    "scaling_delta": v.get("scaling_delta"),
                    "post_scale_over_firestone": v.get("post_scale_over_firestone"),
                    "pre_scale_over_firestone": v.get("pre_scale_over_firestone"),
                }
                for k, v in (agg.get("by_turn") or {}).items()
            }
        },
        "recruit_delta_t9_t12": _recruit_delta_window(agg),
        "symmetric_absolute_fidelity_turns_8_14": fid,
        "post_scale_fidelity": _post_scale_window(fid),
        "replacement_headline_t8_t14": replacement.get("contamination_headline"),
        "valuation_scaling_blocked_pct_full_board": (
            (blocked / n) if n else None
        ),
        "valuation_scaling_blocked_by_turn": by_turn,
        "full_board_replace_rate": raw["full_board_replace_rate"],
        "full_board_replace_by_turn": raw["full_board_replace_by_turn"],
        "full_board_decisions": raw["full_board_decisions"],
        "full_board_sells": raw["full_board_sells"],
        "lobby_dynamics": lobby,
        "headline_end_recruit": summarize_divergence(turn_curves),
        "policy_stats": policy_stats,
        "mechanism": mechanism,
        "lifecycle_funnel": (lifecycle or {}).get("funnel") if lifecycle else None,
        "turn_curves_end_recruit": turn_curves,
    }


def compare_control_treatment(control: Dict, treatment: Dict) -> Dict:
    """Primary Phase 2Q mechanism gates (treatment − control)."""
    c_rd = (control.get("recruit_delta_t9_t12") or {}).get("mean_t9_t12")
    t_rd = (treatment.get("recruit_delta_t9_t12") or {}).get("mean_t9_t12")
    c_rep = control.get("full_board_replace_rate")
    t_rep = treatment.get("full_board_replace_rate")
    c_block = control.get("valuation_scaling_blocked_pct_full_board")
    t_block = treatment.get("valuation_scaling_blocked_pct_full_board")

    def _delta(a, b):
        if a is None or b is None:
            return None
        return float(b) - float(a)

    def _post(arm: Dict, turn: str = "10"):
        return (
            (arm.get("post_scale_fidelity") or {}).get(turn) or {}
        ).get("mean_post_scale_over_firestone")

    c_post = _post(control, "10")
    t_post = _post(treatment, "10")
    c_post14 = _post(control, "14")
    t_post14 = _post(treatment, "14")

    harm = directional_macro_policy_harm(
        control.get("symmetric_absolute_fidelity_turns_8_14") or {},
        treatment.get("symmetric_absolute_fidelity_turns_8_14") or {},
    )

    c_len = (control.get("lobby_dynamics") or {}).get("avg_game_length")
    t_len = (treatment.get("lobby_dynamics") or {}).get("avg_game_length")

    recruit_up = c_rd is not None and t_rd is not None and t_rd > c_rd
    replace_up = c_rep is not None and t_rep is not None and t_rep > c_rep
    block_down = (
        c_block is not None and t_block is not None and t_block < c_block * 0.5
    )
    # Material worsen: treatment substantially farther from 1.0 than control at T10.
    post_ok = True
    if c_post is not None and t_post is not None:
        post_ok = abs(t_post - 1.0) <= abs(c_post - 1.0) + 0.15
    length_ok = True
    if c_len is not None and t_len is not None:
        length_ok = t_len >= c_len - 1.5

    gates = {
        "recruit_delta_t9_t12_increases": recruit_up,
        "full_board_replace_rate_increases": replace_up,
        "scaling_blocked_collapses": block_down,
        "post_scale_macro_not_materially_worse": post_ok,
        "game_length_acceptable": length_ok,
    }
    return {
        "deltas": {
            "recruit_delta_mean_t9_t12": _delta(c_rd, t_rd),
            "full_board_replace_rate": _delta(c_rep, t_rep),
            "valuation_scaling_blocked_pct": _delta(c_block, t_block),
            "post_scale_over_firestone_t10": _delta(c_post, t_post),
            "post_scale_over_firestone_t14": _delta(c_post14, t_post14),
            "mean_game_length": _delta(c_len, t_len),
        },
        "control": {
            "recruit_delta_mean_t9_t12": c_rd,
            "full_board_replace_rate": c_rep,
            "valuation_scaling_blocked_pct": c_block,
            "post_scale_over_firestone_t10": c_post,
            "post_scale_over_firestone_t14": c_post14,
            "mean_game_length": c_len,
        },
        "treatment": {
            "recruit_delta_mean_t9_t12": t_rd,
            "full_board_replace_rate": t_rep,
            "valuation_scaling_blocked_pct": t_block,
            "post_scale_over_firestone_t10": t_post,
            "post_scale_over_firestone_t14": t_post14,
            "mean_game_length": t_len,
        },
        "gates": gates,
        "gates_passed": sum(1 for v in gates.values() if v),
        "gates_total": len(gates),
        "directional_macro_policy_harm": harm,
    }


def diagnose_phase_2q(
    greedy_cmp: Dict,
    phase_2j_cmp: Optional[Dict] = None,
    *,
    phase_2j_mechanism: Optional[Dict] = None,
) -> Dict:
    g = greedy_cmp.get("gates") or {}
    replace_ok = bool(g.get("full_board_replace_rate_increases"))
    block_ok = bool(g.get("scaling_blocked_collapses"))
    post_ok = bool(g.get("post_scale_macro_not_materially_worse"))
    recruit_ok = bool(g.get("recruit_delta_t9_t12_increases"))
    length_ok = bool(g.get("game_length_acceptable", True))

    if replace_ok and block_ok and post_ok and (recruit_ok or length_ok):
        primary = "recruit_value_split_mechanism_confirmed"
    elif replace_ok and block_ok and not post_ok:
        primary = "replacement_unblocked_but_post_scale_macro_collapses"
    elif replace_ok and block_ok:
        primary = "recruit_value_split_partial"
    elif not block_ok:
        primary = "scaling_blocked_did_not_collapse"
    else:
        primary = "inconclusive"

    next_step = {
        "recruit_value_split_mechanism_confirmed": (
            "Mechanism confirmed. Keep α=0.5 untuned. Do not freeze; "
            "do not consume confirm seeds."
        ),
        "replacement_unblocked_but_post_scale_macro_collapses": (
            "Replacement contamination is causal, but naive recruit-value "
            "replacement without adjusting residual timing/budget collapses "
            "post-scale macro (boards sell scaled combat stats for printed "
            "shop units). Next: redesign residual interaction or pace "
            "recruit swaps — not α retune, not confirm burn."
        ),
        "recruit_value_split_partial": (
            "Partial mechanism lift; inspect failed gates before advancing."
        ),
        "scaling_blocked_did_not_collapse": (
            "Unexpected — inspect valuation wiring."
        ),
        "inconclusive": "Inspect failed gates before advancing.",
    }.get(primary, "Inspect failed gates before advancing.")

    return {
        "primary_finding": primary,
        "greedy_gates": greedy_cmp,
        "phase_2j_gates": phase_2j_cmp,
        "phase_2j_mechanism": phase_2j_mechanism,
        "keep_pr_29_hold": True,
        "keep_phase_2j_alpha": 0.5,
        "confirm_seeds_reserved": "11500–11699",
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "recommended_next_step": next_step,
    }
