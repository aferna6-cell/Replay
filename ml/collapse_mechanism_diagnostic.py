"""Phase 2R — replacement-collapse mechanism diagnostic (measurement only).

Instruments every full-board replacement on T8–T14:

  incumbent combat / incumbent recruit-value / candidate recruit
  combat-strength loss of sell→buy→play
  residual scaling added afterward
  next-turn carried strength
  replacement frequency / churn
  death and game-length impact

Does not retune residual scaling, Phase 2J α, or the Phase 2Q toggle default.
Fresh DEV seeds 13700–14199. Confirm 11500–11699 reserved.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
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
INSTRUMENT_TURNS = tuple(range(8, 15))
DECOMP_TURNS = tuple(range(8, 15))
HEADLINE_TURNS = (9, 10, 11, 12)
LOSS_BUCKET_EDGES = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)

# Replacement explains "most" of the post-scale hole when its share ≥ this.
REPLACEMENT_SHARE_THRESHOLD = 0.50

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


def _delta(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _minion_snapshot(m: Dict) -> Dict:
    return {
        "name": m.get("name"),
        "card_id": m.get("card_id"),
        "golden": bool((m.get("tags") or {}).get("PREMIUM") == "1"),
        "combat_raw": combat_raw(m),
        "recruit_raw": recruit_raw(m),
        "attack": float(m.get("attack") or 0),
        "health": float(m.get("health") or 0),
        "recruit_attack": m.get("recruit_attack"),
        "recruit_health": m.get("recruit_health"),
    }


def _histogram(xs: List[float], edges: Sequence[float] = LOSS_BUCKET_EDGES) -> Dict:
    counts = {f"lt_{int(e)}" if i else f"lt_{int(e)}": 0
              for i, e in enumerate(edges)}
    # Stable bucket labels: [0,10), [10,20), ... [640, inf)
    labels = []
    prev = 0.0
    for e in edges:
        labels.append(f"[{int(prev)},{int(e)})")
        prev = e
    labels.append(f">={int(edges[-1])}")
    counts = {lab: 0 for lab in labels}
    for x in xs:
        placed = False
        prev = 0.0
        for i, e in enumerate(edges):
            if x < e:
                counts[labels[i]] += 1
                placed = True
                break
            prev = e
        if not placed:
            counts[labels[-1]] += 1
    n = len(xs)
    return {
        "n": n,
        "counts": counts,
        "shares": {k: (v / n if n else None) for k, v in counts.items()},
    }


def _loss_distribution(xs: List[float]) -> Dict:
    return {
        "n": len(xs),
        "mean": _mean(xs),
        "median": _median(xs),
        "p10": _pctl(xs, 0.10),
        "p25": _pctl(xs, 0.25),
        "p50": _pctl(xs, 0.50),
        "p75": _pctl(xs, 0.75),
        "p90": _pctl(xs, 0.90),
        "p95": _pctl(xs, 0.95),
        "p99": _pctl(xs, 0.99),
        "share_positive_loss": (
            (sum(1 for x in xs if x > 0) / len(xs)) if xs else None
        ),
        "histogram": _histogram(xs),
    }


class CollapseMechanismTracer:
    """Observational sell→buy→play + residual + death tracer (must not mutate)."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.scaling = ScalingBudgetTracer(lobby_id, seed, arm)
        self.replacement_events: List[Dict] = []
        self.seat_turn_rows: List[Dict] = []
        self.combat_rows: List[Dict] = []
        self._obs: Optional[Dict] = None
        self._board_full = False
        self._pending_sells: List[Dict] = []
        self._pending_buys: List[Dict] = []
        self._cur: Optional[Tuple[int, int]] = None
        self._churn: Dict[Tuple[int, int], Dict] = {}
        self._pre_combat: Dict[int, Dict] = {}

    def attach_to_env(self, env: BGEnv) -> None:
        self.scaling.attach_to_env(env)

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id
        self.scaling.begin_lobby(lobby_id, rng_seed, lobby_tribes)

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.scaling.begin_seat_recruit(seat, turn, player)
        self._pending_sells = []
        self._pending_buys = []
        self._cur = (seat, turn)
        if turn in INSTRUMENT_TURNS:
            self._churn[(seat, turn)] = {
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": seat,
                "turn": turn,
                "start_of_recruit_stats": float(player.strength()),
                "start_hp": float(player.hp),
                "full_board_decisions": 0,
                "full_board_sells": 0,
                "completed_replacements": 0,
                "combat_removed": 0.0,
                "recruit_gain_combat": 0.0,
                "recruit_gain_recruit_value": 0.0,
                "unmatched_sells": 0,
            }

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask: List[bool]
    ) -> None:
        self._obs = obs
        board = obs.get("board") or []
        self._board_full = len(board) >= MAX_BOARD
        if self._board_full and turn in INSTRUMENT_TURNS:
            row = self._churn.get((seat, turn))
            if row is not None:
                row["full_board_decisions"] += 1

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int, ended: bool,
        player=None,
    ) -> None:
        if turn not in INSTRUMENT_TURNS or self._obs is None:
            return
        row = self._churn.get((seat, turn))
        obs = self._obs
        board = obs.get("board") or []
        shop = obs.get("shop") or []
        hand = obs.get("hand") or []

        if A_SELL0 <= action < A_SELL0 + N_SELL and self._board_full:
            slot = action - A_SELL0
            if 0 <= slot < len(board):
                inc = _minion_snapshot(board[slot])
                self._pending_sells.append({
                    "incumbent": inc,
                    "board_combat_before": sum(combat_raw(m) for m in board),
                    "board_recruit_before": sum(recruit_raw(m) for m in board),
                })
                if row is not None:
                    row["full_board_sells"] += 1

        elif A_BUY0 <= action < A_BUY0 + N_BUY:
            slot = action - A_BUY0
            if 0 <= slot < len(shop):
                self._pending_buys.append(_minion_snapshot(shop[slot]))

        elif A_PLAY0 <= action < A_PLAY0 + N_PLAY and self._pending_sells:
            slot = action - A_PLAY0
            played = None
            if 0 <= slot < len(hand):
                played = _minion_snapshot(hand[slot])
            sell = self._pending_sells.pop(0)
            candidate = self._pending_buys.pop(0) if self._pending_buys else played
            if candidate is None or played is None:
                return
            inc = sell["incumbent"]
            combat_loss = float(inc["combat_raw"]) - float(played["combat_raw"])
            recruit_delta = (
                float(candidate["recruit_raw"]) - float(inc["recruit_raw"])
            )
            board_after = float(player.strength()) if player is not None else None
            event = {
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": seat,
                "turn": turn,
                "incumbent_name": inc["name"],
                "incumbent_golden": inc["golden"],
                "incumbent_combat": float(inc["combat_raw"]),
                "incumbent_recruit": float(inc["recruit_raw"]),
                "incumbent_attack": inc["attack"],
                "incumbent_health": inc["health"],
                "candidate_name": candidate["name"],
                "candidate_recruit": float(candidate["recruit_raw"]),
                "candidate_combat_at_play": float(played["combat_raw"]),
                "played_recruit": float(played["recruit_raw"]),
                "combat_loss": combat_loss,
                "recruit_delta": recruit_delta,
                "board_combat_before_sell": sell["board_combat_before"],
                "board_combat_after_play": board_after,
                "inflation_ratio": (
                    (float(inc["combat_raw"]) / float(inc["recruit_raw"]))
                    if inc["recruit_raw"] else None
                ),
            }
            self.replacement_events.append(event)
            if row is not None:
                row["completed_replacements"] += 1
                row["combat_removed"] += float(inc["combat_raw"])
                row["recruit_gain_combat"] += float(played["combat_raw"])
                row["recruit_gain_recruit_value"] += float(candidate["recruit_raw"])

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.scaling.end_seat_recruit(seat, turn, player)
        row = self._churn.get((seat, turn))
        if row is not None:
            row["end_of_recruit_stats"] = float(player.strength())
            row["end_hp"] = float(player.hp)
            row["unmatched_sells"] = len(self._pending_sells)
            row["recruit_delta"] = (
                row["end_of_recruit_stats"] - row["start_of_recruit_stats"]
            )
            net = row["recruit_gain_combat"] - row["combat_removed"]
            row["replacement_net_combat"] = net
            row["other_recruit_delta"] = row["recruit_delta"] - net
        self._pending_sells = []
        self._pending_buys = []
        self._obs = None

    def after_scale_all(self, env: BGEnv) -> None:
        self.scaling.after_scale_all(env)
        turn = env.turn
        self._pre_combat = {}
        for seat, player in enumerate(env.players):
            self._pre_combat[seat] = {
                "alive": bool(player.alive),
                "hp": float(player.hp),
                "post_scale": float(player.strength()) if player.board else 0.0,
            }
            row = self._churn.get((seat, turn))
            if row is None:
                continue
            post = self._pre_combat[seat]["post_scale"]
            end = row.get("end_of_recruit_stats")
            row["post_scale_stats"] = post
            row["residual_added"] = (
                (post - float(end)) if end is not None else None
            )
            row["hp_pre_combat"] = float(player.hp)

    def after_combat(self, env: BGEnv) -> None:
        turn = env.turn
        for seat, player in enumerate(env.players):
            pre = self._pre_combat.get(seat) or {}
            was_alive = bool(pre.get("alive"))
            hp_before = pre.get("hp")
            died = was_alive and not player.alive
            damage = None
            if hp_before is not None:
                damage = max(0.0, float(hp_before) - float(player.hp))
            rec = {
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": seat,
                "turn": turn,
                "died": died,
                "alive_after": bool(player.alive),
                "hp_before": hp_before,
                "hp_after": float(player.hp),
                "damage_taken": damage,
                "post_scale_stats": pre.get("post_scale"),
            }
            row = self._churn.get((seat, turn))
            if row is not None:
                row["died_this_combat"] = died
                row["damage_taken"] = damage
                row["alive_after_combat"] = bool(player.alive)
                rec["completed_replacements"] = row.get("completed_replacements")
                rec["combat_removed"] = row.get("combat_removed")
                rec["replacement_net_combat"] = row.get("replacement_net_combat")
                rec["residual_added"] = row.get("residual_added")
            self.combat_rows.append(rec)

    def end_lobby(self, players) -> None:
        self.scaling.end_lobby(players)
        by_seat_turn = {(r["seat"], r["turn"]): r for r in self._churn.values()}
        for (seat, turn), row in by_seat_turn.items():
            nxt = by_seat_turn.get((seat, turn + 1))
            if nxt is not None:
                row["next_turn_carried_strength"] = nxt.get(
                    "start_of_recruit_stats"
                )
                row["survived_to_next_recruit"] = True
            else:
                row["next_turn_carried_strength"] = None
                row["survived_to_next_recruit"] = bool(
                    row.get("alive_after_combat")
                )
        for row in self._churn.values():
            self.seat_turn_rows.append(row)
        self._churn.clear()


def run_collapse_arm(
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
    rows: List[Dict] = []
    replacement_events: List[Dict] = []
    seat_turn_rows: List[Dict] = []
    combat_rows: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        for i in range(lobbies):
            if policy_factory is not None:
                policies = list(policy_factory(i))
            else:
                pol = policy or greedy_policy
                policies = [pol] * 8
            tracer = CollapseMechanismTracer(i, seed + i, arm)
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
            for ev in tracer.replacement_events:
                ev["game_length"] = game_length
            for st_row in tracer.seat_turn_rows:
                st_row["game_length"] = game_length
            all_records.extend(tracer.scaling.records)
            replacement_events.extend(tracer.replacement_events)
            seat_turn_rows.extend(tracer.seat_turn_rows)
            combat_rows.extend(tracer.combat_rows)
            del env

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "records": all_records,
        "rows": rows,
        "replacement_events": replacement_events,
        "seat_turn_rows": seat_turn_rows,
        "combat_rows": combat_rows,
    }


def run_greedy_control(lobbies: int, seed: int) -> Dict:
    return run_collapse_arm(
        lobbies, seed, arm="greedy_control", recruit_value_stats=False,
        policy=greedy_policy,
    )


def run_greedy_treatment(lobbies: int, seed: int) -> Dict:
    return run_collapse_arm(
        lobbies, seed, arm="greedy_treatment", recruit_value_stats=True,
        policy=greedy_policy,
    )


def run_phase_2j_control(
    lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_collapse_arm(
        lobbies, seed, arm="phase_2j_control", recruit_value_stats=False,
        policy_factory=factory,
    )


def run_phase_2j_treatment(
    lobbies: int, seed: int, alpha: float, prior: PersistencePrior,
) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_collapse_arm(
        lobbies, seed, arm="phase_2j_treatment", recruit_value_stats=True,
        policy_factory=factory,
    )


def _counterfactual_residual(end_recruit: float, combat_removed_net: float,
                             budget: Optional[Dict]) -> Optional[float]:
    """Observational residual if replacements had not cratered end-of-recruit.

    Uses the recorded ratio_g / pace_target / clamp; does not re-roll RNG.
    ``combat_removed_net`` is combat_removed − recruit_gain_combat (≥0 when
    selling scaled incumbents for printed shop).
    """
    if budget is None or end_recruit is None:
        return None
    ratio_g = budget.get("ratio_g")
    if ratio_g is None:
        return None
    cf_end = float(end_recruit) + max(0.0, float(combat_removed_net))
    ratio_add = cf_end * (float(ratio_g) - 1.0)
    pace_target = budget.get("pace_target")
    over = max(0.0, cf_end - float(pace_target)) if pace_target is not None else 0.0
    clamp = bool(budget.get("residual_clamp_active"))
    if clamp:
        return max(0.0, ratio_add - over)
    return float(ratio_add)


def summarize_collapse_arm(raw: Dict) -> Dict:
    from ml.fidelity_metrics import (
        aggregate_lobby_dynamics,
        aggregate_turn_curves,
        summarize_divergence,
    )

    records = raw["records"]
    rows = raw["rows"]
    events = raw["replacement_events"]
    seat_turns = raw["seat_turn_rows"]
    combat_rows = raw["combat_rows"]

    agg = aggregate_scaling_budget(records)
    fid = symmetric_absolute_fidelity(records)
    budget_by_lst = {
        (int(r.get("lobby", 0)), int(r["seat"]), int(r["turn"])): r
        for r in records
    }

    losses = [float(e["combat_loss"]) for e in events]
    by_turn_events: Dict[int, List[Dict]] = defaultdict(list)
    for e in events:
        by_turn_events[int(e["turn"])].append(e)

    by_turn_seats: Dict[int, List[Dict]] = defaultdict(list)
    for r in seat_turns:
        by_turn_seats[int(r["turn"])].append(r)

    decomp = {}
    churn = {}
    for t in DECOMP_TURNS:
        seats = by_turn_seats.get(t, [])
        evs = by_turn_events.get(t, [])
        n_seats = len(seats)
        n_repl = len(evs)
        combat_removed = _mean([float(r.get("combat_removed") or 0) for r in seats])
        recruit_gain = _mean([float(r.get("recruit_gain_combat") or 0) for r in seats])
        other = _mean([
            float(r["other_recruit_delta"])
            for r in seats if r.get("other_recruit_delta") is not None
        ])
        residual = _mean([
            float(r["residual_added"])
            for r in seats if r.get("residual_added") is not None
        ])
        start = _mean([
            float(r["start_of_recruit_stats"])
            for r in seats if r.get("start_of_recruit_stats") is not None
        ])
        end = _mean([
            float(r["end_of_recruit_stats"])
            for r in seats if r.get("end_of_recruit_stats") is not None
        ])
        post = _mean([
            float(r["post_scale_stats"])
            for r in seats if r.get("post_scale_stats") is not None
        ])
        carried = _mean([
            float(r["next_turn_carried_strength"])
            for r in seats if r.get("next_turn_carried_strength") is not None
        ])
        repl_net = _mean([
            float(r["replacement_net_combat"])
            for r in seats if r.get("replacement_net_combat") is not None
        ])
        cf_residuals = []
        residual_shrinks = []
        for r in seats:
            bud = budget_by_lst.get(
                (int(r.get("lobby", 0)), int(r["seat"]), int(r["turn"]))
            )
            end_r = r.get("end_of_recruit_stats")
            net_loss = float(r.get("combat_removed") or 0) - float(
                r.get("recruit_gain_combat") or 0
            )
            cf = _counterfactual_residual(end_r, net_loss, bud)
            actual = r.get("residual_added")
            if cf is not None:
                cf_residuals.append(cf)
                if actual is not None:
                    residual_shrinks.append(cf - float(actual))
        decomp[str(t)] = {
            "n_seat_turns": n_seats,
            "n_replacements": n_repl,
            "start_of_recruit": start,
            "combat_removed_by_replacement": combat_removed,
            "recruit_gain_from_replacement": recruit_gain,
            "replacement_net_combat": repl_net,
            "other_recruit_delta": other,
            "end_of_recruit": end,
            "residual_scaling_recovery": residual,
            "post_scale": post,
            "next_turn_carried_strength": carried,
            "counterfactual_residual_if_no_replace_loss": _mean(cf_residuals),
            "residual_shrinkage_from_crater": _mean(residual_shrinks),
            "identity_end": (
                (start or 0) + (repl_net or 0) + (other or 0)
                if start is not None else None
            ),
        }
        full_dec = sum(int(r.get("full_board_decisions") or 0) for r in seats)
        full_sells = sum(int(r.get("full_board_sells") or 0) for r in seats)
        churn[str(t)] = {
            "n_seat_turns": n_seats,
            "n_replacements": n_repl,
            "replacements_per_seat_turn": (n_repl / n_seats) if n_seats else None,
            "full_board_decisions": full_dec,
            "full_board_sells": full_sells,
            "full_board_replace_rate": (
                full_sells / full_dec if full_dec else None
            ),
            "share_seats_with_replacement": (
                (sum(1 for r in seats if r.get("completed_replacements", 0) > 0)
                 / n_seats)
                if n_seats else None
            ),
            "mean_unmatched_sells": _mean([
                float(r.get("unmatched_sells") or 0) for r in seats
            ]),
        }

    loss_by_turn = {
        str(t): _loss_distribution([float(e["combat_loss"]) for e in evs])
        for t, evs in sorted(by_turn_events.items())
    }

    deaths = [c for c in combat_rows if c.get("died")]
    survivors = [c for c in combat_rows if c.get("died") is False]
    death_impact = {
        "n_combat_rows": len(combat_rows),
        "n_deaths": len(deaths),
        "deaths_by_turn": {
            str(t): sum(1 for c in deaths if int(c["turn"]) == t)
            for t in DECOMP_TURNS
        },
        "mean_replacements_on_death_turn": _mean([
            float(c["completed_replacements"])
            for c in deaths if c.get("completed_replacements") is not None
        ]),
        "mean_replacements_on_survivor_turn": _mean([
            float(c["completed_replacements"])
            for c in survivors if c.get("completed_replacements") is not None
        ]),
        "mean_combat_removed_on_death_turn": _mean([
            float(c["combat_removed"])
            for c in deaths if c.get("combat_removed") is not None
        ]),
        "mean_combat_removed_on_survivor_turn": _mean([
            float(c["combat_removed"])
            for c in survivors if c.get("combat_removed") is not None
        ]),
        "mean_damage_taken": _mean([
            float(c["damage_taken"])
            for c in combat_rows if c.get("damage_taken") is not None
        ]),
        "mean_damage_after_replacement_turn": _mean([
            float(c["damage_taken"])
            for c in combat_rows
            if c.get("damage_taken") is not None
            and (c.get("completed_replacements") or 0) > 0
        ]),
        "mean_damage_after_no_replacement_turn": _mean([
            float(c["damage_taken"])
            for c in combat_rows
            if c.get("damage_taken") is not None
            and (c.get("completed_replacements") or 0) == 0
            and c.get("completed_replacements") is not None
        ]),
    }

    window_rows = [
        r for r in seat_turns if int(r.get("turn") or 0) in HEADLINE_TURNS
    ]
    headline = {
        "turns": list(HEADLINE_TURNS),
        "mean_combat_removed": _mean([
            float(r.get("combat_removed") or 0) for r in window_rows
        ]),
        "mean_recruit_gain_combat": _mean([
            float(r.get("recruit_gain_combat") or 0) for r in window_rows
        ]),
        "mean_replacement_net": _mean([
            float(r["replacement_net_combat"])
            for r in window_rows if r.get("replacement_net_combat") is not None
        ]),
        "mean_residual": _mean([
            float(r["residual_added"])
            for r in window_rows if r.get("residual_added") is not None
        ]),
        "mean_residual_shrinkage": _mean([
            (
                (_counterfactual_residual(
                    r.get("end_of_recruit_stats"),
                    float(r.get("combat_removed") or 0) - float(
                        r.get("recruit_gain_combat") or 0),
                    budget_by_lst.get((
                        int(r.get("lobby", 0)), int(r["seat"]), int(r["turn"])
                    )),
                ) or 0.0)
                - float(r["residual_added"])
            )
            for r in window_rows
            if r.get("residual_added") is not None
        ]),
        "n_replacements": sum(
            int(r.get("completed_replacements") or 0) for r in window_rows
        ),
        "n_seat_turns": len(window_rows),
    }

    turn_curves = aggregate_turn_curves(rows)
    lobby = aggregate_lobby_dynamics(rows)
    scaling_by_turn = {
        k: {
            "n": v.get("n"),
            "recruit_delta": v.get("recruit_delta"),
            "scaling_delta": v.get("scaling_delta"),
            "residual_add": v.get("residual_add"),
            "ratio_add": v.get("ratio_add"),
            "pre_scale_over_firestone": v.get("pre_scale_over_firestone"),
            "post_scale_over_firestone": v.get("post_scale_over_firestone"),
            "start_of_recruit_stats": v.get("start_of_recruit_stats"),
            "end_of_recruit_pre_scaling_stats": v.get(
                "end_of_recruit_pre_scaling_stats"
            ),
            "post_scaling_stats": v.get("post_scaling_stats"),
            "firestone_target": v.get("firestone_target"),
        }
        for k, v in (agg.get("by_turn") or {}).items()
    }

    return {
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "n_lobbies": raw["n_lobbies"],
        "seed_base": raw["seed_base"],
        "n_replacement_events": len(events),
        "per_turn_decomposition": decomp,
        "churn_by_turn": churn,
        "replacement_loss_distribution": _loss_distribution(losses),
        "replacement_loss_by_turn": loss_by_turn,
        "headline_t9_t12": headline,
        "death_impact": death_impact,
        "post_scale_fidelity": {
            str(t): (fid.get(str(t)) or {})
            for t in DECOMP_TURNS
        },
        "symmetric_absolute_fidelity_turns_8_14": fid,
        "scaling_by_turn": scaling_by_turn,
        "lobby_dynamics": lobby,
        "headline_end_recruit": summarize_divergence(turn_curves),
        "mean_incumbent_combat": _mean([
            float(e["incumbent_combat"]) for e in events
        ]),
        "mean_incumbent_recruit": _mean([
            float(e["incumbent_recruit"]) for e in events
        ]),
        "mean_candidate_recruit": _mean([
            float(e["candidate_recruit"]) for e in events
        ]),
        "mean_inflation_ratio": _mean([
            float(e["inflation_ratio"])
            for e in events if e.get("inflation_ratio") is not None
        ]),
    }


def _post(arm: Dict, turn: str) -> Optional[float]:
    return (
        (arm.get("post_scale_fidelity") or {}).get(turn) or {}
    ).get("mean_post_scale_over_firestone")


def _decomp(arm: Dict, turn: str) -> Dict:
    return (arm.get("per_turn_decomposition") or {}).get(turn) or {}


def compare_control_treatment(control: Dict, treatment: Dict) -> Dict:
    """Paired control vs 2Q-treatment collapse comparison (report-only)."""
    c_len = (control.get("lobby_dynamics") or {}).get("avg_game_length")
    t_len = (treatment.get("lobby_dynamics") or {}).get("avg_game_length")
    c_rep = control.get("n_replacement_events")
    t_rep = treatment.get("n_replacement_events")
    c_loss = (control.get("replacement_loss_distribution") or {}).get("mean")
    t_loss = (treatment.get("replacement_loss_distribution") or {}).get("mean")
    c_h = control.get("headline_t9_t12") or {}
    t_h = treatment.get("headline_t9_t12") or {}

    paired_post = {}
    paired_alive = {}
    c_alive = (control.get("lobby_dynamics") or {}).get("sim_alive_by_turn") or {}
    t_alive = (treatment.get("lobby_dynamics") or {}).get("sim_alive_by_turn") or {}
    for t in DECOMP_TURNS:
        ts = str(t)
        paired_post[ts] = {
            "control": _post(control, ts),
            "treatment": _post(treatment, ts),
            "delta": _delta(_post(control, ts), _post(treatment, ts)),
        }
        paired_alive[ts] = {
            "control": c_alive.get(ts),
            "treatment": t_alive.get(ts),
            "delta": _delta(c_alive.get(ts), t_alive.get(ts)),
        }

    gap_terms = {}
    for t in DECOMP_TURNS:
        ts = str(t)
        cd = _decomp(control, ts)
        td = _decomp(treatment, ts)
        # control − treatment so a collapse is a positive "hole"
        hole = (
            (float(cd["post_scale"]) - float(td["post_scale"]))
            if cd.get("post_scale") is not None and td.get("post_scale") is not None
            else None
        )
        start_gap = _delta(td.get("start_of_recruit"), cd.get("start_of_recruit"))
        start_hole = (
            (float(cd["start_of_recruit"]) - float(td["start_of_recruit"]))
            if cd.get("start_of_recruit") is not None
            and td.get("start_of_recruit") is not None
            else None
        )
        repl_hole = _delta(
            td.get("replacement_net_combat"), cd.get("replacement_net_combat")
        )
        repl_hole = (
            (float(cd.get("replacement_net_combat") or 0)
             - float(td.get("replacement_net_combat") or 0))
            if (cd.get("replacement_net_combat") is not None
                or td.get("replacement_net_combat") is not None)
            else None
        )
        resid_hole = (
            (float(cd.get("residual_scaling_recovery") or 0)
             - float(td.get("residual_scaling_recovery") or 0))
            if (cd.get("residual_scaling_recovery") is not None
                or td.get("residual_scaling_recovery") is not None)
            else None
        )
        other_hole = (
            (float(cd.get("other_recruit_delta") or 0)
             - float(td.get("other_recruit_delta") or 0))
            if (cd.get("other_recruit_delta") is not None
                or td.get("other_recruit_delta") is not None)
            else None
        )
        shrink_extra = (
            (float(td.get("residual_shrinkage_from_crater") or 0)
             - float(cd.get("residual_shrinkage_from_crater") or 0))
            if (td.get("residual_shrinkage_from_crater") is not None
                or cd.get("residual_shrinkage_from_crater") is not None)
            else None
        )
        gap_terms[ts] = {
            "post_scale_hole_control_minus_treatment": hole,
            "start_of_recruit_hole": start_hole,
            "replacement_net_hole": repl_hole,
            "other_recruit_hole": other_hole,
            "residual_recovery_hole": resid_hole,
            "residual_shrinkage_extra_vs_control": shrink_extra,
            "n_replacements_control": cd.get("n_replacements"),
            "n_replacements_treatment": td.get("n_replacements"),
        }

    t10 = gap_terms.get("10") or {}
    hole10 = t10.get("post_scale_hole_control_minus_treatment")
    repl10 = t10.get("replacement_net_hole")
    resid10 = t10.get("residual_recovery_hole")
    start10 = t10.get("start_of_recruit_hole")

    def _share(part, total):
        if part is None or total is None or abs(total) < 1e-9:
            return None
        return float(part) / float(total)

    window_hole_post = []
    window_repl = []
    window_resid = []
    window_start = []
    for t in HEADLINE_TURNS:
        g = gap_terms.get(str(t)) or {}
        if g.get("post_scale_hole_control_minus_treatment") is not None:
            window_hole_post.append(g["post_scale_hole_control_minus_treatment"])
        if g.get("replacement_net_hole") is not None:
            window_repl.append(g["replacement_net_hole"])
        if g.get("residual_recovery_hole") is not None:
            window_resid.append(g["residual_recovery_hole"])
        if g.get("start_of_recruit_hole") is not None:
            window_start.append(g["start_of_recruit_hole"])

    return {
        "deltas": {
            "n_replacements": _delta(c_rep, t_rep),
            "mean_combat_loss_per_replacement": _delta(c_loss, t_loss),
            "mean_replacement_net_t9_t12": _delta(
                c_h.get("mean_replacement_net"), t_h.get("mean_replacement_net")
            ),
            "mean_combat_removed_t9_t12": _delta(
                c_h.get("mean_combat_removed"), t_h.get("mean_combat_removed")
            ),
            "mean_residual_t9_t12": _delta(
                c_h.get("mean_residual"), t_h.get("mean_residual")
            ),
            "post_scale_over_firestone_t10": _delta(
                _post(control, "10"), _post(treatment, "10")
            ),
            "post_scale_over_firestone_t14": _delta(
                _post(control, "14"), _post(treatment, "14")
            ),
            "mean_game_length": _delta(c_len, t_len),
        },
        "control": {
            "n_replacements": c_rep,
            "mean_combat_loss_per_replacement": c_loss,
            "mean_replacement_net_t9_t12": c_h.get("mean_replacement_net"),
            "mean_combat_removed_t9_t12": c_h.get("mean_combat_removed"),
            "post_scale_over_firestone_t10": _post(control, "10"),
            "post_scale_over_firestone_t14": _post(control, "14"),
            "mean_game_length": c_len,
        },
        "treatment": {
            "n_replacements": t_rep,
            "mean_combat_loss_per_replacement": t_loss,
            "mean_replacement_net_t9_t12": t_h.get("mean_replacement_net"),
            "mean_combat_removed_t9_t12": t_h.get("mean_combat_removed"),
            "post_scale_over_firestone_t10": _post(treatment, "10"),
            "post_scale_over_firestone_t14": _post(treatment, "14"),
            "mean_game_length": t_len,
        },
        "paired_post_scale_firestone": paired_post,
        "paired_alive_curve": paired_alive,
        "gap_decomposition_by_turn": gap_terms,
        "t10_shares_of_post_scale_hole": {
            "replacement_net": _share(repl10, hole10),
            "residual_recovery": _share(resid10, hole10),
            "start_of_recruit_carried": _share(start10, hole10),
        },
        "t9_t12_mean_hole": {
            "post_scale": _mean(window_hole_post),
            "replacement_net": _mean(window_repl),
            "residual_recovery": _mean(window_resid),
            "start_of_recruit_carried": _mean(window_start),
            "replacement_share": _share(_mean(window_repl), _mean(window_hole_post)),
            "residual_share": _share(_mean(window_resid), _mean(window_hole_post)),
        },
        "directional_macro_policy_harm": directional_macro_policy_harm(
            control.get("symmetric_absolute_fidelity_turns_8_14") or {},
            treatment.get("symmetric_absolute_fidelity_turns_8_14") or {},
        ),
    }


def diagnose_phase_2r(
    greedy_cmp: Dict,
    phase_2j_cmp: Optional[Dict] = None,
) -> Dict:
    """Predeclared routing — measurement only, no α / scaling retune."""
    t10 = greedy_cmp.get("t10_shares_of_post_scale_hole") or {}
    win = greedy_cmp.get("t9_t12_mean_hole") or {}
    gaps = greedy_cmp.get("gap_decomposition_by_turn") or {}

    def _share(part, total):
        if part is None or total is None or abs(total) < 1e-9:
            return None
        return float(part) / float(total)

    def _initiated(g: Dict) -> Optional[float]:
        """Replacement net + carried start + crater-induced residual shrink."""
        if not g:
            return None
        parts = [
            g.get("replacement_net_hole") or 0.0,
            max(0.0, g.get("start_of_recruit_hole") or 0.0),
            max(0.0, g.get("residual_shrinkage_extra_vs_control") or 0.0),
        ]
        hole = g.get("post_scale_hole_control_minus_treatment")
        return _share(sum(parts), hole)

    def _independent_residual(g: Dict) -> Optional[float]:
        hole = (g or {}).get("post_scale_hole_control_minus_treatment")
        resid = (g or {}).get("residual_recovery_hole")
        shrink = (g or {}).get("residual_shrinkage_extra_vs_control")
        if resid is None:
            return None
        independent = float(resid) - max(0.0, float(shrink or 0.0))
        return _share(independent, hole)

    t10_gap = gaps.get("10") or {}
    win_initiated = _share(
        (win.get("replacement_net") or 0.0)
        + max(0.0, win.get("start_of_recruit_carried") or 0.0),
        win.get("post_scale"),
    )
    # Mean shrinkage extra over the headline window.
    shrink_parts = []
    for t in HEADLINE_TURNS:
        g = gaps.get(str(t)) or {}
        if g.get("residual_shrinkage_extra_vs_control") is not None:
            shrink_parts.append(float(g["residual_shrinkage_extra_vs_control"]))
    win_shrink = _mean(shrink_parts) if shrink_parts else None
    win_initiated_plus_shrink = _share(
        (win.get("replacement_net") or 0.0)
        + max(0.0, win.get("start_of_recruit_carried") or 0.0)
        + max(0.0, win_shrink or 0.0),
        win.get("post_scale"),
    )
    t10_initiated = _initiated(t10_gap)
    t10_independent_resid = _independent_residual(t10_gap)
    win_independent_resid = None
    if win.get("residual_recovery") is not None:
        win_independent_resid = _share(
            float(win["residual_recovery"]) - max(0.0, win_shrink or 0.0),
            win.get("post_scale"),
        )

    repl_share = win.get("replacement_share")
    resid_share = win.get("residual_share")

    n_t = (greedy_cmp.get("treatment") or {}).get("n_replacements") or 0
    n_c = (greedy_cmp.get("control") or {}).get("n_replacements") or 0
    churn_up = n_t > n_c
    t10_hole = t10_gap.get("post_scale_hole_control_minus_treatment")
    collapse = t10_hole is not None and t10_hole > 50.0

    # Replacement initiates if same-turn net + carried crater + the residual
    # budget that shrinks *because* the board was cratered explain ≥ 50%.
    replacement_explains = (
        (win_initiated_plus_shrink is not None
         and win_initiated_plus_shrink >= REPLACEMENT_SHARE_THRESHOLD)
        or (t10_initiated is not None
            and t10_initiated >= REPLACEMENT_SHARE_THRESHOLD)
    )
    residual_independent = (
        (win_independent_resid is not None
         and win_independent_resid >= REPLACEMENT_SHARE_THRESHOLD)
        or (t10_independent_resid is not None
            and t10_independent_resid >= REPLACEMENT_SHARE_THRESHOLD)
    )

    if collapse and churn_up and replacement_explains:
        primary = "replacement_churn_loss_explains_macro_collapse"
        nxt = (
            "Preserve legitimate accumulated combat value on incumbents "
            "while using unscaled recruit-value for selection. Residual "
            "under-recovery on this run is mostly the cratered-board "
            "budget (ratio_add ∝ current), not an independent pace-formula "
            "defect. Do not retune residual budget or Phase 2J α yet."
        )
    elif collapse and residual_independent and not replacement_explains:
        primary = "residual_pace_coupling_dominates"
        nxt = (
            "Replacement volume is not the main post-scale hole after "
            "crediting crater-induced residual shrink. Inspect residual/"
            "pace coupling (ratio_add vs clamp vs Firestone current). "
            "Do not retune α or burn confirm seeds."
        )
    elif collapse:
        primary = "mixed_replacement_and_residual_coupling"
        nxt = (
            "Neither replacement-initiated share nor independent residual "
            "clears the 50% threshold. Inspect gap_decomposition_by_turn "
            "before designing the next split."
        )
    else:
        primary = "collapse_not_reproduced"
        nxt = (
            "T10 post-scale hole did not reproduce on this DEV band. "
            "Do not advance a combat-preserve design from this run."
        )

    return {
        "primary_finding": primary,
        "replacement_share_threshold": REPLACEMENT_SHARE_THRESHOLD,
        "greedy_replacement_share_t9_t12": repl_share,
        "greedy_residual_share_t9_t12": resid_share,
        "greedy_replacement_initiated_share_t9_t12": win_initiated_plus_shrink,
        "greedy_independent_residual_share_t9_t12": win_independent_resid,
        "greedy_t10_initiated_share": t10_initiated,
        "greedy_t10_independent_residual_share": t10_independent_resid,
        "greedy_t10_shares": t10,
        "greedy_comparison": {
            "deltas": greedy_cmp.get("deltas"),
            "t9_t12_mean_hole": win,
        },
        "phase_2j_comparison_secondary": (
            {
                "deltas": (phase_2j_cmp or {}).get("deltas"),
                "t9_t12_mean_hole": (phase_2j_cmp or {}).get("t9_t12_mean_hole"),
                "t10_shares_of_post_scale_hole": (
                    (phase_2j_cmp or {}).get("t10_shares_of_post_scale_hole")
                ),
                "note": "Phase 2J α=0.5 frozen; report-only, no retune.",
            } if phase_2j_cmp is not None else None
        ),
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_phase_2j_alpha": 0.5,
        "confirm_seeds_reserved": "11500–11699",
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "toggle_default_off": True,
        "measurement_only": True,
        "recommended_next_step": nxt,
    }
