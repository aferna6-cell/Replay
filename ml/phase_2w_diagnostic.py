"""Phase 2W — last/alive board mix + 2Q full-board replacement tracing.

Observational only. Paired greedy control (2Q/2S OFF) vs treatment
(2Q recruit-value + 2S pool) on consumed DEV 14200–14699.
"""

from __future__ import annotations

import math
import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_BUY0,
    A_SELL0,
    BGEnv,
    MAX_BOARD,
    N_BUY,
    N_SELL,
    EnvMinion,
    greedy_policy,
    board_level_abstract_scaling_enabled,
    recruit_value_stats_enabled,
)
from ml.firestone_composition_reference import (
    TIERS,
    _mean,
    _pctl,
    _raw_distribution,
    _tier_hist,
    _tribe_mix,
    _weighted_mean,
    _weighted_share,
    base_card_id,
    is_golden_card_id,
    load_lookup,
    resolve_card,
)
from ml.phase_2w_prereg import (
    HIGH_TIER_MIN,
    LATE_TURNS,
    assert_seed_range_allowed,
    diagnose_phase_2w,
)

METHODOLOGY_VERSION = "2w_v1"


def classify_sim_minion(m, lookup: Dict, slot: int = 0) -> Dict:
    """Printed-tier / printed-raw row from an env minion or observe() dict."""
    if isinstance(m, EnvMinion):
        card_id = str(m.card_id or "")
        name = str(m.name or "")
        golden = bool(m.golden)
        tribes = list(m.tribes or [])
        recruit_atk = int(m.recruit_attack if m.recruit_attack is not None else m.attack)
        recruit_hp = int(m.recruit_health if m.recruit_health is not None else m.health)
        combat_atk = int(m.attack)
        combat_hp = int(m.health)
        env_tier = int(m.tier or 1)
    else:
        card_id = str(m.get("card_id") or "")
        name = str(m.get("name") or "")
        golden = bool(m.get("golden") or is_golden_card_id(card_id))
        tribes = list(m.get("tribes") or [])
        ra, rh = m.get("recruit_attack"), m.get("recruit_health")
        combat_atk = int(m.get("attack") or 0)
        combat_hp = int(m.get("health") or 0)
        try:
            recruit_atk = int(ra) if ra not in (None, "") else combat_atk
            recruit_hp = int(rh) if rh not in (None, "") else combat_hp
        except (TypeError, ValueError):
            recruit_atk, recruit_hp = combat_atk, combat_hp
        env_tier = int(m.get("tier") or 1)

    ck, kb_golden, path = resolve_card(card_id, name, lookup["kb_id"], lookup["kb_name"])
    golden = bool(golden or kb_golden)
    printed_atk = int(ck.attack or 0) if ck else recruit_atk
    printed_hp = int(ck.health or 0) if ck else recruit_hp
    printed_tier = int(ck.tier) if ck and ck.tier is not None else env_tier
    factor = 2 if golden else 1
    printed_raw = float((printed_atk + printed_hp) * factor)
    base = base_card_id(card_id)
    in_pool = bool(
        (base and base in lookup["pool_ids"])
        or (card_id in lookup["pool_ids"])
        or (name in lookup["pool_names"])
    )
    return {
        "card_id": card_id,
        "base_card_id": base,
        "name": name or (ck.name if ck else ""),
        "golden": golden,
        "joined": ck is not None,
        "resolve_path": path,
        "in_active_pool": in_pool,
        "printed_tier": printed_tier,
        "printed_attack": printed_atk,
        "printed_health": printed_hp,
        "printed_raw": printed_raw,
        "recruit_raw": float(recruit_atk + recruit_hp),
        "combat_raw": float(combat_atk + combat_hp),
        "tribes": tribes,
        "archetype": tribes[0] if tribes else "tribeless",
        "board_slot": int(slot),
        "weight": 1.0,
        "unweighted": 1.0,
        "kb_card_id": ck.card_id if ck else None,
    }


def _terminal_board(player) -> List:
    if getattr(player, "board", None):
        return list(player.board)
    return list(getattr(player, "last_board", None) or [])


class FirestoneBoardTracer:
    """Observational last/alive + T12–T14 + full-board replacement tracer."""

    def __init__(self, lobby_id: int, seed: int, arm: str, lookup: Dict):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.lookup = lookup
        self.game_length: Optional[int] = None
        self.last_boards: List[Dict] = []
        self.late_snapshots: List[Dict] = []
        self.replacements: List[Dict] = []
        self.full_board_decisions = 0
        self.full_board_sells = 0
        self._pending_sell: Optional[Dict] = None
        self._board_full = False
        self._pre_board: List[Dict] = []
        self._pre_shop: List[Dict] = []
        self._last_turn: Dict[int, int] = {}

    def begin_lobby(self, lobby_id: int, _rng_seed: int, _tribes: List[str]) -> None:
        self.lobby_id = lobby_id

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        self._pending_sell = None

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask: List[bool]
    ) -> None:
        board = list(obs.get("board") or [])
        shop = list(obs.get("shop") or [])
        self._pre_board = board
        self._pre_shop = shop
        self._board_full = len(board) >= MAX_BOARD
        if self._board_full:
            self.full_board_decisions += 1

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int, ended: bool,
        player=None,
    ) -> None:
        if self._board_full and A_SELL0 <= action < A_SELL0 + N_SELL:
            self.full_board_sells += 1
            slot = action - A_SELL0
            sold = self._pre_board[slot] if 0 <= slot < len(self._pre_board) else None
            if sold is not None:
                self._pending_sell = {
                    "seat": seat,
                    "turn": turn,
                    "sold": classify_sim_minion(sold, self.lookup, slot),
                    "shop": list(self._pre_shop),
                }
            return
        if (
            self._pending_sell is not None
            and A_BUY0 <= action < A_BUY0 + N_BUY
        ):
            slot = action - A_BUY0
            shop = self._pending_sell["shop"]
            bought = shop[slot] if 0 <= slot < len(shop) else None
            if bought is not None:
                cand = classify_sim_minion(bought, self.lookup, slot)
                inc = self._pending_sell["sold"]
                self.replacements.append({
                    "lobby": self.lobby_id,
                    "seed": self.seed,
                    "arm": self.arm,
                    "seat": self._pending_sell["seat"],
                    "turn": int(self._pending_sell["turn"]),
                    "incumbent_name": inc.get("name"),
                    "incumbent_tier": inc.get("printed_tier"),
                    "incumbent_printed_raw": inc.get("printed_raw"),
                    "incumbent_recruit_raw": inc.get("recruit_raw"),
                    "candidate_name": cand.get("name"),
                    "candidate_tier": cand.get("printed_tier"),
                    "candidate_printed_raw": cand.get("printed_raw"),
                    "delta_printed_tier": (
                        None if cand.get("printed_tier") is None
                        or inc.get("printed_tier") is None
                        else float(cand["printed_tier"]) - float(inc["printed_tier"])
                    ),
                    "delta_printed_raw": (
                        None if cand.get("printed_raw") is None
                        or inc.get("printed_raw") is None
                        else float(cand["printed_raw"]) - float(inc["printed_raw"])
                    ),
                    "increases_printed_tier": bool(
                        (cand.get("printed_tier") or 0) > (inc.get("printed_tier") or 0)
                    ),
                    "increases_printed_raw": bool(
                        (cand.get("printed_raw") or 0) > (inc.get("printed_raw") or 0)
                    ),
                })
            self._pending_sell = None
            return
        if ended:
            self._pending_sell = None

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        self._last_turn[seat] = int(turn)
        if int(turn) in LATE_TURNS and getattr(player, "alive", True):
            rows = [
                classify_sim_minion(m, self.lookup, i)
                for i, m in enumerate(list(player.board or []))
            ]
            self.late_snapshots.append({
                "lobby": self.lobby_id,
                "seat": seat,
                "turn": int(turn),
                "alive": True,
                "placement": None,
                "n": len(rows),
                "minions": rows,
            })
        self._pending_sell = None

    def end_lobby(self, players) -> None:
        for seat, pl in enumerate(players):
            board = _terminal_board(pl)
            rows = [
                classify_sim_minion(m, self.lookup, i)
                for i, m in enumerate(board)
            ]
            self.last_boards.append({
                "lobby": self.lobby_id,
                "seed": self.seed,
                "seat": seat,
                "placement": pl.placement,
                "last_turn": self._last_turn.get(seat),
                "late_game": int(self._last_turn.get(seat) or 0) >= min(LATE_TURNS),
                "n": len(rows),
                "minions": rows,
            })
        for snap in self.late_snapshots:
            pl = players[snap["seat"]]
            snap["placement"] = pl.placement


def run_composition_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
    lookup: Optional[Dict] = None,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    lookup = lookup or load_lookup()
    last_boards: List[Dict] = []
    late: List[Dict] = []
    replacements: List[Dict] = []
    lengths: List[float] = []
    full_decisions = 0
    full_sells = 0

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = FirestoneBoardTracer(i, seed + i, arm, lookup)
                env = BGEnv(seed=seed + i)
                recs = env.play_scripted(
                    [greedy_policy] * env.n_players, recruit_tracer=tracer
                )
                game_length = max((r["turn"] for r in recs), default=0)
                tracer.game_length = game_length
                lengths.append(float(game_length))
                last_boards.extend(tracer.last_boards)
                late.extend(tracer.late_snapshots)
                replacements.extend(tracer.replacements)
                full_decisions += tracer.full_board_decisions
                full_sells += tracer.full_board_sells
                del env
                del tracer

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "board_level_abstract_scaling": bool(board_level_abstract_scaling),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "game_lengths": lengths,
        "last_boards": last_boards,
        "late_snapshots": late,
        "replacements": replacements,
        "full_board_decisions": full_decisions,
        "full_board_sells": full_sells,
    }


def run_greedy_control(lobbies: int, seed: int, lookup: Optional[Dict] = None) -> Dict:
    return run_composition_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
        lookup=lookup,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int, lookup: Optional[Dict] = None) -> Dict:
    return run_composition_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
        lookup=lookup,
    )


def _board_minions(boards: Sequence[Dict]) -> List[Dict]:
    out = []
    for b in boards:
        for m in b.get("minions") or []:
            row = dict(m)
            row["weight"] = 1.0
            row["unweighted"] = 1.0
            out.append(row)
    return out


def summarize_board_set(boards: Sequence[Dict], *, label: str) -> Dict:
    rows = [m for m in _board_minions(boards) if m.get("printed_tier") is not None]
    hist = _tier_hist(rows, weighted=False)
    t4 = sum(hist.get(str(t), 0.0) for t in range(4, 7)) + hist.get("7plus", 0.0)
    t5 = sum(hist.get(str(t), 0.0) for t in range(5, 7)) + hist.get("7plus", 0.0)
    t6 = hist.get("6", 0.0) + hist.get("7plus", 0.0)
    sizes = [float(b.get("n") or 0) for b in boards]
    n_joined = sum(1 for m in rows if m.get("joined"))
    raws = [float(m["printed_raw"]) for m in rows if m.get("printed_raw") is not None]
    names = Counter(m.get("name") for m in rows if m.get("name"))
    n_rows = max(1, len(rows))
    card_freq = [
        {"name": k, "share": v / n_rows, "n": v}
        for k, v in names.most_common(20)
    ]
    return {
        "label": label,
        "n_boards": len(boards),
        "n_minions": len(rows),
        "join_rate": n_joined / n_rows if rows else None,
        "tier_histogram": hist,
        "t4_plus_share": t4,
        "t5_plus_share": t5,
        "t6_plus_share": t6,
        "t6_share": hist.get("6", 0.0),
        "mean_printed_tier": _weighted_mean(rows, "printed_tier", weighted=False),
        "mean_printed_raw": _weighted_mean(rows, "printed_raw", weighted=False),
        "printed_raw": _raw_distribution(rows, weighted=False),
        "mean_recruit_raw": _mean([
            float(m["recruit_raw"]) for m in rows if m.get("recruit_raw") is not None
        ]),
        "mean_combat_raw": _mean([
            float(m["combat_raw"]) for m in rows if m.get("combat_raw") is not None
        ]),
        "golden_share": _weighted_share(rows, lambda r: r.get("golden"), weighted=False),
        "mean_board_size": _mean(sizes),
        "tribe_mix": _tribe_mix(rows, weighted=False),
        "card_frequency_top20": card_freq,
    }


def summarize_replacements(events: Sequence[Dict], firestone_mean_tier, firestone_mean_raw) -> Dict:
    if not events:
        return {
            "n": 0,
            "mean_delta_printed_tier": None,
            "mean_delta_printed_raw": None,
            "share_increase_tier": None,
            "share_increase_raw": None,
            "share_cand_t4_plus": None,
            "mean_cand_tier": None,
            "mean_inc_tier": None,
            "mean_cand_printed_raw": None,
            "mean_inc_printed_raw": None,
            "share_cand_tier_above_firestone": None,
            "share_cand_raw_above_firestone": None,
            "share_delta_tier_positive_vs_firestone_like": None,
        }
    d_tier = [float(e["delta_printed_tier"]) for e in events if e.get("delta_printed_tier") is not None]
    d_raw = [float(e["delta_printed_raw"]) for e in events if e.get("delta_printed_raw") is not None]
    cand_tier = [float(e["candidate_tier"]) for e in events if e.get("candidate_tier") is not None]
    inc_tier = [float(e["incumbent_tier"]) for e in events if e.get("incumbent_tier") is not None]
    cand_raw = [float(e["candidate_printed_raw"]) for e in events if e.get("candidate_printed_raw") is not None]
    inc_raw = [float(e["incumbent_printed_raw"]) for e in events if e.get("incumbent_printed_raw") is not None]
    n = float(len(events))
    fs_t = firestone_mean_tier
    fs_r = firestone_mean_raw
    return {
        "n": len(events),
        "mean_delta_printed_tier": _mean(d_tier),
        "mean_delta_printed_raw": _mean(d_raw),
        "median_delta_printed_tier": _pctl(d_tier, 0.5),
        "median_delta_printed_raw": _pctl(d_raw, 0.5),
        "share_increase_tier": sum(1 for e in events if e.get("increases_printed_tier")) / n,
        "share_increase_raw": sum(1 for e in events if e.get("increases_printed_raw")) / n,
        "share_cand_t4_plus": (
            sum(1 for t in cand_tier if t >= HIGH_TIER_MIN) / len(cand_tier)
            if cand_tier else None
        ),
        "mean_cand_tier": _mean(cand_tier),
        "mean_inc_tier": _mean(inc_tier),
        "mean_cand_printed_raw": _mean(cand_raw),
        "mean_inc_printed_raw": _mean(inc_raw),
        "share_cand_tier_above_firestone": (
            None if fs_t is None or not cand_tier
            else sum(1 for t in cand_tier if t > fs_t) / len(cand_tier)
        ),
        "share_cand_raw_above_firestone": (
            None if fs_r is None or not cand_raw
            else sum(1 for r in cand_raw if r > fs_r) / len(cand_raw)
        ),
        "disproportionate_tier_upgrade": bool(
            d_tier and _mean(d_tier) is not None and _mean(d_tier) > 0.25
            and (fs_t is None or (_mean(cand_tier) or 0) > fs_t)
        ),
        "disproportionate_raw_upgrade": bool(
            d_raw and _mean(d_raw) is not None and _mean(d_raw) > 2.0
            and (fs_r is None or (_mean(cand_raw) or 0) > fs_r)
        ),
    }


def _freq_map(top: Sequence[Dict]) -> Dict[str, float]:
    return {r["name"]: float(r.get("share") or 0.0) for r in top if r.get("name")}


def card_frequency_overlap(sim_top: Sequence[Dict], fs_top: Sequence[Dict]) -> Dict:
    a = _freq_map(sim_top)
    b = _freq_map(fs_top)
    names_a, names_b = set(a), set(b)
    inter = names_a & names_b
    union = names_a | names_b
    # Cosine on the union of reported top-N names.
    keys = sorted(union)
    va = [a.get(k, 0.0) for k in keys]
    vb = [b.get(k, 0.0) for k in keys]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    cosine = (dot / (na * nb)) if na > 1e-12 and nb > 1e-12 else None
    return {
        "jaccard_top20": (len(inter) / len(union)) if union else None,
        "overlap_count": len(inter),
        "cosine_top20": cosine,
        "shared_names": sorted(inter),
    }


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(b) - float(a)


def summarize_arm(raw: Dict, firestone: Dict) -> Dict:
    late_last = [
        b for b in raw["last_boards"]
        if b.get("late_game") or int(b.get("last_turn") or 0) >= min(LATE_TURNS)
    ]
    last = summarize_board_set(late_last, label="last_alive_late")
    last_all = summarize_board_set(raw["last_boards"], label="last_alive_all")
    late_by = {}
    for t in LATE_TURNS:
        snaps = [s for s in raw["late_snapshots"] if int(s.get("turn") or 0) == t]
        late_by[str(t)] = summarize_board_set(snaps, label=f"t{t}")
    fs_w = firestone.get("weighted") or {}
    repl = summarize_replacements(
        raw["replacements"],
        fs_w.get("mean_printed_tier"),
        fs_w.get("mean_printed_raw"),
    )
    replace_rate = (
        raw["full_board_sells"] / raw["full_board_decisions"]
        if raw["full_board_decisions"] else None
    )
    return {
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "board_level_abstract_scaling": raw["board_level_abstract_scaling"],
        "n_lobbies": raw["n_lobbies"],
        "mean_game_length": _mean(list(raw["game_lengths"])),
        "last_alive": last,
        "last_alive_all": last_all,
        "late": late_by,
        "replacements": repl,
        "full_board_decisions": raw["full_board_decisions"],
        "full_board_sells": raw["full_board_sells"],
        "full_board_replace_rate": replace_rate,
        "overlap_last_vs_firestone": card_frequency_overlap(
            last.get("card_frequency_top20") or [],
            fs_w.get("card_frequency_top20") or [],
        ),
        "n_last_minion_join_mismatch": sum(
            1 for b in raw["last_boards"]
            for m in (b.get("minions") or [])
            if not m.get("joined")
        ),
    }


def _pack_last_deltas(control: Dict, treatment: Dict, firestone: Dict) -> Dict:
    fs = firestone.get("weighted") or {}
    c = control.get("last_alive") or {}
    t = treatment.get("last_alive") or {}
    return {
        "control": {k: c.get(k) for k in (
            "t4_plus_share", "t5_plus_share", "t6_plus_share", "t6_share",
            "mean_printed_tier", "mean_printed_raw", "golden_share",
            "mean_board_size", "n_boards", "n_minions",
        )},
        "treatment": {k: t.get(k) for k in (
            "t4_plus_share", "t5_plus_share", "t6_plus_share", "t6_share",
            "mean_printed_tier", "mean_printed_raw", "golden_share",
            "mean_board_size", "n_boards", "n_minions",
        )},
        "firestone": {k: fs.get(k) for k in (
            "t4_plus_share", "t5_plus_share", "t6_plus_share", "t6_share",
            "mean_printed_tier", "mean_printed_raw", "golden_share",
            "mean_board_size",
        )},
        "t4_share_treatment_minus_control": _delta(c.get("t4_plus_share"), t.get("t4_plus_share")),
        "t4_share_treatment_minus_firestone": _delta(fs.get("t4_plus_share"), t.get("t4_plus_share")),
        "t4_share_control_minus_firestone": _delta(fs.get("t4_plus_share"), c.get("t4_plus_share")),
        "mean_printed_tier_treatment_minus_control": _delta(
            c.get("mean_printed_tier"), t.get("mean_printed_tier")
        ),
        "mean_printed_tier_treatment_minus_firestone": _delta(
            fs.get("mean_printed_tier"), t.get("mean_printed_tier")
        ),
        "mean_printed_raw_treatment_minus_control": _delta(
            c.get("mean_printed_raw"), t.get("mean_printed_raw")
        ),
        "mean_printed_raw_treatment_minus_firestone": _delta(
            fs.get("mean_printed_raw"), t.get("mean_printed_raw")
        ),
        "t6_share_treatment_minus_control": _delta(c.get("t6_share"), t.get("t6_share")),
        "t6_share_treatment_minus_firestone": _delta(fs.get("t6_share"), t.get("t6_share")),
        "golden_share_treatment_minus_firestone": _delta(
            fs.get("golden_share"), t.get("golden_share")
        ),
    }


def compare_arms(control: Dict, treatment: Dict, firestone: Dict) -> Dict:
    last = _pack_last_deltas(control, treatment, firestone)
    late_delta = {}
    fs = firestone.get("weighted") or {}
    for t in LATE_TURNS:
        key = str(t)
        c = (control.get("late") or {}).get(key) or {}
        tr = (treatment.get("late") or {}).get(key) or {}
        late_delta[key] = {
            "t4_share_treatment_minus_control": _delta(c.get("t4_plus_share"), tr.get("t4_plus_share")),
            "t4_share_treatment_minus_firestone": _delta(fs.get("t4_plus_share"), tr.get("t4_plus_share")),
            "mean_printed_tier_treatment_minus_control": _delta(
                c.get("mean_printed_tier"), tr.get("mean_printed_tier")
            ),
            "mean_printed_tier_treatment_minus_firestone": _delta(
                fs.get("mean_printed_tier"), tr.get("mean_printed_tier")
            ),
            "mean_printed_raw_treatment_minus_firestone": _delta(
                fs.get("mean_printed_raw"), tr.get("mean_printed_raw")
            ),
            "control": {k: c.get(k) for k in (
                "t4_plus_share", "mean_printed_tier", "mean_printed_raw",
                "n_boards", "golden_share",
            )},
            "treatment": {k: tr.get(k) for k in (
                "t4_plus_share", "mean_printed_tier", "mean_printed_raw",
                "n_boards", "golden_share",
            )},
        }
    return {
        "coverage": firestone.get("coverage"),
        "last_alive": last,
        "late": late_delta,
        "replacements": {
            "control": control.get("replacements"),
            "treatment": treatment.get("replacements"),
        },
        "overlap": {
            "control": control.get("overlap_last_vs_firestone"),
            "treatment": treatment.get("overlap_last_vs_firestone"),
        },
        "replace_rate": {
            "control": control.get("full_board_replace_rate"),
            "treatment": treatment.get("full_board_replace_rate"),
        },
        "reconciliation": {
            "firestone_weight_ok": bool(
                (firestone.get("reconciliation") or {}).get("weight_delta", 1) == 0
                or abs(float((firestone.get("reconciliation") or {}).get("weight_delta") or 1)) < 1e-6
            ),
            "firestone_join_plus_unresolved": (
                firestone.get("reconciliation") or {}
            ).get("join_plus_unresolved"),
            "firestone_n_minions": (firestone.get("reconciliation") or {}).get("n_minions"),
            "last_join_mismatch_control": control.get("n_last_minion_join_mismatch"),
            "last_join_mismatch_treatment": treatment.get("n_last_minion_join_mismatch"),
            "n_last_boards_control": (control.get("last_alive") or {}).get("n_boards"),
            "n_last_boards_treatment": (treatment.get("last_alive") or {}).get("n_boards"),
        },
    }


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2w(comparison, non_evaluative=non_evaluative)
