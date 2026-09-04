"""Phase 2T — observational game-length / `_hero_damage` attribution.

Recomputes paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON)
on the already-consumed 14200–14699 DEV band. Instruments combat T7–T14.

Does not change α, scaling math, `_hero_damage`, gates, or defaults.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from hsbg_coach.bg_env import (
    BGEnv,
    EnvMinion,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.phase_2t_prereg import (
    INSTRUMENT_TURNS,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_2t,
)

METHODOLOGY_VERSION = "2t_v1"

# 2S published post-scale / Firestone (greedy 14200–14699). Used as the
# combat-strength-fidelity prior; this hour does not retune those gates.
PHASE_2S_POST_SCALE = {
    "8": {"control": 1.154, "treatment": 1.158},
    "9": {"control": 1.300, "treatment": 1.363},
    "10": {"control": 0.952, "treatment": 1.007},
    "11": {"control": 1.179, "treatment": 1.260},
    "12": {"control": 1.462, "treatment": 1.574},
    "13": {"control": 1.748, "treatment": 1.855},
    "14": {"control": 1.835, "treatment": 1.883},
}


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return float(st.median(xs)) if xs else None


def _board_audit(board: Sequence[EnvMinion]) -> Dict:
    tiers = [int(getattr(m, "tier", 1) or 1) for m in board]
    kws: Counter = Counter()
    for m in board:
        for kw in getattr(m, "keywords", None) or []:
            kws[str(kw)] += 1
        if getattr(m, "golden", False):
            kws["GOLDEN"] += 1
    n = len(tiers)
    strength = 0
    for m in board:
        strength += int(getattr(m, "attack", 0) or 0) + int(
            getattr(m, "health", 0) or 0
        )
    return {
        "n": n,
        "tier_sum": int(sum(tiers)),
        "tier_mean": (float(sum(tiers)) / n) if n else None,
        "strength": int(strength),
        "keywords": dict(kws),
    }


def decompose_hero_damage(
    raw: int,
    winner_tier: int,
    board: Sequence[EnvMinion],
) -> Dict:
    """Observational identity of ``BGEnv._hero_damage`` vs sim.py count-only.

    ``sim.py._damage_to_hero`` returns ``survivor_count + max(tavern_tier, 1)``.
    ``BGEnv._hero_damage`` recovers that count and reweights it by the winner
    board's mean minion tier. Amplification is applied − count-only.
    """
    raw_i = int(raw)
    tier = int(winner_tier)
    applied = int(BGEnv._hero_damage(raw_i, tier, list(board)))
    survivors = max(1, abs(raw_i) - tier)
    audit = _board_audit(board)
    avg_tier = float(audit["tier_mean"] if audit["tier_mean"] is not None else 1.0)
    # sim.py `_damage_to_hero` = survivor_count + max(tavern_tier, 1).
    count_only = int(tier) + int(survivors)
    amplification = applied - count_only
    return {
        "raw": raw_i,
        "winner_tavern_tier": tier,
        "survivor_count": int(survivors),
        "winner_board_n": audit["n"],
        "winner_minion_tier_sum": audit["tier_sum"],
        "winner_minion_tier_mean": audit["tier_mean"],
        "estimated_survivor_tier_sum": float(survivors * avg_tier),
        "count_only_damage": count_only,
        "applied_damage": applied,
        "amplification": int(amplification),
        "raw_abs": abs(raw_i),
    }


def _hit_components(raw: int, winner_tier: int, winner_board: Sequence) -> Dict:
    if int(raw) == 0:
        return {
            "survivor_count": 0,
            "count_only_damage": 0,
            "applied_damage": 0,
            "amplification": 0,
            "winner_minion_tier_sum": 0,
            "winner_minion_tier_mean": None,
            "winner_board_n": len(winner_board),
            "estimated_survivor_tier_sum": 0.0,
        }
    return decompose_hero_damage(raw, winner_tier, winner_board)


class GameLengthDamageTracer:
    """Attach to ``combat_audit_hook``; record fights and HP trajectories."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.fights: List[Dict] = []
        self.hp_rows: List[Dict] = []
        self.eliminations: List[Dict] = []
        self.game_length: Optional[int] = None
        self._alive_at_combat: Dict[int, int] = {}

    def attach_to_env(self, env: BGEnv) -> None:
        env.combat_audit_hook = self.on_fight

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id
        # play_scripted passes lobby_id as the second arg; keep the real seed.

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        turn = int(fight.get("turn") or env.turn)
        kind = fight.get("kind") or "live"
        raw = int(fight.get("raw") or 0)
        applied_hp = int(fight.get("applied") or 0)
        winner_board = fight.get("winner_board") or []
        loser_board = fight.get("loser_board") or []
        winner_tier = int(fight.get("winner_tier") or 1)
        loser_tier = fight.get("loser_tier")
        pre_a = fight.get("pre_hp_a")
        post_a = fight.get("post_hp_a")
        pre_b = fight.get("pre_hp_b")
        post_b = fight.get("post_hp_b")
        comps = _hit_components(raw, winner_tier, winner_board)
        w_audit = _board_audit(winner_board)
        l_audit = _board_audit(loser_board)
        strength_margin = None
        if kind == "live":
            strength_margin = float(w_audit["strength"] - l_audit["strength"])
            if raw < 0:
                # winner is B; margin should be winner − loser already
                pass
        elif kind == "ghost":
            strength_margin = float(w_audit["strength"] - l_audit["strength"])

        loser_pre = None
        if fight.get("loser_seat") == fight.get("seat_a"):
            loser_pre = pre_a
        elif fight.get("loser_seat") == fight.get("seat_b"):
            loser_pre = pre_b
        elif kind == "ghost" and raw < 0:
            loser_pre = pre_a
        lethal = bool(
            applied_hp > 0
            and loser_pre is not None
            and int(loser_pre) > 0
            and int(loser_pre) - applied_hp <= 0
        )
        # Reconcile applied vs HP delta on the losing seat.
        hp_delta = 0
        if pre_a is not None and post_a is not None:
            hp_delta += int(pre_a) - int(post_a)
        if pre_b is not None and post_b is not None:
            hp_delta += int(pre_b) - int(post_b)

        if raw != 0:
            outcome = "win" if raw > 0 else "loss"
            # fight-level: decisive
            fight_outcome = "a_win" if raw > 0 else (
                "b_win" if not fight.get("ghost") else "ghost_win"
            )
            if fight.get("ghost") and raw < 0:
                fight_outcome = "living_loss"
            elif fight.get("ghost") and raw > 0:
                fight_outcome = "living_win"
        elif kind == "bye":
            fight_outcome = "bye"
            outcome = "bye"
        else:
            fight_outcome = "tie"
            outcome = "tie"

        rec = {
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "turn": turn,
            "kind": kind,
            "ghost": bool(fight.get("ghost")),
            "seat_a": fight.get("seat_a"),
            "seat_b": fight.get("seat_b"),
            "raw": raw,
            "applied_hp_loss": applied_hp,
            "hp_delta": hp_delta,
            "pre_hp_a": pre_a,
            "post_hp_a": post_a,
            "pre_hp_b": pre_b,
            "post_hp_b": post_b,
            "winner_seat": fight.get("winner_seat"),
            "loser_seat": fight.get("loser_seat"),
            "winner_tavern_tier": winner_tier,
            "loser_tavern_tier": loser_tier,
            "fight_outcome": fight_outcome,
            "outcome": outcome,
            "lethal": lethal,
            "combat_margin_raw": raw,
            "combat_margin_strength": strength_margin,
            "winner_keywords": w_audit["keywords"],
            "loser_keywords": l_audit["keywords"],
            "winner_strength": w_audit["strength"],
            "loser_strength": l_audit["strength"],
            **{k: comps[k] for k in (
                "survivor_count",
                "count_only_damage",
                "applied_damage",
                "amplification",
                "winner_minion_tier_sum",
                "winner_minion_tier_mean",
                "winner_board_n",
                "estimated_survivor_tier_sum",
            )},
        }
        # Prefer formula applied when a hit landed; bye/tie stay 0.
        if applied_hp <= 0:
            rec["applied_damage"] = 0
            rec["count_only_damage"] = 0
            rec["amplification"] = 0
            rec["survivor_count"] = 0
        self.fights.append(rec)

    def after_combat(self, env: BGEnv) -> None:
        turn = env.turn
        n_alive = sum(1 for p in env.players if p.alive)
        for p in env.players:
            self.hp_rows.append({
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": p.idx,
                "turn": turn,
                "hp": int(p.hp),
                "alive": bool(p.alive),
                "tier": int(p.tier),
                "players_alive": n_alive,
            })
            if (not p.alive) and p.idx not in self._alive_at_combat:
                # already recorded
                continue
        # Record first time a previously-alive seat is dead after combat.
        for p in env.players:
            was = self._alive_at_combat.get(p.idx, True)
            if was and not p.alive:
                self.eliminations.append({
                    "lobby": self.lobby_id,
                    "seat": p.idx,
                    "turn": turn,
                    "hp": int(p.hp),
                    "placement": p.placement,
                })
            self._alive_at_combat[p.idx] = bool(p.alive)

    def end_lobby(self, players) -> None:
        turns = [int(r["turn"]) for r in self.hp_rows]
        self.game_length = max(turns) if turns else 0
        for p in players:
            if p.idx not in {e["seat"] for e in self.eliminations}:
                self.eliminations.append({
                    "lobby": self.lobby_id,
                    "seat": p.idx,
                    "turn": self.game_length,
                    "hp": int(p.hp),
                    "placement": p.placement,
                    "survived": True,
                })


def run_damage_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    fights: List[Dict] = []
    hp_rows: List[Dict] = []
    eliminations: List[Dict] = []
    lengths: List[float] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = GameLengthDamageTracer(i, seed + i, arm)
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
                hp_rows.extend(tracer.hp_rows)
                eliminations.extend(tracer.eliminations)
                del env

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "board_level_abstract_scaling": bool(board_level_abstract_scaling),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "fights": fights,
        "hp_rows": hp_rows,
        "eliminations": eliminations,
        "game_lengths": lengths,
    }


def run_greedy_control(lobbies: int, seed: int) -> Dict:
    return run_damage_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_damage_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _in_window(turn: int) -> bool:
    return int(turn) in INSTRUMENT_TURNS


def summarize_damage_arm(raw: Dict) -> Dict:
    fights_all = raw["fights"]
    fights = [f for f in fights_all if _in_window(f["turn"])]
    hp_rows = [r for r in raw["hp_rows"] if _in_window(r["turn"])]
    lengths = list(raw["game_lengths"])
    elims = raw["eliminations"]

    live = [f for f in fights if f["kind"] == "live"]
    ghost = [f for f in fights if f["kind"] == "ghost"]
    byes = [f for f in fights if f["kind"] == "bye"]
    hits = [f for f in fights if int(f.get("applied_hp_loss") or 0) > 0]
    decisive = [f for f in fights if f["kind"] != "bye" and int(f.get("raw") or 0) != 0]
    ties = [f for f in fights if f["kind"] != "bye" and int(f.get("raw") or 0) == 0]

    n_fights = len(fights)
    n_live = len(live)
    n_resolved = n_live + len(ghost)

    # Alive-at-combat seat-turns from HP rows (post-combat snapshot includes
    # seats that just died this round — use pre-death via fight seats).
    alive_start: Dict[str, int] = Counter()
    hp_alive_mean: Dict[str, List[float]] = defaultdict(list)
    hp_all_mean: Dict[str, List[float]] = defaultdict(list)
    by_lobby_turn_alive: Dict[tuple, int] = {}
    for r in hp_rows:
        key = str(int(r["turn"]))
        hp_all_mean[key].append(float(r["hp"]))
        if r.get("alive") or int(r.get("hp") or 0) > 0:
            hp_alive_mean[key].append(float(r["hp"]))
        by_lobby_turn_alive[(r["lobby"], int(r["turn"]))] = int(
            r.get("players_alive") or 0
        )

    # Seat-turns that entered combat: count unique seats mentioned in fights
    # plus byes, per lobby-turn. Easier: use n_alive before deaths ≈
    # players_alive after + deaths this turn.
    deaths_by_lt: Counter = Counter()
    for e in elims:
        if _in_window(e["turn"]) and not e.get("survived"):
            deaths_by_lt[(e["lobby"], int(e["turn"]))] += 1
    n_alive_seat_turns = 0
    for (lobby, turn), after in by_lobby_turn_alive.items():
        n_alive_seat_turns += after + deaths_by_lt.get((lobby, turn), 0)

    applied_hit = [float(f["applied_hp_loss"]) for f in hits]
    count_hit = [float(f["count_only_damage"]) for f in hits]
    amp_hit = [float(f["amplification"]) for f in hits]
    applied_all = [float(f.get("applied_hp_loss") or 0) for f in fights]
    count_all = [float(f.get("count_only_damage") or 0) for f in fights]
    amp_all = [float(f.get("amplification") or 0) for f in fights]

    total_applied = float(sum(applied_all))
    total_count = float(sum(count_all))
    total_amp = float(sum(amp_all))
    n_lobbies = int(raw["n_lobbies"] or 1)

    p_hit = (
        (len(hits) / n_alive_seat_turns) if n_alive_seat_turns else None
    )
    mean_hit = _mean(applied_hit)
    dpt = (
        (total_applied / n_alive_seat_turns) if n_alive_seat_turns else None
    )

    # Per-turn tables
    by_turn: Dict[str, Dict] = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        ft = [f for f in fights if int(f["turn"]) == t]
        ht = [f for f in ft if int(f.get("applied_hp_loss") or 0) > 0]
        live_t = [f for f in ft if f["kind"] == "live"]
        n_res = len([f for f in ft if f["kind"] != "bye"])
        n_tie = len([f for f in ft if f["kind"] != "bye" and int(f["raw"]) == 0])
        n_win = len([f for f in live_t if int(f["raw"]) > 0])
        n_loss_a = len([f for f in live_t if int(f["raw"]) < 0])
        # fight-level win/loss is symmetric; report A-perspective plus rates
        n_decisive_t = n_res - n_tie
        hp_t = hp_alive_mean.get(key) or []
        by_turn[key] = {
            "n_fights": len(ft),
            "n_live": len(live_t),
            "n_ghost": len([f for f in ft if f["kind"] == "ghost"]),
            "n_bye": len([f for f in ft if f["kind"] == "bye"]),
            "n_hits": len(ht),
            "tie_rate": (n_tie / n_res) if n_res else None,
            "decisive_rate": (n_decisive_t / n_res) if n_res else None,
            "lethal_rate": (
                sum(1 for f in ft if f.get("lethal")) / len(ft) if ft else None
            ),
            "mean_applied": _mean([float(f.get("applied_hp_loss") or 0) for f in ft]),
            "mean_applied_when_hit": _mean([float(f["applied_hp_loss"]) for f in ht]),
            "mean_count_only_when_hit": _mean(
                [float(f["count_only_damage"]) for f in ht]
            ),
            "mean_amplification_when_hit": _mean(
                [float(f["amplification"]) for f in ht]
            ),
            "mean_survivor_count": _mean(
                [float(f["survivor_count"]) for f in ht]
            ),
            "mean_winner_tavern_tier": _mean(
                [float(f["winner_tavern_tier"]) for f in ht]
            ),
            "mean_winner_minion_tier_sum": _mean(
                [float(f["winner_minion_tier_sum"]) for f in ht]
            ),
            "mean_winner_minion_tier_mean": _mean(
                [float(f["winner_minion_tier_mean"] or 0) for f in ht
                 if f.get("winner_minion_tier_mean") is not None]
            ),
            "mean_combat_margin_raw_abs": _mean(
                [abs(float(f.get("raw") or 0)) for f in live_t]
            ),
            "mean_combat_margin_strength_abs": _mean(
                [abs(float(f["combat_margin_strength"]))
                 for f in live_t if f.get("combat_margin_strength") is not None]
            ),
            "mean_hp_alive_after": _mean(hp_t),
            "mean_players_alive_after": _mean(
                [float(r["players_alive"]) for r in hp_rows if int(r["turn"]) == t]
            ),
            "live_a_win_rate": (n_win / len(live_t)) if live_t else None,
            "live_b_win_rate": (n_loss_a / len(live_t)) if live_t else None,
            "live_tie_rate": (
                (len(live_t) - n_win - n_loss_a) / len(live_t) if live_t else None
            ),
        }

    tte = [float(e["turn"]) for e in elims if not e.get("survived")]
    tte_all = [float(e["turn"]) for e in elims]
    hp_t7 = [
        float(r["hp"]) for r in raw["hp_rows"]
        if int(r["turn"]) == 7 and (r.get("alive") or int(r.get("hp") or 0) > 0)
    ]
    # If T7 missing (game ended earlier — rare), use last available pre-window.
    hp_entry = hp_t7

    recon_err = [
        abs(int(f.get("applied_hp_loss") or 0) - int(f.get("hp_delta") or 0))
        for f in fights
    ]
    formula_err = []
    for f in hits:
        formula_err.append(
            abs(int(f["applied_hp_loss"]) - int(f["applied_damage"]))
        )

    return {
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "board_level_abstract_scaling": raw["board_level_abstract_scaling"],
        "n_lobbies": n_lobbies,
        "seed_base": raw["seed_base"],
        "mean_game_length": _mean(lengths),
        "median_game_length": _median(lengths),
        "mean_turns_to_elimination": _mean(tte),
        "mean_turns_to_elimination_incl_survivors": _mean(tte_all),
        "n_eliminations": len(tte),
        "mean_hp_at_t7": _mean(hp_entry),
        "n_fights_t7_t14": n_fights,
        "n_live_fights": n_live,
        "n_ghost_fights": len(ghost),
        "n_byes": len(byes),
        "n_hits": len(hits),
        "n_decisive": len(decisive),
        "n_ties": len(ties),
        "n_alive_seat_turns": n_alive_seat_turns,
        "tie_rate": (len(ties) / n_resolved) if n_resolved else None,
        "decisive_rate": (len(decisive) / n_resolved) if n_resolved else None,
        "ghost_share": (len(ghost) / n_fights) if n_fights else None,
        "lethal_rate": (
            sum(1 for f in fights if f.get("lethal")) / n_fights if n_fights else None
        ),
        "hit_rate_per_alive_seat_turn": p_hit,
        "mean_applied_when_hit": mean_hit,
        "mean_count_only_when_hit": _mean(count_hit),
        "mean_amplification_when_hit": _mean(amp_hit),
        "mean_applied_per_fight": _mean(applied_all),
        "mean_survivor_count_when_hit": _mean(
            [float(f["survivor_count"]) for f in hits]
        ),
        "mean_winner_tavern_tier_when_hit": _mean(
            [float(f["winner_tavern_tier"]) for f in hits]
        ),
        "mean_winner_minion_tier_sum_when_hit": _mean(
            [float(f["winner_minion_tier_sum"]) for f in hits]
        ),
        "mean_winner_minion_tier_mean_when_hit": _mean(
            [float(f["winner_minion_tier_mean"]) for f in hits
             if f.get("winner_minion_tier_mean") is not None]
        ),
        "mean_combat_margin_raw_abs": _mean(
            [abs(float(f.get("raw") or 0)) for f in live]
        ),
        "mean_combat_margin_strength_abs": _mean(
            [abs(float(f["combat_margin_strength"]))
             for f in live if f.get("combat_margin_strength") is not None]
        ),
        "mean_winner_strength": _mean(
            [float(f["winner_strength"]) for f in decisive]
        ),
        "mean_loser_strength": _mean(
            [float(f["loser_strength"]) for f in decisive]
        ),
        "keyword_means_winner": {
            kw: _mean([float((f.get("winner_keywords") or {}).get(kw, 0))
                       for f in decisive])
            for kw in ("DIVINE_SHIELD", "TAUNT", "POISONOUS", "REBORN",
                       "WINDFURY", "CLEAVE", "GOLDEN")
        },
        "total_applied_hp_t7_t14": total_applied,
        "total_count_only_t7_t14": total_count,
        "total_amplification_t7_t14": total_amp,
        "mean_applied_per_lobby": total_applied / n_lobbies,
        "mean_count_only_per_lobby": total_count / n_lobbies,
        "mean_amplification_per_lobby": total_amp / n_lobbies,
        "mean_applied_per_alive_seat_turn": dpt,
        "hp_flow_identity_ok": (
            abs(total_applied - total_count - total_amp) < 1e-6
        ),
        "max_hp_delta_reconcile_err": max(recon_err) if recon_err else 0,
        "max_formula_reconcile_err": max(formula_err) if formula_err else 0,
        "per_turn": by_turn,
        "example_fights": fights[:40],
    }


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(b) - float(a)


def compare_control_treatment(control: Dict, treatment: Dict) -> Dict:
    keys = (
        "mean_game_length",
        "mean_turns_to_elimination",
        "mean_hp_at_t7",
        "tie_rate",
        "decisive_rate",
        "lethal_rate",
        "ghost_share",
        "hit_rate_per_alive_seat_turn",
        "mean_applied_when_hit",
        "mean_count_only_when_hit",
        "mean_amplification_when_hit",
        "mean_applied_per_fight",
        "mean_survivor_count_when_hit",
        "mean_winner_tavern_tier_when_hit",
        "mean_winner_minion_tier_sum_when_hit",
        "mean_winner_minion_tier_mean_when_hit",
        "mean_combat_margin_raw_abs",
        "mean_combat_margin_strength_abs",
        "mean_winner_strength",
        "mean_loser_strength",
        "mean_applied_per_lobby",
        "mean_count_only_per_lobby",
        "mean_amplification_per_lobby",
        "mean_applied_per_alive_seat_turn",
        "n_hits",
        "n_decisive",
        "n_ties",
    )
    deltas = {k: _delta(control.get(k), treatment.get(k)) for k in keys}

    per_turn = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        c = (control.get("per_turn") or {}).get(key) or {}
        t_ = (treatment.get("per_turn") or {}).get(key) or {}
        per_turn[key] = {
            m: _delta(c.get(m), t_.get(m))
            for m in (
                "tie_rate", "decisive_rate", "lethal_rate",
                "mean_applied_when_hit", "mean_count_only_when_hit",
                "mean_amplification_when_hit", "mean_survivor_count",
                "mean_winner_tavern_tier", "mean_winner_minion_tier_mean",
                "mean_combat_margin_raw_abs", "mean_combat_margin_strength_abs",
                "mean_hp_alive_after", "mean_players_alive_after",
            )
        }

    attr = attribute_shortening(control, treatment)
    return {
        "deltas": deltas,
        "control": {
            k: control.get(k) for k in (
                "mean_game_length", "mean_turns_to_elimination",
                "mean_hp_at_t7", "tie_rate", "decisive_rate", "lethal_rate",
                "mean_applied_when_hit", "mean_count_only_when_hit",
                "mean_amplification_when_hit",
                "mean_applied_per_alive_seat_turn",
                "mean_survivor_count_when_hit",
                "mean_winner_tavern_tier_when_hit",
                "mean_winner_minion_tier_mean_when_hit",
                "n_lobbies", "n_fights_t7_t14", "n_hits",
            )
        },
        "treatment": {
            k: treatment.get(k) for k in (
                "mean_game_length", "mean_turns_to_elimination",
                "mean_hp_at_t7", "tie_rate", "decisive_rate", "lethal_rate",
                "mean_applied_when_hit", "mean_count_only_when_hit",
                "mean_amplification_when_hit",
                "mean_applied_per_alive_seat_turn",
                "mean_survivor_count_when_hit",
                "mean_winner_tavern_tier_when_hit",
                "mean_winner_minion_tier_mean_when_hit",
                "n_lobbies", "n_fights_t7_t14", "n_hits",
            )
        },
        "per_turn_delta": per_turn,
        "attribution": attr,
    }


def attribute_shortening(control: Dict, treatment: Dict) -> Dict:
    """Decompose −Δ game length into (a) combat outcome, (b) amplification, (c).

    HP-flow identity on T7–T14 hits:
        applied = count_only + amplification
    Extra per-alive-seat-turn drain splits the same way. First-order turns
    use control dpt as the clock. Lifecycle residual is actual shortening
    minus implied shortening from extra dpt (and T7 HP entry gap).
    """
    c_len = control.get("mean_game_length")
    t_len = treatment.get("mean_game_length")
    actual = (
        float(c_len) - float(t_len)
        if c_len is not None and t_len is not None
        else None
    )
    c_dpt = control.get("mean_applied_per_alive_seat_turn")
    t_dpt = treatment.get("mean_applied_per_alive_seat_turn")
    c_count_hit = control.get("mean_count_only_when_hit")
    t_count_hit = treatment.get("mean_count_only_when_hit")
    c_amp_hit = control.get("mean_amplification_when_hit")
    t_amp_hit = treatment.get("mean_amplification_when_hit")
    c_p = control.get("hit_rate_per_alive_seat_turn")
    t_p = treatment.get("hit_rate_per_alive_seat_turn")
    c_hit = control.get("mean_applied_when_hit")
    t_hit = treatment.get("mean_applied_when_hit")

    d_dpt = _delta(c_dpt, t_dpt)
    implied = None
    if actual is not None and c_dpt and float(c_dpt) > 1e-9 and d_dpt is not None:
        # Extra dpt shortens remaining post-T7 life. Scale by control
        # remaining length after T7 (game_length − 6 recruit/combat cycles
        # before the window, clamped).
        remaining_c = max(1.0, float(c_len) - 6.0) if c_len else 1.0
        implied = remaining_c * (float(d_dpt) / float(c_dpt))

    # Shift-share of Δdpt: Δp * hit_c + p_c * Δhit + cross.
    d_p = _delta(c_p, t_p)
    d_hit = _delta(c_hit, t_hit)
    d_count = _delta(c_count_hit, t_count_hit)
    d_amp = _delta(c_amp_hit, t_amp_hit)
    term_rate = None
    term_count = None
    term_amp = None
    if c_p is not None and c_hit is not None:
        term_rate = (float(d_p) * float(c_hit)) if d_p is not None else None
        if d_count is not None:
            term_count = float(c_p) * float(d_count)
        if d_amp is not None:
            term_amp = float(c_p) * float(d_amp)
    # Rate term is combat-outcome (more/less hits from ties / pairing).
    # Count-only-when-hit term is also combat-outcome (more survivors).
    # Amp-when-hit is damage-model (tier weighting).
    combat_dpt = None
    amp_dpt = None
    if term_rate is not None and term_count is not None:
        combat_dpt = term_rate + term_count
    elif term_rate is not None:
        combat_dpt = term_rate
    if term_amp is not None:
        amp_dpt = term_amp

    share_combat = None
    share_amp = None
    if d_dpt is not None and abs(float(d_dpt)) > 1e-9:
        if combat_dpt is not None:
            share_combat = float(combat_dpt) / float(d_dpt)
        if amp_dpt is not None:
            share_amp = float(amp_dpt) / float(d_dpt)

    implied_combat = None
    implied_amp = None
    if implied is not None and share_combat is not None:
        implied_combat = float(implied) * float(share_combat)
    if implied is not None and share_amp is not None:
        implied_amp = float(implied) * float(share_amp)

    residual_turns = None
    if actual is not None and implied is not None:
        residual_turns = float(actual) - float(implied)
    share_life = None
    if actual is not None and abs(float(actual)) > 1e-9 and residual_turns is not None:
        share_life = float(residual_turns) / float(actual)

    # Combat-strength fidelity prior from published 2S T8–T14 post-scale.
    healthy = all(
        float(v["treatment"]) + 1e-9 >= float(v["control"])
        for v in PHASE_2S_POST_SCALE.values()
    )
    # Also require this run's winner boards are not cratered vs control.
    w_delta = _delta(
        control.get("mean_winner_strength"),
        treatment.get("mean_winner_strength"),
    )
    if w_delta is not None and float(w_delta) < -50:
        healthy = False

    hp_t7_delta = _delta(control.get("mean_hp_at_t7"), treatment.get("mean_hp_at_t7"))

    return {
        "actual_shortening_turns": actual,
        "control_mean_game_length": c_len,
        "treatment_mean_game_length": t_len,
        "control_dpt": c_dpt,
        "treatment_dpt": t_dpt,
        "delta_dpt": d_dpt,
        "delta_hit_rate": d_p,
        "delta_applied_when_hit": d_hit,
        "delta_count_only_when_hit": d_count,
        "delta_amplification_when_hit": d_amp,
        "dpt_term_hit_rate": term_rate,
        "dpt_term_count_only": term_count,
        "dpt_term_amplification": term_amp,
        "dpt_from_combat_outcome": combat_dpt,
        "dpt_from_amplification": amp_dpt,
        "share_of_extra_hp_from_combat_outcome": share_combat,
        "share_of_extra_hp_from_amplification": share_amp,
        "implied_shortening_from_extra_dpt": implied,
        "implied_shortening_from_combat_outcome": implied_combat,
        "implied_shortening_from_amplification": implied_amp,
        "lifecycle_residual_turns": residual_turns,
        "share_of_shortening_unexplained_lifecycle": share_life,
        "mean_hp_at_t7_delta": hp_t7_delta,
        "combat_strength_fidelity_healthy": healthy,
        "phase_2s_post_scale_treatment_ge_control_t8_t14": healthy,
        "share_dominant_threshold": SHARE_DOMINANT,
        "hp_flow_identity_control": control.get("hp_flow_identity_ok"),
        "hp_flow_identity_treatment": treatment.get("hp_flow_identity_ok"),
    }


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2t(comparison, non_evaluative=non_evaluative)
