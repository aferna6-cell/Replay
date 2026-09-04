"""Phase 2O — midgame residual scaling-budget diagnostic (measurement only).

Decomposes Firestone target vs start-recruit / recruit delta / scaling add /
remaining gap for turns 7–14. Does not change scaling math, α, pool, economy,
card effects, combat, or PPO.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import BGEnv, greedy_policy
from hsbg_coach.board_opportunity_policy import policies_for_lobby
from hsbg_coach.pace import board_stats
from hsbg_coach.persistence_prior import PersistencePrior

METHODOLOGY_VERSION = "2o_v1"

INSTRUMENT_TURNS = tuple(range(7, 15))
SYMMETRIC_FIDELITY_TURNS = tuple(range(8, 15))

PHASE_2O_SEED = 12200
PHASE_2O_LOBBIES = 500

# Confirm + prior reserved ranges (must not be consumed by Phase 2O).
FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),
    (11000, 11499),  # 2n_v1 consumed
    (11500, 11699),  # confirmation — reserved
    (11700, 12199),  # 2n_v2/v3 consumed
)

ROUTING_TABLE = (
    ("pre_scale_near_firestone_scaling_wrong",
     "scaling formula defect"),
    ("pre_scale_far_below_post_still_far",
     "target-gap bridge defect"),
    ("recruit_contribution_collapsed",
     "recruit/effect-value fidelity"),
    ("just_leveled_factor_explains_deficit",
     "leveling-growth penalty"),
    ("greedy_healthy_phase_2j_low",
     "policy issue"),
    ("both_arms_low",
     "simulator-level issue"),
    ("late_scaling_overcompensates",
     "growth timing/budget redistribution"),
)

# Prospective (next-eval) directional policy-harm metric — NOT applied to 2n_v3.
PROSPECTIVE_MACRO_POLICY_HARM = {
    "name": "directional_macro_policy_harm",
    "definition": (
        "macro_policy_harm = |treatment_ratio - 1| - |greedy_ratio - 1|; "
        "harmful when treatment moves substantially farther from Firestone "
        "than greedy, not merely when it differs from greedy."
    ),
    "note": (
        "Do not retroactively pass/fail 2n_v3 with this metric. "
        "Predeclare for the next fresh evaluation after any Phase 2P fix."
    ),
}


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    lo, hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2O seed range {lo}–{hi} overlaps forbidden "
                f"{flo}–{fhi}")


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


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


class ScalingBudgetTracer:
    """Observational recruit/scaling tracer for Phase 2O (must not mutate)."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.records: List[Dict] = []
        self._pending: Dict[Tuple[int, int], Dict] = {}
        self._budget_by_seat: Dict[int, Dict] = {}

    def begin_lobby(self, lobby_id: int, _rng_seed: int,
                    lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        if turn not in INSTRUMENT_TURNS:
            return
        self._pending[(seat, turn)] = {
            "start_of_recruit_stats": float(player.strength()),
            "start_tavern_tier": float(player.tier),
        }

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        if turn not in INSTRUMENT_TURNS:
            return
        key = (seat, turn)
        base = self._pending.pop(key, {
            "start_of_recruit_stats": None,
            "start_tavern_tier": float(player.tier),
        })
        self._pending[key] = {
            **base,
            "end_of_recruit_pre_scaling_stats": float(player.strength()),
            "tavern_tier": float(player.tier),
        }

    def _on_scaling_audit(self, env, player, seat: int, budget: Dict) -> None:
        if env.turn not in INSTRUMENT_TURNS:
            return
        self._budget_by_seat[int(seat)] = dict(budget)

    def attach_to_env(self, env: BGEnv) -> None:
        env.scaling_audit_hook = self._on_scaling_audit

    def after_scale_all(self, env: BGEnv) -> None:
        turn = env.turn
        if turn not in INSTRUMENT_TURNS:
            self._budget_by_seat.clear()
            return
        for seat, player in enumerate(env.players):
            if not player.alive and seat not in self._budget_by_seat:
                # Still record seats that had recruit this turn.
                pass
            key = (seat, turn)
            pending = self._pending.pop(key, None)
            budget = self._budget_by_seat.pop(seat, None)
            if pending is None and budget is None:
                continue
            start = (pending or {}).get("start_of_recruit_stats")
            pre = None
            if budget is not None:
                pre = budget.get("end_of_recruit_pre_scaling_stats")
            elif pending is not None:
                pre = pending.get("end_of_recruit_pre_scaling_stats")
            post = float(player.strength()) if player.board else (
                float(pre) if pre is not None else 0.0)
            firestone = (budget or {}).get("firestone_target")
            recruit_delta = None
            if start is not None and pre is not None:
                recruit_delta = float(pre) - float(start)
            scaling_delta = None
            if pre is not None:
                scaling_delta = float(post) - float(pre)
            remaining = None
            if firestone is not None:
                remaining = float(firestone) - float(post)
            just_leveled = bool((budget or {}).get("just_leveled", 0.0))
            tier = float((budget or pending or {}).get("tavern_tier")
                         or player.tier)
            row = {
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": seat,
                "turn": turn,
                "tavern_tier": tier,
                "just_leveled": just_leveled,
                "start_of_recruit_stats": start,
                "end_of_recruit_pre_scaling_stats": pre,
                "recruit_delta": recruit_delta,
                "firestone_target": firestone,
                "growth_factor": (budget or {}).get("growth_factor"),
                "ratio_g": (budget or {}).get("ratio_g"),
                "ratio_add": (budget or {}).get("ratio_add"),
                "pace_target": (budget or {}).get("pace_target"),
                "over": (budget or {}).get("over"),
                "residual_add": (budget or {}).get("residual_add"),
                "residual_clamp_active": bool(
                    (budget or {}).get("residual_clamp_active", 0.0)),
                "post_scaling_stats": post,
                "scaling_delta": scaling_delta,
                "pre_scale_over_firestone": (
                    (float(pre) / float(firestone))
                    if pre is not None and firestone else None),
                "post_scale_over_firestone": (
                    (float(post) / float(firestone))
                    if firestone else None),
                "remaining_target_gap_after_scaling": remaining,
            }
            self.records.append(row)
        self._budget_by_seat.clear()

    def end_lobby(self, players) -> None:
        return


def run_scaling_budget_arm(
        lobbies: int, seed: int, *, arm: str,
        policy_factory: Optional[Callable[[int], Sequence[Callable]]] = None,
        policy: Optional[Callable] = None,
        scaling_mode: str = "residual") -> Dict:
    """Run lobbies and collect scaling-budget records + end-recruit rows."""
    assert_seed_range_allowed(seed, lobbies)
    all_records: List[Dict] = []
    rows: List[Dict] = []
    for i in range(lobbies):
        if policy_factory is not None:
            policies = list(policy_factory(i))
        else:
            pol = policy or greedy_policy
            policies = [pol] * 8
        tracer = ScalingBudgetTracer(lobby_id=i, seed=seed + i, arm=arm)
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
        all_records.extend(tracer.records)
        del env
    return {
        "arm": arm,
        "n_lobbies": lobbies,
        "seed_base": seed,
        "records": all_records,
        "rows": rows,
    }


def run_greedy_arm(lobbies: int, seed: int) -> Dict:
    return run_scaling_budget_arm(
        lobbies, seed, arm="greedy", policy=greedy_policy)


def run_phase_2j_arm(lobbies: int, seed: int, alpha: float,
                     prior: PersistencePrior) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_scaling_budget_arm(
        lobbies, seed, arm="phase_2j", policy_factory=factory)


def _agg_bucket(recs: List[Dict]) -> Dict:
    def col(name: str) -> List[float]:
        return [float(r[name]) for r in recs if r.get(name) is not None]

    firestone = _mean(col("firestone_target"))
    start = _mean(col("start_of_recruit_stats"))
    pre = _mean(col("end_of_recruit_pre_scaling_stats"))
    recruit = _mean(col("recruit_delta"))
    scaling = _mean(col("scaling_delta"))
    post = _mean(col("post_scaling_stats"))
    gap = _mean(col("remaining_target_gap_after_scaling"))
    return {
        "n": len(recs),
        "firestone_target": firestone,
        "start_of_recruit_stats": start,
        "end_of_recruit_pre_scaling_stats": pre,
        "recruit_delta": recruit,
        "growth_factor": _mean(col("growth_factor")),
        "ratio_g": _mean(col("ratio_g")),
        "ratio_add": _mean(col("ratio_add")),
        "pace_target": _mean(col("pace_target")),
        "over": _mean(col("over")),
        "residual_add": _mean(col("residual_add")),
        "scaling_delta": scaling,
        "post_scaling_stats": post,
        "pre_scale_over_firestone": _mean(col("pre_scale_over_firestone")),
        "post_scale_over_firestone": _mean(col("post_scale_over_firestone")),
        "remaining_target_gap_after_scaling": gap,
        "remaining_gap_p50": _pctl(col("remaining_target_gap_after_scaling"), 0.5),
        "remaining_gap_p90": _pctl(col("remaining_target_gap_after_scaling"), 0.9),
        "decomposition": {
            "firestone_target": firestone,
            "start_of_recruit": start,
            "recruit_contribution": recruit,
            "scaling_contribution": scaling,
            "post_scale": post,
            "unfilled_target_gap": gap,
        },
    }


def aggregate_scaling_budget(records: List[Dict]) -> Dict:
    """Aggregate by turn / just_leveled / tavern_tier."""
    by_turn: Dict[str, List[Dict]] = defaultdict(list)
    by_turn_level: Dict[str, List[Dict]] = defaultdict(list)
    by_turn_tier: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        t = str(int(r["turn"]))
        by_turn[t].append(r)
        leveled = "just_leveled" if r.get("just_leveled") else "did_not_level"
        by_turn_level[f"{t}|{leveled}"].append(r)
        tier = int(r.get("tavern_tier") or 0)
        by_turn_tier[f"{t}|T{tier}"].append(r)

    return {
        "by_turn": {k: _agg_bucket(v) for k, v in sorted(by_turn.items())},
        "by_turn_level": {
            k: _agg_bucket(v) for k, v in sorted(by_turn_level.items())},
        "by_turn_tier": {
            k: _agg_bucket(v) for k, v in sorted(by_turn_tier.items())},
        "n_records": len(records),
    }


def symmetric_absolute_fidelity(records: List[Dict],
                                turns: Sequence[int] = SYMMETRIC_FIDELITY_TURNS
                                ) -> Dict:
    """Symmetric absolute stats fidelity for turns 8–14 (undershoot + overshoot).

    Preserves historical Phase 2B *upper-bound* gates elsewhere; this report
    explicitly surfaces underscaling that those gates cannot catch.
    """
    out = {}
    for t in turns:
        bucket = [r for r in records if int(r["turn"]) == t]
        post_ratio = [
            float(r["post_scale_over_firestone"])
            for r in bucket if r.get("post_scale_over_firestone") is not None]
        pre_ratio = [
            float(r["pre_scale_over_firestone"])
            for r in bucket if r.get("pre_scale_over_firestone") is not None]
        mean_post = _mean(post_ratio)
        mean_pre = _mean(pre_ratio)
        out[str(t)] = {
            "n": len(bucket),
            "mean_pre_scale_over_firestone": mean_pre,
            "mean_post_scale_over_firestone": mean_post,
            "abs_distance_from_one_post": (
                abs(mean_post - 1.0) if mean_post is not None else None),
            "undershoot": (
                mean_post is not None and mean_post < 1.0),
            "overshoot": (
                mean_post is not None and mean_post > 1.0),
            "firestone_target": _mean([
                float(r["firestone_target"]) for r in bucket
                if r.get("firestone_target") is not None]),
            "mean_post_scaling_stats": _mean([
                float(r["post_scaling_stats"]) for r in bucket
                if r.get("post_scaling_stats") is not None]),
            "mean_remaining_target_gap": _mean([
                float(r["remaining_target_gap_after_scaling"]) for r in bucket
                if r.get("remaining_target_gap_after_scaling") is not None]),
            "note": (
                "Symmetric absolute fidelity; Phase 2B historical upper bounds "
                "are preserved separately and not retuned here."),
        }
    return out


def directional_macro_policy_harm(
        greedy_fidelity: Dict, treatment_fidelity: Dict,
        turns: Sequence[int] = (10, 12, 14)) -> Dict:
    """Prospective control: harm = distance-to-1(treatment) − distance-to-1(greedy)."""
    rows = {}
    for t in turns:
        g = (greedy_fidelity.get(str(t)) or {}).get(
            "mean_post_scale_over_firestone")
        tr = (treatment_fidelity.get(str(t)) or {}).get(
            "mean_post_scale_over_firestone")
        if g is None or tr is None:
            rows[str(t)] = {"unmeasured": True}
            continue
        g_dist = abs(g - 1.0)
        t_dist = abs(tr - 1.0)
        rows[str(t)] = {
            "greedy_ratio": g,
            "treatment_ratio": tr,
            "greedy_distance": g_dist,
            "treatment_distance": t_dist,
            "macro_policy_harm": t_dist - g_dist,
            "treatment_closer_to_firestone": t_dist < g_dist,
        }
    return {
        "definition": PROSPECTIVE_MACRO_POLICY_HARM,
        "by_turn": rows,
        "applied_to_2n_v3": False,
    }


def route_phase_2o_finding(
        greedy_agg: Dict, treatment_agg: Dict,
        greedy_fid: Dict, treatment_fid: Dict) -> Dict:
    """Predeclared routing from Phase 2O measurement patterns."""
    findings: List[str] = []

    def _turn(arm_agg: Dict, t: int) -> Dict:
        return (arm_agg.get("by_turn") or {}).get(str(t)) or {}

    # Focus on T10 headline + midgame band T9–T12.
    t10_g = _turn(greedy_agg, 10)
    t10_t = _turn(treatment_agg, 10)
    mid_turns = (9, 10, 11, 12)

    def _post_ratio(fid: Dict, t: int) -> Optional[float]:
        return (fid.get(str(t)) or {}).get("mean_post_scale_over_firestone")

    def _pre_ratio_agg(agg: Dict, t: int) -> Optional[float]:
        return _turn(agg, t).get("pre_scale_over_firestone")

    both_low = True
    greedy_ok_t_low = False
    for t in mid_turns:
        rg = _post_ratio(greedy_fid, t)
        rt = _post_ratio(treatment_fid, t)
        if rg is None or rt is None:
            continue
        if rg >= 0.85 or rt >= 0.85:
            both_low = False
        if rg >= 0.85 and rt < 0.75:
            greedy_ok_t_low = True

    # Target-gap bridge: pre far below AND post still far below at T10.
    for label, bucket in (("greedy", t10_g), ("phase_2j", t10_t)):
        pre_r = bucket.get("pre_scale_over_firestone")
        post_r = bucket.get("post_scale_over_firestone")
        gap = bucket.get("remaining_target_gap_after_scaling")
        firestone = bucket.get("firestone_target") or 0.0
        if (pre_r is not None and post_r is not None
                and pre_r < 0.75 and post_r < 0.75
                and gap is not None and firestone
                and gap > 0.25 * firestone):
            findings.append("pre_scale_far_below_post_still_far")
            break

    # Pre near Firestone but scaling pushes wrong.
    for bucket in (t10_g, t10_t):
        pre_r = bucket.get("pre_scale_over_firestone")
        post_r = bucket.get("post_scale_over_firestone")
        if (pre_r is not None and post_r is not None
                and abs(pre_r - 1.0) <= 0.15 and abs(post_r - 1.0) > 0.25):
            findings.append("pre_scale_near_firestone_scaling_wrong")
            break

    # Recruit contribution collapsed relative to Firestone step T9→T10.
    for bucket in (t10_g, t10_t):
        recruit = bucket.get("recruit_delta")
        firestone = bucket.get("firestone_target") or 0.0
        start = bucket.get("start_of_recruit_stats")
        if (recruit is not None and firestone and start is not None
                and recruit < 0.15 * firestone and start < 0.6 * firestone):
            findings.append("recruit_contribution_collapsed")
            break

    # Just-leveled 0.6× factor explains most deficit.
    for arm_agg in (greedy_agg, treatment_agg):
        leveled = (arm_agg.get("by_turn_level") or {}).get("10|just_leveled") or {}
        not_lev = (arm_agg.get("by_turn_level") or {}).get("10|did_not_level") or {}
        if not leveled.get("n") or not not_lev.get("n"):
            continue
        gap_l = leveled.get("remaining_target_gap_after_scaling")
        gap_n = not_lev.get("remaining_target_gap_after_scaling")
        share = leveled["n"] / max(1, leveled["n"] + not_lev["n"])
        if (gap_l is not None and gap_n is not None and share >= 0.25
                and gap_l > gap_n * 1.35):
            findings.append("just_leveled_factor_explains_deficit")
            break

    if greedy_ok_t_low:
        findings.append("greedy_healthy_phase_2j_low")
    if both_low:
        findings.append("both_arms_low")

    # Late overcompensation T13–14.
    for fid in (greedy_fid, treatment_fid):
        r13 = _post_ratio(fid, 13)
        r14 = _post_ratio(fid, 14)
        r11 = _post_ratio(fid, 11)
        if (r13 is not None and r14 is not None and r11 is not None
                and r11 < 0.7 and r14 > 1.15):
            findings.append("late_scaling_overcompensates")
            break

    # Deduplicate preserving order.
    seen = set()
    ordered = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    primary = ordered[0] if ordered else "inconclusive"
    next_step = dict(ROUTING_TABLE).get(primary, "inspect decomposition manually")
    return {
        "primary_finding": primary,
        "all_findings": ordered,
        "recommended_next_step": next_step,
        "routing_table": [
            {"finding": f, "next_step": s} for f, s in ROUTING_TABLE],
        "hypothesis": (
            "Strong prior: pre_scale_far_below_post_still_far → "
            "target-gap bridge defect (residual_add ∝ undersized current)."
        ),
    }


def t10_headline_decomposition(agg: Dict) -> Dict:
    """Human-readable T10 decomposition block."""
    t10 = (agg.get("by_turn") or {}).get("10") or {}
    d = t10.get("decomposition") or {}
    return {
        "turn": 10,
        "n": t10.get("n"),
        "firestone_target": d.get("firestone_target"),
        "start_of_recruit": d.get("start_of_recruit"),
        "end_recruit": t10.get("end_of_recruit_pre_scaling_stats"),
        "recruit_contribution": d.get("recruit_contribution"),
        "scaling_contribution": d.get("scaling_contribution"),
        "ratio_add": t10.get("ratio_add"),
        "residual_add": t10.get("residual_add"),
        "pace_target": t10.get("pace_target"),
        "over": t10.get("over"),
        "post_scale": d.get("post_scale"),
        "unfilled_target_gap": d.get("unfilled_target_gap"),
        "pre_scale_over_firestone": t10.get("pre_scale_over_firestone"),
        "post_scale_over_firestone": t10.get("post_scale_over_firestone"),
    }
