"""Phase 2I seeded opportunity decision-margin diagnostic (2i_v2).

Measurement-only audit of why frozen Phase 2H λ=12 rejects seeded core exposures.
Uses 2c_v3 exposure units: core name × shop generation × seeded current target.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hsbg_coach.build_path import Archetype, load_archetypes
from hsbg_coach.tempo_margin_audit import (
    DecisionSnapshot,
    TempoMarginAuditCollector,
    break_even_lambda,
    directional_break_even_bucket,
)

from .composition_diagnostic import (
    METHODOLOGY_VERSION as PHASE_2C_VERSION,
    _is_relevant_at_offer,
    _legally_buyable_cores,
    _target_meets_view_threshold,
    _winner_for_lobby,
    _lobby_tribes,
    _core_set,
    _archetype_eligible,
    aggregate_diagnostics,
)

METHODOLOGY_VERSION = "2i_v2"
VIEW = "seeded_current_target"

FAILURE_CODES = (
    "A_REPLACEMENT_COST_DOMINATES",
    "B_RAW_STAT_COMPETITOR_DOMINATES",
    "C_BUILD_SIGNAL_TOO_SMALL",
    "D_BUILD_SIGNAL_NONDISCRIMINATIVE",
    "E_ALTERNATE_CORE_SELECTED",
    "F_ECONOMY_LEGALITY_LOSS",
    "G_TARGET_CHANGED",
    "H_OTHER",
)


@dataclass
class TrackedExposure:
    lobby: int
    archetype_key: str
    core_name: str
    turn: int
    shop_generation: int
    core_have: int
    tier: int
    board_full_at_open: bool
    target_at_open: Optional[str]
    fulfilled: bool = False
    closed: bool = False
    last_pre_buyable: bool = True
    core_frequency: float = 0.0
    decision_audit_indices: List[int] = field(default_factory=list)
    decisive_audit_index: Optional[int] = None
    decisive_event_index: Optional[int] = None
    close_reason: Optional[str] = None

    @property
    def exposure_key(self) -> Tuple[int, str, str, int, int]:
        return (self.lobby, self.archetype_key, self.core_name,
                self.turn, self.shop_generation)


@dataclass
class _ActiveGen:
    turn: int
    shop_generation: int
    target_key: Optional[str]
    cores: Dict[str, TrackedExposure] = field(default_factory=dict)


def _snapshot_to_dict(snap: DecisionSnapshot) -> Dict:
    d = asdict(snap)
    return d


def _core_still_buyable(ev: Dict, core_name: str) -> bool:
    pre_shop = ev.get("pre_shop") or []
    legal = ev.get("legal_buy_slots") or []
    buyable = _legally_buyable_cores(pre_shop, legal, {core_name})
    return core_name in buyable


def classify_rejection(
        exp: TrackedExposure,
        snap: Optional[DecisionSnapshot]) -> Tuple[str, Dict[str, Any]]:
    """Assign exactly one primary failure category."""
    decomp: Dict[str, Any] = {"exposure_key": list(exp.exposure_key)}
    if snap is None:
        decomp["reason"] = "no_audit_at_decisive_event"
        return "H_OTHER", decomp

    cs = snap.core_scores.get(exp.core_name)
    chosen = snap.chosen
    decomp["core_net"] = cs.core_net_value if cs else None
    decomp["chosen_type"] = chosen.action_type if chosen else None
    decomp["chosen_net"] = chosen.net_value if chosen else None
    decomp["decision_margin"] = (
        (cs.core_net_value - chosen.net_value)
        if cs and chosen else None)

    if (chosen and chosen.is_target_core and chosen.candidate_name
            and chosen.candidate_name != exp.core_name):
        decomp["alternate_core"] = chosen.candidate_name
        return "E_ALTERNATE_CORE_SELECTED", decomp

    if (exp.target_at_open and snap.target_archetype
            and snap.target_archetype != exp.target_at_open):
        decomp["target_at_decisive"] = snap.target_archetype
        return "G_TARGET_CHANGED", decomp

    if cs and cs.board_full:
        free_v = cs.core_free_slot_value or 0.0
        repl_v = cs.core_actual_replacement_value
        if free_v > 0 and (repl_v is None or repl_v <= 0):
            decomp["core_free_slot_value"] = free_v
            decomp["core_replacement_value"] = repl_v
            return "A_REPLACEMENT_COST_DOMINATES", decomp

    if cs and chosen and cs.build_gain > 0:
        if chosen.build_gain >= cs.build_gain:
            decomp["chosen_build_gain"] = chosen.build_gain
            decomp["core_build_gain"] = cs.build_gain
            return "D_BUILD_SIGNAL_NONDISCRIMINATIVE", decomp

    if cs and cs.build_component < 2.0:
        decomp["build_component"] = cs.build_component
        return "C_BUILD_SIGNAL_TOO_SMALL", decomp

    if cs and chosen and (cs.core_net_value or 0) > 0:
        raw_disadv = chosen.raw_component - cs.candidate_raw
        if raw_disadv > cs.build_component:
            decomp["raw_disadvantage"] = raw_disadv
            return "B_RAW_STAT_COMPETITOR_DOMINATES", decomp

    if cs and (cs.core_net_value or 0) <= 0:
        if cs.board_full and (cs.core_free_slot_value or 0) > 0:
            return "A_REPLACEMENT_COST_DOMINATES", decomp
        if chosen and chosen.action_type in ("roll", "level", "end", "greedy_buy"):
            return "B_RAW_STAT_COMPETITOR_DOMINATES", decomp

    if chosen and chosen.action_type in ("roll", "level", "end"):
        if cs and (cs.core_net_value or 0) > 0:
            decomp["note"] = "positive_core_at_shop_exit"
            return "B_RAW_STAT_COMPETITOR_DOMINATES", decomp
        return "F_ECONOMY_LEGALITY_LOSS", decomp

    return "H_OTHER", decomp


def is_composition_progress_failure(code: str) -> bool:
    return code != "E_ALTERNATE_CORE_SELECTED"


def analyze_margin_exposures(
        traces: Dict,
        audit: TempoMarginAuditCollector,
        audit_event_links: List[Optional[int]],
        archetypes: Optional[List[Archetype]] = None,
) -> Dict:
    """Walk winner events, latch 2c_v3 exposures, link audit, classify rejections."""
    archetypes = archetypes or load_archetypes()
    exposures: List[TrackedExposure] = []
    lobbies = traces["lobbies"]
    link_offset = 0

    for lobby in range(lobbies):
        winner = _winner_for_lobby(traces, lobby)
        lobby_event_count = sum(
            1 for ev in traces["events"] if ev["lobby"] == lobby)
        if winner is None:
            link_offset += lobby_event_count
            continue
        winner_seat = winner["seat"]
        lobby_tribes = _lobby_tribes(traces, lobby)
        final_target_key = (winner.get("target") or {}).get("archetype_key")

        eligible = [a for a in archetypes
                    if _archetype_eligible(a, lobby_tribes)]
        active: Dict[str, _ActiveGen] = {}
        prev_gen: Dict[str, Tuple[int, int]] = {}

        local_idx = 0
        for ev in traces["events"]:
            if ev["lobby"] != lobby:
                continue
            link_idx = link_offset + local_idx
            audit_idx = (audit_event_links[link_idx]
                         if link_idx < len(audit_event_links) else None)
            local_idx += 1

            if ev["seat"] != winner_seat:
                continue

            turn = ev["turn"]
            shop_gen = ev.get("shop_generation", 0)
            gen_key = (turn, shop_gen)
            target_before = ev.get("target_before")
            tier = ev.get("tavern_tier")

            for arch in eligible:
                core = _core_set(arch)
                akey = arch.key

                if akey in prev_gen and prev_gen[akey] != gen_key:
                    _close_active_gen(active.get(akey), exposures,
                                      audit_event_links, link_idx,
                                      "generation_change")
                    active.pop(akey, None)
                prev_gen[akey] = gen_key

                pre_shop = ev.get("pre_shop") or []
                legal_slots = ev.get("legal_buy_slots") or []
                buyable = _legally_buyable_cores(pre_shop, legal_slots, core)

                if buyable and _is_relevant_at_offer(
                        arch, VIEW, target_before, final_target_key, tier):
                    ag = active.get(akey)
                    if ag is None or (ag.turn, ag.shop_generation) != gen_key:
                        if ag is not None:
                            _close_active_gen(ag, exposures, audit_event_links,
                                              link_idx, "generation_change")
                        board = ev.get("board_before") or []
                        ag = _ActiveGen(
                            turn=turn, shop_generation=shop_gen,
                            target_key=(target_before or {}).get("archetype_key"))
                        active[akey] = ag
                    for name in buyable:
                        if name not in ag.cores:
                            exp = TrackedExposure(
                                lobby=lobby, archetype_key=akey,
                                core_name=name, turn=turn,
                                shop_generation=shop_gen,
                                core_have=(target_before or {}).get("core_have") or 0,
                                tier=int(tier or 1),
                                board_full_at_open=len(board) >= 7,
                                target_at_open=(target_before or {}).get(
                                    "archetype_key"),
                                core_frequency=float(arch.core.get(name, 0.0)),
                                last_pre_buyable=True,
                            )
                            ag.cores[name] = exp
                            exposures.append(exp)

                ag = active.get(akey)
                if ag is not None:
                    for exp in ag.cores.values():
                        if exp.fulfilled or exp.closed:
                            continue
                        now_buyable = _core_still_buyable(ev, exp.core_name)
                        if exp.last_pre_buyable and not now_buyable:
                            audit_for_loss = (exp.decision_audit_indices[-1]
                                              if exp.decision_audit_indices
                                              else audit_idx)
                            _close_exposure(
                                exp,
                                reason="first_loss_of_buyability",
                                audit_idx=audit_for_loss,
                                event_idx=link_idx,
                            )
                        elif now_buyable and audit_idx is not None:
                            exp.decision_audit_indices.append(audit_idx)
                        exp.last_pre_buyable = now_buyable

                if ev["action"] == "buy" and ev.get("card"):
                    bought = ev["card"]["name"]
                    ag = active.get(akey)
                    if ag is not None and bought in ag.cores:
                        exp = ag.cores[bought]
                        exp.fulfilled = True
                        _close_exposure(
                            exp,
                            reason="fulfilled",
                            audit_idx=audit_idx,
                            event_idx=link_idx,
                        )

                if ev["action"] in ("roll", "end"):
                    _close_active_gen(active.get(akey), exposures,
                                      audit_event_links, link_idx, ev["action"])
                    active.pop(akey, None)
                    if ev["action"] == "end":
                        prev_gen.pop(akey, None)

            if ev["action"] == "end":
                prev_gen.clear()

        link_offset += local_idx

        for akey in list(active.keys()):
            _close_active_gen(active.get(akey), exposures, audit_event_links,
                              max(link_offset - 1, 0), "flush")
            active.pop(akey, None)

    return _summarize_exposures(exposures, audit, traces)


def _close_exposure(
        exp: TrackedExposure,
        *,
        reason: str,
        audit_idx: Optional[int],
        event_idx: int) -> None:
    if exp.closed:
        return
    exp.closed = True
    exp.close_reason = reason
    exp.decisive_event_index = event_idx
    if audit_idx is not None:
        exp.decisive_audit_index = audit_idx
    elif exp.decision_audit_indices:
        exp.decisive_audit_index = exp.decision_audit_indices[-1]


def _close_active_gen(ag: Optional[_ActiveGen], exposures: List[TrackedExposure],
                      links: List[Optional[int]], event_idx: int,
                      reason: str) -> None:
    if ag is None:
        return
    for exp in ag.cores.values():
        if exp.fulfilled or exp.closed:
            continue
        audit_idx = (exp.decision_audit_indices[-1]
                     if exp.decision_audit_indices
                     else (links[event_idx] if event_idx < len(links) else None))
        _close_exposure(exp, reason=reason, audit_idx=audit_idx, event_idx=event_idx)
    ag.cores.clear()


def _frequency_quartile(freq: float, boundaries: List[float]) -> str:
    if not boundaries:
        return "q_unknown"
    if freq <= boundaries[0]:
        return "q1_lowest"
    if freq <= boundaries[1]:
        return "q2"
    if freq <= boundaries[2]:
        return "q3"
    return "q4_highest"


def _summarize_exposures(exposures: List[TrackedExposure],
                         audit: TempoMarginAuditCollector,
                         traces: Dict) -> Dict:
    fulfilled = [e for e in exposures if e.fulfilled]
    rejected = [e for e in exposures if not e.fulfilled]

    records: List[Dict] = []
    cause_counts = Counter()
    comp_fail_counts = Counter()
    break_even_lambdas: List[float] = []
    break_even_buckets = Counter()

    margins: List[float] = []
    raw_gaps: List[float] = []
    build_bonuses: List[float] = []
    repl_costs: List[float] = []
    core_net_positive = 0
    core_ranked_first_with_build = 0
    core_ranked_second_with_build = 0
    core_ranked_third_plus_with_build = 0
    core_ranked_first_no_build = 0
    core_ranked_second_no_build = 0
    core_ranked_third_plus_no_build = 0
    rejected_despite_positive = 0
    lost_board_full_only = 0
    rejected_frequencies: List[float] = []

    for exp in rejected:
        snap = (audit.snapshots[exp.decisive_audit_index]
                if exp.decisive_audit_index is not None
                and exp.decisive_audit_index < len(audit.snapshots)
                else None)
        code, decomp = classify_rejection(exp, snap)
        cause_counts[code] += 1
        if is_composition_progress_failure(code):
            comp_fail_counts[code] += 1

        cs = snap.core_scores.get(exp.core_name) if snap else None
        chosen = snap.chosen if snap else None
        margin = decomp.get("decision_margin")
        if margin is not None:
            margins.append(margin)
        freq = exp.core_frequency
        if cs and cs.core_frequency:
            freq = cs.core_frequency
        rejected_frequencies.append(freq)
        if cs:
            if cs.candidate_raw and chosen:
                raw_gaps.append(chosen.raw_component - cs.candidate_raw)
            build_bonuses.append(cs.build_component)
            if cs.replacement_raw:
                repl_costs.append(cs.replacement_raw)
            if (cs.core_net_value or 0) > 0:
                core_net_positive += 1
            if (cs.core_net_value or 0) > 0 and margin is not None and margin < 0:
                rejected_despite_positive += 1
            rk = cs.rank_with_build or 99
            if rk == 1:
                core_ranked_first_with_build += 1
            elif rk == 2:
                core_ranked_second_with_build += 1
            else:
                core_ranked_third_plus_with_build += 1
            rkn = cs.rank_without_build or 99
            if rkn == 1:
                core_ranked_first_no_build += 1
            elif rkn == 2:
                core_ranked_second_no_build += 1
            else:
                core_ranked_third_plus_no_build += 1
            if cs.board_full and (cs.core_free_slot_value or 0) > 0:
                if (cs.core_actual_replacement_value or 0) <= 0:
                    lost_board_full_only += 1

            if chosen and cs:
                current_lambda = snap.lambda_build if snap else 12.0
                core_slope = cs.build_gain - cs.replacement_build_value
                chosen_slope = (chosen.build_gain
                                - chosen.replacement_build_value)
                lam = break_even_lambda(
                    core_raw=cs.candidate_raw, core_build=cs.build_gain,
                    repl_raw=cs.replacement_raw,
                    repl_build=cs.replacement_build_value,
                    chosen_raw=chosen.raw_component,
                    chosen_build=chosen.build_gain,
                    chosen_repl_raw=chosen.replacement_raw,
                    chosen_repl_build=chosen.replacement_build_value)
                bucket = directional_break_even_bucket(
                    lam,
                    current_lambda=current_lambda,
                    core_slope=core_slope,
                    chosen_slope=chosen_slope)
                break_even_buckets[bucket] += 1
                if lam is not None:
                    break_even_lambdas.append(lam)

        records.append({
            "lobby": exp.lobby,
            "archetype_key": exp.archetype_key,
            "core_name": exp.core_name,
            "turn": exp.turn,
            "shop_generation": exp.shop_generation,
            "tier": exp.tier,
            "core_have": exp.core_have,
            "core_frequency": freq,
            "board_full_at_open": exp.board_full_at_open,
            "primary_cause": code,
            "composition_progress_failure": is_composition_progress_failure(code),
            "close_reason": exp.close_reason,
            "decomposition": decomp,
        })

    freq_boundaries: List[float] = []
    if len(rejected_frequencies) >= 4:
        freq_boundaries = list(st.quantiles(rejected_frequencies, n=4))[:3]
    for r in records:
        r["core_frequency_quartile"] = _frequency_quartile(
            r["core_frequency"], freq_boundaries)

    diag_2c = aggregate_diagnostics(traces)
    seeded_2c = ((diag_2c.get("winner_decision_funnel") or {})
                 .get("seeded_current_target") or {})
    agg_2c = seeded_2c.get("aggregate_funnel") or {}

    n_total = len(exposures)
    n_rej = len(rejected)
    comp_failures = sum(1 for r in records if r["composition_progress_failure"])

    def _pct(n: int, d: int) -> Optional[float]:
        return n / d if d else None

    def _breakdown_by(field: str, items: List[Dict]) -> Dict:
        groups: Dict[Any, Counter] = defaultdict(Counter)
        for r in items:
            groups[r.get(field, "?")][r["primary_cause"]] += 1
        return {k: dict(v) for k, v in groups.items()}

    mean_raw_gap = st.mean(raw_gaps) if raw_gaps else None

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "phase_2c_methodology_version": PHASE_2C_VERSION,
        "view": VIEW,
        "funnel": {
            "seeded_legally_buyable_exposures": n_total,
            "fulfilled": len(fulfilled),
            "rejected": n_rej,
            "raw_rejection_rate": _pct(n_rej, n_total),
            "composition_progress_failures": comp_failures,
            "composition_progress_failure_rate": _pct(comp_failures, n_rej),
            "by_primary_cause": dict(cause_counts),
            "composition_progress_by_cause": dict(comp_fail_counts),
            "by_close_reason": dict(Counter(r["close_reason"] for r in records)),
        },
        "reconciliation_2c_v3": {
            "tracked_exposures": n_total,
            "phase_2c_legally_buyable_exposures": agg_2c.get(
                "legally_buyable_exposures"),
            "phase_2c_fulfilled": agg_2c.get("fulfilled_exposures"),
            "phase_2c_rejected": agg_2c.get("rejected_exposures"),
            "counts_match": (
                n_total == agg_2c.get("legally_buyable_exposures")
                and len(fulfilled) == agg_2c.get("fulfilled_exposures")
                and n_rej == agg_2c.get("rejected_exposures")),
        },
        "headline_metrics": {
            "mean_chosen_minus_core_raw_gap": mean_raw_gap,
            "mean_core_raw_advantage": (-mean_raw_gap if mean_raw_gap is not None
                                        else None),
            "median_chosen_minus_core_raw_gap": (
                st.median(raw_gaps) if raw_gaps else None),
            "mean_lambda_build_bonus": (
                st.mean(build_bonuses) if build_bonuses else None),
            "mean_replacement_raw_stat_cost": (
                st.mean(repl_costs) if repl_costs else None),
            "mean_core_vs_chosen_margin": (
                st.mean(margins) if margins else None),
            "median_core_vs_chosen_margin": (
                st.median(margins) if margins else None),
            "pct_core_transitions_net_positive": _pct(
                core_net_positive, n_rej),
            "pct_core_ranked_first_with_build": _pct(
                core_ranked_first_with_build, n_rej),
            "pct_core_ranked_second_with_build": _pct(
                core_ranked_second_with_build, n_rej),
            "pct_core_ranked_third_plus_with_build": _pct(
                core_ranked_third_plus_with_build, n_rej),
            "pct_core_ranked_first_without_build": _pct(
                core_ranked_first_no_build, n_rej),
            "pct_core_ranked_second_without_build": _pct(
                core_ranked_second_no_build, n_rej),
            "pct_core_ranked_third_plus_without_build": _pct(
                core_ranked_third_plus_no_build, n_rej),
            "pct_rejected_despite_positive_core_net": _pct(
                rejected_despite_positive, n_rej),
            "pct_exposures_lost_board_full_only": _pct(
                lost_board_full_only, n_rej),
        },
        "break_even_lambda": {
            "median": st.median(break_even_lambdas) if break_even_lambdas else None,
            "p25": (st.quantiles(break_even_lambdas, n=4)[0]
                    if len(break_even_lambdas) >= 4 else None),
            "p75": (st.quantiles(break_even_lambdas, n=4)[2]
                    if len(break_even_lambdas) >= 4 else None),
            "pct_helpful_higher_lambda_le_24": _pct(
                break_even_buckets.get("helpful_higher_lambda_le_24", 0), n_rej),
            "pct_helpful_higher_lambda_gt_24": _pct(
                break_even_buckets.get("helpful_higher_lambda_gt_24", 0), n_rej),
            "pct_helpful_lower_lambda_only": _pct(
                break_even_buckets.get("helpful_lower_lambda_only", 0), n_rej),
            "pct_no_lambda_effect": _pct(
                break_even_buckets.get("no_lambda_effect", 0), n_rej),
            "pct_no_finite_helpful_higher_lambda": _pct(
                break_even_buckets.get("no_finite_helpful_higher_lambda", 0),
                n_rej),
            "bucket_counts": dict(break_even_buckets),
        },
        "breakdown_by_tier": _breakdown_by("tier", records),
        "breakdown_by_core_have": _breakdown_by("core_have", records),
        "breakdown_by_board_full": _breakdown_by("board_full_at_open", records),
        "breakdown_by_archetype": _breakdown_by("archetype_key", records),
        "breakdown_by_core_frequency_quartile": _breakdown_by(
            "core_frequency_quartile", records),
        "rejected_exposure_records": records,
    }
