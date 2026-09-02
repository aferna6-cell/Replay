"""Phase 2K post-assembly residual composition-gap diagnostic (2k_v1).

Measurement-only: for each (lobby, winner_seat, archetype) that first reaches
persistent 2+ cores at end-of-recruit under the frozen Phase 2J policy, account
for every unit of missing final weighted core-coverage mass.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hsbg_coach.build_path import Archetype, load_archetypes
from ml.composition_diagnostic import (
    METHODOLOGY_VERSION as PHASE_2C_VERSION,
    _archetype_eligible,
    _core_set,
    _legally_buyable_cores,
    _lobby_tribes,
    _max_core_count,
    _winner_for_lobby,
)

METHODOLOGY_VERSION = "2k_v1"
VIEW = "post_assembly_frozen_target"

FROZEN_ALPHA = 0.5
FROZEN_PRIOR_HASH = (
    "9b31c93a8d89f3d9a1e4301ea7926ec6ed7c1f659ed6939c5705977141721951"
)
PHASE_2J_PRIOR_PATH = "results/sim_fidelity_phase_2j/persistence_prior.json"

PHASE_2K_SEED = 9000
PHASE_2K_LOBBIES = 500
PHASE_2K_EXPAND_THROUGH = 9999
PHASE_2K_MIN_STATES = 40
# Reserved confirmation seeds — rejected by default runner.
FORBIDDEN_CONFIRM_SEED = 8000
FORBIDDEN_CONFIRM_LOBBIES = 200

FATE_CODES = (
    "A_NEVER_AVAILABLE_POST_ASSEMBLY",
    "B_AVAILABLE_NOT_BOUGHT",
    "C_BOUGHT_NOT_DEPLOYED",
    "D_DEPLOYED_THEN_LOST",
    "E_EXISTING_CORE_LOST",
    "F_TARGET_SWITCH",
    "G_TRANSFORM_TRIPLE_DISCOVER_PATH",
    "H_UNRESOLVED",
)


def core_weights(arch: Archetype) -> Dict[str, float]:
    total = sum(arch.core.values()) or 1.0
    return {name: float(w) / total for name, w in arch.core.items()}


def weighted_coverage(board: List[Dict], weights: Dict[str, float]) -> float:
    names = {c.get("name") for c in (board or []) if c.get("name")}
    return sum(weights.get(n, 0.0) for n in names)


def _board_core_names(board: List[Dict], core: Set[str]) -> Set[str]:
    return {c.get("name") for c in (board or [])
            if c.get("name") in core}


@dataclass
class CoreCardTrace:
    name: str
    weight: float
    present_at_first_2: bool = False
    legally_offered_after: bool = False
    purchased_after: bool = False
    played_after: bool = False
    survived_recruit_end: bool = False
    survived_1_turn: bool = False
    survived_2_turns: bool = False
    present_final: bool = False
    tripled_or_transformed: bool = False
    primary_fate: Optional[str] = None  # set only if not present_final


@dataclass
class PostAssemblyState:
    lobby: int
    seat: int
    archetype_key: str
    entry_turn: int
    entry_tier: int
    coverage_at_first_2: float
    remaining_weight_at_first_2: float
    cards: Dict[str, CoreCardTrace] = field(default_factory=dict)
    # Trajectory after entry
    turn_first_3: Optional[int] = None
    turn_first_4: Optional[int] = None
    max_core_count: int = 2
    final_core_count: int = 0
    max_coverage: float = 0.0
    final_coverage: float = 0.0
    target_switch_turn: Optional[int] = None
    survived_2_plus_1_turn: bool = False
    survived_2_plus_2_turns: bool = False
    # Weighted funnel after first-2
    weight_legally_available_after: float = 0.0
    weight_purchased_after: float = 0.0
    weight_deployed_after: float = 0.0
    weight_retained_2_turns: float = 0.0
    weight_present_final: float = 0.0


def _classify_missing_card(card: CoreCardTrace, *,
                          target_switched: bool) -> str:
    if card.present_final:
        return ""  # no missing mass
    if card.tripled_or_transformed and not card.present_final:
        # Triple may keep golden on board under same name — if still missing:
        return "G_TRANSFORM_TRIPLE_DISCOVER_PATH"
    if card.present_at_first_2:
        return "E_EXISTING_CORE_LOST"
    if target_switched and not card.purchased_after:
        return "F_TARGET_SWITCH"
    if not card.legally_offered_after:
        return "A_NEVER_AVAILABLE_POST_ASSEMBLY"
    if not card.purchased_after:
        return "B_AVAILABLE_NOT_BOUGHT"
    if not card.played_after:
        return "C_BOUGHT_NOT_DEPLOYED"
    if card.played_after and not card.present_final:
        return "D_DEPLOYED_THEN_LOST"
    return "H_UNRESOLVED"


def analyze_lobby_post_assembly(
        traces: Dict, lobby: int,
        archetypes: Optional[List[Archetype]] = None,
) -> List[PostAssemblyState]:
    """Find first 2+ end-of-recruit cohorts for the winner seat."""
    archetypes = archetypes or load_archetypes()
    winner = _winner_for_lobby(traces, lobby)
    if winner is None:
        return []
    seat = winner["seat"]
    tribes = _lobby_tribes(traces, lobby)
    results: List[PostAssemblyState] = []

    # Index turn summaries for this seat
    turn_ends = [
        ts for ts in traces.get("turn_summaries") or []
        if ts["lobby"] == lobby and ts["seat"] == seat
    ]
    turn_ends.sort(key=lambda t: t["turn"])

    for arch in archetypes:
        if not _archetype_eligible(arch, tribes):
            continue
        core = _core_set(arch)
        weights = core_weights(arch)

        entry_ts = None
        for ts in turn_ends:
            board = ts.get("board_after_recruit") or []
            tgt = ts.get("target") or {}
            if tgt.get("archetype_key") != arch.key:
                continue
            if _max_core_count(board, core) >= 2:
                entry_ts = ts
                break
        if entry_ts is None:
            continue

        entry_turn = int(entry_ts["turn"])
        entry_board = entry_ts.get("board_after_recruit") or []
        present0 = _board_core_names(entry_board, core)
        cov0 = weighted_coverage(entry_board, weights)

        state = PostAssemblyState(
            lobby=lobby, seat=seat, archetype_key=arch.key,
            entry_turn=entry_turn,
            entry_tier=int(entry_ts.get("tavern_tier") or 1),
            coverage_at_first_2=cov0,
            remaining_weight_at_first_2=max(0.0, 1.0 - cov0),
            max_coverage=cov0,
            max_core_count=_max_core_count(entry_board, core),
        )
        for name, w in weights.items():
            state.cards[name] = CoreCardTrace(
                name=name, weight=w,
                present_at_first_2=(name in present0),
            )

        # Walk events after entry_turn's recruit (events with turn > entry_turn,
        # plus remaining actions on entry_turn after end are none — entry is end).
        # Also walk subsequent turn summaries.
        frozen = arch.key
        target_switched = False
        bought_after: Set[str] = set()
        played_after: Set[str] = set()
        offered_after: Set[str] = set()
        # Track board presence across post-entry turn ends for survival
        presence_by_turn: Dict[int, Set[str]] = {
            entry_turn: set(present0)
        }

        for ev in traces.get("events") or []:
            if ev["lobby"] != lobby or ev["seat"] != seat:
                continue
            turn = int(ev["turn"])
            if turn < entry_turn:
                continue
            # On entry turn, only observe target_after-style signals after end —
            # entry is defined at end-of-recruit, so skip same-turn pre-end noise
            # for acquisition; post-entry acquisitions start next turn.
            if turn == entry_turn and ev["action"] != "end":
                continue

            tgt = ev.get("target_before") or {}
            if (not target_switched and tgt.get("archetype_key")
                    and tgt.get("archetype_key") != frozen):
                target_switched = True
                state.target_switch_turn = turn

            pre_shop = ev.get("pre_shop") or []
            legal = ev.get("legal_buy_slots") or []
            buyable = _legally_buyable_cores(pre_shop, legal, core)
            # Availability while frozen target still relevant OR always count
            # legal offers of missing cores (policy could buy regardless).
            if not target_switched or tgt.get("archetype_key") == frozen:
                for name in buyable:
                    if name not in present0 or name not in presence_by_turn.get(
                            turn, set()):
                        offered_after.add(name)
                        state.cards[name].legally_offered_after = True

            if ev["action"] == "buy" and ev.get("card"):
                name = ev["card"]["name"]
                if name in core and turn > entry_turn:
                    bought_after.add(name)
                    state.cards[name].purchased_after = True
                    # crude triple detection: golden appearing
                    if ev["card"].get("golden"):
                        state.cards[name].tripled_or_transformed = True

            if ev["action"] == "play" and ev.get("card"):
                name = ev["card"]["name"]
                if name in core and turn > entry_turn:
                    played_after.add(name)
                    state.cards[name].played_after = True

            if ev["action"] in ("buy", "play") and ev.get("card"):
                name = ev["card"].get("name")
                if name in core:
                    board_a = ev.get("board_after") or []
                    hand_a = ev.get("hand_after") or []
                    ng = sum(1 for c in list(board_a) + list(hand_a)
                             if c.get("name") == name and not c.get("golden"))
                    g = sum(1 for c in list(board_a) + list(hand_a)
                            if c.get("name") == name and c.get("golden"))
                    if g > 0 and ng <= 1:
                        state.cards[name].tripled_or_transformed = True

        for ts in turn_ends:
            turn = int(ts["turn"])
            if turn < entry_turn:
                continue
            board = ts.get("board_after_recruit") or []
            present = _board_core_names(board, core)
            presence_by_turn[turn] = present
            count = _max_core_count(board, core)
            cov = weighted_coverage(board, weights)
            state.max_core_count = max(state.max_core_count, count)
            state.max_coverage = max(state.max_coverage, cov)
            if count >= 3 and state.turn_first_3 is None:
                state.turn_first_3 = turn
            if count >= 4 and state.turn_first_4 is None:
                state.turn_first_4 = turn
            if turn == entry_turn + 1 and count >= 2:
                state.survived_2_plus_1_turn = True
            if turn == entry_turn + 2 and count >= 2:
                state.survived_2_plus_2_turns = True
            for name, card in state.cards.items():
                if name in present:
                    if turn >= entry_turn:
                        card.survived_recruit_end = True
                    if turn >= entry_turn + 1:
                        card.survived_1_turn = True
                    if turn >= entry_turn + 2:
                        card.survived_2_turns = True

        final_board = winner.get("final_board") or []
        final_present = _board_core_names(final_board, core)
        state.final_core_count = _max_core_count(final_board, core)
        state.final_coverage = weighted_coverage(final_board, weights)
        state.max_coverage = max(state.max_coverage, state.final_coverage)

        for name, card in state.cards.items():
            card.present_final = name in final_present
            # Mark offered for cores we somehow bought
            if card.purchased_after:
                card.legally_offered_after = True
            if card.present_final and not card.present_at_first_2:
                # Must have been deployed at some point
                card.played_after = True
                card.survived_recruit_end = True

        for name, card in state.cards.items():
            if card.present_final:
                card.primary_fate = None
            else:
                card.primary_fate = _classify_missing_card(
                    card, target_switched=target_switched)

        # Weighted funnel (mass of cores not present at first-2)
        missing0 = [c for c in state.cards.values() if not c.present_at_first_2]
        state.weight_legally_available_after = sum(
            c.weight for c in missing0 if c.legally_offered_after)
        state.weight_purchased_after = sum(
            c.weight for c in missing0 if c.purchased_after)
        state.weight_deployed_after = sum(
            c.weight for c in missing0 if c.played_after)
        state.weight_retained_2_turns = sum(
            c.weight for c in missing0 if c.survived_2_turns)
        state.weight_present_final = sum(
            c.weight for c in state.cards.values() if c.present_final)

        results.append(state)

    return results


def analyze_post_assembly_gap(traces: Dict) -> Dict:
    """Run Phase 2K accounting across all lobbies."""
    archetypes = load_archetypes()
    states: List[PostAssemblyState] = []
    for lobby in range(traces["lobbies"]):
        states.extend(analyze_lobby_post_assembly(traces, lobby, archetypes))

    # Deduplicate by (lobby, seat, archetype) — analyze already one per arch
    keys = [(s.lobby, s.seat, s.archetype_key) for s in states]
    assert len(keys) == len(set(keys)), "duplicate post-assembly cohort keys"

    fate_mass = Counter()
    fate_counts = Counter()
    total_missing_mass = 0.0
    for s in states:
        for card in s.cards.values():
            if card.present_final:
                continue
            w = card.weight
            total_missing_mass += w
            fate = card.primary_fate or "H_UNRESOLVED"
            fate_mass[fate] += w
            fate_counts[fate] += 1

    n = len(states)
    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    funnel = {
        "n_post_assembly_states": n,
        "reached_3_plus": sum(1 for s in states if s.turn_first_3 is not None),
        "reached_4_plus": sum(1 for s in states if s.turn_first_4 is not None),
        "survived_2plus_1_turn": sum(1 for s in states if s.survived_2_plus_1_turn),
        "survived_2plus_2_turns": sum(
            1 for s in states if s.survived_2_plus_2_turns),
        "mean_final_core_count": _mean([s.final_core_count for s in states]),
        "mean_final_coverage": _mean([s.final_coverage for s in states]),
    }

    distributions = {
        "turn_first_2": [s.entry_turn for s in states],
        "turn_first_3": [s.turn_first_3 for s in states if s.turn_first_3],
        "turn_first_4": [s.turn_first_4 for s in states if s.turn_first_4],
        "max_core_count": [s.max_core_count for s in states],
        "final_core_count": [s.final_core_count for s in states],
        "coverage_at_first_2": [s.coverage_at_first_2 for s in states],
        "max_coverage": [s.max_coverage for s in states],
        "final_coverage": [s.final_coverage for s in states],
        "coverage_peak_minus_final": [
            s.max_coverage - s.final_coverage for s in states],
    }
    dist_summary = {
        k: {
            "n": len(v),
            "mean": _mean(v),
            "min": min(v) if v else None,
            "max": max(v) if v else None,
        }
        for k, v in distributions.items()
    }

    # Aggregate weighted funnel (mean over states)
    weighted_funnel = {
        "mean_remaining_core_weight_at_first_2": _mean(
            [s.remaining_weight_at_first_2 for s in states]),
        "mean_weight_legally_available_after": _mean(
            [s.weight_legally_available_after for s in states]),
        "mean_weight_purchased_after": _mean(
            [s.weight_purchased_after for s in states]),
        "mean_weight_deployed_after": _mean(
            [s.weight_deployed_after for s in states]),
        "mean_weight_retained_2_turns": _mean(
            [s.weight_retained_2_turns for s in states]),
        "mean_weight_present_final": _mean(
            [s.weight_present_final for s in states]),
    }

    missing_mass_total = sum(fate_mass.values())
    fate_share = {
        code: (fate_mass[code] / missing_mass_total if missing_mass_total else 0.0)
        for code in FATE_CODES
    }

    # Integrity: assigned missing mass ≈ sum(1 - final_coverage)
    expected_missing = sum(1.0 - s.final_coverage for s in states)
    mass_reconcile = {
        "sum_missing_final_coverage": expected_missing,
        "sum_fate_mass": missing_mass_total,
        "abs_diff": abs(expected_missing - missing_mass_total),
        "within_tolerance": abs(expected_missing - missing_mass_total) < 1e-6 * max(1, n),
    }

    records = []
    for s in states:
        records.append({
            "lobby": s.lobby,
            "seat": s.seat,
            "archetype_key": s.archetype_key,
            "entry_turn": s.entry_turn,
            "entry_tier": s.entry_tier,
            "coverage_at_first_2": s.coverage_at_first_2,
            "max_coverage": s.max_coverage,
            "final_coverage": s.final_coverage,
            "coverage_peak_minus_final": s.max_coverage - s.final_coverage,
            "max_core_count": s.max_core_count,
            "final_core_count": s.final_core_count,
            "turn_first_3": s.turn_first_3,
            "turn_first_4": s.turn_first_4,
            "target_switch_turn": s.target_switch_turn,
            "card_fates": {
                name: {
                    "weight": c.weight,
                    "present_final": c.present_final,
                    "primary_fate": c.primary_fate,
                    "present_at_first_2": c.present_at_first_2,
                    "legally_offered_after": c.legally_offered_after,
                    "purchased_after": c.purchased_after,
                    "played_after": c.played_after,
                }
                for name, c in s.cards.items()
            },
        })

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "phase_2c_methodology_version": PHASE_2C_VERSION,
        "n_states": n,
        "funnel": funnel,
        "distribution_summary": dist_summary,
        "weighted_funnel": weighted_funnel,
        "missing_coverage_mass_by_cause": dict(fate_mass),
        "missing_coverage_mass_share_by_cause": fate_share,
        "missing_coverage_card_counts_by_cause": dict(fate_counts),
        "mass_reconciliation": mass_reconcile,
        "dominant_cause": (
            max(fate_share.items(), key=lambda x: x[1])[0] if n else None),
        "dominant_share": (
            max(fate_share.values()) if n and fate_share else None),
        "state_records": records,
    }
