"""Phase 2L availability decomposition (2l_v2).

Measurement-only: decompose Phase 2K's A_NEVER_AVAILABLE_POST_ASSEMBLY mass using
*exact* ``BGEnv.build_pool`` catalogue membership, then calibrate the sampler on an
unconditioned eligible cohort (not the never-legal A3 selection).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from hsbg_coach import cards as cards_mod
from hsbg_coach.bg_env import POOL_COPIES, SHOP_SLOTS, build_pool
from hsbg_coach.build_path import Archetype, load_archetypes
from ml.composition_diagnostic import (
    METHODOLOGY_VERSION as PHASE_2C_VERSION,
    _archetype_eligible,
    _core_set,
    _lobby_tribes,
    _max_core_count,
    _winner_for_lobby,
)
from ml.post_assembly_gap_diagnostic import (
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    PHASE_2J_PRIOR_PATH,
    core_weights,
    weighted_coverage,
)

METHODOLOGY_VERSION = "2l_v2"

PHASE_2L_SEED = 10200
PHASE_2L_LOBBIES = 500
PHASE_2L_EXPAND_THROUGH = 10999
PHASE_2L_MIN_STATES = 40

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
)

SUBFATE_CODES = (
    "A1_NOT_IN_EXACT_CATALOGUE",
    "A2_NEVER_TIER_ELIGIBLE",
    "A3_TIER_ELIGIBLE_ZERO_RAW",
    "A4_RAW_BUT_ZERO_LEGAL",
    "A5_OTHER",
)

# Why a card is absent from exact build_pool catalogue (A1 subtypes).
EXCLUSION_REASONS = (
    "MISSING_KB_OR_TIER_OR_STATS",
    "TRIBE_EXCLUDED",
    "BUILD_POOL_EXCLUDED",  # KB tribe-ok but build_pool still excludes (skin/dedup/etc.)
)


@lru_cache(maxsize=1)
def _kb_by_name() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for ck in cards_mod.load_kb().values():
        if ck.name and ck.name not in out:
            out[ck.name] = ck
    return out


def card_tier(name: str) -> Optional[int]:
    ck = _kb_by_name().get(name)
    if ck is None or ck.tier is None:
        return None
    return int(ck.tier)


@lru_cache(maxsize=64)
def exact_catalogue_names(lobby_tribes: Tuple[str, ...]) -> frozenset:
    """Exact simulator catalogue for this lobby (``build_pool``)."""
    return frozenset(m.name for m in build_pool(lobby_tribes=list(lobby_tribes)))


def tribe_eligible(name: str, lobby_tribes: List[str]) -> Optional[bool]:
    """None if missing from KB; else whether tribe filter alone would pass."""
    ck = _kb_by_name().get(name)
    if ck is None:
        return None
    allowed = set(lobby_tribes or [])
    trs = ck.tribes or []
    if not trs or "All" in trs or any(t in allowed for t in trs):
        return True
    return False


def catalogue_exclusion_reason(
        name: str, lobby_tribes: List[str],
        catalogue_names: Set[str]) -> Optional[str]:
    """If ``name`` not in exact catalogue, classify why."""
    if name in catalogue_names:
        return None
    ck = _kb_by_name().get(name)
    if (ck is None or ck.tier is None or not (1 <= int(ck.tier) <= 6)
            or ck.attack is None or ck.health is None):
        return "MISSING_KB_OR_TIER_OR_STATS"
    if tribe_eligible(name, lobby_tribes) is False:
        return "TRIBE_EXCLUDED"
    # Skin / dedup / embedding-thin fallback path / other build_pool filters.
    return "BUILD_POOL_EXCLUDED"


def slot_draw_probability(
        card_name: str, *, tavern_tier: int,
        catalogue: List, ) -> float:
    """P(a single shop slot is ``card_name``) under static POOL_COPIES weights."""
    ct = card_tier(card_name)
    if ct is None or ct > tavern_tier:
        return 0.0
    eligible = [m for m in catalogue if m.tier <= tavern_tier]
    if not eligible:
        return 0.0
    weights = {m.name: POOL_COPIES.get(m.tier, 1) for m in eligible}
    total = sum(weights.values()) or 1.0
    return weights.get(card_name, 0) / total


def expected_raw_one_deal(p_slot: float, n_slots: int) -> float:
    return n_slots * p_slot


def p_zero_one_deal(p_slot: float, n_slots: int) -> float:
    """P(card absent from all slots) under independent with-replacement draws."""
    if n_slots <= 0:
        return 1.0
    return (1.0 - p_slot) ** n_slots


@dataclass
class CardAvailabilityTrace:
    name: str
    weight: float
    card_tier: Optional[int]
    in_exact_catalogue: bool
    exclusion_reason: Optional[str]
    kb_tribe_eligible: Optional[bool]
    present_at_first_2: bool
    present_final: bool
    n_post_shop_observations: int = 0
    n_tier_eligible_obs: int = 0
    n_raw_appearances: int = 0
    n_legal_appearances: int = 0
    n_raw_but_illegal: int = 0
    # Per-deal expected / zero-prob product for tier-eligible deals only
    expected_raw_sum: float = 0.0
    p_zero_expected: float = 1.0  # product over tier-eligible deals
    subfate: Optional[str] = None


@dataclass
class AvailabilityState:
    lobby: int
    seat: int
    archetype_key: str
    entry_turn: int
    lobby_tribes: List[str]
    catalogue_size: int
    cards: Dict[str, CardAvailabilityTrace] = field(default_factory=dict)
    final_coverage: float = 0.0


def _raw_core_names(pre_shop: List[Dict], core: Set[str]) -> Set[str]:
    return {c.get("name") for c in (pre_shop or [])
            if c.get("name") in core}


def _legal_core_names(pre_shop: List[Dict], legal_slots: List[int],
                      core: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for slot in legal_slots or []:
        if slot < len(pre_shop):
            name = pre_shop[slot].get("name")
            if name in core:
                out.add(name)
    return out


def analyze_lobby_availability(
        traces: Dict, lobby: int,
        archetypes: Optional[List[Archetype]] = None,
) -> List[AvailabilityState]:
    archetypes = archetypes or load_archetypes()
    winner = _winner_for_lobby(traces, lobby)
    if winner is None:
        return []
    seat = winner["seat"]
    tribes = _lobby_tribes(traces, lobby)
    tribes_key = tuple(tribes)
    catalogue = list(build_pool(lobby_tribes=tribes))
    catalogue_names = set(exact_catalogue_names(tribes_key))
    turn_ends = sorted(
        [ts for ts in traces.get("turn_summaries") or []
         if ts["lobby"] == lobby and ts["seat"] == seat],
        key=lambda t: t["turn"])
    results: List[AvailabilityState] = []

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
        present0 = {c.get("name") for c in entry_board if c.get("name") in core}
        final_board = winner.get("final_board") or []
        final_present = {c.get("name") for c in final_board if c.get("name") in core}

        state = AvailabilityState(
            lobby=lobby, seat=seat, archetype_key=arch.key,
            entry_turn=entry_turn, lobby_tribes=list(tribes),
            catalogue_size=len(catalogue_names),
            final_coverage=weighted_coverage(final_board, weights),
        )
        for name, w in weights.items():
            in_cat = name in catalogue_names
            state.cards[name] = CardAvailabilityTrace(
                name=name, weight=w, card_tier=card_tier(name),
                in_exact_catalogue=in_cat,
                exclusion_reason=(
                    None if in_cat
                    else catalogue_exclusion_reason(name, tribes, catalogue_names)),
                kb_tribe_eligible=tribe_eligible(name, tribes),
                present_at_first_2=(name in present0),
                present_final=(name in final_present),
            )

        seen_gens: Set[Tuple[int, int]] = set()
        for ev in traces.get("events") or []:
            if ev["lobby"] != lobby or ev["seat"] != seat:
                continue
            turn = int(ev["turn"])
            if turn < entry_turn:
                continue
            if turn == entry_turn and ev["action"] != "end":
                continue
            gen = int(ev.get("shop_generation") or 0)
            gkey = (turn, gen)
            if gkey in seen_gens:
                continue
            seen_gens.add(gkey)

            pre_shop = ev.get("pre_shop") or []
            legal_slots = ev.get("legal_buy_slots") or []
            tavern = int(ev.get("tavern_tier") or 1)
            n_slots = len(pre_shop) or int(SHOP_SLOTS.get(tavern, 6))
            raw_names = _raw_core_names(pre_shop, core)
            legal_names = _legal_core_names(pre_shop, legal_slots, core)

            for name, card in state.cards.items():
                if card.present_final and card.present_at_first_2:
                    continue
                card.n_post_shop_observations += 1
                ct = card.card_tier
                # Tier eligibility only meaningful for exact-catalogue cards.
                tier_ok = (
                    card.in_exact_catalogue
                    and ct is not None
                    and tavern >= ct)
                if tier_ok:
                    card.n_tier_eligible_obs += 1
                    p_slot = slot_draw_probability(
                        name, tavern_tier=tavern, catalogue=catalogue)
                    card.expected_raw_sum += expected_raw_one_deal(p_slot, n_slots)
                    card.p_zero_expected *= p_zero_one_deal(p_slot, n_slots)
                if name in raw_names:
                    card.n_raw_appearances += 1
                    if name not in legal_names:
                        card.n_raw_but_illegal += 1
                if name in legal_names:
                    card.n_legal_appearances += 1

        for card in state.cards.values():
            if card.present_final or card.n_legal_appearances > 0:
                card.subfate = None
                continue
            # Never-legal missing mass decomposition (exact catalogue first).
            if not card.in_exact_catalogue:
                card.subfate = "A1_NOT_IN_EXACT_CATALOGUE"
            elif card.n_tier_eligible_obs == 0:
                card.subfate = "A2_NEVER_TIER_ELIGIBLE"
            elif card.n_raw_appearances == 0:
                card.subfate = "A3_TIER_ELIGIBLE_ZERO_RAW"
            elif card.n_raw_appearances > 0 and card.n_legal_appearances == 0:
                card.subfate = "A4_RAW_BUT_ZERO_LEGAL"
            else:
                card.subfate = "A5_OTHER"

        results.append(state)
    return results


def analyze_availability_decomposition(traces: Dict) -> Dict:
    archetypes = load_archetypes()
    states: List[AvailabilityState] = []
    for lobby in range(traces["lobbies"]):
        states.extend(analyze_lobby_availability(traces, lobby, archetypes))

    keys = [(s.lobby, s.seat, s.archetype_key) for s in states]
    assert len(keys) == len(set(keys))

    n = len(states)
    sub_mass = Counter()
    sub_counts = Counter()
    exclusion_mass = Counter()
    exclusion_counts = Counter()

    never_legal_mass = 0.0
    a1_mass = a2_mass = a3_mass = a4_mass = 0.0

    # Unconditioned sampler cohort: missing final, exact catalogue, tier-eligible
    calib_cards = 0
    calib_expected_raw = 0.0
    calib_observed_raw = 0.0
    calib_expected_ge1 = 0.0  # Σ (1 − P_zero) over card-windows
    calib_observed_ge1 = 0
    calib_zero_obs = 0
    calib_p_zero_sum = 0.0
    kb_ok_but_pool_excludes_mass = 0.0

    for s in states:
        for card in s.cards.values():
            if (card.exclusion_reason == "BUILD_POOL_EXCLUDED"
                    and not card.present_final):
                kb_ok_but_pool_excludes_mass += card.weight

            # Sampler calibration cohort (unconditioned on never-legal / A3)
            if (not card.present_final
                    and card.in_exact_catalogue
                    and card.n_tier_eligible_obs > 0):
                calib_cards += 1
                calib_expected_raw += card.expected_raw_sum
                calib_observed_raw += card.n_raw_appearances
                calib_expected_ge1 += 1.0 - card.p_zero_expected
                if card.n_raw_appearances >= 1:
                    calib_observed_ge1 += 1
                if card.n_raw_appearances == 0:
                    calib_zero_obs += 1
                calib_p_zero_sum += card.p_zero_expected

            if card.subfate is None:
                continue
            w = card.weight
            never_legal_mass += w
            sub_mass[card.subfate] += w
            sub_counts[card.subfate] += 1
            if card.subfate == "A1_NOT_IN_EXACT_CATALOGUE":
                a1_mass += w
                reason = card.exclusion_reason or "MISSING_KB_OR_TIER_OR_STATS"
                exclusion_mass[reason] += w
                exclusion_counts[reason] += 1
            elif card.subfate == "A2_NEVER_TIER_ELIGIBLE":
                a2_mass += w
            elif card.subfate == "A3_TIER_ELIGIBLE_ZERO_RAW":
                a3_mass += w
            elif card.subfate == "A4_RAW_BUT_ZERO_LEGAL":
                a4_mass += w

    def _share(x: float) -> Optional[float]:
        return x / never_legal_mass if never_legal_mass else None

    sub_share = {c: (sub_mass[c] / never_legal_mass if never_legal_mass else 0.0)
                 for c in SUBFATE_CODES}
    exclusion_share = {
        r: (exclusion_mass[r] / a1_mass if a1_mass else 0.0)
        for r in EXCLUSION_REASONS
    }

    total_missing = sum(1.0 - s.final_coverage for s in states)

    records = []
    for s in states:
        records.append({
            "lobby": s.lobby,
            "seat": s.seat,
            "archetype_key": s.archetype_key,
            "entry_turn": s.entry_turn,
            "catalogue_size": s.catalogue_size,
            "final_coverage": s.final_coverage,
            "cards": {
                name: {
                    "weight": c.weight,
                    "card_tier": c.card_tier,
                    "in_exact_catalogue": c.in_exact_catalogue,
                    "exclusion_reason": c.exclusion_reason,
                    "kb_tribe_eligible": c.kb_tribe_eligible,
                    "present_final": c.present_final,
                    "n_tier_eligible_obs": c.n_tier_eligible_obs,
                    "n_raw_appearances": c.n_raw_appearances,
                    "n_legal_appearances": c.n_legal_appearances,
                    "n_raw_but_illegal": c.n_raw_but_illegal,
                    "expected_raw_sum": c.expected_raw_sum,
                    "p_zero_expected": c.p_zero_expected,
                    "subfate": c.subfate,
                }
                for name, c in s.cards.items()
            },
        })

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "phase_2c_methodology_version": PHASE_2C_VERSION,
        "n_states": n,
        "never_legal_missing_mass": never_legal_mass,
        "total_missing_final_coverage_mass": total_missing,
        "never_legal_share_of_total_missing": (
            never_legal_mass / total_missing if total_missing else None),
        "subfate_mass": dict(sub_mass),
        "subfate_share_of_never_legal": sub_share,
        "subfate_card_counts": dict(sub_counts),
        "a1_exclusion_breakdown": {
            "mass": dict(exclusion_mass),
            "share_of_a1": exclusion_share,
            "card_counts": dict(exclusion_counts),
            "kb_says_tribe_ok_but_build_pool_excludes_mass": (
                kb_ok_but_pool_excludes_mass),
        },
        "headlines": {
            # Primary 2l_v2 headlines over never-legal missing mass
            "pct_not_in_exact_simulator_catalogue": _share(a1_mass),
            "pct_exact_catalogue_tier_eligible_zero_raw": _share(a3_mass),
            "pct_raw_but_never_legal": _share(a4_mass),
            "pct_never_tier_eligible": _share(a2_mass),
            # Absolute masses
            "not_in_exact_catalogue_mass": a1_mass,
            "tier_eligible_zero_raw_mass": a3_mass,
            "raw_but_zero_legal_mass": a4_mass,
        },
        "sampler_calibration_unconditioned": {
            "cohort": (
                "missing-final cores that exist in exact build_pool catalogue "
                "AND have ≥1 tier-eligible post-assembly shop observation"),
            "n_card_windows": calib_cards,
            "sum_expected_raw_appearances": calib_expected_raw,
            "sum_observed_raw_appearances": calib_observed_raw,
            "expected_cards_with_ge1_appearance": calib_expected_ge1,
            "observed_cards_with_ge1_appearance": calib_observed_ge1,
            "observed_zero_offer_rate": (
                calib_zero_obs / calib_cards if calib_cards else None),
            "expected_zero_offer_rate": (
                calib_p_zero_sum / calib_cards if calib_cards else None),
            "note": (
                "Unconditioned on never-legal/A3 selection. Expected uses static "
                "POOL_COPIES catalogue weights with with-replacement slot approx; "
                "not live-pool depletion. Do not treat A3's observed-zero count as "
                "sampler underperformance evidence."),
        },
        # Backward-compat alias — runner previously keyed sampler_diagnostic
        "sampler_diagnostic": {
            "deprecated": True,
            "see": "sampler_calibration_unconditioned",
            "selection_bias_warning": (
                "2l_v1 summed expected_raw over never-legal cards (A3-conditioned). "
                "That comparison is selection-biased and must not be used as "
                "confirmatory evidence of sampler underperformance."),
        },
        "dominant_subfate": (
            max(sub_share.items(), key=lambda x: x[1])[0] if n else None),
        "dominant_share": (
            max(sub_share.values()) if n and sub_share else None),
        "state_records": records,
    }
