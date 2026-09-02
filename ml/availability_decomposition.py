"""Phase 2L availability decomposition (2l_v1).

Measurement-only: decompose Phase 2K's A_NEVER_AVAILABLE_POST_ASSEMBLY mass into
tier eligibility, raw shop appearance, legal-buy appearance, pool/card-data, and
economy/mask losses. Does not change policy or simulator.
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

METHODOLOGY_VERSION = "2l_v1"

PHASE_2L_SEED = 10200
PHASE_2L_LOBBIES = 500
PHASE_2L_EXPAND_THROUGH = 10999
PHASE_2L_MIN_STATES = 40

# Preserve prior phase confirmations / diagnostics.
FORBIDDEN_RANGES = (
    (8000, 8199),   # Phase 2J confirmation
    (9000, 9999),   # Phase 2K diagnostic
    (10000, 10199), # reserved future intervention
)

SUBFATE_CODES = (
    "A1_NOT_IN_LOBBY_POOL",
    "A2_NEVER_TIER_ELIGIBLE",
    "A3_TIER_ELIGIBLE_ZERO_RAW",
    "A4_RAW_BUT_ZERO_LEGAL",
    "A5_OTHER",
)


@lru_cache(maxsize=1)
def _name_to_tier() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ck in cards_mod.load_kb().values():
        if ck.name and ck.tier is not None and ck.name not in out:
            out[ck.name] = int(ck.tier)
    return out


@lru_cache(maxsize=1)
def _name_to_tribes() -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    for ck in cards_mod.load_kb().values():
        if ck.name and ck.name not in out:
            out[ck.name] = tuple(ck.tribes or [])
    return out


def card_tier(name: str) -> Optional[int]:
    return _name_to_tier().get(name)


def card_in_lobby_pool(name: str, lobby_tribes: List[str]) -> bool:
    """Whether ``build_pool`` could include this card for the lobby."""
    tribes = _name_to_tribes().get(name)
    tier = card_tier(name)
    if tribes is None or tier is None or not (1 <= tier <= 6):
        return False
    allowed = set(lobby_tribes or [])
    if not tribes or "All" in tribes or any(t in allowed for t in tribes):
        return True
    return False


def expected_raw_appearances(
        *, card_name: str, tavern_tier: int, n_shop_slots: int,
        lobby_tribes: List[str]) -> float:
    """Rough expected count of raw appearances in one shop deal (with-replacement approx).

    Uses static catalogue composition and POOL_COPIES as initial weights — not
    exact live pool state, but a comparable order-of-magnitude diagnostic.
    """
    ct = card_tier(card_name)
    if ct is None or ct > tavern_tier:
        return 0.0
    if not card_in_lobby_pool(card_name, lobby_tribes):
        return 0.0
    catalogue = build_pool(lobby_tribes=lobby_tribes)
    eligible = [m for m in catalogue if m.tier <= tavern_tier]
    if not eligible:
        return 0.0
    weights = {m.name: POOL_COPIES.get(m.tier, 1) for m in eligible}
    total = sum(weights.values()) or 1.0
    p = weights.get(card_name, 0) / total
    # Approx: expected slots containing the card under with-replacement draw.
    return n_shop_slots * p


@dataclass
class CardAvailabilityTrace:
    name: str
    weight: float
    card_tier: Optional[int]
    in_lobby_pool: bool
    present_at_first_2: bool
    present_final: bool
    n_post_shop_observations: int = 0
    n_tier_eligible_obs: int = 0
    n_raw_appearances: int = 0
    n_legal_appearances: int = 0
    n_raw_but_illegal: int = 0  # raw in shop while buys masked
    expected_raw_sum: float = 0.0
    subfate: Optional[str] = None  # only if never legal after first-2 and missing final


@dataclass
class AvailabilityState:
    lobby: int
    seat: int
    archetype_key: str
    entry_turn: int
    lobby_tribes: List[str]
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
            final_coverage=weighted_coverage(final_board, weights),
        )
        for name, w in weights.items():
            state.cards[name] = CardAvailabilityTrace(
                name=name, weight=w, card_tier=card_tier(name),
                in_lobby_pool=card_in_lobby_pool(name, tribes),
                present_at_first_2=(name in present0),
                present_final=(name in final_present),
            )

        # Deduplicate shop observations by (turn, shop_generation)
        seen_gens: Set[Tuple[int, int]] = set()
        for ev in traces.get("events") or []:
            if ev["lobby"] != lobby or ev["seat"] != seat:
                continue
            turn = int(ev["turn"])
            if turn < entry_turn:
                continue
            if turn == entry_turn and ev["action"] != "end":
                # Cohort begins at end-of-recruit; shop obs start next turn.
                continue
            gen = int(ev.get("shop_generation") or 0)
            gkey = (turn, gen)
            # Count each generation once (first action in that gen).
            if gkey in seen_gens:
                # Still allow buy/legal updates? Use first observation only for
                # appearance counting to avoid double-counting same shop.
                continue
            seen_gens.add(gkey)

            pre_shop = ev.get("pre_shop") or []
            legal_slots = ev.get("legal_buy_slots") or []
            tavern = int(ev.get("tavern_tier") or 1)
            n_slots = len(pre_shop) or int(SHOP_SLOTS.get(tavern, 6))
            buys_legal = bool(legal_slots)  # empty → gold/hand blocked all buys
            raw_names = _raw_core_names(pre_shop, core)
            legal_names = _legal_core_names(pre_shop, legal_slots, core)

            for name, card in state.cards.items():
                if card.present_final and card.present_at_first_2:
                    continue
                card.n_post_shop_observations += 1
                ct = card.card_tier
                tier_ok = ct is not None and tavern >= ct
                if tier_ok:
                    card.n_tier_eligible_obs += 1
                    card.expected_raw_sum += expected_raw_appearances(
                        card_name=name, tavern_tier=tavern,
                        n_shop_slots=n_slots, lobby_tribes=tribes)
                if name in raw_names:
                    card.n_raw_appearances += 1
                    if name not in legal_names:
                        card.n_raw_but_illegal += 1
                if name in legal_names:
                    card.n_legal_appearances += 1

        for card in state.cards.values():
            if card.present_final:
                card.subfate = None
                continue
            if card.n_legal_appearances > 0:
                # Not in the 2K "never available" bucket.
                card.subfate = None
                continue
            # Decompose never-legal missing mass:
            if not card.in_lobby_pool:
                card.subfate = "A1_NOT_IN_LOBBY_POOL"
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
    # Headlines over never-legal missing mass
    never_legal_mass = 0.0
    tier_ok_zero_raw_mass = 0.0
    raw_zero_legal_mass = 0.0
    never_tier_mass = 0.0
    not_in_pool_mass = 0.0

    expected_raw_total = 0.0
    observed_raw_total = 0.0

    for s in states:
        for card in s.cards.values():
            if card.subfate is None:
                continue
            w = card.weight
            never_legal_mass += w
            sub_mass[card.subfate] += w
            sub_counts[card.subfate] += 1
            if card.subfate == "A3_TIER_ELIGIBLE_ZERO_RAW":
                tier_ok_zero_raw_mass += w
            elif card.subfate == "A4_RAW_BUT_ZERO_LEGAL":
                raw_zero_legal_mass += w
            elif card.subfate == "A2_NEVER_TIER_ELIGIBLE":
                never_tier_mass += w
            elif card.subfate == "A1_NOT_IN_LOBBY_POOL":
                not_in_pool_mass += w
            expected_raw_total += card.expected_raw_sum
            observed_raw_total += card.n_raw_appearances

    def _share(x: float) -> Optional[float]:
        return x / never_legal_mass if never_legal_mass else None

    sub_share = {c: (sub_mass[c] / never_legal_mass if never_legal_mass else 0.0)
                 for c in SUBFATE_CODES}

    total_missing = sum(1.0 - s.final_coverage for s in states)
    # Never-legal mass should be ≤ total missing (other fates exist in 2K)
    records = []
    for s in states:
        records.append({
            "lobby": s.lobby,
            "seat": s.seat,
            "archetype_key": s.archetype_key,
            "entry_turn": s.entry_turn,
            "final_coverage": s.final_coverage,
            "cards": {
                name: {
                    "weight": c.weight,
                    "card_tier": c.card_tier,
                    "in_lobby_pool": c.in_lobby_pool,
                    "present_final": c.present_final,
                    "n_tier_eligible_obs": c.n_tier_eligible_obs,
                    "n_raw_appearances": c.n_raw_appearances,
                    "n_legal_appearances": c.n_legal_appearances,
                    "n_raw_but_illegal": c.n_raw_but_illegal,
                    "expected_raw_sum": c.expected_raw_sum,
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
        "headlines": {
            "pct_never_legal_mass_tier_eligible_zero_raw": _share(
                tier_ok_zero_raw_mass),
            "pct_never_legal_mass_raw_but_zero_legal": _share(
                raw_zero_legal_mass),
            "pct_never_legal_mass_never_tier_eligible": _share(never_tier_mass),
            "pct_never_legal_mass_not_in_lobby_pool": _share(not_in_pool_mass),
            "tier_eligible_zero_raw_mass": tier_ok_zero_raw_mass,
            "raw_but_zero_legal_mass": raw_zero_legal_mass,
        },
        "sampler_diagnostic": {
            "sum_expected_raw_appearances_never_legal_cards": expected_raw_total,
            "sum_observed_raw_appearances_never_legal_cards": observed_raw_total,
            "observed_over_expected": (
                observed_raw_total / expected_raw_total
                if expected_raw_total > 1e-12 else None),
            "note": (
                "Expected uses static POOL_COPIES catalogue weights per shop deal; "
                "not exact live-pool simulation."),
        },
        "dominant_subfate": (
            max(sub_share.items(), key=lambda x: x[1])[0] if n else None),
        "dominant_share": (
            max(sub_share.values()) if n and sub_share else None),
        "state_records": records,
    }
