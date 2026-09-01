"""Phase 2C composition assembly funnel and failure classification (v2).

Methodology fixes (v2):
- Global shop availability (all seats) is separate from winner decision funnel.
- Winner opportunities use pre-action shop + legal buy slots, deduped by shop generation.
- Rejection = legally buyable core survived until shop generation ended without purchase.
- Archetype relevance: current-target vs final-target hindsight views.
- Tribe-ineligible archetypes excluded from decision-failure denominators.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from hsbg_coach.build_path import Archetype, load_archetypes

METHODOLOGY_VERSION = "2c_v2"
RETAIN_TURNS = 2

INVALIDATED_PRIOR = {
    "note": "Invalidated by seat/shop opportunity-scope bugs in first Phase 2C run",
    "classification_totals": {
        "B_AVAILABLE_NOT_BOUGHT": 2516,
        "A_IMPOSSIBLE": 1185,
        "C_BOUGHT_NOT_RETAINED": 99,
    },
    "recommended_intervention": "build_aware_recruit_policy",
}


def _core_set(arch: Archetype) -> Set[str]:
    return set(arch.core.keys())


def _names_on_board_hand(board: List[Dict], hand: Optional[List[Dict]] = None) -> Set[str]:
    names = {c["name"] for c in board if c.get("name")}
    if hand:
        names |= {c["name"] for c in hand if c.get("name")}
    return names


def _max_core_count(board: List[Dict], core: Set[str]) -> int:
    return sum(1 for c in board if c.get("name") in core)


def _archetype_eligible(arch: Archetype, lobby_tribes: List[str]) -> bool:
    if not arch.tribe:
        return True
    return arch.tribe in lobby_tribes


def _legally_buyable_cores(pre_shop: List[Dict], legal_buy_slots: List[int],
                           core: Set[str]) -> Dict[str, Dict]:
    """Map core name -> card dict for slots that are legally buyable."""
    out: Dict[str, Dict] = {}
    for slot in legal_buy_slots:
        if slot < len(pre_shop):
            card = pre_shop[slot]
            name = card.get("name")
            if name in core and name not in out:
                out[name] = card
    return out


@dataclass
class _ShopGenState:
    """Tracks one shop generation for winner decision funnel."""
    turn: int
    shop_generation: int
    target_key: Optional[str]
    legally_buyable: Dict[str, Dict] = field(default_factory=dict)
    purchased: Set[str] = field(default_factory=set)


@dataclass
class _WinnerFunnelState:
    core: Set[str]
    lobby_tribes: List[str]
    # Distinct relevant cores legally buyable across the game (deduped by name).
    distinct_legally_available: Set[str] = field(default_factory=set)
    shop_generations_with_core: int = 0
    legally_buyable_exposures: int = 0
    purchased: int = 0
    rejected_at_shop_exit: int = 0
    plays: int = 0
    retained_2_turns: int = 0
    max_core_reached: int = 0
    turn_first_2: Optional[int] = None
    turn_first_4: Optional[int] = None
    final_core: int = 0
    final_coverage: float = 0.0
    rejections: List[Dict] = field(default_factory=list)
    _active_gen: Optional[_ShopGenState] = None
    _purchased_turn: Dict[str, int] = field(default_factory=dict)
    _seen_gen_keys: Set[Tuple[int, int]] = field(default_factory=set)

    def _gen_key(self, turn: int, shop_generation: int) -> Tuple[int, int]:
        return (turn, shop_generation)

    def open_generation(self, turn: int, shop_generation: int,
                        target_key: Optional[str],
                        legally_buyable: Dict[str, Dict]) -> None:
        if not legally_buyable:
            return
        gkey = self._gen_key(turn, shop_generation)
        if self._active_gen and (
                self._active_gen.turn, self._active_gen.shop_generation) == gkey:
            for name, card in legally_buyable.items():
                if name not in self._active_gen.legally_buyable:
                    self._active_gen.legally_buyable[name] = card
                    self.legally_buyable_exposures += 1
                    self.distinct_legally_available.add(name)
            return
        if gkey not in self._seen_gen_keys:
            self._seen_gen_keys.add(gkey)
            self.shop_generations_with_core += 1
        self.legally_buyable_exposures += len(legally_buyable)
        for name in legally_buyable:
            self.distinct_legally_available.add(name)
        if self._active_gen and self._active_gen.legally_buyable:
            self._close_generation()
        self._active_gen = _ShopGenState(
            turn=turn, shop_generation=shop_generation,
            target_key=target_key, legally_buyable=dict(legally_buyable))

    def note_purchase(self, name: str, turn: int) -> None:
        if self._active_gen and name in self._active_gen.legally_buyable:
            self._active_gen.purchased.add(name)
        self.purchased += 1
        self._purchased_turn[name] = turn

    def _close_generation(self) -> None:
        gen = self._active_gen
        if gen is None:
            return
        for name, card in gen.legally_buyable.items():
            if name not in gen.purchased:
                self.rejected_at_shop_exit += 1
                self.rejections.append({
                    "turn": gen.turn,
                    "shop_generation": gen.shop_generation,
                    "core_offered": name,
                    "core_stats": card.get("attack", 0) + card.get("health", 0),
                    "target_at_offer": gen.target_key,
                })
        self._active_gen = None

    def close_generation(self) -> None:
        self._close_generation()

    def flush(self) -> None:
        self.close_generation()

    def track_retention(self, turn: int, board: List[Dict],
                        hand: Optional[List[Dict]] = None) -> None:
        present = _names_on_board_hand(board, hand)
        for name, bought_turn in list(self._purchased_turn.items()):
            if name in present and turn - bought_turn >= RETAIN_TURNS:
                self.retained_2_turns += 1
                del self._purchased_turn[name]


@dataclass
class _GlobalAvailabilityState:
    """All-seat shop exposure deduped by (seat, turn, shop_generation, core_name)."""
    core: Set[str]
    seen_exposures: Set[Tuple[int, int, int, str]] = field(default_factory=set)
    seen_generations: Set[Tuple[int, int, int]] = field(default_factory=set)
    distinct_cores_seen: Set[str] = field(default_factory=set)
    total_exposures: int = 0

    def note_shop(self, seat: int, turn: int, shop_generation: int,
                  pre_shop: List[Dict]) -> None:
        gen_key = (seat, turn, shop_generation)
        for card in pre_shop:
            name = card.get("name")
            if name not in self.core:
                continue
            exp_key = (seat, turn, shop_generation, name)
            if exp_key in self.seen_exposures:
                continue
            self.seen_exposures.add(exp_key)
            self.seen_generations.add(gen_key)
            self.distinct_cores_seen.add(name)
            self.total_exposures += 1


def _winner_for_lobby(traces: Dict, lobby: int) -> Optional[Dict]:
    return next(
        (p for p in traces["player_finals"]
         if p["lobby"] == lobby and p["placement"] == 1), None)


def _lobby_tribes(traces: Dict, lobby: int) -> List[str]:
    meta = next((m for m in traces.get("lobby_meta", [])
                 if m["lobby"] == lobby), None)
    if meta:
        return meta.get("lobby_tribes") or []
    winner = _winner_for_lobby(traces, lobby)
    return (winner or {}).get("lobby_tribes") or []


def _is_relevant_core(arch: Archetype, view: str,
                      target_before: Optional[Dict],
                      final_target_key: Optional[str]) -> bool:
    if view == "current_target":
        if not target_before:
            return False
        return target_before.get("archetype_key") == arch.key
    if view == "final_target_hindsight":
        return final_target_key == arch.key
    return False


def analyze_winner_funnel(traces: Dict, lobby: int, arch: Archetype,
                          view: str) -> Optional[_WinnerFunnelState]:
    """Winner-only decision funnel for a relevant archetype in one lobby."""
    core = _core_set(arch)
    lobby_tribes = _lobby_tribes(traces, lobby)
    if not _archetype_eligible(arch, lobby_tribes):
        return None

    winner = _winner_for_lobby(traces, lobby)
    if winner is None:
        return None
    winner_seat = winner["seat"]
    final_target_key = (winner.get("target") or {}).get("archetype_key")

    if view == "final_target_hindsight" and final_target_key != arch.key:
        return None

    state = _WinnerFunnelState(core=core, lobby_tribes=lobby_tribes)
    prev_gen_key: Optional[Tuple[int, int, int]] = None

    for ev in traces["events"]:
        if ev["lobby"] != lobby or ev["seat"] != winner_seat:
            continue

        turn = ev["turn"]
        shop_gen = ev.get("shop_generation", 0)
        gen_key = (turn, shop_gen)
        target_before = ev.get("target_before")

        if prev_gen_key is not None and gen_key != prev_gen_key:
            state.close_generation()
        prev_gen_key = gen_key

        pre_shop = ev.get("pre_shop") or []
        legal_slots = ev.get("legal_buy_slots") or []
        buyable = _legally_buyable_cores(pre_shop, legal_slots, core)

        if buyable and _is_relevant_core(arch, view, target_before, final_target_key):
            state.open_generation(
                turn, shop_gen,
                (target_before or {}).get("archetype_key"),
                buyable)

        if ev["action"] == "buy" and ev.get("card"):
            name = ev["card"]["name"]
            if name in core and _is_relevant_core(
                    arch, view, target_before, final_target_key):
                state.note_purchase(name, turn)
        elif ev["action"] == "play" and ev.get("card"):
            if ev["card"]["name"] in core:
                state.plays += 1

        if ev["action"] in ("roll", "end"):
            state.close_generation()
            if ev["action"] == "end":
                prev_gen_key = None

        state.track_retention(turn, ev.get("board_after") or [],
                              ev.get("hand_after"))

    state.flush()

    for ts in traces["turn_summaries"]:
        if ts["lobby"] != lobby or ts["seat"] != winner_seat:
            continue
        board = ts.get("board_after_recruit") or []
        count = _max_core_count(board, core)
        state.max_core_reached = max(state.max_core_reached, count)
        if count >= 2 and state.turn_first_2 is None:
            state.turn_first_2 = ts["turn"]
        if count >= 4 and state.turn_first_4 is None:
            state.turn_first_4 = ts["turn"]

    wboard = winner.get("final_board") or []
    state.final_core = _max_core_count(wboard, core)
    tgt = winner.get("target")
    if tgt and tgt.get("archetype_key") == arch.key:
        state.final_coverage = tgt.get("coverage", 0.0)
    else:
        wnames = {c.get("name") for c in wboard}
        have_w = sum(arch.core.get(n, 0.0) for n in wnames if n in core)
        denom = sum(arch.core.values()) or 1.0
        state.final_coverage = have_w / denom

    return state


def _current_target_keys(traces: Dict, lobby: int, winner_seat: int) -> Set[str]:
    keys: Set[str] = set()
    for ev in traces["events"]:
        if ev["lobby"] != lobby or ev["seat"] != winner_seat:
            continue
        tk = (ev.get("target_before") or {}).get("archetype_key")
        if tk:
            keys.add(tk)
    return keys


def analyze_global_availability(traces: Dict, lobby: int,
                                arch: Archetype) -> _GlobalAvailabilityState:
    """All-seat shop exposure for pool/starvation diagnosis."""
    core = _core_set(arch)
    state = _GlobalAvailabilityState(core=core)
    first_in_gen: Dict[Tuple[int, int, int], Dict] = {}

    for ev in traces["events"]:
        if ev["lobby"] != lobby:
            continue
        gen_key = (ev["seat"], ev["turn"], ev.get("shop_generation", 0))
        if gen_key not in first_in_gen:
            first_in_gen[gen_key] = ev

    for gen_key, ev in first_in_gen.items():
        seat, turn, shop_gen = gen_key
        state.note_shop(seat, turn, shop_gen, ev.get("pre_shop") or [])
    return state


def _classify_winner_funnel(state: _WinnerFunnelState,
                            winner_is_archetype: bool) -> str:
    if winner_is_archetype and state.final_coverage >= 0.4 and state.final_core >= 4:
        return "E_SUCCESSFULLY_ASSEMBLED"
    if state.max_core_reached >= 2 and state.final_coverage < 0.15:
        return "D_ASSEMBLED_NO_PAYOFF"
    if state.purchased > 0 and state.max_core_reached < 2:
        return "C_BOUGHT_NOT_RETAINED"
    if state.legally_buyable_exposures > 0 and state.purchased == 0:
        return "B_AVAILABLE_NOT_BOUGHT"
    if len(state.distinct_legally_available) >= 2 and state.purchased == 0:
        return "B_AVAILABLE_NOT_BOUGHT"
    if not state.distinct_legally_available:
        return "A_IMPOSSIBLE"
    if state.max_core_reached >= 2:
        return "D_ASSEMBLED_NO_PAYOFF"
    return "A_IMPOSSIBLE"


def _funnel_summary(states: List[_WinnerFunnelState]) -> Dict:
    n = len(states)
    if n == 0:
        return {}
    lobbies_2plus = sum(1 for s in states if len(s.distinct_legally_available) >= 2)
    lobbies_4plus = sum(1 for s in states if len(s.distinct_legally_available) >= 4)
    return {
        "n_lobbies_evaluated": n,
        "distinct_core_legally_available": sum(
            len(s.distinct_legally_available) for s in states),
        "shop_generations_with_buyable_core": sum(
            s.shop_generations_with_core for s in states),
        "legally_buyable_exposures": sum(s.legally_buyable_exposures for s in states),
        "purchased": sum(s.purchased for s in states),
        "rejected_at_shop_exit": sum(s.rejected_at_shop_exit for s in states),
        "played": sum(s.plays for s in states),
        "retained_2_turns": sum(s.retained_2_turns for s in states),
        "reached_2_core": sum(1 for s in states if s.max_core_reached >= 2),
        "reached_4_core": sum(1 for s in states if s.max_core_reached >= 4),
        "pct_lobbies_2plus_distinct_core_available": lobbies_2plus / n,
        "pct_lobbies_4plus_distinct_core_available": lobbies_4plus / n,
        "purchase_rate_when_legally_buyable": (
            sum(s.purchased for s in states)
            / max(1, sum(s.legally_buyable_exposures for s in states))),
        "mean_final_coverage_winner": st.mean([s.final_coverage for s in states]),
        "mean_max_core_pieces": st.mean([s.max_core_reached for s in states]),
    }


def _aggregate_view(traces: Dict, view: str) -> Dict:
    archetypes = load_archetypes()
    lobbies = traces["lobbies"]
    by_arch: Dict[str, Dict] = {}
    all_states: List[_WinnerFunnelState] = []
    classifications: Counter = Counter()
    ineligible_count = 0

    for arch in archetypes:
        arch_states: List[_WinnerFunnelState] = []
        arch_cls: Counter = Counter()
        global_states: List[_GlobalAvailabilityState] = []

        for lobby in range(lobbies):
            tribes = _lobby_tribes(traces, lobby)
            if not _archetype_eligible(arch, tribes):
                ineligible_count += 1
                continue
            g = analyze_global_availability(traces, lobby, arch)
            global_states.append(g)

            winner = _winner_for_lobby(traces, lobby)
            winner_seat = winner["seat"] if winner else None
            if view == "current_target":
                if winner_seat is None or arch.key not in _current_target_keys(
                        traces, lobby, winner_seat):
                    continue

            wf = analyze_winner_funnel(traces, lobby, arch, view)
            if wf is None:
                continue
            arch_states.append(wf)
            all_states.append(wf)
            winner = _winner_for_lobby(traces, lobby)
            winner_is = ((winner.get("target") or {}).get("archetype_key") == arch.key
                         if winner else False)
            cls = _classify_winner_funnel(wf, winner_is)
            arch_cls[cls] += 1
            classifications[cls] += 1

        if not arch_states and not global_states:
            continue

        rejections = []
        for s in arch_states:
            rejections.extend(s.rejections)

        by_arch[arch.key] = {
            "name": arch.name,
            "tribe": arch.tribe,
            "core_cards": list(arch.core.keys()),
            "classification": dict(arch_cls),
            "winner_decision_funnel": _funnel_summary(arch_states),
            "global_availability": {
                "mean_distinct_cores_seen_per_lobby": (
                    st.mean([len(g.distinct_cores_seen) for g in global_states])
                    if global_states else 0),
                "mean_shop_exposures_per_lobby": (
                    st.mean([g.total_exposures for g in global_states])
                    if global_states else 0),
                "pct_lobbies_2plus_core_seen": (
                    sum(1 for g in global_states if len(g.distinct_cores_seen) >= 2)
                    / len(global_states) if global_states else 0),
                "pct_lobbies_4plus_core_seen": (
                    sum(1 for g in global_states if len(g.distinct_cores_seen) >= 4)
                    / len(global_states) if global_states else 0),
            },
            "top_rejections_at_shop_exit": _top_rejections(rejections),
        }

    return {
        "view": view,
        "view_label": (
            "current-target (infer_target at offer time)"
            if view == "current_target"
            else "final-target hindsight (winner eventual archetype)"),
        "classification_totals": dict(classifications),
        "n_lobby_archetype_ineligible": ineligible_count,
        "aggregate_funnel": _funnel_summary(all_states),
        "by_archetype": by_arch,
    }


def _top_rejections(rejections: List[Dict], n: int = 5) -> List[Dict]:
    if not rejections:
        return []
    keys = Counter(r["core_offered"] for r in rejections)
    out = []
    for core, cnt in keys.most_common(n):
        sample = next(r for r in rejections if r["core_offered"] == core)
        out.append({"core_offered": core, "count": cnt, "example": sample})
    return out


def aggregate_diagnostics(traces: Dict) -> Dict:
    """Full Phase 2C v2 report from traced rollouts."""
    current = _aggregate_view(traces, "current_target")
    hindsight = _aggregate_view(traces, "final_target_hindsight")

    winners = [p for p in traces["player_finals"] if p["placement"] == 1]
    final_cov = [p["target"]["coverage"] for p in winners if p.get("target")]

    recommendation = recommend_intervention(current, hindsight)

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "invalidated_prior_results": INVALIDATED_PRIOR,
        "n_lobbies": traces["lobbies"],
        "n_archetypes": len(load_archetypes()),
        "n_events": len(traces["events"]),
        "sim_final_winner_coverage_mean": st.mean(final_cov) if final_cov else 0.0,
        "sim_final_winner_coverage_median": st.median(final_cov) if final_cov else 0.0,
        "winner_decision_funnel": {
            "current_target": current,
            "final_target_hindsight": hindsight,
        },
        "recommended_phase_2d_intervention": recommendation,
    }


def recommend_intervention(current_view: Dict, hindsight_view: Dict) -> Dict:
    """Pick exactly one Phase 2D intervention from corrected failure modes."""
    # Primary evidence: current-target winner funnel (not hindsight).
    funnel = current_view.get("aggregate_funnel") or {}
    cls = current_view.get("classification_totals") or {}
    h_funnel = hindsight_view.get("aggregate_funnel") or {}

    legally_buyable = funnel.get("legally_buyable_exposures", 0)
    purchased = funnel.get("purchased", 0)
    rejected = funnel.get("rejected_at_shop_exit", 0)

    a_count = cls.get("A_IMPOSSIBLE", 0)
    b_count = cls.get("B_AVAILABLE_NOT_BOUGHT", 0)
    c_count = cls.get("C_BOUGHT_NOT_RETAINED", 0)
    d_count = cls.get("D_ASSEMBLED_NO_PAYOFF", 0)

    rationale_parts = [
        f"Corrected v2 funnel (current-target view): "
        f"{legally_buyable} legally-buyable exposures, "
        f"{purchased} purchased, {rejected} rejected at shop exit.",
        f"Classification (eligible lobby×archetype): "
        f"B={b_count}, A={a_count}, D={d_count}, C={c_count}.",
        f"Prior 2,516-case B_AVAILABLE_NOT_BOUGHT headline invalidated "
        f"(see invalidated_prior_results).",
    ]

    if legally_buyable == 0:
        winner = "shop_pool_fidelity"
        phase = "Phase 2D = shop/pool fidelity"
        rationale_parts.append(
            "No legally-buyable relevant core exposures in current-target view — "
            "shop may be starving compositions.")
    elif rejected >= purchased and rejected > legally_buyable * 0.3:
        winner = "build_aware_recruit_policy"
        phase = "Phase 2D = build-aware recruit policy / evaluator"
        rationale_parts.append(
            "Most legally-buyable relevant cores are rejected at shop exit without "
            "purchase — build-aware recruiting is the strongest measured gap.")
    elif a_count >= b_count and a_count >= d_count:
        winner = "shop_pool_fidelity"
        phase = "Phase 2D = shop/pool fidelity"
        rationale_parts.append(
            "Relevant cores rarely become legally buyable for the winner.")
    elif d_count >= b_count:
        winner = "card_effect_fidelity"
        phase = "Phase 2D = targeted real card-effect implementation"
        rationale_parts.append(
            "Assembly-without-payoff dominates after proper scoping.")
    elif c_count > max(a_count, b_count, d_count):
        winner = "build_aware_recruit_policy"
        phase = "Phase 2D = build-aware recruit policy (retention)"
        rationale_parts.append("Core pieces bought but not retained on board.")
    else:
        winner = "card_effect_fidelity"
        phase = "Phase 2D = targeted card-effect implementation"
        rationale_parts.append("Mixed funnel after corrected scoping.")

    h_cov = h_funnel.get("mean_final_coverage_winner")
    if h_cov is not None:
        rationale_parts.append(
            f"Hindsight final-target mean coverage: {h_cov:.3f} (labeled hindsight only).")

    return {
        "intervention": winner,
        "phase_2d_title": phase,
        "rationale": " ".join(rationale_parts),
        "classification_totals_current_target": cls,
        "classification_totals_hindsight": (
            hindsight_view.get("classification_totals") or {}),
        "funnel_current_target": funnel,
        "funnel_hindsight": h_funnel,
    }
