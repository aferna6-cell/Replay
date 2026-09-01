"""Phase 2C composition assembly funnel and failure classification."""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from hsbg_coach.build_path import Archetype, load_archetypes

BUY_COST = 3
RETAIN_TURNS = 2


def _core_set(arch: Archetype) -> Set[str]:
    return set(arch.core.keys())


def _names_on_board_hand(board: List[Dict], hand: Optional[List[Dict]] = None) -> Set[str]:
    names = {c["name"] for c in board if c.get("name")}
    if hand:
        names |= {c["name"] for c in hand if c.get("name")}
    return names


def _max_core_count(board: List[Dict], core: Set[str]) -> int:
    return sum(1 for c in board if c.get("name") in core)


class _LobbyArchetypeState:
    def __init__(self, core: Set[str]):
        self.core = core
        self.shop_appearances = 0
        self.affordable_offers = 0
        self.lobbies_with_2plus_offered = False
        self.lobbies_with_4plus_obtainable = False
        self.offers_by_tier: Counter = Counter()
        self.purchases = 0
        self.plays = 0
        self.offer_events = 0
        self.buy_other_events = 0
        self.retained_2_turns = 0
        self.max_core_reached = 0
        self.turn_first_2: Optional[int] = None
        self.turn_first_4: Optional[int] = None
        self.final_core = 0
        self.final_coverage = 0.0
        self.opportunity_losses: List[Dict] = []
        self._purchased_turn: Dict[str, int] = {}
        self._core_offered_this_turn: Set[str] = set()
        self._current_turn: Optional[int] = None

    def flush_turn(self) -> None:
        if len(self._core_offered_this_turn) >= 2:
            self.lobbies_with_2plus_offered = True
        if len(self._core_offered_this_turn) >= 4:
            self.lobbies_with_4plus_obtainable = True
        self._core_offered_this_turn.clear()
        self._current_turn = None


def _track_retention(state: _LobbyArchetypeState, turn: int, board: List[Dict],
                     hand: Optional[List[Dict]] = None) -> None:
    present = _names_on_board_hand(board, hand)
    for name, bought_turn in list(state._purchased_turn.items()):
        if name in present and turn - bought_turn >= RETAIN_TURNS:
            state.retained_2_turns += 1
            del state._purchased_turn[name]


def analyze_archetype_lobby(arch: Archetype, traces: Dict, lobby: int) -> Tuple[str, _LobbyArchetypeState]:
    """Build funnel state for one archetype in one lobby; return classification."""
    core = _core_set(arch)
    state = _LobbyArchetypeState(core)

    winner = next(
        (p for p in traces["player_finals"]
         if p["lobby"] == lobby and p["placement"] == 1), None)
    winner_seat = winner["seat"] if winner else None
    winner_target_key = (winner.get("target") or {}).get("archetype_key") if winner else None

    for ev in traces["events"]:
        if ev["lobby"] != lobby:
            continue
        if state._current_turn != ev["turn"]:
            state.flush_turn()
            state._current_turn = ev["turn"]

        shop = ev.get("shop_offered") or []
        for card in shop:
            name = card.get("name")
            if name in core:
                state.shop_appearances += 1
                state.offers_by_tier[card.get("tier", 0)] += 1
                state._core_offered_this_turn.add(name)
                if ev["gold_before"] >= BUY_COST:
                    state.affordable_offers += 1
                    state.offer_events += 1

        if winner_seat is not None and ev["seat"] != winner_seat:
            continue

        if ev["action"] == "buy" and ev.get("card"):
            bought_name = ev["card"]["name"]
            if bought_name in core:
                state.purchases += 1
                state._purchased_turn[bought_name] = ev["turn"]
            elif state.offer_events > 0 and any(c.get("name") in core for c in shop):
                alt = ev["card"]
                core_in_shop = [c for c in shop if c.get("name") in core
                                and ev["gold_before"] >= BUY_COST]
                if core_in_shop:
                    state.buy_other_events += 1
                    core_c = core_in_shop[0]
                    state.opportunity_losses.append({
                        "turn": ev["turn"],
                        "seat": ev["seat"],
                        "core_offered": core_c["name"],
                        "core_stats": (core_c.get("attack", 0) + core_c.get("health", 0)),
                        "bought_instead": bought_name,
                        "bought_stats": (alt.get("attack", 0) + alt.get("health", 0)),
                        "bought_tier": alt.get("tier"),
                        "core_tier": core_c.get("tier"),
                    })
        elif ev["action"] == "play" and ev.get("card"):
            if ev["card"]["name"] in core:
                state.plays += 1

        _track_retention(state, ev["turn"], ev.get("board_after") or [],
                         ev.get("hand_after"))

    state.flush_turn()

    for ts in traces["turn_summaries"]:
        if ts["lobby"] != lobby:
            continue
        if ts["seat"] != winner_seat and winner_seat is not None:
            continue
        board = ts.get("board_after_recruit") or []
        count = _max_core_count(board, core)
        state.max_core_reached = max(state.max_core_reached, count)
        if count >= 2 and state.turn_first_2 is None:
            state.turn_first_2 = ts["turn"]
        if count >= 4 and state.turn_first_4 is None:
            state.turn_first_4 = ts["turn"]

    if winner:
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

    classification = _classify(state, winner_target_key == arch.key)
    return classification, state


def _classify(state: _LobbyArchetypeState, winner_is_archetype: bool) -> str:
    if winner_is_archetype and state.final_coverage >= 0.4 and state.final_core >= 4:
        return "E_SUCCESSFULLY_ASSEMBLED"
    if state.max_core_reached >= 2 and state.final_coverage < 0.15:
        return "D_ASSEMBLED_NO_PAYOFF"
    if state.purchases > 0 and state.max_core_reached < 2:
        return "C_BOUGHT_NOT_RETAINED"
    if state.purchases == 0 and state.affordable_offers > 0:
        return "B_AVAILABLE_NOT_BOUGHT"
    if state.lobbies_with_2plus_offered or state.shop_appearances >= 2:
        if state.purchases == 0:
            return "B_AVAILABLE_NOT_BOUGHT"
    if state.shop_appearances == 0 or not state.lobbies_with_2plus_offered:
        return "A_IMPOSSIBLE"
    if state.max_core_reached >= 4:
        return "D_ASSEMBLED_NO_PAYOFF"
    if state.max_core_reached >= 2:
        return "D_ASSEMBLED_NO_PAYOFF"
    return "A_IMPOSSIBLE"


def aggregate_diagnostics(traces: Dict) -> Dict:
    """Full Phase 2C report from traced rollouts."""
    archetypes = load_archetypes()
    lobbies = traces["lobbies"]
    by_arch: Dict[str, Dict] = {}

    for arch in archetypes:
        classifications: Counter = Counter()
        states: List[_LobbyArchetypeState] = []
        for lobby in range(lobbies):
            cls, st_state = analyze_archetype_lobby(arch, traces, lobby)
            classifications[cls] += 1
            states.append(st_state)

        n = len(states)
        lobbies_2_offered = sum(1 for s in states if s.lobbies_with_2plus_offered)
        lobbies_4_obtain = sum(1 for s in states if s.lobbies_with_4plus_obtainable)
        opp = []
        for s in states:
            opp.extend(s.opportunity_losses)

        purchase_rate = (
            sum(s.purchases for s in states) / max(1, sum(s.offer_events for s in states)))
        play_rate = (
            sum(s.plays for s in states) / max(1, sum(s.purchases for s in states)))
        retain_rate = (
            sum(s.retained_2_turns for s in states) / max(1, sum(s.purchases for s in states)))

        by_arch[arch.key] = {
            "name": arch.name,
            "tribe": arch.tribe,
            "board_count": arch.board_count,
            "core_cards": list(arch.core.keys()),
            "n_lobbies": n,
            "classification": dict(classifications),
            "availability": {
                "mean_core_shop_appearances_per_lobby": (
                    sum(s.shop_appearances for s in states) / n if n else 0),
                "pct_lobbies_2plus_core_offered": lobbies_2_offered / n if n else 0,
                "pct_lobbies_4plus_core_obtainable": lobbies_4_obtain / n if n else 0,
                "core_offer_rate_by_tier": dict(
                    sum((s.offers_by_tier for s in states), Counter())),
            },
            "conversion": {
                "purchase_rate_when_offered_affordable": purchase_rate,
                "play_rate_after_purchase": play_rate,
                "retention_rate_2plus_turns": retain_rate,
            },
            "assembly": {
                "mean_max_core_pieces": st.mean([s.max_core_reached for s in states]) if states else 0,
                "mean_final_core_pieces_winner": st.mean([s.final_core for s in states]) if states else 0,
                "mean_final_coverage_winner": st.mean([s.final_coverage for s in states]) if states else 0,
                "median_turn_first_2_core": _median_int([s.turn_first_2 for s in states]),
                "median_turn_first_4_core": _median_int([s.turn_first_4 for s in states]),
            },
            "opportunity_loss_top": _top_opportunity_losses(opp),
            "funnel": {
                "core_in_shop": sum(s.shop_appearances for s in states),
                "affordable": sum(s.affordable_offers for s in states),
                "purchased": sum(s.purchases for s in states),
                "played": sum(s.plays for s in states),
                "retained_2_turns": sum(s.retained_2_turns for s in states),
                "reached_2_core": sum(1 for s in states if s.max_core_reached >= 2),
                "reached_4_core": sum(1 for s in states if s.max_core_reached >= 4),
                "final_coverage_mean": st.mean([s.final_coverage for s in states]) if states else 0,
            },
        }

    winners = [p for p in traces["player_finals"] if p["placement"] == 1]
    final_cov = [p["target"]["coverage"] for p in winners if p.get("target")]
    recommendation = recommend_intervention(by_arch, classifications_total(by_arch))

    return {
        "n_lobbies": lobbies,
        "n_archetypes": len(by_arch),
        "n_events": len(traces["events"]),
        "sim_final_winner_coverage_mean": st.mean(final_cov) if final_cov else 0.0,
        "sim_final_winner_coverage_median": st.median(final_cov) if final_cov else 0.0,
        "by_archetype": by_arch,
        "recommended_phase_2d_intervention": recommendation,
    }


def _median_int(vals: List[Optional[int]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return st.median(xs) if xs else None


def _top_opportunity_losses(losses: List[Dict], n: int = 5) -> List[Dict]:
    if not losses:
        return []
    keys = Counter(
        (l["core_offered"], l.get("bought_instead")) for l in losses)
    out = []
    for (core, bought), cnt in keys.most_common(n):
        sample = next(l for l in losses
                      if l["core_offered"] == core and l.get("bought_instead") == bought)
        out.append({
            "core_offered": core,
            "bought_instead": bought,
            "count": cnt,
            "mean_stat_delta": st.mean(
                l.get("bought_stats", 0) - l.get("core_stats", 0)
                for l in losses
                if l["core_offered"] == core and l.get("bought_instead") == bought),
            "example": sample,
        })
    return out


def classifications_total(by_arch: Dict[str, Dict]) -> Counter:
    total: Counter = Counter()
    for data in by_arch.values():
        total.update(data.get("classification", {}))
    return total


def recommend_intervention(by_arch: Dict[str, Dict], totals: Counter) -> Dict:
    """Pick exactly one Phase 2D intervention from measured failure modes."""
    scores = {
        "shop_pool_fidelity": 0.0,
        "build_aware_policy": 0.0,
        "card_effect_fidelity": 0.0,
        "triple_discover_fidelity": 0.0,
    }
    reasons = []

    for key, data in by_arch.items():
        cls = data.get("classification", {})
        n = data.get("n_lobbies", 1)
        avail = data.get("availability", {})
        conv = data.get("conversion", {})
        asm = data.get("assembly", {})

        scores["shop_pool_fidelity"] += cls.get("A_IMPOSSIBLE", 0) / n
        scores["build_aware_policy"] += cls.get("B_AVAILABLE_NOT_BOUGHT", 0) / n
        scores["card_effect_fidelity"] += cls.get("D_ASSEMBLED_NO_PAYOFF", 0) / n
        if avail.get("pct_lobbies_4plus_core_obtainable", 0) < 0.2:
            scores["shop_pool_fidelity"] += 0.5

        opp = data.get("opportunity_loss_top") or []
        if opp and opp[0].get("mean_stat_delta", 0) > 0:
            scores["build_aware_policy"] += 0.3

    a_count = totals.get("A_IMPOSSIBLE", 0)
    b_count = totals.get("B_AVAILABLE_NOT_BOUGHT", 0)
    d_count = totals.get("D_ASSEMBLED_NO_PAYOFF", 0)
    c_count = totals.get("C_BOUGHT_NOT_RETAINED", 0)

    if b_count >= a_count and b_count >= d_count:
        winner = "build_aware_recruit_policy"
        phase = "Phase 2D = build-aware recruit policy / evaluator"
        rationale = (
            f"Available-but-not-bought dominates ({b_count} lobby-archetype cases vs "
            f"impossible={a_count}, no-payoff={d_count}). Greedy often prefers slightly "
            f"larger off-comp bodies when core pieces are offered.")
    elif a_count >= b_count and a_count >= d_count:
        winner = "shop_pool_fidelity"
        phase = "Phase 2D = shop/pool fidelity"
        rationale = (
            f"Availability failure dominates ({a_count} lobby-archetype cases). "
            f"Required core pieces are rarely offered often enough to assemble comps.")
    elif d_count >= b_count:
        winner = "card_effect_fidelity"
        phase = "Phase 2D = targeted real card-effect implementation"
        rationale = (
            f"Assembly-without-payoff dominates ({d_count} cases). Core pieces appear "
            f"on boards but infer_target coverage stays near zero — missing mechanics "
            f"likely make synergy pieces weak.")
    elif c_count > max(a_count, b_count, d_count):
        winner = "build_aware_recruit_policy"
        phase = "Phase 2D = build-aware recruit policy (retention)"
        rationale = (
            f"Bought-but-not-retained dominates ({c_count} cases). Policy buys core "
            f"pieces then sells/replaces them.")
    else:
        winner = "card_effect_fidelity"
        phase = "Phase 2D = targeted card-effect implementation"
        rationale = "Mixed funnel; largest measured gap is assembly without payoff."

    return {
        "intervention": winner,
        "phase_2d_title": phase,
        "rationale": rationale,
        "failure_mode_scores": scores,
        "classification_totals": dict(totals),
    }
