"""Phase 2F post-purchase core lifecycle diagnosis (measurement-only).

For each fulfilled seeded exposure, trace the purchased core from buy through
disappearance or game end and assign a mutually exclusive lifecycle fate.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from hsbg_coach.bg_env import MAX_BOARD
from hsbg_coach.build_path import Archetype, load_archetypes

from ml.composition_diagnostic import (
    METHODOLOGY_VERSION as PHASE_2C_METHODOLOGY,
    RETAIN_TURNS,
    _WinnerFunnelState,
    _core_set,
    _current_target_keys,
    _is_relevant_at_offer,
    _legally_buyable_cores,
    _max_core_count,
    _names_on_board_hand,
    _target_meets_view_threshold,
    _winner_for_lobby,
    _archetype_eligible,
    _lobby_tribes,
)

METHODOLOGY_VERSION = "2f_v1"
PRIMARY_VIEW = "seeded_current_target"

FATE_LABELS = {
    "A_BOUGHT_STUCK_IN_HAND": "Purchased but never played.",
    "B_PLAYED_THEN_SOLD_SAME_TURN": "Reaches board, then removed before recruit ends (same turn).",
    "C_PLAYED_THEN_SOLD_LATER": "Survives recruit turn, then gets replaced later.",
    "D_TARGET_SWITCH": "Core survives, but infer_target changes before second-piece assembly.",
    "E_SEED_PIECE_LOST": "New core survives, but the original seeded core disappears.",
    "F_TRANSFORMED_TRIPLED": "Card identity changes through triple/discover mechanics.",
    "G_TWO_CORE_TRANSIENT": "Two cores coexist during an action-level state but not at end-of-recruit.",
    "H_TWO_CORE_PERSISTENT": "Two+ cores coexist through end-of-recruit.",
}


@dataclass
class FulfilledPurchase:
    purchase_id: str
    event_index: int
    lobby: int
    seat: int
    seed: int
    turn: int
    shop_generation: int
    archetype_key: str
    purchased_core: str
    seed_cores_at_buy: List[str]
    board_full_at_buy: bool
    hand_size_after_buy: int
    target_at_buy: Optional[Dict]
    card_at_buy: Dict


@dataclass
class _ActiveInstance:
    purchase: FulfilledPurchase
    zone: str  # "hand" | "board"
    played_turn: Optional[int] = None
    sold_turn: Optional[int] = None
    sold_same_turn: bool = False
    tripled: bool = False


def _seed_cores_present(board: List[Dict], hand: List[Dict], core: Set[str],
                        exclude: Optional[str] = None) -> List[str]:
    names = sorted(
        n for n in _names_on_board_hand(board, hand)
        if n in core and n != exclude)
    return names


def _target_key(target: Optional[Dict]) -> Optional[str]:
    return (target or {}).get("archetype_key")


def _count_name(board: List[Dict], hand: List[Dict], name: str,
                golden: Optional[bool] = None) -> int:
    cards = list(board or []) + list(hand or [])
    n = 0
    for c in cards:
        if c.get("name") != name:
            continue
        if golden is None or bool(c.get("golden")) == golden:
            n += 1
    return n


def _collect_purchases_for_state(traces: Dict, lobby: int, arch: Archetype,
                                 view: str, winner_seat: int,
                                 events: List[Tuple[int, Dict]]) -> List[FulfilledPurchase]:
    core = _core_set(arch)
    state = _WinnerFunnelState(core=core, lobby_tribes=_lobby_tribes(traces, lobby))
    purchases: List[FulfilledPurchase] = []
    prev_gen_key: Optional[Tuple[int, int]] = None

    for event_index, ev in events:
        turn = ev["turn"]
        shop_gen = ev.get("shop_generation", 0)
        gen_key = (turn, shop_gen)
        target_before = ev.get("target_before")
        tavern_tier = ev.get("tavern_tier")

        if prev_gen_key is not None and gen_key != prev_gen_key:
            state.close_generation()
        prev_gen_key = gen_key

        pre_shop = ev.get("pre_shop") or []
        legal_slots = ev.get("legal_buy_slots") or []
        buyable = _legally_buyable_cores(pre_shop, legal_slots, core)
        winner = _winner_for_lobby(traces, lobby)
        final_target_key = (winner.get("target") or {}).get("archetype_key") if winner else None

        if buyable and _is_relevant_at_offer(
                arch, view, target_before, final_target_key, tavern_tier):
            state.open_generation(
                turn, shop_gen,
                (target_before or {}).get("archetype_key"),
                buyable)

        if ev["action"] == "buy" and ev.get("card"):
            name = ev["card"]["name"]
            before_fulfilled = state.fulfilled_exposures
            state.note_core_purchase(name, turn)
            if (state.fulfilled_exposures > before_fulfilled
                    and name in core):
                board_before = ev.get("board_before") or []
                hand_before = ev.get("hand_before") or []
                purchases.append(FulfilledPurchase(
                    purchase_id=f"{lobby}:{winner_seat}:{event_index}",
                    event_index=event_index,
                    lobby=lobby,
                    seat=winner_seat,
                    seed=ev["seed"],
                    turn=turn,
                    shop_generation=shop_gen,
                    archetype_key=arch.key,
                    purchased_core=name,
                    seed_cores_at_buy=_seed_cores_present(
                        board_before, hand_before, core, exclude=name),
                    board_full_at_buy=len(board_before) >= MAX_BOARD,
                    hand_size_after_buy=len(ev.get("hand_after") or []),
                    target_at_buy=target_before,
                    card_at_buy=dict(ev["card"]),
                ))
        elif ev["action"] in ("roll", "end"):
            state.close_generation()
            if ev["action"] == "end":
                prev_gen_key = None

    state.flush()
    return purchases


def collect_fulfilled_seeded_purchases(traces: Dict,
                                       view: str = PRIMARY_VIEW) -> List[FulfilledPurchase]:
    """All fulfilled seeded exposures on winner seats (Phase 2C latch replay)."""
    out: List[FulfilledPurchase] = []
    archetypes = load_archetypes()
    lobbies = traces["lobbies"]

    seat_events: Dict[Tuple[int, int], List[Tuple[int, Dict]]] = {}
    for idx, ev in enumerate(traces["events"]):
        key = (ev["lobby"], ev["seat"])
        seat_events.setdefault(key, []).append((idx, ev))

    for arch in archetypes:
        for lobby in range(lobbies):
            if not _archetype_eligible(arch, _lobby_tribes(traces, lobby)):
                continue
            winner = _winner_for_lobby(traces, lobby)
            if winner is None:
                continue
            winner_seat = winner["seat"]
            if view != "final_target_hindsight":
                if arch.key not in _current_target_keys(
                        traces, lobby, winner_seat, view):
                    continue
            events = seat_events.get((lobby, winner_seat), [])
            out.extend(_collect_purchases_for_state(
                traces, lobby, arch, view, winner_seat, events))
    return out


def _turn_end_boards(traces: Dict, lobby: int, seat: int) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for ts in traces["turn_summaries"]:
        if ts["lobby"] == lobby and ts["seat"] == seat:
            out[ts["turn"]] = ts.get("board_after_recruit") or []
    return out


def _trace_lifecycle(purchase: FulfilledPurchase,
                     seat_events: List[Tuple[int, Dict]],
                     turn_end_boards: Dict[int, List[Dict]],
                     core: Set[str]) -> Dict:
    """Walk events after purchase; return trajectory + fate."""
    inst = _ActiveInstance(purchase=purchase, zone="hand")
    instances: Deque[_ActiveInstance] = deque([inst])
    by_name: Dict[str, Deque[_ActiveInstance]] = {purchase.purchased_core: instances}

    played_turn: Optional[int] = None
    sell_turn: Optional[int] = None
    sell_action: Optional[str] = None
    max_core_action = 0
    max_core_recruit_end = 0
    had_two_core_action = False
    had_two_core_recruit_end = False
    target_at_buy = _target_key(purchase.target_at_buy)
    target_after_purchase: Optional[str] = None
    target_next_turn: Optional[str] = None
    target_switched = False
    seed_cores_at_buy = set(purchase.seed_cores_at_buy)
    seed_piece_lost_turn: Optional[int] = None
    both_cores_coexist_turn: Optional[int] = None
    survived_recruit_end = False
    survived_1_turn = False
    survived_2_turns = False
    core_have_trajectory: List[Dict] = []

    post_events = [(i, ev) for i, ev in seat_events if i > purchase.event_index]

    for event_index, ev in post_events:
        turn = ev["turn"]
        target_key = _target_key(ev.get("target_before"))
        if target_after_purchase is None:
            target_after_purchase = target_key
        if target_next_turn is None and turn > purchase.turn:
            target_next_turn = target_key

        if (target_at_buy and target_key and target_key != target_at_buy
                and target_key != purchase.archetype_key):
            target_switched = True

        board_after = ev.get("board_after") or []
        hand_after = ev.get("hand_after") or []
        core_count = _max_core_count(board_after, core)
        max_core_action = max(max_core_action, core_count)
        if core_count >= 2:
            had_two_core_action = True
            if both_cores_coexist_turn is None:
                both_cores_coexist_turn = turn

        present = _names_on_board_hand(board_after, hand_after)
        if seed_cores_at_buy and not seed_cores_at_buy.intersection(present):
            if seed_piece_lost_turn is None and purchase.purchased_core in present:
                seed_piece_lost_turn = turn

        if ev["action"] == "play" and ev.get("card"):
            name = ev["card"]["name"]
            q = by_name.get(name)
            if q:
                for active in list(q):
                    if active.zone == "hand" and active.sold_turn is None:
                        active.zone = "board"
                        active.played_turn = turn
                        if active.purchase.purchase_id == purchase.purchase_id:
                            played_turn = turn
                        break

        elif ev["action"] == "sell" and ev.get("card"):
            name = ev["card"]["name"]
            q = by_name.get(name)
            if q:
                for active in list(q):
                    if active.zone == "board" and active.sold_turn is None:
                        active.sold_turn = turn
                        active.sold_same_turn = (active.played_turn == turn)
                        if active.purchase.purchase_id == purchase.purchase_id:
                            sell_turn = turn
                            sell_action = f"sell:{name}:turn{turn}"
                        break

        # Triple: three non-golden copies merge; mark FIFO instances tripled.
        if ev["action"] in ("buy", "play"):
            name = (ev.get("card") or {}).get("name")
            if name:
                non_golden = _count_name(board_after, hand_after, name, golden=False)
                golden = _count_name(board_after, hand_after, name, golden=True)
                if golden > 0 and non_golden <= 1:
                    q = by_name.get(name)
                    if q:
                        for active in list(q):
                            if not active.tripled and active.sold_turn is None:
                                active.tripled = True
                                if active.purchase.purchase_id == purchase.purchase_id:
                                    pass  # fate F set below

        if ev["action"] == "end":
            end_board = turn_end_boards.get(turn, board_after)
            end_count = _max_core_count(end_board, core)
            max_core_recruit_end = max(max_core_recruit_end, end_count)
            if end_count >= 2:
                had_two_core_recruit_end = True
            tgt = ev.get("target_before") or {}
            core_have_trajectory.append({
                "turn": turn,
                "core_have": tgt.get("core_have"),
                "target_key": _target_key(tgt),
                "core_on_board": end_count,
            })
            present_end = {c.get("name") for c in end_board}
            if purchase.purchased_core in present_end:
                survived_recruit_end = True
                if turn - purchase.turn >= 1:
                    survived_1_turn = True
                if turn - purchase.turn >= 2:
                    survived_2_turns = True

    inst_state = instances[0]
    fate = _classify_fate(
        played_turn=played_turn,
        sell_turn=sell_turn,
        sold_same_turn=inst_state.sold_same_turn,
        tripled=inst_state.tripled,
        had_two_core_action=had_two_core_action,
        had_two_core_recruit_end=had_two_core_recruit_end,
        target_switched=target_switched,
        seed_piece_lost=seed_piece_lost_turn is not None,
        seed_cores_at_buy=bool(seed_cores_at_buy),
    )

    return {
        "purchase_id": purchase.purchase_id,
        "lobby": purchase.lobby,
        "seat": purchase.seat,
        "seed": purchase.seed,
        "turn": purchase.turn,
        "shop_generation": purchase.shop_generation,
        "archetype_key": purchase.archetype_key,
        "purchased_core": purchase.purchased_core,
        "seed_cores_at_buy": list(purchase.seed_cores_at_buy),
        "board_full_at_buy": purchase.board_full_at_buy,
        "hand_size_after_buy": purchase.hand_size_after_buy,
        "played_turn": played_turn,
        "both_cores_coexist_turn": both_cores_coexist_turn,
        "survived_recruit_end": survived_recruit_end,
        "survived_1_turn": survived_1_turn,
        "survived_2_turns": survived_2_turns,
        "sell_turn": sell_turn,
        "sell_action": sell_action,
        "target_at_buy": target_at_buy,
        "target_after_purchase": target_after_purchase,
        "target_next_turn": target_next_turn,
        "target_switched": target_switched,
        "seed_piece_lost_turn": seed_piece_lost_turn,
        "max_core_action": max_core_action,
        "max_core_recruit_end": max_core_recruit_end,
        "core_have_trajectory": core_have_trajectory,
        "fate": fate,
        "fate_label": FATE_LABELS[fate],
    }


def _classify_fate(*, played_turn: Optional[int], sell_turn: Optional[int],
                   sold_same_turn: bool, tripled: bool,
                   had_two_core_action: bool, had_two_core_recruit_end: bool,
                   target_switched: bool, seed_piece_lost: bool,
                   seed_cores_at_buy: bool) -> str:
    """Mutually exclusive fate with fixed priority."""
    if had_two_core_recruit_end:
        return "H_TWO_CORE_PERSISTENT"
    if had_two_core_action and not had_two_core_recruit_end:
        return "G_TWO_CORE_TRANSIENT"
    if tripled:
        return "F_TRANSFORMED_TRIPLED"
    if played_turn is not None and sell_turn is not None and sold_same_turn:
        return "B_PLAYED_THEN_SOLD_SAME_TURN"
    if played_turn is not None and sell_turn is not None:
        return "C_PLAYED_THEN_SOLD_LATER"
    if seed_cores_at_buy and seed_piece_lost:
        return "E_SEED_PIECE_LOST"
    if played_turn is None:
        return "A_BOUGHT_STUCK_IN_HAND"
    if target_switched:
        return "D_TARGET_SWITCH"
    # Played, not sold, never assembled 2+ cores at recruit end — partial retention.
    return "C_PLAYED_THEN_SOLD_LATER"


def analyze_core_lifecycles(traces: Dict,
                            view: str = PRIMARY_VIEW) -> Dict:
    purchases = collect_fulfilled_seeded_purchases(traces, view=view)
    seat_events: Dict[Tuple[int, int], List[Tuple[int, Dict]]] = {}
    for idx, ev in enumerate(traces["events"]):
        key = (ev["lobby"], ev["seat"])
        seat_events.setdefault(key, []).append((idx, ev))

    arch_core: Dict[str, Set[str]] = {
        a.key: _core_set(a) for a in load_archetypes()}

    records = []
    for p in purchases:
        core = arch_core[p.archetype_key]
        turn_ends = _turn_end_boards(traces, p.lobby, p.seat)
        rec = _trace_lifecycle(
            p, seat_events.get((p.lobby, p.seat), []), turn_ends, core)
        records.append(rec)

    fate_totals = Counter(r["fate"] for r in records)
    n = len(records)
    played = sum(1 for r in records if r["played_turn"] is not None)
    coexist = sum(1 for r in records if r["both_cores_coexist_turn"] is not None)
    end_survive = sum(1 for r in records if r["survived_recruit_end"])
    one_turn = sum(1 for r in records if r["survived_1_turn"])
    two_turn = sum(1 for r in records if r["survived_2_turns"])

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "phase_2c_methodology": PHASE_2C_METHODOLOGY,
        "view": view,
        "n_fulfilled_purchases": n,
        "fate_totals": dict(fate_totals),
        "fate_labels": FATE_LABELS,
        "funnel": {
            "fulfilled_purchases": n,
            "played": played,
            "coexist_with_seed_core": coexist,
            "survive_end_of_recruit": end_survive,
            "survive_1_turn": one_turn,
            "survive_2_turns": two_turn,
        },
        "purchases": records,
    }


def lifecycle_meets_fulfillment_count(traces: Dict, view: str = PRIMARY_VIEW) -> bool:
    """Lifecycle cohort size matches Phase 2C fulfilled_exposures (sanity check)."""
    from ml.composition_diagnostic import aggregate_diagnostics
    lifecycle_n = len(collect_fulfilled_seeded_purchases(traces, view=view))
    diag = aggregate_diagnostics(traces)
    funnel = (diag["winner_decision_funnel"][view].get("aggregate_funnel") or {})
    return lifecycle_n == funnel.get("fulfilled_exposures", -1)
