"""Phase 2M shop/pool rules audit (measurement-only).

Research question: is post-assembly scarcity caused by incorrect simulator
catalogue/pool rules, incorrect live-pool accounting, or expected scarcity
under a correctly implemented finite shared pool?

Does **not** change pool/shop behavior — only observational hooks + analysis.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from hsbg_coach import cards as cards_mod
from hsbg_coach.bg_env import (
    BGEnv,
    POOL_COPIES,
    SHOP_SLOTS,
    TRIBES,
    build_pool,
)
from hsbg_coach.board_opportunity_policy import policies_for_lobby
from hsbg_coach.build_path import Archetype, load_archetypes
from hsbg_coach.persistence_prior import PersistencePrior
from ml.availability_decomposition import (
    FROZEN_ALPHA,
    analyze_availability_decomposition,
    catalogue_exclusion_reason,
)
from ml.composition_trace import RecruitTracer

METHODOLOGY_VERSION = "2m_v1"

# Reuse Phase 2L diagnostic DEV for continuity (measurement only).
PHASE_2L_SEED = 10200
PHASE_2L_LOBBIES = 500
PHASE_2M_SEED = PHASE_2L_SEED
PHASE_2M_LOBBIES = PHASE_2L_LOBBIES

# Reserved for Phase 2N+ — do NOT consume in 2M.
RESERVED_INTERVENTION_SEED = 11000
RESERVED_INTERVENTION_LOBBIES = 500    # 11000–11499
RESERVED_CONFIRM_SEED = 11500
RESERVED_CONFIRM_LOBBIES = 200         # 11500–11699

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (11000, 11499),
    (11500, 11699),
)

# Current Battlegrounds reference — document only; do not patch simulator here.
REF_POOL_COPIES = {1: 15, 2: 15, 3: 13, 4: 11, 5: 9, 6: 7}
REF_SHOP_SLOTS_MINION_CLASSIC = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6}
REF_SHOP_SLOTS_WITH_SPELLS = {1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: 7}


def p_zero_live_deal(card_remaining: int, eligible_total: int,
                     n_slots: int) -> float:
    """Exact P(card absent from a without-replacement deal of ``n_slots``)."""
    if n_slots <= 0:
        return 1.0
    if card_remaining <= 0 or eligible_total <= 0:
        return 1.0
    other = eligible_total - card_remaining
    if other < 0:
        return 1.0
    draws = min(n_slots, eligible_total)
    if other < draws:
        return 0.0
    p = 1.0
    rem_other = float(other)
    rem_total = float(eligible_total)
    for _ in range(draws):
        p *= rem_other / rem_total
        rem_other -= 1.0
        rem_total -= 1.0
    return float(p)


def expected_raw_live_deal(card_remaining: int, eligible_total: int,
                           n_slots: int) -> float:
    """E[appearances] in one without-replacement deal."""
    if card_remaining <= 0 or eligible_total <= 0 or n_slots <= 0:
        return 0.0
    draws = min(n_slots, eligible_total)
    return draws * (card_remaining / float(eligible_total))


def all_archetype_core_names(
        archetypes: Optional[List[Archetype]] = None) -> Set[str]:
    archetypes = archetypes or load_archetypes()
    out: Set[str] = set()
    for arch in archetypes:
        out.update(arch.core.keys())
    return out


def audit_catalogue_synchronization(
        archetypes: Optional[List[Archetype]] = None) -> Dict:
    """Every archetype core → KB / tier / stats / full build_pool membership."""
    archetypes = archetypes or load_archetypes()
    kb = cards_mod.load_kb()
    by_name: Dict[str, object] = {}
    for ck in kb.values():
        if ck.name and ck.name not in by_name:
            by_name[ck.name] = ck
    full_cat = {m.name for m in build_pool(lobby_tribes=list(TRIBES))}

    rows = []
    counts = Counter()
    for arch in archetypes:
        for name, freq in arch.core.items():
            ck = by_name.get(name)
            if ck is None:
                reason = "MISSING_FROM_KB"
            elif ck.tier is None or not (1 <= int(ck.tier) <= 6):
                reason = "MISSING_OR_INVALID_TIER"
            elif ck.attack is None or ck.health is None:
                reason = "MISSING_STATS"
            elif name not in full_cat:
                reason = catalogue_exclusion_reason(
                    name, list(TRIBES), full_cat) or "BUILD_POOL_EXCLUDED"
            else:
                reason = "IN_EXACT_CATALOGUE"
            counts[reason] += 1
            rows.append({
                "archetype_key": arch.key,
                "archetype_name": arch.name,
                "tribe": arch.tribe,
                "card": name,
                "core_frequency": freq,
                "status": reason,
                "kb_tier": None if ck is None else ck.tier,
                "kb_tribes": None if ck is None else list(ck.tribes or []),
            })

    n = len(rows)
    return {
        "n_core_slots": n,
        "n_unique_cores": len({r["card"] for r in rows}),
        "status_counts": dict(counts),
        "status_share": {k: v / n for k, v in counts.items()} if n else {},
        "n_missing_from_kb": int(counts.get("MISSING_FROM_KB", 0)),
        "n_in_exact_catalogue": int(counts.get("IN_EXACT_CATALOGUE", 0)),
        "missing_from_kb_names": sorted({
            r["card"] for r in rows if r["status"] == "MISSING_FROM_KB"}),
        "rows": rows,
    }


def audit_pool_contract() -> Dict:
    return {
        "simulator": {
            "POOL_COPIES": dict(POOL_COPIES),
            "SHOP_SLOTS": dict(SHOP_SLOTS),
            "draw": (
                "weighted by remaining live copies among catalogue minions "
                "with tier <= tavern and pool > 0; without replacement within deal"),
            "freeze": (
                "frozen shop skips return+redraw; does not top-up incomplete "
                "frozen shops after buys/tier-up"),
            "roll": "unfreeze; return shop copies; redraw SHOP_SLOTS[tier]",
            "buy": "shop→hand; copy stays out of pool",
            "sell": "board→pool (+1 or +3 if golden)",
            "triple": (
                "3 non-golden → 1 golden in hand; discover = extra _draw; "
                "base copies not returned (golden returns as 3 on sell)"),
            "elimination": (
                "DEAD PLAYERS DO NOT RETURN board/hand/shop copies to pool"),
        },
        "reference_current_bg": {
            "POOL_COPIES": dict(REF_POOL_COPIES),
            "SHOP_SLOTS_minion_classic": dict(REF_SHOP_SLOTS_MINION_CLASSIC),
            "SHOP_SLOTS_with_spells_patch34_2": dict(REF_SHOP_SLOTS_WITH_SPELLS),
            "shared_pool": (
                "minions at/below tavern tier from remaining shared pool"),
            "elimination_return": (
                "eliminated players' minions expected to return to shared pool"),
            "freeze_topup": (
                "incomplete freeze / tier-up tops up new slots"),
            "sources": [
                "https://hearthstone.wiki.gg/wiki/Battlegrounds",
                "Hearthstone patch 34.2 notes (tavern card counts with spells)",
            ],
        },
    }


def audit_rule_mismatches(contract: Optional[Dict] = None) -> Dict:
    """Enumerate demonstrated simulator↔rules mismatches (document only)."""
    contract = contract or audit_pool_contract()
    sim_copies = contract["simulator"]["POOL_COPIES"]
    ref_copies = contract["reference_current_bg"]["POOL_COPIES"]
    mismatches = []

    for tier in sorted(set(sim_copies) | set(ref_copies)):
        s, r = sim_copies.get(tier), ref_copies.get(tier)
        if s != r:
            mismatches.append({
                "id": f"pool_copies_tier_{tier}",
                "area": "per_tier_copy_counts",
                "simulator": s,
                "reference": r,
                "mismatch": True,
                "note": f"Tier {tier} copy count differs",
            })

    sim_slots = contract["simulator"]["SHOP_SLOTS"]
    ref_classic = contract["reference_current_bg"]["SHOP_SLOTS_minion_classic"]
    ref_spells = contract["reference_current_bg"]["SHOP_SLOTS_with_spells_patch34_2"]
    mismatches.append({
        "id": "shop_slots_vs_classic_minion",
        "area": "shop_slot_counts",
        "simulator": dict(sim_slots),
        "reference": dict(ref_classic),
        "mismatch": sim_slots != ref_classic,
        "note": (
            "Matches classic minion-only table. Modern taverns also offer "
            "spells with larger card counts — sim has no tavern spells."),
    })
    mismatches.append({
        "id": "shop_slots_vs_spell_era",
        "area": "shop_slot_counts",
        "simulator": dict(sim_slots),
        "reference": dict(ref_spells),
        "mismatch": sim_slots != ref_spells,
        "note": "Spell-era sizes differ; relevant only if 2N adds spells.",
    })
    mismatches.append({
        "id": "elimination_no_return_to_pool",
        "area": "pool_lifecycle_accounting",
        "simulator": "no return of board/hand/shop on death",
        "reference": "eliminated players' minions expected to return to pool",
        "mismatch": True,
        "note": "Concrete accounting divergence",
    })
    mismatches.append({
        "id": "freeze_no_topup",
        "area": "freeze_return_to_pool",
        "simulator": "frozen shop kept as-is; no top-up",
        "reference": "incomplete freeze / tier-up tops up new slots",
        "mismatch": True,
        "note": "Documented behavioral mismatch",
    })
    mismatches.append({
        "id": "no_tier_7",
        "area": "catalogue",
        "simulator": "MAX_TIER=6; no T7",
        "reference": "T7 exists in some modes (5 copies)",
        "mismatch": True,
        "note": "Out of scope unless lobby mode requires T7",
    })

    demonstrated = [m for m in mismatches if m["mismatch"]]
    return {
        "n_documented_checks": len(mismatches),
        "n_demonstrated_mismatches": len(demonstrated),
        "mismatches": mismatches,
        "demonstrated_ids": [m["id"] for m in demonstrated],
    }


class PoolDealTracer(RecruitTracer):
    """RecruitTracer + per-deal live pool snapshots (observational)."""

    def __init__(self, lobby_id: int, seed: int, track_names: Set[str]):
        super().__init__(lobby_id, seed)
        self.track_names = set(track_names)
        self.deal_events: List[Dict] = []

    def on_deal(self, env: BGEnv, player, meta: Dict) -> None:
        self.deal_events.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "seat": player.idx,
            "turn": env.turn,
            "reason": meta["reason"],
            "frozen_skip": meta["frozen_skip"],
            "tavern_tier": meta["tavern_tier"],
            "n_slots": meta["n_slots"],
            "dealt_names": list(meta["dealt_names"]),
            "eligible_total_copies": meta["eligible_total_copies"],
            "card_remaining": dict(meta["card_remaining"]),
            "alive_players": sum(1 for q in env.players if q.alive),
            "pool_total_copies": int(sum(env._pool.values())),
        })


def run_board_opp_with_pool_audit(
        lobbies: int, seed: int, prior: PersistencePrior,
        track_names: Optional[Set[str]] = None,
        alpha: float = FROZEN_ALPHA) -> Dict:
    """Frozen 2J rollouts with live pool deal snapshots for ``track_names``."""
    track_names = track_names or all_archetype_core_names()
    all_events: List[Dict] = []
    all_turn_summaries: List[Dict] = []
    all_player_finals: List[Dict] = []
    all_deals: List[Dict] = []
    lobby_meta: List[Dict] = []

    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        policies = policies_for_lobby(alpha, prior, 8)
        env = BGEnv(seed=lobby_seed, scaling_mode="residual")
        tracer = PoolDealTracer(
            lobby_id=lobby_i, seed=lobby_seed, track_names=track_names)
        env._pool_audit_track_names = frozenset(track_names)
        env.pool_deal_hook = tracer.on_deal
        env.play_scripted(list(policies), recruit_tracer=tracer)
        game_length = env.turn
        for pf in tracer.player_finals:
            pf["game_length"] = game_length
        all_events.extend(tracer.events)
        all_turn_summaries.extend(tracer.turn_summaries)
        all_player_finals.extend(tracer.player_finals)
        all_deals.extend(tracer.deal_events)
        lobby_meta.append({
            "lobby": lobby_i,
            "seed": lobby_seed,
            "lobby_tribes": list(env.lobby_tribes),
            "game_length": env.turn,
            "final_pool_total": int(sum(env._pool.values())),
            "initial_pool_total_expected": int(
                sum(POOL_COPIES[m.tier] for m in env._catalogue.values())),
            "catalogue_size": len(env._catalogue),
        })
        del env

    return {
        "lobbies": lobbies,
        "seed": seed,
        "scaling_mode": "residual",
        "events": all_events,
        "turn_summaries": all_turn_summaries,
        "player_finals": all_player_finals,
        "lobby_meta": lobby_meta,
        "deal_events": all_deals,
        "track_names": sorted(track_names),
    }


@dataclass
class CardWindowLive:
    lobby: int
    seat: int
    archetype_key: str
    card: str
    weight: float
    card_tier: Optional[int]
    entry_turn: int
    n_tier_eligible_deals: int = 0
    n_raw_appearances: int = 0
    expected_raw_live: float = 0.0
    p_zero_live: float = 1.0
    mean_remaining_at_deal: float = 0.0
    mean_eligible_total: float = 0.0
    remaining_sum: float = 0.0
    eligible_sum: float = 0.0


def _post_assembly_eligible_windows(traces: Dict) -> List[CardWindowLive]:
    deals_by_ls: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for d in traces.get("deal_events") or []:
        if d.get("frozen_skip"):
            continue
        deals_by_ls[(d["lobby"], d["seat"])].append(d)

    avail = analyze_availability_decomposition(traces)
    windows: List[CardWindowLive] = []

    for rec in avail["state_records"]:
        lobby, seat = rec["lobby"], rec["seat"]
        entry = int(rec["entry_turn"])
        deals = [d for d in deals_by_ls.get((lobby, seat), [])
                 if int(d["turn"]) >= entry]

        for name, c in (rec.get("cards") or {}).items():
            if c.get("present_final"):
                continue
            if not c.get("in_exact_catalogue"):
                continue
            ct = c.get("card_tier")
            if ct is None:
                continue
            w = CardWindowLive(
                lobby=lobby, seat=seat, archetype_key=rec["archetype_key"],
                card=name, weight=float(c["weight"]), card_tier=ct,
                entry_turn=entry,
            )
            for d in deals:
                tavern = int(d["tavern_tier"])
                if tavern < int(ct):
                    continue
                rem = int((d.get("card_remaining") or {}).get(name, 0))
                tot = int(d.get("eligible_total_copies") or 0)
                n_slots = int(d.get("n_slots") or 0)
                w.n_tier_eligible_deals += 1
                w.remaining_sum += rem
                w.eligible_sum += tot
                w.expected_raw_live += expected_raw_live_deal(rem, tot, n_slots)
                w.p_zero_live *= p_zero_live_deal(rem, tot, n_slots)
                dealt = d.get("dealt_names") or []
                w.n_raw_appearances += sum(1 for x in dealt if x == name)
            if w.n_tier_eligible_deals == 0:
                continue
            w.mean_remaining_at_deal = w.remaining_sum / w.n_tier_eligible_deals
            w.mean_eligible_total = w.eligible_sum / w.n_tier_eligible_deals
            windows.append(w)
    return windows


def calibrate_live_pool(traces: Dict) -> Dict:
    windows = _post_assembly_eligible_windows(traces)
    n = len(windows)
    if n == 0:
        return {
            "n_card_windows": 0,
            "sum_expected_raw_live": 0.0,
            "sum_observed_raw": 0.0,
            "expected_windows_with_ge1": 0.0,
            "observed_windows_with_ge1": 0,
            "observed_zero_offer_rate": None,
            "expected_zero_offer_rate": None,
            "log_prob_all_observed_zeros": None,
            "by_card_tier": {},
            "by_archetype": {},
            "by_entry_turn": {},
            "most_surprising_zeros": [],
        }

    sum_exp = sum(w.expected_raw_live for w in windows)
    sum_obs = float(sum(w.n_raw_appearances for w in windows))
    exp_ge1 = sum(1.0 - w.p_zero_live for w in windows)
    obs_ge1 = sum(1 for w in windows if w.n_raw_appearances >= 1)
    zero_obs = sum(1 for w in windows if w.n_raw_appearances == 0)
    mean_p_zero = sum(w.p_zero_live for w in windows) / n
    log_p = sum(math.log(max(w.p_zero_live, 1e-300)) for w in windows)

    def _bucket(key_fn):
        groups: Dict[str, List[CardWindowLive]] = defaultdict(list)
        for w in windows:
            groups[str(key_fn(w))].append(w)
        out = {}
        for k, ws in sorted(groups.items()):
            nn = len(ws)
            out[k] = {
                "n": nn,
                "sum_expected_raw": sum(x.expected_raw_live for x in ws),
                "sum_observed_raw": sum(x.n_raw_appearances for x in ws),
                "observed_zero_rate": sum(
                    1 for x in ws if x.n_raw_appearances == 0) / nn,
                "expected_zero_rate": sum(x.p_zero_live for x in ws) / nn,
                "mean_remaining": sum(x.mean_remaining_at_deal for x in ws) / nn,
            }
        return out

    surprising = sorted(
        [w for w in windows if w.n_raw_appearances == 0],
        key=lambda w: w.p_zero_live)[:25]

    return {
        "cohort": (
            "missing-final cores in exact catalogue with ≥1 post-assembly "
            "tier-eligible live shop deal"),
        "n_card_windows": n,
        "sum_expected_raw_live": sum_exp,
        "sum_observed_raw": sum_obs,
        "expected_windows_with_ge1": exp_ge1,
        "observed_windows_with_ge1": obs_ge1,
        "observed_zero_offer_rate": zero_obs / n,
        "expected_zero_offer_rate": mean_p_zero,
        "log_prob_all_observed_zeros": log_p,
        "by_card_tier": _bucket(lambda w: w.card_tier),
        "by_archetype": _bucket(lambda w: w.archetype_key),
        "by_entry_turn": _bucket(lambda w: w.entry_turn),
        "most_surprising_zeros": [
            {
                "lobby": w.lobby, "seat": w.seat,
                "archetype_key": w.archetype_key, "card": w.card,
                "card_tier": w.card_tier, "entry_turn": w.entry_turn,
                "n_deals": w.n_tier_eligible_deals,
                "expected_raw_live": w.expected_raw_live,
                "p_zero_live": w.p_zero_live,
                "mean_remaining": w.mean_remaining_at_deal,
                "mean_eligible_total": w.mean_eligible_total,
                "weight": w.weight,
            }
            for w in surprising
        ],
    }


def analyze_shop_pool_audit(traces: Dict) -> Dict:
    catalogue = audit_catalogue_synchronization()
    contract = audit_pool_contract()
    rules = audit_rule_mismatches(contract)
    avail = analyze_availability_decomposition(traces)
    live = calibrate_live_pool(traces)

    conservation = {
        "n_lobbies": traces["lobbies"],
        "note": (
            "Elimination does not return copies; final pool + alive holdings "
            "is expected below initial pool total."),
        "lobby_final_pool_mean": (
            sum(m["final_pool_total"] for m in traces["lobby_meta"])
            / max(len(traces["lobby_meta"]), 1)),
        "lobby_initial_pool_mean": (
            sum(m["initial_pool_total_expected"] for m in traces["lobby_meta"])
            / max(len(traces["lobby_meta"]), 1)),
    }

    headlines = {
        "pct_cores_missing_from_kb": (
            catalogue["status_share"].get("MISSING_FROM_KB")),
        "pct_cores_in_exact_catalogue": (
            catalogue["status_share"].get("IN_EXACT_CATALOGUE")),
        "n_demonstrated_rule_mismatches": rules["n_demonstrated_mismatches"],
        "live_observed_zero_offer_rate": live.get("observed_zero_offer_rate"),
        "live_expected_zero_offer_rate": live.get("expected_zero_offer_rate"),
        "live_sum_expected_raw": live.get("sum_expected_raw_live"),
        "live_sum_observed_raw": live.get("sum_observed_raw"),
        "phase_2l_a3_share": (avail.get("headlines") or {}).get(
            "pct_exact_catalogue_tier_eligible_zero_raw"),
        "phase_2l_a1_share": (avail.get("headlines") or {}).get(
            "pct_not_in_exact_simulator_catalogue"),
        "phase_2l_a4_share": (avail.get("headlines") or {}).get(
            "pct_raw_but_never_legal"),
    }

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_states_phase_2l": avail.get("n_states"),
        "catalogue_synchronization": {
            k: v for k, v in catalogue.items() if k != "rows"
        },
        "catalogue_rows": catalogue["rows"],
        "pool_contract": contract,
        "rule_mismatches": rules,
        "pool_conservation": conservation,
        "phase_2l_headlines": avail.get("headlines"),
        "phase_2l_a1_exclusion": avail.get("a1_exclusion_breakdown"),
        "live_calibration": live,
        "headlines": headlines,
        "reserved_seeds": {
            "diagnostic_dev": (
                f"{PHASE_2M_SEED}–{PHASE_2M_SEED + PHASE_2M_LOBBIES - 1}"),
            "intervention_reserved": (
                f"{RESERVED_INTERVENTION_SEED}–"
                f"{RESERVED_INTERVENTION_SEED + RESERVED_INTERVENTION_LOBBIES - 1}"),
            "confirmation_reserved": (
                f"{RESERVED_CONFIRM_SEED}–"
                f"{RESERVED_CONFIRM_SEED + RESERVED_CONFIRM_LOBBIES - 1}"),
            "consumed_in_2m": "diagnostic_dev only (reuse 2L)",
        },
    }
