"""Phase 2M shop/pool rules audit (measurement-only).

Research question: is post-assembly scarcity caused by incorrect simulator
catalogue/pool rules, incorrect live-pool accounting, or expected scarcity
under a correctly implemented finite shared pool?

Does **not** change pool/shop behavior — only observational hooks + analysis.
"""

from __future__ import annotations

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

METHODOLOGY_VERSION = "2m_v2"

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


def p_hit_live_deal(card_remaining: int, eligible_total: int,
                    n_slots: int) -> float:
    """P(card appears ≥1 time in a without-replacement deal)."""
    return 1.0 - p_zero_live_deal(card_remaining, eligible_total, n_slots)


def is_post_assembly_deal(deal_turn: int, entry_turn: int) -> bool:
    """True iff the deal occurs strictly after the first-2 cohort entry turn.

    Phase 2L entry is the *end* of the first recruit turn that finishes with
    2+ cores. Shops on the entry turn itself are pre-assembly (and positively
    selected — they may be the shop that offered the assembling core).
    """
    return int(deal_turn) > int(entry_turn)


def filter_post_assembly_deals(deals: List[Dict], entry_turn: int) -> List[Dict]:
    return [d for d in deals if is_post_assembly_deal(d.get("turn", -1), entry_turn)]


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
    from hsbg_coach.bg_env import PHASE_2N_DEATH_RETURN, PHASE_2N_FREEZE_TOPUP
    return {
        "simulator": {
            "POOL_COPIES": dict(POOL_COPIES),
            "SHOP_SLOTS": dict(SHOP_SLOTS),
            "draw": (
                "weighted by remaining live copies among catalogue minions "
                "with tier <= tavern and pool > 0; without replacement within deal"),
            "freeze": (
                "frozen shop skips full return+redraw; "
                + ("incomplete shops top up to SHOP_SLOTS[tier] (Phase 2N-B)"
                   if PHASE_2N_FREEZE_TOPUP else
                   "does not top-up incomplete frozen shops")
            ),
            "roll": "unfreeze; return shop copies; redraw SHOP_SLOTS[tier]",
            "buy": "shop→hand; copy stays out of pool",
            "sell": "board→pool (+1 or +3 if golden)",
            "triple": (
                "3 non-golden → 1 golden in hand; discover = extra _draw; "
                "base copies not returned (golden returns as 3 on sell)"),
            "elimination": (
                "DEAD PLAYERS RETURN board/hand/shop copies to pool (Phase 2N-B)"
                if PHASE_2N_DEATH_RETURN else
                "DEAD PLAYERS DO NOT RETURN board/hand/shop copies to pool"),
            "phase_2n_death_return": PHASE_2N_DEATH_RETURN,
            "phase_2n_freeze_topup": PHASE_2N_FREEZE_TOPUP,
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
    """Enumerate simulator↔rules diffs; separate actionable vs contextual."""
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
                "phase_2n_actionable": True,
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
        "phase_2n_actionable": False,
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
        "phase_2n_actionable": False,
        "severity": "contextual",
        "note": (
            "Spell-era tavern card counts differ. Contextual/out-of-scope "
            "until the simulator models tavern spells."),
    })
    mismatches.append({
        "id": "elimination_no_return_to_pool",
        "area": "pool_lifecycle_accounting",
        "simulator": contract["simulator"]["elimination"],
        "reference": "eliminated players' minions expected to return to pool",
        "mismatch": not bool(contract["simulator"].get("phase_2n_death_return")),
        "phase_2n_actionable": not bool(
            contract["simulator"].get("phase_2n_death_return")),
        "note": "Phase 2N-B enables death return when flag is on",
    })
    mismatches.append({
        "id": "freeze_no_topup",
        "area": "freeze_return_to_pool",
        "simulator": contract["simulator"]["freeze"],
        "reference": "incomplete freeze / tier-up tops up new slots",
        "mismatch": not bool(contract["simulator"].get("phase_2n_freeze_topup")),
        "phase_2n_actionable": not bool(
            contract["simulator"].get("phase_2n_freeze_topup")),
        "note": "Phase 2N-B enables freeze top-up when flag is on",
    })
    mismatches.append({
        "id": "no_tier_7",
        "area": "catalogue",
        "simulator": "MAX_TIER=6; no T7",
        "reference": "T7 exists in some modes (5 copies)",
        "mismatch": True,
        "phase_2n_actionable": False,
        "severity": "contextual",
        "note": (
            "Contextual/out-of-scope for a deliberate standard Tier-6 "
            "simulator; do not add T7 solely to match a global table."),
    })

    demonstrated = [m for m in mismatches if m["mismatch"]]
    actionable = [m for m in demonstrated if m.get("phase_2n_actionable")]
    contextual = [m for m in demonstrated if not m.get("phase_2n_actionable")]
    return {
        "n_documented_checks": len(mismatches),
        "n_demonstrated_mismatches": len(demonstrated),
        "n_phase_2n_actionable": len(actionable),
        "n_contextual_out_of_scope": len(contextual),
        "mismatches": mismatches,
        "demonstrated_ids": [m["id"] for m in demonstrated],
        "phase_2n_actionable_ids": [m["id"] for m in actionable],
        "contextual_ids": [m["id"] for m in contextual],
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
class DealObservation:
    """One (card × deal) conditional prediction under the exact pre-deal pool."""
    lobby: int
    seat: int
    archetype_key: str
    card: str
    card_tier: int
    entry_turn: int
    deal_turn: int
    deal_reason: str
    expected_raw: float
    p_hit: float
    observed_raw: int
    observed_hit: int
    card_remaining: int
    eligible_total: int
    n_slots: int
    present_final: bool
    subfate: Optional[str]


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
    p_zero_live: float = 1.0  # descriptive only — adaptive product
    mean_remaining_at_deal: float = 0.0
    mean_eligible_total: float = 0.0
    remaining_sum: float = 0.0
    eligible_sum: float = 0.0


def _iter_post_assembly_states(traces: Dict):
    """Yield (rec, deals_strictly_after_entry) for each 2L post-assembly state."""
    deals_by_ls: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for d in traces.get("deal_events") or []:
        if d.get("frozen_skip"):
            continue
        deals_by_ls[(d["lobby"], d["seat"])].append(d)

    avail = analyze_availability_decomposition(traces)
    for rec in avail["state_records"]:
        lobby, seat = rec["lobby"], rec["seat"]
        entry = int(rec["entry_turn"])
        deals = filter_post_assembly_deals(
            deals_by_ls.get((lobby, seat), []), entry)
        yield rec, deals, avail


def _collect_deal_observations(
        rec: Dict, name: str, c: Dict, deals: List[Dict]
) -> List[DealObservation]:
    if not c.get("in_exact_catalogue"):
        return []
    ct = c.get("card_tier")
    if ct is None:
        return []
    out: List[DealObservation] = []
    for d in deals:
        tavern = int(d["tavern_tier"])
        if tavern < int(ct):
            continue
        rem = int((d.get("card_remaining") or {}).get(name, 0))
        tot = int(d.get("eligible_total_copies") or 0)
        n_slots = int(d.get("n_slots") or 0)
        dealt = d.get("dealt_names") or []
        obs_raw = sum(1 for x in dealt if x == name)
        out.append(DealObservation(
            lobby=rec["lobby"], seat=rec["seat"],
            archetype_key=rec["archetype_key"], card=name,
            card_tier=int(ct), entry_turn=int(rec["entry_turn"]),
            deal_turn=int(d["turn"]), deal_reason=str(d.get("reason") or ""),
            expected_raw=expected_raw_live_deal(rem, tot, n_slots),
            p_hit=p_hit_live_deal(rem, tot, n_slots),
            observed_raw=int(obs_raw),
            observed_hit=1 if obs_raw >= 1 else 0,
            card_remaining=rem, eligible_total=tot, n_slots=n_slots,
            present_final=bool(c.get("present_final")),
            subfate=c.get("subfate"),
        ))
    return out


def _window_from_deals(rec: Dict, name: str, c: Dict, deals: List[Dict]
                       ) -> Optional[CardWindowLive]:
    """Descriptive card-window aggregate (adaptive P_zero is secondary only)."""
    if not c.get("in_exact_catalogue"):
        return None
    ct = c.get("card_tier")
    if ct is None:
        return None
    w = CardWindowLive(
        lobby=rec["lobby"], seat=rec["seat"],
        archetype_key=rec["archetype_key"],
        card=name, weight=float(c["weight"]), card_tier=ct,
        entry_turn=int(rec["entry_turn"]),
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
        return None
    w.mean_remaining_at_deal = w.remaining_sum / w.n_tier_eligible_deals
    w.mean_eligible_total = w.eligible_sum / w.n_tier_eligible_deals
    return w


def _lobby_bootstrap_ci(
        lobby_deltas: List[float], n_boot: int = 2000, seed: int = 2
) -> Dict:
    """Percentile bootstrap CI for the mean of per-lobby deltas."""
    import random
    if not lobby_deltas:
        return {"n_lobbies": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    n = len(lobby_deltas)
    means = []
    for _ in range(n_boot):
        sample = [lobby_deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "n_lobbies": n,
        "mean": sum(lobby_deltas) / n,
        "ci95": [lo, hi],
    }


def _summarize_deal_level(obs: List[DealObservation], cohort: str) -> Dict:
    """Primary calibration: per-deal conditional E[raw]/P(hit) vs observed."""
    n = len(obs)
    if n == 0:
        return {
            "cohort": cohort,
            "n_deal_card_observations": 0,
            "sum_expected_raw": 0.0,
            "sum_observed_raw": 0.0,
            "sum_expected_hit_probability": 0.0,
            "sum_observed_hit_deals": 0,
            "raw_ratio_obs_over_exp": None,
            "hit_ratio_obs_over_exp": None,
            "lobby_clustered": {},
            "by_card_tier": {},
        }

    sum_exp_raw = sum(o.expected_raw for o in obs)
    sum_obs_raw = float(sum(o.observed_raw for o in obs))
    sum_exp_hit = sum(o.p_hit for o in obs)
    sum_obs_hit = float(sum(o.observed_hit for o in obs))

    # Per-lobby aggregates for clustered inference
    by_lobby: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"exp_raw": 0.0, "obs_raw": 0.0, "exp_hit": 0.0, "obs_hit": 0.0})
    for o in obs:
        b = by_lobby[o.lobby]
        b["exp_raw"] += o.expected_raw
        b["obs_raw"] += o.observed_raw
        b["exp_hit"] += o.p_hit
        b["obs_hit"] += o.observed_hit

    raw_deltas = [b["obs_raw"] - b["exp_raw"] for b in by_lobby.values()]
    hit_deltas = [b["obs_hit"] - b["exp_hit"] for b in by_lobby.values()]

    def _tier_bucket():
        groups: Dict[str, List[DealObservation]] = defaultdict(list)
        for o in obs:
            groups[str(o.card_tier)].append(o)
        out = {}
        for k, xs in sorted(groups.items()):
            er = sum(x.expected_raw for x in xs)
            or_ = float(sum(x.observed_raw for x in xs))
            eh = sum(x.p_hit for x in xs)
            oh = float(sum(x.observed_hit for x in xs))
            out[k] = {
                "n": len(xs),
                "sum_expected_raw": er,
                "sum_observed_raw": or_,
                "sum_expected_hit_probability": eh,
                "sum_observed_hit_deals": oh,
            }
        return out

    return {
        "cohort": cohort,
        "n_deal_card_observations": n,
        "n_lobbies": len(by_lobby),
        "sum_expected_raw": sum_exp_raw,
        "sum_observed_raw": sum_obs_raw,
        "sum_expected_hit_probability": sum_exp_hit,
        "sum_observed_hit_deals": sum_obs_hit,
        "raw_ratio_obs_over_exp": (
            sum_obs_raw / sum_exp_raw if sum_exp_raw > 1e-12 else None),
        "hit_ratio_obs_over_exp": (
            sum_obs_hit / sum_exp_hit if sum_exp_hit > 1e-12 else None),
        "lobby_clustered": {
            "raw_obs_minus_exp": _lobby_bootstrap_ci(raw_deltas),
            "hit_obs_minus_exp": _lobby_bootstrap_ci(hit_deltas),
            "note": (
                "Bootstrap mean(obs−exp) across lobbies; CI accounts for "
                "within-lobby correlation of card×deal observations."),
        },
        "by_card_tier": _tier_bucket(),
    }


def _summarize_windows_descriptive(windows: List[CardWindowLive],
                                   cohort: str) -> Dict:
    """Secondary/descriptive card-window metrics (adaptive P_zero not primary)."""
    n = len(windows)
    if n == 0:
        return {
            "cohort": cohort,
            "role": "descriptive_secondary",
            "n_card_windows": 0,
            "sum_expected_raw_live": 0.0,
            "sum_observed_raw": 0.0,
            "observed_zero_offer_rate": None,
            "adaptive_product_expected_zero_rate": None,
            "note": (
                "Adaptive product(P_zero) along the realized trajectory is "
                "not an ex-ante zero probability when hits change later pool "
                "state; deal-level calibration is primary."),
        }

    zero_obs = sum(1 for w in windows if w.n_raw_appearances == 0)
    return {
        "cohort": cohort,
        "role": "descriptive_secondary",
        "n_card_windows": n,
        "sum_expected_raw_live": sum(w.expected_raw_live for w in windows),
        "sum_observed_raw": float(sum(w.n_raw_appearances for w in windows)),
        "observed_zero_offer_rate": zero_obs / n,
        "adaptive_product_expected_zero_rate": (
            sum(w.p_zero_live for w in windows) / n),
        "note": (
            "observed_zero_offer_rate is descriptive. "
            "adaptive_product_expected_zero_rate is demoted — not a clean "
            "ex-ante P(zero) under adaptive pool trajectories."),
    }


def calibrate_live_pool(traces: Dict) -> Dict:
    """2m_v2 live-pool calibration.

    Primary: deal-level conditional predictions (exact pre-deal pool state).
    Post-assembly deals use ``turn > entry_turn`` only.
    """
    uncond_obs: List[DealObservation] = []
    missing_obs: List[DealObservation] = []
    a3_obs: List[DealObservation] = []
    uncond_win: List[CardWindowLive] = []
    a3_win: List[CardWindowLive] = []
    n_entry_turn_deals_excluded = 0

    # Count excluded entry-turn deals for integrity reporting
    deals_by_ls: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for d in traces.get("deal_events") or []:
        if d.get("frozen_skip"):
            continue
        deals_by_ls[(d["lobby"], d["seat"])].append(d)

    for rec, deals, _avail in _iter_post_assembly_states(traces):
        entry = int(rec["entry_turn"])
        all_deals = deals_by_ls.get((rec["lobby"], rec["seat"]), [])
        n_entry_turn_deals_excluded += sum(
            1 for d in all_deals if int(d["turn"]) == entry)

        for name, c in (rec.get("cards") or {}).items():
            obs = _collect_deal_observations(rec, name, c, deals)
            if not obs:
                continue
            uncond_obs.extend(obs)
            if not c.get("present_final"):
                missing_obs.extend(obs)
            if c.get("subfate") == "A3_TIER_ELIGIBLE_ZERO_RAW":
                a3_obs.extend(obs)

            w = _window_from_deals(rec, name, c, deals)
            if w is None:
                continue
            uncond_win.append(w)
            if c.get("subfate") == "A3_TIER_ELIGIBLE_ZERO_RAW":
                a3_win.append(w)

    primary = _summarize_deal_level(
        uncond_obs,
        "deal-level: exact-catalogue cores × post-assembly deals "
        "(turn > entry_turn); NOT conditioned on missing-final")
    missing_deal = _summarize_deal_level(
        missing_obs,
        "deal-level missing-final subset (selection-biased; secondary)")
    a3_deal = _summarize_deal_level(
        a3_obs,
        "deal-level A3 zero-raw windows (observed hits near 0 by definition)")

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "post_assembly_deal_boundary": "turn > entry_turn",
        "n_entry_turn_deals_excluded_from_calib": n_entry_turn_deals_excluded,
        "primary_deal_level": primary,
        "missing_final_deal_level": missing_deal,
        "a3_deal_level": a3_deal,
        "descriptive_card_windows": {
            "unconditioned": _summarize_windows_descriptive(
                uncond_win,
                "descriptive card-windows (adaptive P_zero demoted)"),
            "a3_zero_raw": _summarize_windows_descriptive(
                a3_win,
                "descriptive A3 card-windows (observed zero ≈ 1 by definition)"),
        },
        # Convenience aliases for headlines / decision
        "sum_expected_raw": primary.get("sum_expected_raw"),
        "sum_observed_raw": primary.get("sum_observed_raw"),
        "sum_expected_hit_probability": primary.get(
            "sum_expected_hit_probability"),
        "sum_observed_hit_deals": primary.get("sum_observed_hit_deals"),
        "raw_ratio_obs_over_exp": primary.get("raw_ratio_obs_over_exp"),
        "hit_ratio_obs_over_exp": primary.get("hit_ratio_obs_over_exp"),
        "lobby_clustered": primary.get("lobby_clustered"),
        "n_deal_card_observations": primary.get("n_deal_card_observations"),
        "note": (
            "2m_v2 primary = deal-level ΣE(raw)/ΣP(hit) vs observed, with "
            "lobby-clustered bootstrap. Entry-turn deals excluded. Adaptive "
            "whole-window product(P_zero) is descriptive only."),
    }


def analyze_shop_pool_audit(traces: Dict) -> Dict:
    catalogue = audit_catalogue_synchronization()
    contract = audit_pool_contract()
    rules = audit_rule_mismatches(contract)
    avail = analyze_availability_decomposition(traces)
    live = calibrate_live_pool(traces)
    primary = live.get("primary_deal_level") or {}
    a3_desc = (live.get("descriptive_card_windows") or {}).get(
        "a3_zero_raw") or {}

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
        "n_phase_2n_actionable_mismatches": rules["n_phase_2n_actionable"],
        # Primary deal-level
        "live_sum_expected_raw": primary.get("sum_expected_raw"),
        "live_sum_observed_raw": primary.get("sum_observed_raw"),
        "live_sum_expected_hit_probability": primary.get(
            "sum_expected_hit_probability"),
        "live_sum_observed_hit_deals": primary.get("sum_observed_hit_deals"),
        "live_raw_ratio_obs_over_exp": primary.get("raw_ratio_obs_over_exp"),
        "live_hit_ratio_obs_over_exp": primary.get("hit_ratio_obs_over_exp"),
        "live_n_deal_card_observations": primary.get(
            "n_deal_card_observations"),
        # Descriptive only
        "a3_descriptive_observed_zero_rate": a3_desc.get(
            "observed_zero_offer_rate"),
        "a3_descriptive_adaptive_expected_zero_rate": a3_desc.get(
            "adaptive_product_expected_zero_rate"),
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
