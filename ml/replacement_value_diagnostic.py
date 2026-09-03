"""Phase 2P — replacement-value / scaling-contamination diagnostic.

Measurement only. Quantifies how often abstract scaling makes a full board's
weakest incumbent incomparable to fresh Tavern candidates.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from hsbg_coach import cards as cards_mod
from hsbg_coach.bg_env import (
    A_BUY0,
    A_END,
    A_FREEZE,
    A_LEVEL,
    A_PLAY0,
    A_ROLL,
    A_SELL0,
    BGEnv,
    MAX_BOARD,
    N_BUY,
    N_PLAY,
    N_SELL,
    greedy_policy,
)
from hsbg_coach.board_opportunity_policy import policies_for_lobby
from hsbg_coach.build_path import infer_target, path_value
from hsbg_coach.persistence_prior import PersistencePrior
from hsbg_coach.synergy import load_embeddings
from hsbg_coach.tempo_board_policy import _held_names, _shop_build_gain

METHODOLOGY_VERSION = "2p_v2"

INSTRUMENT_TURNS = tuple(range(7, 15))
PHASE_2P_SEED = 12700
PHASE_2P_LOBBIES = 500

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),
    (11000, 11499),
    (11500, 11699),
    (11700, 12199),
    (12200, 12699),
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    lo, hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2P seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return float(st.median(xs)) if xs else None


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


def _raw_stats(m: Dict) -> float:
    return float((m.get("attack") or 0) + (m.get("health") or 0))


def _decode_action(action: int) -> str:
    if A_BUY0 <= action < A_BUY0 + N_BUY:
        return "buy"
    if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
        return "play"
    if A_SELL0 <= action < A_SELL0 + N_SELL:
        return "sell"
    if action == A_ROLL:
        return "roll"
    if action == A_LEVEL:
        return "level"
    if action == A_FREEZE:
        return "freeze"
    if action == A_END:
        return "end"
    return "unknown"


def _kb_views() -> tuple[Dict[str, object], Dict[str, object]]:
    kb = cards_mod.load_kb()
    by_id = dict(kb)
    by_name = cards_mod.by_name(kb)
    return by_id, by_name


def _is_golden(m: Dict) -> bool:
    """Golden copies expose PREMIUM in observe() tags; EnvMinion also has .golden."""
    tags = m.get("tags") or {}
    if tags.get("PREMIUM") == "1":
        return True
    return bool(m.get("golden"))


def _base_card_view(m: Dict, kb_id: Dict[str, object], kb_name: Dict[str, object]) -> Dict:
    """KB printed baseline, with golden natural baseline = 2× normal printed.

    Phase 2P 2p_v2: abstract scaling must be measured against the *natural*
    printed stats of the observed copy (golden doubles attack/health), not the
    non-golden KB entry alone.
    """
    is_golden = _is_golden(m)
    factor = 2 if is_golden else 1
    ck = kb_id.get(m.get("card_id")) or kb_name.get(m.get("name"))
    if ck is None:
        # Fallback: treat current attack/health as already-natural if KB missing.
        atk = int(m.get("attack") or 0)
        hp = int(m.get("health") or 0)
        # If golden but KB missing, we cannot recover the normal printed; keep
        # current as both normal/natural to avoid inventing a false halving.
        normal_atk, normal_hp = atk, hp
        natural_atk, natural_hp = atk, hp
        return {
            "card_id": m.get("card_id"),
            "name": m.get("name"),
            "is_golden": is_golden,
            "normal_attack": normal_atk,
            "normal_health": normal_hp,
            "normal_raw": float(normal_atk + normal_hp),
            "attack": natural_atk,
            "health": natural_hp,
            "raw": float(natural_atk + natural_hp),
            "keywords": list(m.get("keywords") or []),
            "text": "",
            "tribes": list(m.get("tribes") or []),
            "text_present": False,
        }
    normal_atk = int(ck.attack or 0)
    normal_hp = int(ck.health or 0)
    natural_atk = normal_atk * factor
    natural_hp = normal_hp * factor
    return {
        "card_id": ck.card_id,
        "name": ck.name,
        "is_golden": is_golden,
        "normal_attack": normal_atk,
        "normal_health": normal_hp,
        "normal_raw": float(normal_atk + normal_hp),
        "attack": natural_atk,
        "health": natural_hp,
        "raw": float(natural_atk + natural_hp),
        "keywords": list(ck.keywords),
        "text": ck.text,
        "tribes": list(ck.tribes),
        "text_present": bool(ck.text),
    }


def _candidate_build_meta(obs: Dict, shop_card: Dict, embeddings: Dict[str, List[float]]) -> Dict:
    board = obs.get("board") or []
    fit = infer_target(board)
    if fit is None:
        return {
            "target_archetype": None,
            "target_core": False,
            "shop_build_gain": 0.0,
            "path_value_adjustment": 0.0,
            "path_reason": None,
            "card2vec_in_vocab": shop_card.get("name") in embeddings,
        }
    held = _held_names(obs)
    tier = int(obs.get("tavern_tier") or 1)
    build_gain = _shop_build_gain(shop_card.get("name"), fit, tier, held)
    adj, reason = path_value(
        board,
        shop_card.get("name"),
        tier,
        candidate_tribe=((shop_card.get("tribes") or [None])[0]),
    )
    return {
        "target_archetype": fit.arch.key,
        "target_core": shop_card.get("name") in fit.arch.core,
        "shop_build_gain": float(build_gain),
        "path_value_adjustment": float(adj),
        "path_reason": reason,
        "card2vec_in_vocab": shop_card.get("name") in embeddings,
    }


def _state_row(
    *,
    lobby: int,
    seed: int,
    arm: str,
    seat: int,
    turn: int,
    obs: Dict,
    mask: List[bool],
    action_counts: Dict[str, int],
    kb_id: Dict[str, object],
    kb_name: Dict[str, object],
) -> Optional[Dict]:
    board = obs.get("board") or []
    shop = obs.get("shop") or []
    buy_slots = [i for i in range(min(len(shop), N_BUY)) if mask[A_BUY0 + i]]
    if turn not in INSTRUMENT_TURNS or len(board) < MAX_BOARD or not buy_slots:
        return None

    weakest_idx = min(range(len(board)), key=lambda i: _raw_stats(board[i]))
    weakest = board[weakest_idx]
    weakest_current_raw = _raw_stats(weakest)
    weakest_base = _base_card_view(weakest, kb_id, kb_name)
    weakest_normal_printed_raw = float(weakest_base["normal_raw"])
    weakest_natural_printed_raw = float(weakest_base["raw"])
    weakest_golden = bool(weakest_base["is_golden"])
    inflation = (
        (weakest_current_raw / weakest_natural_printed_raw)
        if weakest_natural_printed_raw else None
    )

    best_slot = max(buy_slots, key=lambda i: _raw_stats(shop[i]))
    best_shop = shop[best_slot]
    best_current_raw = _raw_stats(best_shop)
    # Shop deals are non-golden; printed = natural for candidates.
    best_base = _base_card_view(best_shop, kb_id, kb_name)
    best_printed_raw = float(best_base["raw"])

    current_rule_accepts = best_printed_raw > weakest_current_raw
    base_scale_accepts = best_printed_raw > weakest_natural_printed_raw
    return {
        "lobby": lobby,
        "seed": seed,
        "arm": arm,
        "seat": seat,
        "turn": turn,
        "gold": float(obs.get("gold") or 0),
        "legal_buy_slots": list(buy_slots),
        "rolls_so_far": int(action_counts.get("roll", 0)),
        "buys_so_far": int(action_counts.get("buy", 0)),
        "sells_so_far": int(action_counts.get("sell", 0)),
        "board_full": True,
        "weakest_board_slot": weakest_idx,
        "weakest_board_name": weakest.get("name"),
        "weakest_board_card_id": weakest.get("card_id"),
        "weakest_board_golden": weakest_golden,
        "weakest_board_current_raw": weakest_current_raw,
        "weakest_board_normal_printed_raw": weakest_normal_printed_raw,
        "weakest_board_natural_printed_raw": weakest_natural_printed_raw,
        # Alias kept for aggregators / 2p_v1 field readers: natural printed.
        "weakest_board_printed_raw": weakest_natural_printed_raw,
        "weakest_board_current_attack": weakest.get("attack"),
        "weakest_board_current_health": weakest.get("health"),
        "weakest_board_printed_attack": weakest_base["attack"],
        "weakest_board_printed_health": weakest_base["health"],
        "weakest_board_keywords": list(weakest_base["keywords"]),
        "weakest_board_inflation_ratio": inflation,
        "best_shop_slot": best_slot,
        "best_shop_name": best_shop.get("name"),
        "best_shop_card_id": best_shop.get("card_id"),
        "best_shop_current_raw": best_current_raw,
        "best_shop_printed_raw": best_printed_raw,
        "best_shop_current_attack": best_shop.get("attack"),
        "best_shop_current_health": best_shop.get("health"),
        "best_shop_printed_attack": best_base["attack"],
        "best_shop_printed_health": best_base["health"],
        "best_shop_keywords": list(best_base["keywords"]),
        "best_shop_text_present": bool(best_base["text_present"]),
        "best_shop_rules_text": best_base["text"],
        "best_shop_gt_weakest_scaled": bool(current_rule_accepts),
        "best_shop_gt_weakest_printed": bool(base_scale_accepts),
        "current_rule_accepts": bool(current_rule_accepts),
        "base_scale_accepts": bool(base_scale_accepts),
        "scaling_blocked_upgrade": bool(base_scale_accepts and not current_rule_accepts),
        "actual_action_kind": None,
        "actual_action_id": None,
        "pending_candidate_name": None,
        "pending_candidate_slot": None,
        "pending_source": None,
        "_dedupe_key": (
            seat,
            turn,
            tuple((m.get("name"), m.get("attack"), m.get("health"),
                   (m.get("tags") or {}).get("PREMIUM")) for m in board),
            tuple((shop[i].get("name"), shop[i].get("attack"), shop[i].get("health"))
                  for i in buy_slots),
            float(obs.get("gold") or 0),
        ),
    }


def _candidate_rows(
    *,
    state_idx: int,
    state_row: Dict,
    obs: Dict,
    kb_id: Dict[str, object],
    kb_name: Dict[str, object],
    embeddings: Dict[str, List[float]],
) -> List[Dict]:
    out: List[Dict] = []
    shop = obs.get("shop") or []
    weakest_scaled = float(state_row["weakest_board_current_raw"])
    weakest_printed = float(state_row["weakest_board_natural_printed_raw"])
    for slot in state_row["legal_buy_slots"]:
        if slot >= len(shop):
            continue
        sm = shop[slot]
        base = _base_card_view(sm, kb_id, kb_name)
        meta = _candidate_build_meta(obs, sm, embeddings)
        cand_printed = float(base["raw"])
        cand_current = _raw_stats(sm)
        current_accepts = cand_printed > weakest_scaled
        base_accepts = cand_printed > weakest_printed
        scaling_blocked = base_accepts and not current_accepts
        build_signal = bool(
            meta["target_core"]
            or meta["shop_build_gain"] > 0
            or meta["path_value_adjustment"] < 0
        )
        reject_bucket = None
        if not current_accepts:
            if scaling_blocked:
                reject_bucket = "A_SCALING_BLOCKED_UPGRADE"
            elif build_signal:
                reject_bucket = "B_BUILD_OR_CORE_VALUE"
            else:
                reject_bucket = "C_NEITHER"
        if current_accepts:
            continue
        out.append({
            "state_index": state_idx,
            "lobby": state_row["lobby"],
            "seed": state_row["seed"],
            "arm": state_row["arm"],
            "seat": state_row["seat"],
            "turn": state_row["turn"],
            "shop_slot": slot,
            "candidate_name": sm.get("name"),
            "candidate_card_id": sm.get("card_id"),
            "candidate_current_raw": cand_current,
            "candidate_printed_raw": cand_printed,
            "candidate_keywords": list(base["keywords"]),
            "candidate_text_present": bool(base["text_present"]),
            "candidate_tribes": list(base["tribes"]),
            "target_archetype": meta["target_archetype"],
            "candidate_target_core": bool(meta["target_core"]),
            "candidate_shop_build_gain": float(meta["shop_build_gain"]),
            "candidate_path_value_adjustment": float(meta["path_value_adjustment"]),
            "candidate_card2vec_in_vocab": bool(meta["card2vec_in_vocab"]),
            "weakest_board_name": state_row["weakest_board_name"],
            "weakest_board_golden": bool(state_row.get("weakest_board_golden")),
            "weakest_board_scaled_raw": weakest_scaled,
            "weakest_board_normal_printed_raw": float(
                state_row["weakest_board_normal_printed_raw"]),
            "weakest_board_natural_printed_raw": weakest_printed,
            "weakest_board_printed_raw": weakest_printed,
            "candidate_gt_weakest_scaled": bool(current_accepts),
            "candidate_gt_weakest_printed": bool(base_accepts),
            "current_rule_accepts": bool(current_accepts),
            "base_scale_accepts": bool(base_accepts),
            "scaling_blocked_upgrade": bool(scaling_blocked),
            "reject_bucket": reject_bucket,
        })
    return out


class ReplacementValueTracer:
    """Observational tracer for full-board replacement decisions."""

    def __init__(self, lobby_id: int, seed: int, arm: str, policies: Optional[Sequence] = None):
        self.lobby_id = lobby_id
        self.seed = seed
        self.arm = arm
        self.policies = list(policies or [])
        self.state_rows: List[Dict] = []
        self.candidate_rows: List[Dict] = []
        self._kb_id, self._kb_name = _kb_views()
        self._embeddings = load_embeddings()
        self._last_state_idx: Optional[int] = None
        self._action_counts: Dict[tuple[int, int], Counter] = defaultdict(Counter)
        self._seen_keys: set[tuple] = set()

    def begin_lobby(self, lobby_id: int, _rng_seed: int, _lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id

    def before_action(self, seat: int, turn: int, shop_generation: int, obs: Dict, mask: List[bool]) -> None:
        counts = self._action_counts[(seat, turn)]
        row = _state_row(
            lobby=self.lobby_id,
            seed=self.seed,
            arm=self.arm,
            seat=seat,
            turn=turn,
            obs=obs,
            mask=mask,
            action_counts=counts,
            kb_id=self._kb_id,
            kb_name=self._kb_name,
        )
        self._last_state_idx = None
        if row is None:
            return
        dedupe_key = row.pop("_dedupe_key", None)
        if dedupe_key in self._seen_keys:
            return
        if dedupe_key is not None:
            self._seen_keys.add(dedupe_key)
        idx = len(self.state_rows)
        self.state_rows.append(row)
        self.candidate_rows.extend(
            _candidate_rows(
                state_idx=idx,
                state_row=row,
                obs=obs,
                kb_id=self._kb_id,
                kb_name=self._kb_name,
                embeddings=self._embeddings,
            )
        )
        self._last_state_idx = idx

    def after_action(self, seat: int, turn: int, shop_generation: int, action: int, ended: bool, player=None) -> None:
        kind = _decode_action(action)
        self._action_counts[(seat, turn)][kind] += 1
        if self._last_state_idx is None:
            return
        row = self.state_rows[self._last_state_idx]
        row["actual_action_kind"] = kind
        row["actual_action_id"] = int(action)
        if seat < len(self.policies):
            pending = getattr(self.policies[seat], "pending", None)
            if pending is not None:
                row["pending_candidate_name"] = pending.candidate_name
                row["pending_candidate_slot"] = pending.candidate_slot
                row["pending_source"] = pending.source
        self._last_state_idx = None

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        return

    def end_lobby(self, players) -> None:
        return


def run_replacement_value_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    policy_factory=None,
    policy=None,
    scaling_mode: str = "residual",
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    state_rows: List[Dict] = []
    candidate_rows: List[Dict] = []
    for i in range(lobbies):
        policies = list(policy_factory(i)) if policy_factory else [policy or greedy_policy] * 8
        tracer = ReplacementValueTracer(i, seed + i, arm, policies=policies)
        env = BGEnv(seed=seed + i, scaling_mode=scaling_mode)
        env.play_scripted(policies, recruit_tracer=tracer)
        state_rows.extend(tracer.state_rows)
        candidate_rows.extend(tracer.candidate_rows)
        del env
    return {
        "arm": arm,
        "seed_base": seed,
        "n_lobbies": lobbies,
        "state_rows": state_rows,
        "candidate_rows": candidate_rows,
    }


def run_greedy_arm(lobbies: int, seed: int) -> Dict:
    return run_replacement_value_arm(lobbies, seed, arm="greedy", policy=greedy_policy)


def run_phase_2j_arm(lobbies: int, seed: int, alpha: float, prior: PersistencePrior) -> Dict:
    def factory(_i: int):
        return policies_for_lobby(alpha, prior, 8)

    return run_replacement_value_arm(
        lobbies,
        seed,
        arm="phase_2j",
        policy_factory=factory,
    )


def _aggregate_state_bucket(bucket: List[Dict]) -> Dict:
    if not bucket:
        return {
            "n_full_board_states": 0,
            "scaling_blocked_upgrade_states": 0,
            "pct_scaling_blocked_upgrade_states": None,
            "p_best_shop_gt_weakest_scaled": None,
            "p_best_shop_gt_weakest_printed": None,
            "median_weakest_board_scaled_raw": None,
            "median_weakest_board_normal_printed_raw": None,
            "median_weakest_board_natural_printed_raw": None,
            "median_weakest_board_printed_raw": None,
            "median_best_shop_printed_raw": None,
            "median_board_inflation_ratio": None,
            "p90_board_inflation_ratio": None,
            "share_weakest_golden": None,
            "mean_gold": None,
            "action_counts": {},
        }
    action_counts = Counter(r.get("actual_action_kind") or "unknown" for r in bucket)
    weakest_scaled = [float(r["weakest_board_current_raw"]) for r in bucket]
    weakest_normal = [
        float(r.get("weakest_board_normal_printed_raw")
              if r.get("weakest_board_normal_printed_raw") is not None
              else r["weakest_board_printed_raw"])
        for r in bucket
    ]
    weakest_natural = [
        float(r.get("weakest_board_natural_printed_raw")
              if r.get("weakest_board_natural_printed_raw") is not None
              else r["weakest_board_printed_raw"])
        for r in bucket
    ]
    best_printed = [float(r["best_shop_printed_raw"]) for r in bucket]
    infl = [
        float(r["weakest_board_inflation_ratio"])
        for r in bucket
        if r.get("weakest_board_inflation_ratio") is not None
    ]
    blocked = sum(1 for r in bucket if r.get("scaling_blocked_upgrade"))
    accept_scaled = sum(1 for r in bucket if r.get("best_shop_gt_weakest_scaled"))
    accept_printed = sum(1 for r in bucket if r.get("best_shop_gt_weakest_printed"))
    golden_n = sum(1 for r in bucket if r.get("weakest_board_golden"))
    return {
        "n_full_board_states": len(bucket),
        "scaling_blocked_upgrade_states": blocked,
        "pct_scaling_blocked_upgrade_states": (blocked / len(bucket)) if bucket else None,
        "p_best_shop_gt_weakest_scaled": (accept_scaled / len(bucket)) if bucket else None,
        "p_best_shop_gt_weakest_printed": (accept_printed / len(bucket)) if bucket else None,
        "median_weakest_board_scaled_raw": _median(weakest_scaled),
        "median_weakest_board_normal_printed_raw": _median(weakest_normal),
        "median_weakest_board_natural_printed_raw": _median(weakest_natural),
        # Alias: natural printed (golden-aware).
        "median_weakest_board_printed_raw": _median(weakest_natural),
        "median_best_shop_printed_raw": _median(best_printed),
        "median_board_inflation_ratio": _median(infl),
        "p90_board_inflation_ratio": _pctl(infl, 0.9),
        "share_weakest_golden": golden_n / len(bucket),
        "mean_gold": _mean([float(r["gold"]) for r in bucket]),
        "action_counts": dict(action_counts),
    }


def aggregate_state_rows(rows: List[Dict]) -> Dict:
    by_turn: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_turn[str(int(row["turn"]))].append(row)

    out: Dict[str, Dict] = {}
    for turn, bucket in sorted(by_turn.items()):
        all_states = _aggregate_state_bucket(bucket)
        nongolden = [r for r in bucket if not r.get("weakest_board_golden")]
        nongolden_states = _aggregate_state_bucket(nongolden)
        out[turn] = {
            **all_states,
            "all_full_board_states": all_states,
            "nongolden_weakest_states": nongolden_states,
        }
    return out


def overall_contamination_headline(state_summary_by_turn: Dict) -> Dict:
    """Weighted headline across instrumented turns for all vs non-golden weakest."""
    def _weighted(prefix: str) -> Dict:
        total = 0
        blocked = 0
        gt_scaled = 0.0
        gt_printed = 0.0
        for turn, row in state_summary_by_turn.items():
            bucket = row.get(prefix) or row
            n = int(bucket.get("n_full_board_states") or 0)
            if n <= 0:
                continue
            total += n
            blocked += int(bucket.get("scaling_blocked_upgrade_states") or 0)
            gt_scaled += float(bucket.get("p_best_shop_gt_weakest_scaled") or 0) * n
            gt_printed += float(bucket.get("p_best_shop_gt_weakest_printed") or 0) * n
        return {
            "n_full_board_states": total,
            "pct_scaling_blocked_upgrade_states": (blocked / total) if total else None,
            "p_best_shop_gt_weakest_scaled": (gt_scaled / total) if total else None,
            "p_best_shop_gt_weakest_printed": (gt_printed / total) if total else None,
        }

    return {
        "all_full_board_states": _weighted("all_full_board_states"),
        "nongolden_weakest_states": _weighted("nongolden_weakest_states"),
    }


def aggregate_candidate_rows(rows: List[Dict]) -> Dict:
    by_turn: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_turn[str(int(row["turn"]))].append(row)

    out: Dict[str, Dict] = {}
    for turn, bucket in sorted(by_turn.items()):
        rejected = [r for r in bucket if not r.get("current_rule_accepts")]
        buckets = Counter(r.get("reject_bucket") or "accepted" for r in rejected)
        out[turn] = {
            "n_candidates": len(bucket),
            "n_rejected_current_rule": len(rejected),
            "reject_bucket_counts": dict(buckets),
            "reject_bucket_shares": {
                k: (v / len(rejected)) if rejected else None for k, v in buckets.items()
            },
            "mean_candidate_shop_build_gain_rejected": _mean(
                [float(r["candidate_shop_build_gain"]) for r in rejected]
            ),
            "share_target_core_rejected": (
                sum(1 for r in rejected if r.get("candidate_target_core")) / len(rejected)
                if rejected else None
            ),
            "share_text_present_rejected": (
                sum(1 for r in rejected if r.get("candidate_text_present")) / len(rejected)
                if rejected else None
            ),
            "share_card2vec_in_vocab_rejected": (
                sum(1 for r in rejected if r.get("candidate_card2vec_in_vocab")) / len(rejected)
                if rejected else None
            ),
        }
    return out


def summarize_arm(raw: Dict) -> Dict:
    state_summary = aggregate_state_rows(raw["state_rows"])
    return {
        "arm": raw["arm"],
        "seed_base": raw["seed_base"],
        "n_lobbies": raw["n_lobbies"],
        "n_full_board_states": len(raw["state_rows"]),
        "n_candidate_records": len(raw["candidate_rows"]),
        "state_summary_by_turn": state_summary,
        "candidate_summary_by_turn": aggregate_candidate_rows(raw["candidate_rows"]),
        "contamination_headline": overall_contamination_headline(state_summary),
    }


def diagnose_contamination(greedy: Dict, phase_2j: Dict) -> Dict:
    def _turn(summary: Dict, t: int, *, nongolden: bool = False) -> Dict:
        row = (summary.get("state_summary_by_turn") or {}).get(str(t)) or {}
        if nongolden:
            return row.get("nongolden_weakest_states") or {}
        return row.get("all_full_board_states") or row

    g10 = _turn(greedy, 10)
    j10 = _turn(phase_2j, 10)
    g10_ng = _turn(greedy, 10, nongolden=True)
    j10_ng = _turn(phase_2j, 10, nongolden=True)

    def _dominant(row: Dict) -> bool:
        return (row.get("pct_scaling_blocked_upgrade_states") or 0) >= 0.35

    finding = "inconclusive"
    next_step = "inspect candidate/state rows manually"
    if _dominant(g10) or _dominant(j10):
        finding = "scaling_contamination_dominant"
        next_step = (
            "Phase 2Q: separate recruit valuation from scaled combat strength "
            "(printed/base replacement space vs abstract combat-strength bridge)."
        )
    elif (
        (g10.get("p_best_shop_gt_weakest_printed") or 0) >= 0.35
        and (g10.get("p_best_shop_gt_weakest_scaled") or 0) <= 0.10
    ):
        finding = "scaling_blocks_most_raw_upgrades"
        next_step = (
            "Phase 2Q: quantify printed-vs-scaled replacement valuation and "
            "test a representation split before adding more card-effect logic."
        )
    else:
        finding = "build_or_effect_signal_nontrivial"
        next_step = (
            "Inspect rejected candidate B-bucket and separate missing synergy/"
            "effect valuation from scaling contamination."
        )

    survives_nongolden = _dominant(g10_ng) or _dominant(j10_ng)
    return {
        "primary_finding": finding,
        "recommended_next_step": next_step,
        "survives_nongolden_weakest_filter": survives_nongolden,
        "t10": {
            "greedy_all": g10,
            "phase_2j_all": j10,
            "greedy_nongolden_weakest": g10_ng,
            "phase_2j_nongolden_weakest": j10_ng,
            # Back-compat aliases used by earlier 2p_v1 readers/tests.
            "greedy": g10,
            "phase_2j": j10,
        },
        "contamination_headline": {
            "greedy": greedy.get("contamination_headline"),
            "phase_2j": phase_2j.get("contamination_headline"),
        },
    }
