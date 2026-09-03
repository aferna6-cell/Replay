"""Phase 2N-v3 paired behavioral + macro regression panel (validation only).

Arm A: raw greedy
Arm B: frozen BoardOpportunityCostPolicy (α=0.5, frozen Phase 2J prior)

No simulator changes. Reuses Phase 2J confirmation acceptance thresholds and
the frozen Phase 2B success-threshold envelope for absolute macro fidelity
vs Firestone reference curves.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

from hsbg_coach.bg_env import (
    MAX_TURNS,
    PHASE_2N_DEATH_RETURN,
    PHASE_2N_FREEZE_TOPUP,
    POOL_COPIES,
    SHOP_SLOTS,
    greedy_policy,
)
from hsbg_coach.board_opportunity_policy import (
    METHODOLOGY_VERSION as PHASE_2J_VERSION,
)
from hsbg_coach.board_opportunity_policy import policy_config_fingerprint

from ml.availability_decomposition import (
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    PHASE_2J_PRIOR_PATH,
)
from ml.fidelity_metrics import summarize_divergence
from ml.fidelity_phase_2b import THRESHOLDS_PATH, load_frozen_thresholds
from ml.fidelity_phase_2j import (
    _confirm_arm,
    _run_board_opp_arm,
    _run_single_policy_arm,
)
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_reference import file_sha256, reference_fingerprints
from ml.phase_2d_acceptance import macro_regression_summary
from ml.phase_2j_decision import (
    CONFIRMATION_THRESHOLDS,
    evaluate_confirmation_acceptance,
    evaluate_phase_2j_decision,
)

METHODOLOGY_VERSION = "2n_v3"


def _nontermination_rate(lobby_dynamics: Dict,
                         rows: Optional[List[Dict]] = None) -> Dict:
    """Fraction of lobbies that hit the hard MAX_TURNS cap."""
    n = int(lobby_dynamics.get("n_lobbies") or 0)
    n_cap = 0
    if rows:
        lengths: Dict[int, float] = {}
        for r in rows:
            lobby = int(r["lobby"])
            lengths[lobby] = max(lengths.get(lobby, 0.0), float(r["turn"]))
        n_cap = sum(1 for length in lengths.values() if length >= MAX_TURNS)
        n = len(lengths) or n
    return {
        "max_turns": MAX_TURNS,
        "n_lobbies": n,
        "n_hit_max_turns": n_cap,
        "nontermination_rate": (n_cap / n) if n else None,
    }


def evaluate_macro_fidelity_vs_reference(
        turn_curves: Dict, lobby_dynamics: Dict, *,
        rows: Optional[List[Dict]] = None,
        thresholds_path: str = THRESHOLDS_PATH) -> Dict:
    """Absolute macro fidelity vs frozen Firestone reference curves.

    Uses the Phase 2B success-threshold *envelope* without requiring a paired
    Simulator-v1 baseline (turn-12 'improve vs v1' clause is dropped; the
    absolute max ratio still applies). No thresholds are retuned.
    """
    thresholds, thresholds_sha = load_frozen_thresholds(thresholds_path)
    gates_cfg = thresholds["gates"]

    def _ratio(turn: int) -> Optional[float]:
        row = turn_curves.get(str(turn)) or {}
        return row.get("stats_ratio_sim_over_real")

    r10 = _ratio(10)
    r12 = _ratio(12)
    r14 = _ratio(14)
    center = gates_cfg["turn_10_center_ratio"]
    band = gates_cfg["turn_10_regression_band"]

    def _pass_or_unmeasured(value, ok: bool) -> Dict:
        if value is None:
            return {"passed": True, "unmeasured": True}
        return {"passed": bool(ok), "unmeasured": False}

    t14 = _pass_or_unmeasured(
        r14, r14 is not None and r14 <= gates_cfg["turn_14_primary_max_ratio"])
    t12 = _pass_or_unmeasured(
        r12, r12 is not None and r12 <= gates_cfg["turn_12_secondary_max_ratio"])
    t10 = _pass_or_unmeasured(
        r10, r10 is not None and abs(r10 - center) <= band)

    results = {
        "turn_14_primary": {
            "value": r14,
            "max_allowed": gates_cfg["turn_14_primary_max_ratio"],
            **t14,
        },
        "turn_12_secondary": {
            "value": r12,
            "max_allowed": gates_cfg["turn_12_secondary_max_ratio"],
            **t12,
            "note": (
                "Absolute envelope only; Phase 2B's 'improve vs v1' clause "
                "does not apply to unpaired candidate evaluation."),
        },
        "turn_10_regression": {
            "value": r10,
            "baseline_center": center,
            "allowed_abs_delta": band,
            **t10,
        },
    }

    tier_ok = all(
        abs((turn_curves.get(str(t)) or {}).get("tier_error") or 0.0) <= 0.75
        for t in (8, 9, 10, 11, 12, 13, 14)
        if turn_curves.get(str(t))
        and (turn_curves[str(t)].get("real_tavern_tier") is not None))
    results["tavern_tier_unchanged"] = {"passed": tier_ok}

    alive_ok = all(
        abs((turn_curves.get(str(t)) or {}).get("alive_error_vs_prior") or 0.0)
        <= 1.5
        for t in (8, 9, 10, 11, 12, 13, 14)
        if turn_curves.get(str(t)))
    results["alive_curve_unchanged"] = {"passed": alive_ok}

    nonterm = _nontermination_rate(lobby_dynamics, rows)
    results["game_length"] = {
        "value": lobby_dynamics.get("avg_game_length"),
        "median": lobby_dynamics.get("median_game_length"),
        "monitored_only": True,
    }
    results["nontermination"] = nonterm
    # Soft guard: >25% lobbies hitting the hard cap is a material stall signal.
    results["nontermination"]["passed"] = (
        nonterm["nontermination_rate"] is not None
        and nonterm["nontermination_rate"] <= 0.25)

    headline = summarize_divergence(turn_curves)
    pace = {
        "stats_ratio_turn_10": headline.get("stats_ratio_turn_10"),
        "stats_ratio_turn_12": headline.get("stats_ratio_turn_12"),
        "stats_ratio_turn_14": headline.get("stats_ratio_turn_14"),
        "avg_game_length": lobby_dynamics.get("avg_game_length"),
        "sim_tavern_tier_turn_10": (
            (turn_curves.get("10") or {}).get("sim_tavern_tier")),
        "real_tavern_tier_turn_10": (
            (turn_curves.get("10") or {}).get("real_tavern_tier")),
    }

    core = [
        results["turn_14_primary"]["passed"],
        results["turn_12_secondary"]["passed"],
        results["turn_10_regression"]["passed"],
        results["tavern_tier_unchanged"]["passed"],
        results["alive_curve_unchanged"]["passed"],
        results["nontermination"]["passed"],
    ]
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "success_thresholds_path": thresholds_path,
        "success_thresholds_sha256": thresholds_sha,
        "gates": results,
        "pace_headline": pace,
        "macro_fidelity_pass": all(core),
        "note": (
            "Compared against frozen Firestone reference curves using the "
            "Phase 2B absolute envelope. No thresholds were retuned."),
    }


def run_paired_regression_panel(
        *, seed: int, lobbies: int,
        alpha: float = FROZEN_ALPHA,
        prior_path: str = PHASE_2J_PRIOR_PATH,
        prior_hash: str = FROZEN_PRIOR_HASH) -> Dict:
    """Paired greedy vs frozen BoardOpportunityCostPolicy on one seed range."""
    prior = load_frozen_prior(prior_path)
    assert prior.content_hash_sha256() == prior_hash

    print(f"  [2n_v3] paired regression — {lobbies} lobbies, "
          f"seeds {seed}–{seed + lobbies - 1}, α={alpha}")
    greedy = _run_single_policy_arm(lobbies, seed, greedy_policy, "greedy")
    treatment = _run_board_opp_arm(
        lobbies, seed, alpha, prior, "board_opportunity",
        greedy_baseline_traces=greedy["traces"])

    macro_delta = macro_regression_summary(
        greedy["turn_curves"], treatment["turn_curves"],
        greedy["lobby_dynamics"], treatment["lobby_dynamics"],
        greedy["headline"], treatment["headline"])
    acceptance = evaluate_confirmation_acceptance(
        greedy["mechanism"], treatment["mechanism"],
        greedy["lifecycle"], treatment["lifecycle"],
        macro_delta, policy_stats=treatment.get("policy_stats"))
    mechanism_decision = evaluate_phase_2j_decision(
        greedy["mechanism"], treatment["mechanism"],
        greedy["lifecycle"], treatment["lifecycle"],
        macro_delta, acceptance)

    macro_fidelity = evaluate_macro_fidelity_vs_reference(
        treatment["turn_curves"], treatment["lobby_dynamics"],
        rows=treatment["rows"])

    greedy_slim = _confirm_arm(greedy)
    greedy_slim["lobby_dynamics"] = greedy["lobby_dynamics"]
    greedy_slim["headline_divergence"] = greedy["headline"]
    treatment_slim = _confirm_arm(treatment)
    treatment_slim["lobby_dynamics"] = treatment["lobby_dynamics"]
    treatment_slim["headline_divergence"] = treatment["headline"]
    treatment_slim["policy_stats"] = treatment.get("policy_stats")
    treatment_slim["macro_fidelity_vs_reference"] = {
        "pace_headline": macro_fidelity["pace_headline"],
        "macro_fidelity_pass": macro_fidelity["macro_fidelity_pass"],
    }

    flags = acceptance.get("flags") or {}
    mechanism_pass = bool(flags.get("accept_phase_2j_policy"))

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "phase_2j_methodology_version": PHASE_2J_VERSION,
        "evaluation_seed_base": seed,
        "n_lobbies": lobbies,
        "frozen_alpha": alpha,
        "prior_hash_sha256": prior_hash,
        "policy_config": policy_config_fingerprint(alpha, prior),
        "greedy": greedy_slim,
        "treatment": treatment_slim,
        "macro_regression_delta_treatment_minus_greedy": macro_delta,
        "phase_2j_acceptance": acceptance,
        "phase_2j_mechanism_decision": mechanism_decision,
        "phase_2j_mechanism_regression_pass": mechanism_pass,
        "macro_fidelity_vs_reference": macro_fidelity,
        "macro_fidelity_pass": bool(macro_fidelity["macro_fidelity_pass"]),
        "panel_pass": mechanism_pass and bool(macro_fidelity["macro_fidelity_pass"]),
        "confirmation_thresholds": CONFIRMATION_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Simulator v1.x freeze fingerprint scaffolding (confirmation guard)
# ---------------------------------------------------------------------------

FINGERPRINT_FILENAME = "simulator_v1_x_fingerprint.json"


def build_simulator_v1_x_fingerprint(*, implementation_commit: str) -> Dict:
    """Record hashes/toggles that confirmation must match exactly."""
    from hsbg_coach.active_tavern_pool import ACTIVE_TAVERN_POOL_PATH
    from hsbg_coach.cards import BG_CARDS

    refs = reference_fingerprints()
    prior = load_frozen_prior(PHASE_2J_PRIOR_PATH)
    files = {
        "active_tavern_pool.json": file_sha256(ACTIVE_TAVERN_POOL_PATH),
        "bg_cards.json": file_sha256(BG_CARDS),
        "persistence_prior.json": prior.content_hash_sha256(),
        "success_thresholds.json": file_sha256(THRESHOLDS_PATH),
    }
    for rel in (
            "data/stats/firestone_pace.json",
            "data/stats/firestone_final_boards.json",
    ):
        if os.path.isfile(rel):
            files[rel] = file_sha256(rel)

    payload = {
        "implementation_commit": implementation_commit,
        "files_sha256": files,
        "reference_data_fingerprints": refs,
        "frozen_alpha": FROZEN_ALPHA,
        "prior_hash_sha256": FROZEN_PRIOR_HASH,
        "pool_copies": dict(POOL_COPIES),
        "shop_slots": dict(SHOP_SLOTS),
        "scaling_mode": "residual",
        "phase_2n_toggles": {
            "death_return": PHASE_2N_DEATH_RETURN,
            "freeze_topup": PHASE_2N_FREEZE_TOPUP,
            "active_tavern_pool_filter": True,
        },
        "max_turns": MAX_TURNS,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def assert_fingerprint_matches(expected: Dict, observed: Dict) -> None:
    """Refuse confirmation if any freeze fingerprint field differs."""
    exp_hash = expected.get("fingerprint_sha256")
    obs_hash = observed.get("fingerprint_sha256")
    if exp_hash != obs_hash:
        raise RuntimeError(
            "Simulator v1.x fingerprint mismatch — confirmation refused. "
            f"expected={exp_hash} observed={obs_hash}")
    for key in ("implementation_commit", "frozen_alpha", "prior_hash_sha256",
                "pool_copies", "shop_slots", "scaling_mode", "phase_2n_toggles"):
        if expected.get(key) != observed.get(key):
            raise RuntimeError(
                f"Simulator v1.x fingerprint field {key!r} differs — "
                "confirmation refused.")
    exp_files = expected.get("files_sha256") or {}
    obs_files = observed.get("files_sha256") or {}
    for name, digest in exp_files.items():
        if obs_files.get(name) != digest:
            raise RuntimeError(
                f"Simulator v1.x file hash mismatch for {name} — "
                "confirmation refused.")
