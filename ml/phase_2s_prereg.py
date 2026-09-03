"""Phase 2S preregistration — board-level abstract scaling (spec only).

No simulator behavior. Constants lock the default-OFF representation, frozen
α / residual math, fresh DEV band, and routing gates before any implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

# Proposed toggle — must not exist on BGEnv until a later implementation PR.
FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2s_v0_prereg"
PHASE_2S_SEED = 14200
PHASE_2S_LOBBIES = 500
FROZEN_ALPHA = 0.5

# Confirm remains reserved. 2N–2R DEV bands are consumed. 2S starts above 14199.
FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),
    (11000, 11499),
    (11500, 11699),  # confirmation — reserved
    (11700, 12199),  # 2N
    (12200, 12699),  # 2O
    (12700, 13199),  # 2P
    (13200, 13699),  # 2Q
    (13700, 14199),  # 2R
)

# Primary gates (greedy control vs treatment on 14200–14699).
GATE_REPLACE_RATE_MIN = 0.10
GATE_T10_POST_SCALE_MIN = 0.85
GATE_T10_POST_SCALE_DELTA_FLOOR = -0.10  # treatment − control
GATE_GAME_LENGTH_DELTA_FLOOR = -0.50
GATE_MEAN_COMBAT_LOSS_MAX = 20.0

# Hold stack — 2S must not merge or un-HOLD these.
HOLD_PRS = (29, 33, 34)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    if seed <= 14199:
        raise ValueError(f"Phase 2S DEV seed must be >14199, got {seed}")
    lo, hi = seed, seed + lobbies - 1
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2S seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def evaluate_phase_2s_gates(greedy_cmp: Dict) -> Dict:
    """Apply predeclared 2S gates to a 2R-style control/treatment comparison."""
    d = greedy_cmp.get("deltas") or {}
    t = greedy_cmp.get("treatment") or {}
    replace = t.get("full_board_replace_rate")
    t10 = t.get("post_scale_over_firestone_t10")
    t10_delta = d.get("post_scale_over_firestone_t10")
    length_delta = d.get("mean_game_length")
    mean_loss = t.get("mean_combat_loss_per_replacement")

    gates = {
        "replacement_rate_held": (
            replace is not None and float(replace) >= GATE_REPLACE_RATE_MIN
        ),
        "post_scale_t10_near_firestone": (
            t10 is not None and float(t10) >= GATE_T10_POST_SCALE_MIN
        ),
        "post_scale_t10_not_materially_worse": (
            t10_delta is not None and float(t10_delta) >= GATE_T10_POST_SCALE_DELTA_FLOOR
        ),
        "game_length_not_shortened": (
            length_delta is not None
            and float(length_delta) >= GATE_GAME_LENGTH_DELTA_FLOOR
        ),
        "combat_loss_on_replace_contained": (
            mean_loss is not None and float(mean_loss) <= GATE_MEAN_COMBAT_LOSS_MAX
        ),
    }
    n_pass = sum(1 for v in gates.values() if v)
    if all(gates.values()):
        route = "board_level_scaling_recovers_macro"
    elif gates["replacement_rate_held"] and not gates["post_scale_t10_near_firestone"]:
        route = "representation_insufficient"
    elif not gates["replacement_rate_held"]:
        route = "selection_regressed"
    else:
        route = "inconclusive"
    return {
        "gates": gates,
        "gates_passed": n_pass,
        "gates_total": len(gates),
        "route": route,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_pr_34_hold": True,
        "feature_toggle_default_off": True,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
    }


def diagnose_phase_2s(greedy_cmp: Optional[Dict] = None) -> Dict:
    """Prereg routing stub — no measurement until an implementation PR."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "preregistered_not_run",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2s_seed": PHASE_2S_SEED,
        "phase_2s_lobbies": PHASE_2S_LOBBIES,
        "keep_hold_prs": list(HOLD_PRS),
        "no_merge": True,
    }
    if greedy_cmp is not None:
        out.update(evaluate_phase_2s_gates(greedy_cmp))
    return out
