"""Phase 2L decision tree — route from availability subfate dominance."""

from __future__ import annotations

from typing import Dict

METHODOLOGY_VERSION = "2l_v1"
DOMINANCE_THRESHOLD = 0.50

BRANCH_MAP = {
    "A1_NOT_IN_LOBBY_POOL": (
        "Phase 2M: pool/card-data / tribe-filtering fidelity"),
    "A2_NEVER_TIER_ELIGIBLE": (
        "Phase 2M: leveling/tempo path — cores locked behind tavern tier"),
    "A3_TIER_ELIGIBLE_ZERO_RAW": (
        "Phase 2M: shop/pool generation — tier-eligible cores never appear raw"),
    "A4_RAW_BUT_ZERO_LEGAL": (
        "Phase 2M: legality/economy/action-mask — do NOT touch the pool; "
        "raw offers exist but buys are masked (gold/hand)"),
}


def evaluate_phase_2l_decision(analysis: Dict) -> Dict:
    n = analysis.get("n_states") or 0
    share = analysis.get("subfate_share_of_never_legal") or {}
    headlines = analysis.get("headlines") or {}

    if n == 0:
        return {
            "decision_branch": "insufficient_sample",
            "recommended_next_step": (
                "No post-assembly states — expand DEV before Phase 2M."),
            "dominant_subfate": None,
            "dominant_share": None,
        }

    ranked = sorted(share.items(), key=lambda x: -x[1])
    top, top_share = ranked[0] if ranked else (None, 0.0)

    if top_share > DOMINANCE_THRESHOLD and top in BRANCH_MAP:
        return {
            "decision_branch": top.lower(),
            "recommended_next_step": BRANCH_MAP[top],
            "dominant_subfate": top,
            "dominant_share": round(top_share, 4),
            "n_states": n,
            "headlines": headlines,
        }

    return {
        "decision_branch": "mixed_availability_modes",
        "recommended_next_step": (
            "Mixed availability subfates — expand DEV; do not implement Phase 2M "
            "from a weak story. Compare zero-raw vs raw-but-illegal headlines."),
        "dominant_subfate": top,
        "dominant_share": round(top_share or 0.0, 4),
        "n_states": n,
        "cause_distribution": dict(share),
        "headlines": headlines,
    }
