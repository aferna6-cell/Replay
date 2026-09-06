"""Phase 3U ranking admission gate (measurement-only).

This gate deliberately does not invent numeric thresholds. Sample-size, noise,
and material-effect criteria must be frozen from independent calibration or a
prospective power analysis before candidate allocation scores are visible.
The basis must also carry machine-checkable provenance so merely labelling a
threshold "independent" cannot authorize ranking.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional

ADMISSION_VERSION = "3u_admission_v2"
REQUIRED_PLAN_FIELDS = (
    "minimum_transitions",
    "minimum_trajectories",
    "measurement_error_metric",
    "measurement_error_limit",
    "material_effect_metric",
    "material_effect_threshold",
    "threshold_basis",
    "threshold_basis_reference",
    "threshold_basis_method",
    "threshold_basis_sha256",
    "frozen_before_candidate_scoring",
)
ALLOWED_THRESHOLD_BASES = {
    "independent_calibration",
    "prospective_power_analysis",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_admission_plan(plan: Optional[Mapping]) -> Dict:
    """Validate a preregistered admission plan without looking at candidate scores."""
    if not plan:
        return {"valid": False, "blocker": "admission_plan_missing"}
    missing = [field for field in REQUIRED_PLAN_FIELDS if field not in plan]
    if missing:
        return {"valid": False, "blocker": "admission_plan_incomplete", "missing": missing}
    if not bool(plan["frozen_before_candidate_scoring"]):
        return {"valid": False, "blocker": "thresholds_not_frozen_prospectively"}
    if plan["threshold_basis"] not in ALLOWED_THRESHOLD_BASES:
        return {"valid": False, "blocker": "threshold_basis_not_independent"}
    if not _nonempty_text(plan["threshold_basis_reference"]):
        return {"valid": False, "blocker": "threshold_basis_reference_missing"}
    if not _nonempty_text(plan["threshold_basis_method"]):
        return {"valid": False, "blocker": "threshold_basis_method_missing"}
    digest = plan["threshold_basis_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
        return {"valid": False, "blocker": "threshold_basis_digest_invalid"}
    for field in ("minimum_transitions", "minimum_trajectories"):
        value = plan[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return {"valid": False, "blocker": f"invalid_{field}"}
    for field in ("measurement_error_limit", "material_effect_threshold"):
        value = plan[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return {"valid": False, "blocker": f"invalid_{field}"}
    return {
        "valid": True,
        "blocker": None,
        "admission_version": ADMISSION_VERSION,
        "threshold_basis_reference": plan["threshold_basis_reference"],
        "threshold_basis_sha256": digest.lower(),
    }


def evaluate_ranking_admission(*, schema_result: Mapping, plan: Optional[Mapping]) -> Dict:
    """Authorize ranking only after schema + prospective admission criteria pass.

    This function accepts no candidate scores by design, preventing thresholds
    from being selected or altered after seeing which allocation rule wins.
    """
    plan_result = validate_admission_plan(plan)
    blockers = []
    if not bool(schema_result.get("valid")):
        blockers.append("schema_invalid")
    if not bool(schema_result.get("schema_ready")):
        blockers.append("conserved_pool_evidence_incomplete")
    if not plan_result["valid"]:
        blockers.append(plan_result["blocker"])

    row_count = int(schema_result.get("row_count", 0))
    trajectory_count = int(schema_result.get("trajectory_count", 0))
    if plan_result["valid"]:
        if row_count < int(plan["minimum_transitions"]):
            blockers.append("minimum_transitions_not_met")
        if trajectory_count < int(plan["minimum_trajectories"]):
            blockers.append("minimum_trajectories_not_met")

    return {
        "admission_version": ADMISSION_VERSION,
        "ranking_ready": not blockers,
        "blockers": blockers,
        "candidate_scores_examined": False,
        "thresholds_require_independent_basis": True,
        "threshold_basis_provenance_verified": bool(plan_result["valid"]),
    }
