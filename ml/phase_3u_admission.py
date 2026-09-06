"""Phase 3U ranking admission gate (measurement-only).

This gate deliberately does not invent numeric thresholds. Sample-size, noise,
and material-effect criteria must be frozen from independent calibration or a
prospective power analysis before candidate allocation scores are visible.
The basis must carry machine-checkable provenance, be bound to the immutable
external evidence source admitted by the Phase 3U schema, and prove that the
observations used to calibrate thresholds are disjoint from ranking evidence.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional

ADMISSION_VERSION = "3u_admission_v4"
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
    "calibration_source_reference",
    "calibration_source_sha256",
    "calibration_evidence_overlap_count",
    "overlap_check_reference",
    "overlap_check_sha256",
    "evidence_source_reference",
    "evidence_source_sha256",
    "frozen_before_candidate_scoring",
)
ALLOWED_THRESHOLD_BASES = {
    "independent_calibration",
    "prospective_power_analysis",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.lower()))


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
    if not _valid_digest(digest):
        return {"valid": False, "blocker": "threshold_basis_digest_invalid"}

    if not _nonempty_text(plan["calibration_source_reference"]):
        return {"valid": False, "blocker": "calibration_source_reference_missing"}
    calibration_digest = plan["calibration_source_sha256"]
    if not _valid_digest(calibration_digest):
        return {"valid": False, "blocker": "calibration_source_digest_invalid"}
    if not _nonempty_text(plan["overlap_check_reference"]):
        return {"valid": False, "blocker": "overlap_check_reference_missing"}
    overlap_digest = plan["overlap_check_sha256"]
    if not _valid_digest(overlap_digest):
        return {"valid": False, "blocker": "overlap_check_digest_invalid"}
    overlap_count = plan["calibration_evidence_overlap_count"]
    if isinstance(overlap_count, bool) or not isinstance(overlap_count, int) or overlap_count < 0:
        return {"valid": False, "blocker": "invalid_calibration_evidence_overlap_count"}
    if overlap_count != 0:
        return {"valid": False, "blocker": "calibration_evidence_overlap_detected"}

    if not _nonempty_text(plan["evidence_source_reference"]):
        return {"valid": False, "blocker": "evidence_source_reference_missing"}
    evidence_digest = plan["evidence_source_sha256"]
    if not _valid_digest(evidence_digest):
        return {"valid": False, "blocker": "evidence_source_digest_invalid"}

    calibration_reference = plan["calibration_source_reference"].strip()
    evidence_reference = plan["evidence_source_reference"].strip()
    if calibration_reference == evidence_reference:
        return {"valid": False, "blocker": "calibration_source_not_independent"}
    if calibration_digest.lower() == evidence_digest.lower():
        return {"valid": False, "blocker": "calibration_source_not_independent"}

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
        "calibration_source_reference": calibration_reference,
        "calibration_source_sha256": calibration_digest.lower(),
        "calibration_evidence_overlap_count": overlap_count,
        "overlap_check_reference": plan["overlap_check_reference"].strip(),
        "overlap_check_sha256": overlap_digest.lower(),
        "evidence_source_reference": evidence_reference,
        "evidence_source_sha256": evidence_digest.lower(),
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
    if not bool(schema_result.get("source_provenance_verified")):
        blockers.append("evidence_source_provenance_missing")
    if not plan_result["valid"]:
        blockers.append(plan_result["blocker"])

    if plan_result["valid"]:
        if schema_result.get("source_reference") != plan_result["evidence_source_reference"]:
            blockers.append("evidence_source_reference_mismatch")
        schema_digest = str(schema_result.get("source_sha256", "")).lower()
        if schema_digest != plan_result["evidence_source_sha256"]:
            blockers.append("evidence_source_digest_mismatch")

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
        "calibration_evidence_independence_verified": bool(plan_result["valid"]),
        "evidence_source_binding_verified": bool(plan_result["valid"]) and not any(
            blocker.startswith("evidence_source_") for blocker in blockers
        ),
    }
