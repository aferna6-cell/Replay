"""Phase 3U ranking admission v5: executable split-overlap verification.

Measurement-only hardening layered on :mod:`ml.phase_3u_admission`.  v4 requires
an overlap-proof reference/digest and a declared zero overlap count.  v5 does
not trust that count alone: it recomputes calibration/evaluation overlap from
stable observation identities carried by immutable source-bound manifests.

No candidate scores are accepted here and no simulator behavior is changed.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional

from ml.phase_3u_admission import evaluate_ranking_admission as evaluate_v4

ADMISSION_VERSION = "3u_admission_v5"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.lower()))


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _stable_ids(values: Iterable[object]) -> tuple[str, ...]:
    ids = tuple(str(value).strip() for value in values)
    if not ids or any(not value for value in ids):
        raise ValueError("observation manifest IDs must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("observation manifest IDs must be unique")
    return ids


def verify_overlap_manifests(
    *,
    calibration_manifest: Optional[Mapping],
    evidence_manifest: Optional[Mapping],
    plan: Mapping,
) -> Dict:
    """Recompute calibration/evaluation overlap from immutable observation IDs.

    Each manifest must identify the exact source reference/digest already frozen
    in the admission plan and expose unique stable ``observation_ids``.  The
    verifier computes set intersection directly; a declared overlap count is
    only accepted when it equals that independently recomputed count.
    """
    if not calibration_manifest or not evidence_manifest:
        return {"valid": False, "blocker": "observation_manifests_missing"}

    required = ("source_reference", "source_sha256", "manifest_reference", "manifest_sha256", "observation_ids")
    for label, manifest in (("calibration", calibration_manifest), ("evidence", evidence_manifest)):
        missing = [field for field in required if field not in manifest]
        if missing:
            return {"valid": False, "blocker": f"{label}_manifest_incomplete", "missing": missing}
        if not _nonempty_text(manifest["manifest_reference"]):
            return {"valid": False, "blocker": f"{label}_manifest_reference_missing"}
        if not _valid_digest(manifest["manifest_sha256"]):
            return {"valid": False, "blocker": f"{label}_manifest_digest_invalid"}
        if not _valid_digest(manifest["source_sha256"]):
            return {"valid": False, "blocker": f"{label}_manifest_source_digest_invalid"}

    if calibration_manifest["source_reference"] != plan.get("calibration_source_reference"):
        return {"valid": False, "blocker": "calibration_manifest_source_reference_mismatch"}
    if str(calibration_manifest["source_sha256"]).lower() != str(plan.get("calibration_source_sha256", "")).lower():
        return {"valid": False, "blocker": "calibration_manifest_source_digest_mismatch"}
    if evidence_manifest["source_reference"] != plan.get("evidence_source_reference"):
        return {"valid": False, "blocker": "evidence_manifest_source_reference_mismatch"}
    if str(evidence_manifest["source_sha256"]).lower() != str(plan.get("evidence_source_sha256", "")).lower():
        return {"valid": False, "blocker": "evidence_manifest_source_digest_mismatch"}

    try:
        calibration_ids = _stable_ids(calibration_manifest["observation_ids"])
        evidence_ids = _stable_ids(evidence_manifest["observation_ids"])
    except ValueError as exc:
        return {"valid": False, "blocker": "invalid_observation_manifest_ids", "detail": str(exc)}

    overlap = sorted(set(calibration_ids) & set(evidence_ids))
    declared = plan.get("calibration_evidence_overlap_count")
    if declared != len(overlap):
        return {
            "valid": False,
            "blocker": "declared_overlap_count_mismatch",
            "declared_overlap_count": declared,
            "computed_overlap_count": len(overlap),
        }
    if overlap:
        return {
            "valid": False,
            "blocker": "calibration_evidence_overlap_detected",
            "computed_overlap_count": len(overlap),
            "overlap_ids": overlap[:20],
        }
    return {
        "valid": True,
        "blocker": None,
        "computed_overlap_count": 0,
        "calibration_observation_count": len(calibration_ids),
        "evidence_observation_count": len(evidence_ids),
        "calibration_manifest_reference": calibration_manifest["manifest_reference"].strip(),
        "calibration_manifest_sha256": calibration_manifest["manifest_sha256"].lower(),
        "evidence_manifest_reference": evidence_manifest["manifest_reference"].strip(),
        "evidence_manifest_sha256": evidence_manifest["manifest_sha256"].lower(),
    }


def evaluate_ranking_admission_v5(
    *,
    schema_result: Mapping,
    plan: Optional[Mapping],
    calibration_manifest: Optional[Mapping],
    evidence_manifest: Optional[Mapping],
) -> Dict:
    """Authorize ranking only when v4 passes and manifest overlap is recomputed."""
    base = evaluate_v4(schema_result=schema_result, plan=plan)
    blockers = list(base.get("blockers") or [])
    if plan:
        manifest_result = verify_overlap_manifests(
            calibration_manifest=calibration_manifest,
            evidence_manifest=evidence_manifest,
            plan=plan,
        )
    else:
        manifest_result = {"valid": False, "blocker": "observation_manifests_missing"}
    if not manifest_result["valid"]:
        blockers.append(manifest_result["blocker"])
    return {
        **base,
        "admission_version": ADMISSION_VERSION,
        "ranking_ready": not blockers,
        "blockers": blockers,
        "overlap_recomputed_from_observation_ids": bool(manifest_result["valid"]),
        "manifest_overlap_verification": manifest_result,
        "candidate_scores_examined": False,
    }
