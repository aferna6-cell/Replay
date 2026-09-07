"""Phase 3U ranking admission v6: bind source SHA-256 to artifact bytes.

Measurement-only hardening layered on :mod:`ml.phase_3u_admission_v5`.  v5
cryptographically binds observation manifests and recomputes split overlap, but
its ``source_sha256`` values are identifiers supplied by the manifest/plan.
v6 requires the actual immutable calibration and evaluation source artifact
bytes and recomputes those digests before ranking admission can clear.

No candidate scores are accepted here and no simulator behavior is changed.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional

from ml.phase_3u_admission_v5 import evaluate_ranking_admission_v5 as evaluate_v5

ADMISSION_VERSION = "3u_admission_v6"


def compute_source_sha256(content: bytes) -> str:
    """Return SHA-256 of the exact source artifact bytes."""
    if not isinstance(content, bytes):
        raise TypeError("source artifact content must be bytes")
    return hashlib.sha256(content).hexdigest()


def verify_source_artifacts(
    *,
    calibration_source_content: Optional[bytes],
    evidence_source_content: Optional[bytes],
    calibration_manifest: Optional[Mapping],
    evidence_manifest: Optional[Mapping],
) -> Dict:
    """Verify manifest source digests against the exact supplied artifact bytes."""
    if calibration_source_content is None or evidence_source_content is None:
        return {"valid": False, "blocker": "source_artifact_content_missing"}
    if not calibration_manifest or not evidence_manifest:
        return {"valid": False, "blocker": "observation_manifests_missing"}

    try:
        calibration_digest = compute_source_sha256(calibration_source_content)
        evidence_digest = compute_source_sha256(evidence_source_content)
    except TypeError as exc:
        return {"valid": False, "blocker": "source_artifact_content_not_bytes", "detail": str(exc)}

    declared_calibration = str(calibration_manifest.get("source_sha256", "")).lower()
    declared_evidence = str(evidence_manifest.get("source_sha256", "")).lower()
    if calibration_digest != declared_calibration:
        return {
            "valid": False,
            "blocker": "calibration_source_content_digest_mismatch",
            "declared_source_sha256": declared_calibration,
            "computed_source_sha256": calibration_digest,
        }
    if evidence_digest != declared_evidence:
        return {
            "valid": False,
            "blocker": "evidence_source_content_digest_mismatch",
            "declared_source_sha256": declared_evidence,
            "computed_source_sha256": evidence_digest,
        }

    return {
        "valid": True,
        "blocker": None,
        "calibration_source_sha256": calibration_digest,
        "evidence_source_sha256": evidence_digest,
        "source_artifact_bytes_cryptographically_bound": True,
    }


def evaluate_ranking_admission_v6(
    *,
    schema_result: Mapping,
    plan: Optional[Mapping],
    calibration_manifest: Optional[Mapping],
    evidence_manifest: Optional[Mapping],
    calibration_source_content: Optional[bytes],
    evidence_source_content: Optional[bytes],
) -> Dict:
    """Authorize ranking only when v5 and exact source-byte verification pass."""
    base = evaluate_v5(
        schema_result=schema_result,
        plan=plan,
        calibration_manifest=calibration_manifest,
        evidence_manifest=evidence_manifest,
    )
    blockers = list(base.get("blockers") or [])
    source_result = verify_source_artifacts(
        calibration_source_content=calibration_source_content,
        evidence_source_content=evidence_source_content,
        calibration_manifest=calibration_manifest,
        evidence_manifest=evidence_manifest,
    )
    if not source_result["valid"]:
        blockers.append(source_result["blocker"])
    return {
        **base,
        "admission_version": ADMISSION_VERSION,
        "ranking_ready": not blockers,
        "blockers": blockers,
        "source_artifact_verification": source_result,
        "source_artifact_bytes_cryptographically_bound": bool(source_result["valid"]),
        "candidate_scores_examined": False,
    }
