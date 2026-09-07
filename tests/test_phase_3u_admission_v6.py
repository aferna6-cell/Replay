import hashlib

from ml.phase_3u_admission_v5 import compute_manifest_sha256
from ml.phase_3u_admission_v6 import (
    compute_source_sha256,
    evaluate_ranking_admission_v6,
    verify_source_artifacts,
)


CAL_BYTES = b"independent calibration observations\nrow-1\nrow-2\n"
EVAL_BYTES = b"independent evaluation observations\nrow-a\nrow-b\n"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _schema():
    return {
        "valid": True,
        "schema_ready": True,
        "row_count": 20,
        "trajectory_count": 5,
        "source_provenance_verified": True,
        "source_reference": "external://phase3u/observations-v1",
        "source_sha256": _sha(EVAL_BYTES),
    }


def _plan(**overrides):
    plan = {
        "minimum_transitions": 10,
        "minimum_trajectories": 3,
        "measurement_error_metric": "body_stat_absolute_error",
        "measurement_error_limit": 1.0,
        "material_effect_metric": "paired_body_stat_error_delta",
        "material_effect_threshold": 2.0,
        "threshold_basis": "independent_calibration",
        "threshold_basis_reference": "external://phase3u/calibration-plan-v1",
        "threshold_basis_method": "double-entry reconstruction error on held-out observed transitions",
        "threshold_basis_sha256": "a" * 64,
        "calibration_source_reference": "external://phase3u/calibration-observations-v1",
        "calibration_source_sha256": _sha(CAL_BYTES),
        "calibration_evidence_overlap_count": 0,
        "overlap_check_reference": "external://phase3u/split-overlap-proof-v1",
        "overlap_check_sha256": "d" * 64,
        "evidence_source_reference": "external://phase3u/observations-v1",
        "evidence_source_sha256": _sha(EVAL_BYTES),
        "frozen_before_candidate_scoring": True,
    }
    plan.update(overrides)
    return plan


def _with_manifest_digest(manifest):
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
    return manifest


def _cal():
    return _with_manifest_digest({
        "source_reference": "external://phase3u/calibration-observations-v1",
        "source_sha256": _sha(CAL_BYTES),
        "manifest_reference": "external://phase3u/calibration-manifest-v1",
        "observation_ids": ["cal-1", "cal-2"],
    })


def _eval():
    return _with_manifest_digest({
        "source_reference": "external://phase3u/observations-v1",
        "source_sha256": _sha(EVAL_BYTES),
        "manifest_reference": "external://phase3u/evidence-manifest-v1",
        "observation_ids": ["eval-1", "eval-2"],
    })


def test_exact_source_artifact_bytes_can_clear_v6_gate():
    out = evaluate_ranking_admission_v6(
        schema_result=_schema(),
        plan=_plan(),
        calibration_manifest=_cal(),
        evidence_manifest=_eval(),
        calibration_source_content=CAL_BYTES,
        evidence_source_content=EVAL_BYTES,
    )
    assert out["ranking_ready"] is True
    assert out["source_artifact_bytes_cryptographically_bound"] is True
    assert out["candidate_scores_examined"] is False


def test_missing_source_artifact_bytes_hold_even_when_v5_inputs_are_valid():
    out = evaluate_ranking_admission_v6(
        schema_result=_schema(),
        plan=_plan(),
        calibration_manifest=_cal(),
        evidence_manifest=_eval(),
        calibration_source_content=None,
        evidence_source_content=EVAL_BYTES,
    )
    assert out["ranking_ready"] is False
    assert "source_artifact_content_missing" in out["blockers"]


def test_calibration_source_tampering_is_recomputed_and_rejected():
    proof = verify_source_artifacts(
        calibration_source_content=CAL_BYTES + b"tampered",
        evidence_source_content=EVAL_BYTES,
        calibration_manifest=_cal(),
        evidence_manifest=_eval(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "calibration_source_content_digest_mismatch"
    assert proof["computed_source_sha256"] != proof["declared_source_sha256"]


def test_evidence_source_tampering_is_recomputed_and_rejected():
    proof = verify_source_artifacts(
        calibration_source_content=CAL_BYTES,
        evidence_source_content=EVAL_BYTES + b"tampered",
        calibration_manifest=_cal(),
        evidence_manifest=_eval(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "evidence_source_content_digest_mismatch"


def test_source_content_must_be_exact_bytes_not_reencoded_text():
    proof = verify_source_artifacts(
        calibration_source_content="not bytes",
        evidence_source_content=EVAL_BYTES,
        calibration_manifest=_cal(),
        evidence_manifest=_eval(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "source_artifact_content_not_bytes"


def test_source_hash_is_byte_exact_and_line_ending_sensitive():
    assert compute_source_sha256(b"a\nb\n") != compute_source_sha256(b"a\r\nb\r\n")
