import hashlib

from ml.phase_3u_admission_v5 import compute_manifest_sha256
from ml.phase_3u_admission_v7 import evaluate_ranking_admission_v7


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


def _plan():
    return {
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


def _trusted_reconciliation(manifest):
    return {
        "source_parser_identity_manifest_bound": True,
        "parser_artifact_config_bound": True,
        "execution_provenance_bound": True,
        "ranking_admissible": True,
        "candidate_scoring_performed": False,
        "source_sha256": manifest["source_sha256"],
        "observation_ids": list(manifest["observation_ids"]),
    }


def _evaluate(cal_reconciliation, eval_reconciliation, cal=None, evidence=None):
    cal = cal or _cal()
    evidence = evidence or _eval()
    return evaluate_ranking_admission_v7(
        schema_result=_schema(),
        plan=_plan(),
        calibration_manifest=cal,
        evidence_manifest=evidence,
        calibration_source_content=CAL_BYTES,
        evidence_source_content=EVAL_BYTES,
        calibration_reconciliation=cal_reconciliation,
        evidence_reconciliation=eval_reconciliation,
    )


def test_v6_ready_inputs_still_hold_without_parser_execution_provenance():
    cal = _cal()
    evidence = _eval()
    untrusted = {
        **_trusted_reconciliation(cal),
        "execution_provenance_bound": False,
        "ranking_admissible": False,
        "ranking_block_reason": "parser_callable_not_loaded_from_digest_bound_artifact",
    }
    out = _evaluate(untrusted, _trusted_reconciliation(evidence), cal, evidence)
    assert out["ranking_ready"] is False
    assert "calibration_parser_execution_provenance_unbound" in out["blockers"]
    assert out["parser_execution_provenance_verified"] is False
    assert out["candidate_scores_examined"] is False


def test_missing_reconciliation_holds_even_when_v6_would_clear():
    evidence = _eval()
    out = _evaluate(None, _trusted_reconciliation(evidence))
    assert out["ranking_ready"] is False
    assert "calibration_parser_reconciliation_missing" in out["blockers"]


def test_reconciliation_source_digest_must_match_frozen_manifest():
    cal = _cal()
    evidence = _eval()
    tampered = {**_trusted_reconciliation(cal), "source_sha256": "0" * 64}
    out = _evaluate(tampered, _trusted_reconciliation(evidence), cal, evidence)
    assert out["ranking_ready"] is False
    assert "calibration_parser_source_digest_mismatch" in out["blockers"]


def test_reconciliation_observation_order_must_match_frozen_manifest():
    cal = _cal()
    evidence = _eval()
    reordered = {**_trusted_reconciliation(evidence), "observation_ids": ["eval-2", "eval-1"]}
    out = _evaluate(_trusted_reconciliation(cal), reordered, cal, evidence)
    assert out["ranking_ready"] is False
    assert "evidence_parser_manifest_observation_mismatch" in out["blockers"]


def test_candidate_scoring_inside_parser_provenance_gate_is_rejected():
    cal = _cal()
    evidence = _eval()
    contaminated = {**_trusted_reconciliation(cal), "candidate_scoring_performed": True}
    out = _evaluate(contaminated, _trusted_reconciliation(evidence), cal, evidence)
    assert out["ranking_ready"] is False
    assert "calibration_parser_candidate_scoring_detected" in out["blockers"]


def test_only_fully_bound_reconciliations_can_clear_final_composition():
    cal = _cal()
    evidence = _eval()
    out = _evaluate(_trusted_reconciliation(cal), _trusted_reconciliation(evidence), cal, evidence)
    assert out["ranking_ready"] is True
    assert out["admission_version"] == "3u_admission_v7"
    assert out["parser_execution_provenance_verified"] is True
    assert out["candidate_scores_examined"] is False
