from ml.phase_3u_admission_v5 import (
    compute_manifest_sha256,
    evaluate_ranking_admission_v5,
    verify_overlap_manifests,
)


def _schema():
    return {
        "valid": True,
        "schema_ready": True,
        "row_count": 20,
        "trajectory_count": 5,
        "source_provenance_verified": True,
        "source_reference": "external://phase3u/observations-v1",
        "source_sha256": "b" * 64,
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
        "calibration_source_sha256": "c" * 64,
        "calibration_evidence_overlap_count": 0,
        "overlap_check_reference": "external://phase3u/split-overlap-proof-v1",
        "overlap_check_sha256": "d" * 64,
        "evidence_source_reference": "external://phase3u/observations-v1",
        "evidence_source_sha256": "b" * 64,
        "frozen_before_candidate_scoring": True,
    }
    plan.update(overrides)
    return plan


def _with_digest(out):
    out["manifest_sha256"] = compute_manifest_sha256(out)
    return out


def _cal(ids=("cal-1", "cal-2"), **overrides):
    out = {
        "source_reference": "external://phase3u/calibration-observations-v1",
        "source_sha256": "c" * 64,
        "manifest_reference": "external://phase3u/calibration-manifest-v1",
        "observation_ids": list(ids),
    }
    out.update(overrides)
    return _with_digest(out)


def _eval(ids=("eval-1", "eval-2"), **overrides):
    out = {
        "source_reference": "external://phase3u/observations-v1",
        "source_sha256": "b" * 64,
        "manifest_reference": "external://phase3u/evidence-manifest-v1",
        "observation_ids": list(ids),
    }
    out.update(overrides)
    return _with_digest(out)


def test_v5_requires_observation_manifests_even_when_v4_plan_passes():
    out = evaluate_ranking_admission_v5(
        schema_result=_schema(), plan=_plan(), calibration_manifest=None, evidence_manifest=None
    )
    assert out["ranking_ready"] is False
    assert "observation_manifests_missing" in out["blockers"]
    assert out["candidate_scores_examined"] is False


def test_disjoint_source_bound_manifests_can_clear_v5_overlap_gate():
    proof = verify_overlap_manifests(
        calibration_manifest=_cal(), evidence_manifest=_eval(), plan=_plan()
    )
    assert proof["valid"] is True
    assert proof["computed_overlap_count"] == 0
    assert proof["manifest_contents_cryptographically_bound"] is True
    out = evaluate_ranking_admission_v5(
        schema_result=_schema(), plan=_plan(), calibration_manifest=_cal(), evidence_manifest=_eval()
    )
    assert out["ranking_ready"] is True
    assert out["overlap_recomputed_from_observation_ids"] is True


def test_shared_observation_id_is_detected_even_if_plan_declares_zero_overlap():
    proof = verify_overlap_manifests(
        calibration_manifest=_cal(("shared", "cal-2")),
        evidence_manifest=_eval(("shared", "eval-2")),
        plan=_plan(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "declared_overlap_count_mismatch"
    assert proof["computed_overlap_count"] == 1


def test_nonzero_declared_overlap_and_real_overlap_still_holds():
    proof = verify_overlap_manifests(
        calibration_manifest=_cal(("shared",)),
        evidence_manifest=_eval(("shared",)),
        plan=_plan(calibration_evidence_overlap_count=1),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "calibration_evidence_overlap_detected"


def test_manifest_source_binding_is_machine_checked():
    proof = verify_overlap_manifests(
        calibration_manifest=_cal(source_sha256="9" * 64),
        evidence_manifest=_eval(),
        plan=_plan(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "calibration_manifest_source_digest_mismatch"


def test_duplicate_or_blank_observation_ids_are_rejected():
    duplicate = verify_overlap_manifests(
        calibration_manifest=_cal(("cal-1", "cal-1")), evidence_manifest=_eval(), plan=_plan()
    )
    assert duplicate["valid"] is False
    assert duplicate["blocker"] == "invalid_observation_manifest_ids"
    blank = verify_overlap_manifests(
        calibration_manifest=_cal(("",)), evidence_manifest=_eval(), plan=_plan()
    )
    assert blank["valid"] is False
    assert blank["blocker"] == "invalid_observation_manifest_ids"


def test_manifest_digest_is_recomputed_from_actual_contents():
    calibration = _cal()
    calibration["observation_ids"].append("tampered-after-freeze")
    proof = verify_overlap_manifests(
        calibration_manifest=calibration,
        evidence_manifest=_eval(),
        plan=_plan(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "calibration_manifest_content_digest_mismatch"
    assert proof["computed_manifest_sha256"] != proof["declared_manifest_sha256"]


def test_manifest_digest_preserves_observation_order_in_canonical_hash():
    first = _cal(("cal-1", "cal-2"))
    second = _cal(("cal-2", "cal-1"))
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_noncanonical_manifest_content_is_rejected():
    calibration = _cal()
    calibration["unsupported"] = float("nan")
    calibration["manifest_sha256"] = "e" * 64
    proof = verify_overlap_manifests(
        calibration_manifest=calibration,
        evidence_manifest=_eval(),
        plan=_plan(),
    )
    assert proof["valid"] is False
    assert proof["blocker"] == "calibration_manifest_not_canonicalizable"
