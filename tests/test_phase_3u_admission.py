from ml.phase_3u_admission import evaluate_ranking_admission, validate_admission_plan


def _schema(rows=20, trajectories=5, ready=True, **overrides):
    out = {
        "valid": True,
        "schema_ready": ready,
        "row_count": rows,
        "trajectory_count": trajectories,
        "source_provenance_verified": True,
        "source_reference": "external://phase3u/observations-v1",
        "source_sha256": "b" * 64,
    }
    out.update(overrides)
    return out


def _plan(**overrides):
    plan = {
        "minimum_transitions": 10,
        "minimum_trajectories": 3,
        "measurement_error_metric": "body_stat_absolute_error",
        "measurement_error_limit": 1.0,
        "material_effect_metric": "paired_body_stat_error_delta",
        "material_effect_threshold": 2.0,
        "threshold_basis": "independent_calibration",
        "threshold_basis_reference": "external://phase3u/calibration-v1",
        "threshold_basis_method": "double-entry reconstruction error on held-out observed transitions",
        "threshold_basis_sha256": "a" * 64,
        "evidence_source_reference": "external://phase3u/observations-v1",
        "evidence_source_sha256": "b" * 64,
        "frozen_before_candidate_scoring": True,
    }
    plan.update(overrides)
    return plan


def test_missing_plan_holds():
    result = evaluate_ranking_admission(schema_result=_schema(), plan=None)
    assert result["ranking_ready"] is False
    assert "admission_plan_missing" in result["blockers"]
    assert result["candidate_scores_examined"] is False


def test_thresholds_must_be_frozen_before_scoring():
    result = validate_admission_plan(_plan(frozen_before_candidate_scoring=False))
    assert result == {"valid": False, "blocker": "thresholds_not_frozen_prospectively"}


def test_threshold_basis_must_be_independent():
    result = validate_admission_plan(_plan(threshold_basis="candidate_score_separation"))
    assert result == {"valid": False, "blocker": "threshold_basis_not_independent"}


def test_threshold_basis_requires_reference_and_method():
    assert validate_admission_plan(_plan(threshold_basis_reference="")) == {
        "valid": False,
        "blocker": "threshold_basis_reference_missing",
    }
    assert validate_admission_plan(_plan(threshold_basis_method="  ")) == {
        "valid": False,
        "blocker": "threshold_basis_method_missing",
    }


def test_threshold_basis_requires_sha256_digest():
    assert validate_admission_plan(_plan(threshold_basis_sha256="candidate-result")) == {
        "valid": False,
        "blocker": "threshold_basis_digest_invalid",
    }


def test_evidence_source_binding_fields_are_required():
    assert validate_admission_plan(_plan(evidence_source_reference="")) == {
        "valid": False,
        "blocker": "evidence_source_reference_missing",
    }
    assert validate_admission_plan(_plan(evidence_source_sha256="bad")) == {
        "valid": False,
        "blocker": "evidence_source_digest_invalid",
    }


def test_incomplete_conserved_pool_evidence_holds():
    result = evaluate_ranking_admission(schema_result=_schema(ready=False), plan=_plan())
    assert result["ranking_ready"] is False
    assert "conserved_pool_evidence_incomplete" in result["blockers"]


def test_minimum_transition_count_is_enforced():
    result = evaluate_ranking_admission(schema_result=_schema(rows=9), plan=_plan())
    assert result["ranking_ready"] is False
    assert "minimum_transitions_not_met" in result["blockers"]


def test_minimum_trajectory_count_is_enforced():
    result = evaluate_ranking_admission(schema_result=_schema(trajectories=2), plan=_plan())
    assert result["ranking_ready"] is False
    assert "minimum_trajectories_not_met" in result["blockers"]


def test_complete_prospectively_frozen_plan_can_admit_ranking():
    result = evaluate_ranking_admission(schema_result=_schema(), plan=_plan())
    assert result["ranking_ready"] is True
    assert result["blockers"] == []
    assert result["candidate_scores_examined"] is False
    assert result["threshold_basis_provenance_verified"] is True
    assert result["evidence_source_binding_verified"] is True


def test_invalid_provenance_keeps_ranking_closed():
    result = evaluate_ranking_admission(
        schema_result=_schema(),
        plan=_plan(threshold_basis_sha256="0" * 63),
    )
    assert result["ranking_ready"] is False
    assert "threshold_basis_digest_invalid" in result["blockers"]
    assert result["threshold_basis_provenance_verified"] is False


def test_missing_schema_source_provenance_keeps_ranking_closed():
    result = evaluate_ranking_admission(
        schema_result=_schema(source_provenance_verified=False),
        plan=_plan(),
    )
    assert result["ranking_ready"] is False
    assert "evidence_source_provenance_missing" in result["blockers"]
    assert result["evidence_source_binding_verified"] is False


def test_evidence_source_reference_or_digest_mismatch_keeps_ranking_closed():
    ref_result = evaluate_ranking_admission(
        schema_result=_schema(source_reference="external://phase3u/observations-v2"),
        plan=_plan(),
    )
    assert ref_result["ranking_ready"] is False
    assert "evidence_source_reference_mismatch" in ref_result["blockers"]
    assert ref_result["evidence_source_binding_verified"] is False

    digest_result = evaluate_ranking_admission(
        schema_result=_schema(source_sha256="c" * 64),
        plan=_plan(),
    )
    assert digest_result["ranking_ready"] is False
    assert "evidence_source_digest_mismatch" in digest_result["blockers"]
    assert digest_result["evidence_source_binding_verified"] is False


def test_nonpositive_numeric_thresholds_are_rejected():
    assert validate_admission_plan(_plan(measurement_error_limit=0))["valid"] is False
    assert validate_admission_plan(_plan(material_effect_threshold=-1))["valid"] is False


def test_sample_requirements_must_be_positive_integers():
    assert validate_admission_plan(_plan(minimum_transitions=0))["valid"] is False
    assert validate_admission_plan(_plan(minimum_trajectories=1.5))["valid"] is False
