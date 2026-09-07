"""Phase 3U ranking admission v7: compose evidence admission with parser provenance.

Measurement-only hardening layered on :mod:`ml.phase_3u_admission_v6`.
v6 proves the frozen plan/manifests and exact source artifact bytes satisfy the
ranking-admission contract, but parser reconciliation is a separate boundary.
A source may therefore be byte-bound while the executable parser that produced
its admitted observations is still untrusted.

v7 is the final ranking surface. Reconciliation mappings may be checked for
internal consistency against the frozen source/manifests, but their provenance
booleans are still caller-supplied claims until a reviewed digest-bound loader
constructs the executed parser from the frozen artifact/config. Therefore v7
must fail closed today even when those claims are all true.

No candidate scores are accepted here and no simulator behavior is changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Optional

from ml.phase_3u_admission_v6 import evaluate_ranking_admission_v6 as evaluate_v6

ADMISSION_VERSION = "3u_admission_v7"
EXECUTION_PROVENANCE_GATE_IMPLEMENTED = False
EXECUTION_PROVENANCE_BLOCKER = "digest_bound_parser_loader_not_implemented"


def _validate_reconciliation(
    *,
    label: str,
    result: Optional[Mapping],
    manifest: Optional[Mapping],
) -> Dict:
    """Validate one parser-provenance claim against its frozen manifest."""
    if not result:
        return {"valid": False, "blocker": f"{label}_parser_reconciliation_missing"}
    if not manifest:
        return {"valid": False, "blocker": f"{label}_manifest_missing"}

    if not bool(result.get("source_parser_identity_manifest_bound")):
        return {"valid": False, "blocker": f"{label}_parser_identity_manifest_unbound"}
    if not bool(result.get("parser_artifact_config_bound")):
        return {"valid": False, "blocker": f"{label}_parser_artifact_config_unbound"}
    if not bool(result.get("execution_provenance_bound")):
        return {"valid": False, "blocker": f"{label}_parser_execution_provenance_unbound"}
    if not bool(result.get("ranking_admissible")):
        return {"valid": False, "blocker": f"{label}_parser_not_ranking_admissible"}
    if bool(result.get("candidate_scoring_performed")):
        return {"valid": False, "blocker": f"{label}_parser_candidate_scoring_detected"}

    result_sha = str(result.get("source_sha256", "")).lower()
    manifest_sha = str(manifest.get("source_sha256", "")).lower()
    if not result_sha or result_sha != manifest_sha:
        return {"valid": False, "blocker": f"{label}_parser_source_digest_mismatch"}

    result_ids = result.get("observation_ids")
    manifest_ids = manifest.get("observation_ids")
    if not isinstance(result_ids, Sequence) or isinstance(result_ids, (str, bytes)):
        return {"valid": False, "blocker": f"{label}_parser_observation_ids_invalid"}
    if not isinstance(manifest_ids, Sequence) or isinstance(manifest_ids, (str, bytes)):
        return {"valid": False, "blocker": f"{label}_manifest_observation_ids_invalid"}
    if list(result_ids) != list(manifest_ids):
        return {"valid": False, "blocker": f"{label}_parser_manifest_observation_mismatch"}

    return {
        "valid": True,
        "blocker": None,
        "source_sha256": result_sha,
        "observation_count": len(result_ids),
        "execution_provenance_claimed": True,
        "ranking_admissible_claimed": True,
    }


def evaluate_ranking_admission_v7(
    *,
    schema_result: Mapping,
    plan: Optional[Mapping],
    calibration_manifest: Optional[Mapping],
    evidence_manifest: Optional[Mapping],
    calibration_source_content: Optional[bytes],
    evidence_source_content: Optional[bytes],
    calibration_reconciliation: Optional[Mapping],
    evidence_reconciliation: Optional[Mapping],
) -> Dict:
    """Fail closed until execution provenance is established by a real loader."""
    base = evaluate_v6(
        schema_result=schema_result,
        plan=plan,
        calibration_manifest=calibration_manifest,
        evidence_manifest=evidence_manifest,
        calibration_source_content=calibration_source_content,
        evidence_source_content=evidence_source_content,
    )
    blockers = list(base.get("blockers") or [])

    calibration_parser = _validate_reconciliation(
        label="calibration",
        result=calibration_reconciliation,
        manifest=calibration_manifest,
    )
    evidence_parser = _validate_reconciliation(
        label="evidence",
        result=evidence_reconciliation,
        manifest=evidence_manifest,
    )
    for result in (calibration_parser, evidence_parser):
        if not result["valid"]:
            blockers.append(result["blocker"])

    # Critical scientific boundary: the mappings above are ordinary caller-supplied
    # data. Until a reviewed wrapper loads and executes the parser from the exact
    # digest-bound artifact/config, True booleans inside those mappings cannot prove
    # execution provenance. Keep ranking mechanically closed rather than allowing a
    # synthetic/self-asserted mapping to clear the final gate.
    if not EXECUTION_PROVENANCE_GATE_IMPLEMENTED:
        blockers.append(EXECUTION_PROVENANCE_BLOCKER)

    execution_provenance_verified = (
        EXECUTION_PROVENANCE_GATE_IMPLEMENTED
        and bool(calibration_parser["valid"])
        and bool(evidence_parser["valid"])
    )

    return {
        **base,
        "admission_version": ADMISSION_VERSION,
        "ranking_ready": not blockers,
        "blockers": blockers,
        "calibration_parser_reconciliation": calibration_parser,
        "evidence_parser_reconciliation": evidence_parser,
        "parser_execution_provenance_verified": execution_provenance_verified,
        "candidate_scores_examined": False,
    }
