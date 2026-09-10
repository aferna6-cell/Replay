"""Fail-closed validation for Phase 3U external-reference intake manifests.

This module validates acquisition/provenance only. It MUST NOT inspect Replay
parser output, board reconstructions, model scores, simulator output, or held-out
confirmation data. The contract mirrors the frozen independent-reference rules
in GitHub issue #67.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_SPLITS = {"calibration", "evaluation"}
_ALLOWED_CHECKPOINT_KINDS = {"board_mutation", "recruit_end", "combat_start"}
_REQUIRED_LOBBY_FIELDS = (
    "lobby_id",
    "split",
    "raw_sha256",
    "reference_sha256",
    "checkpoint_sha256",
    "sync_sha256",
    "session_id",
    "player_id",
    "patch",
    "platform",
    "privacy_reviewed",
    "independent_reference",
    "checkpoints",
)
_REQUIRED_CHECKPOINT_FIELDS = (
    "checkpoint_id",
    "turn",
    "kind",
    "reference_timestamp",
    "sync_within_tolerance",
    "simultaneous_board",
    "all_required_fields_resolved",
    "replay_derived_repair",
)


def _failure(code: str, location: str, detail: str) -> Dict[str, str]:
    return {"code": code, "location": location, "detail": detail}


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def validate_manifest(
    manifest: Mapping[str, Any], *, require_full_corpus: bool = False
) -> List[Dict[str, str]]:
    """Return fail-closed intake failures; an empty list means acquisition QA passes.

    ``require_full_corpus=False`` is for a prospective dry run. It validates the
    same invariants but does not require 50 calibration + 200 evaluation lobbies.
    ``require_full_corpus=True`` additionally freezes those exact split counts.
    """

    failures: List[Dict[str, str]] = []

    if manifest.get("protocol_version") != "phase3u-external-reference-v1":
        failures.append(
            _failure("PROTOCOL_VERSION", "manifest", "unexpected protocol_version")
        )
    if manifest.get("simulator_seed_budget") != 0:
        failures.append(
            _failure("SEED_BUDGET", "manifest", "simulator_seed_budget must be 0")
        )
    if manifest.get("confirmation_11500_11699_consumed") is not False:
        failures.append(
            _failure(
                "CONFIRMATION_SEEDS",
                "manifest",
                "reserved confirmation seeds 11500-11699 must remain unused",
            )
        )

    lobbies = manifest.get("lobbies")
    if not isinstance(lobbies, list):
        return failures + [_failure("LOBBIES_MISSING", "manifest", "lobbies must be a list")]

    split_counts = defaultdict(int)
    raw_shas: List[str] = []
    reference_shas: List[str] = []
    checkpoint_shas: List[str] = []
    sync_shas: List[str] = []
    split_by_session: defaultdict[str, set[str]] = defaultdict(set)
    split_by_player: defaultdict[str, set[str]] = defaultdict(set)
    lobby_ids: List[str] = []

    for i, lobby in enumerate(lobbies):
        loc = f"lobbies[{i}]"
        if not isinstance(lobby, Mapping):
            failures.append(_failure("LOBBY_TYPE", loc, "lobby entry must be an object"))
            continue

        for field in _REQUIRED_LOBBY_FIELDS:
            if field not in lobby:
                failures.append(_failure("LOBBY_FIELD_MISSING", loc, field))

        split = lobby.get("split")
        if split not in _ALLOWED_SPLITS:
            failures.append(_failure("SPLIT_INVALID", loc, f"invalid split: {split!r}"))
        else:
            split_counts[split] += 1

        lobby_id = lobby.get("lobby_id")
        if isinstance(lobby_id, str) and lobby_id:
            lobby_ids.append(lobby_id)
        else:
            failures.append(_failure("LOBBY_ID", loc, "lobby_id must be non-empty"))

        for field, bucket in (
            ("raw_sha256", raw_shas),
            ("reference_sha256", reference_shas),
            ("checkpoint_sha256", checkpoint_shas),
            ("sync_sha256", sync_shas),
        ):
            value = lobby.get(field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                failures.append(_failure("SHA256_INVALID", loc, field))
            else:
                bucket.append(value)

        if lobby.get("privacy_reviewed") is not True:
            failures.append(_failure("PRIVACY_REVIEW", loc, "privacy_reviewed must be true"))
        if lobby.get("independent_reference") is not True:
            failures.append(
                _failure(
                    "REFERENCE_NOT_INDEPENDENT",
                    loc,
                    "reference must be generated independently of Replay parser/state",
                )
            )

        session_id = lobby.get("session_id")
        player_id = lobby.get("player_id")
        if isinstance(session_id, str) and session_id and split in _ALLOWED_SPLITS:
            split_by_session[session_id].add(split)
        if isinstance(player_id, str) and player_id and split in _ALLOWED_SPLITS:
            split_by_player[player_id].add(split)

        checkpoints = lobby.get("checkpoints")
        if not isinstance(checkpoints, list):
            failures.append(_failure("CHECKPOINTS_MISSING", loc, "checkpoints must be a list"))
            continue
        if not checkpoints:
            failures.append(_failure("CHECKPOINTS_EMPTY", loc, "independent observer must enumerate D_g"))

        checkpoint_ids: List[str] = []
        for j, checkpoint in enumerate(checkpoints):
            c_loc = f"{loc}.checkpoints[{j}]"
            if not isinstance(checkpoint, Mapping):
                failures.append(_failure("CHECKPOINT_TYPE", c_loc, "checkpoint must be an object"))
                continue
            for field in _REQUIRED_CHECKPOINT_FIELDS:
                if field not in checkpoint:
                    failures.append(_failure("CHECKPOINT_FIELD_MISSING", c_loc, field))

            checkpoint_id = checkpoint.get("checkpoint_id")
            if isinstance(checkpoint_id, str) and checkpoint_id:
                checkpoint_ids.append(checkpoint_id)
            else:
                failures.append(_failure("CHECKPOINT_ID", c_loc, "checkpoint_id must be non-empty"))

            turn = checkpoint.get("turn")
            if not isinstance(turn, int) or isinstance(turn, bool) or not 5 <= turn <= 10:
                failures.append(_failure("TURN_RANGE", c_loc, "turn must be integer T5-T10"))
            if checkpoint.get("kind") not in _ALLOWED_CHECKPOINT_KINDS:
                failures.append(_failure("CHECKPOINT_KIND", c_loc, "unsupported checkpoint kind"))
            if not isinstance(checkpoint.get("reference_timestamp"), (int, float)):
                failures.append(_failure("REFERENCE_TIME", c_loc, "reference_timestamp must be numeric"))
            if checkpoint.get("sync_within_tolerance") is not True:
                failures.append(_failure("SYNC_AMBIGUOUS", c_loc, "checkpoint is not uniquely aligned"))
            if checkpoint.get("simultaneous_board") is not True:
                failures.append(_failure("NON_SIMULTANEOUS", c_loc, "board labels do not describe one state"))
            if checkpoint.get("all_required_fields_resolved") is not True:
                failures.append(
                    _failure(
                        "REFERENCE_FIELD_UNRESOLVED",
                        c_loc,
                        "required identity/controller/position/ATK/HEALTH is unresolved",
                    )
                )
            if checkpoint.get("replay_derived_repair") is not False:
                failures.append(
                    _failure(
                        "REPLAY_DERIVED_REPAIR",
                        c_loc,
                        "reference/checkpoint may not be repaired from Replay-derived state",
                    )
                )

        for duplicate in sorted(_duplicates(checkpoint_ids)):
            failures.append(_failure("CHECKPOINT_DUPLICATE", loc, duplicate))

    for duplicate in sorted(_duplicates(lobby_ids)):
        failures.append(_failure("LOBBY_DUPLICATE", "manifest", duplicate))
    for label, values in (
        ("RAW_SHA_DUPLICATE", raw_shas),
        ("REFERENCE_SHA_DUPLICATE", reference_shas),
        ("CHECKPOINT_SHA_DUPLICATE", checkpoint_shas),
        ("SYNC_SHA_DUPLICATE", sync_shas),
    ):
        for duplicate in sorted(_duplicates(values)):
            failures.append(_failure(label, "manifest", duplicate))

    for session_id, splits in split_by_session.items():
        if len(splits) > 1:
            failures.append(_failure("SESSION_SPLIT_LEAKAGE", "manifest", session_id))
    for player_id, splits in split_by_player.items():
        if len(splits) > 1:
            failures.append(_failure("PLAYER_SPLIT_LEAKAGE", "manifest", player_id))

    if require_full_corpus:
        if split_counts["calibration"] != 50:
            failures.append(
                _failure(
                    "CALIBRATION_COUNT",
                    "manifest",
                    f"expected exactly 50 calibration lobbies; got {split_counts['calibration']}",
                )
            )
        if split_counts["evaluation"] != 200:
            failures.append(
                _failure(
                    "EVALUATION_COUNT",
                    "manifest",
                    f"expected exactly 200 evaluation lobbies; got {split_counts['evaluation']}",
                )
            )

    return failures


def accepted(manifest: Mapping[str, Any], *, require_full_corpus: bool = False) -> bool:
    """Convenience predicate; acceptance means zero fail-closed acquisition errors."""

    return not validate_manifest(manifest, require_full_corpus=require_full_corpus)
