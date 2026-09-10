from copy import deepcopy

from hsbg_coach.external_reference_intake import accepted, validate_manifest


HEX = "a" * 64


def _lobby(split="calibration", suffix="1"):
    return {
        "lobby_id": f"lobby-{suffix}",
        "split": split,
        "raw_sha256": (suffix[-1] if suffix[-1] in "0123456789abcdef" else "a") * 64,
        "reference_sha256": "b" * 63 + suffix[-1],
        "checkpoint_sha256": "c" * 63 + suffix[-1],
        "sync_sha256": "d" * 63 + suffix[-1],
        "session_id": f"session-{suffix}",
        "player_id": f"player-{suffix}",
        "patch": "frozen-patch",
        "platform": "windows",
        "privacy_reviewed": True,
        "independent_reference": True,
        "checkpoints": [
            {
                "checkpoint_id": f"cp-{suffix}",
                "turn": 5,
                "kind": "board_mutation",
                "reference_timestamp": 1.25,
                "sync_within_tolerance": True,
                "simultaneous_board": True,
                "all_required_fields_resolved": True,
                "replay_derived_repair": False,
            }
        ],
    }


def _manifest(lobbies=None):
    return {
        "protocol_version": "phase3u-external-reference-v1",
        "simulator_seed_budget": 0,
        "confirmation_11500_11699_consumed": False,
        "lobbies": [_lobby()] if lobbies is None else lobbies,
    }


def _codes(manifest, **kwargs):
    return {failure["code"] for failure in validate_manifest(manifest, **kwargs)}


def test_valid_dry_run_is_accepted():
    manifest = _manifest()
    assert validate_manifest(manifest) == []
    assert accepted(manifest)


def test_reserved_confirmation_seeds_and_simulator_budget_fail_closed():
    manifest = _manifest()
    manifest["simulator_seed_budget"] = 1
    manifest["confirmation_11500_11699_consumed"] = True
    assert {"SEED_BUDGET", "CONFIRMATION_SEEDS"} <= _codes(manifest)


def test_reference_must_be_independent_and_privacy_reviewed():
    manifest = _manifest()
    manifest["lobbies"][0]["independent_reference"] = False
    manifest["lobbies"][0]["privacy_reviewed"] = False
    assert {"REFERENCE_NOT_INDEPENDENT", "PRIVACY_REVIEW"} <= _codes(manifest)


def test_unresolved_or_replay_repaired_checkpoint_remains_failure():
    manifest = _manifest()
    checkpoint = manifest["lobbies"][0]["checkpoints"][0]
    checkpoint["all_required_fields_resolved"] = False
    checkpoint["replay_derived_repair"] = True
    assert {"REFERENCE_FIELD_UNRESOLVED", "REPLAY_DERIVED_REPAIR"} <= _codes(manifest)


def test_ambiguous_sync_and_non_simultaneous_board_fail():
    manifest = _manifest()
    checkpoint = manifest["lobbies"][0]["checkpoints"][0]
    checkpoint["sync_within_tolerance"] = False
    checkpoint["simultaneous_board"] = False
    assert {"SYNC_AMBIGUOUS", "NON_SIMULTANEOUS"} <= _codes(manifest)


def test_cross_split_session_and_player_leakage_fail():
    calibration = _lobby("calibration", "1")
    evaluation = _lobby("evaluation", "2")
    evaluation["session_id"] = calibration["session_id"]
    evaluation["player_id"] = calibration["player_id"]
    codes = _codes(_manifest([calibration, evaluation]))
    assert {"SESSION_SPLIT_LEAKAGE", "PLAYER_SPLIT_LEAKAGE"} <= codes


def test_duplicate_raw_or_reference_artifacts_fail():
    first = _lobby("calibration", "1")
    second = _lobby("calibration", "2")
    second["raw_sha256"] = first["raw_sha256"]
    second["reference_sha256"] = first["reference_sha256"]
    codes = _codes(_manifest([first, second]))
    assert {"RAW_SHA_DUPLICATE", "REFERENCE_SHA_DUPLICATE"} <= codes


def test_checkpoint_turn_must_be_t5_through_t10():
    manifest = _manifest()
    manifest["lobbies"][0]["checkpoints"][0]["turn"] = 11
    assert "TURN_RANGE" in _codes(manifest)


def test_full_corpus_mode_requires_exact_frozen_split_counts():
    codes = _codes(_manifest(), require_full_corpus=True)
    assert {"CALIBRATION_COUNT", "EVALUATION_COUNT"} <= codes


def test_full_corpus_exact_50_200_can_pass_without_scoring_parser_output():
    lobbies = []
    for i in range(250):
        split = "calibration" if i < 50 else "evaluation"
        suffix = f"{i + 1:064x}"
        lobby = _lobby(split, str((i % 9) + 1))
        lobby["lobby_id"] = f"lobby-{i}"
        lobby["session_id"] = f"session-{i}"
        lobby["player_id"] = f"player-{i}"
        lobby["raw_sha256"] = f"{i + 1:064x}"
        lobby["reference_sha256"] = f"{i + 251:064x}"
        lobby["checkpoint_sha256"] = f"{i + 501:064x}"
        lobby["sync_sha256"] = f"{i + 751:064x}"
        lobby["checkpoints"][0]["checkpoint_id"] = f"cp-{i}"
        lobbies.append(lobby)
    manifest = _manifest(lobbies)
    assert validate_manifest(manifest, require_full_corpus=True) == []
