from __future__ import annotations

from importlib import metadata as importlib_metadata

import ml.fidelity_reference as fidelity_reference


def _missing_distribution(_name: str) -> str:
    raise importlib_metadata.PackageNotFoundError


def _missing_numpy_env():
    raise ModuleNotFoundError("No module named 'numpy'", name="numpy")


def _missing_other_env():
    raise ModuleNotFoundError("No module named 'unexpected_dep'", name="unexpected_dep")


def test_fidelity_runtime_fingerprint_tolerates_missing_ml_distributions(monkeypatch):
    monkeypatch.setattr(
        fidelity_reference.importlib_metadata,
        "version",
        _missing_distribution,
    )

    runtime = fidelity_reference.fidelity_runtime_fingerprint()

    assert runtime["python_version"]
    assert runtime["python_full"]
    assert runtime["platform"]
    assert runtime["torch_version"] is None
    assert runtime["numpy_version"] is None
    assert runtime["torch_device"] is None
    assert runtime["torch_cuda_available"] is None
    assert runtime["provenance_scope"] == "simulator_fidelity"
    assert runtime["ml_runtime_imported"] is False


def test_fidelity_env_config_tolerates_missing_numpy(monkeypatch):
    monkeypatch.setattr(fidelity_reference, "env_config", _missing_numpy_env)

    env = fidelity_reference.fidelity_env_config()

    assert env["env"] == "hsbg_coach.bg_env.BGEnv"
    assert env["n_players"] == 8
    assert env["field_size"] == 7
    assert env["agent_seat"] == 0
    assert env["n_actions"] > 0
    assert env["max_turns"] > 0
    assert env["max_decisions"] == 400


def test_fidelity_env_config_does_not_hide_unrelated_missing_dependency(monkeypatch):
    monkeypatch.setattr(fidelity_reference, "env_config", _missing_other_env)

    try:
        fidelity_reference.fidelity_env_config()
    except ModuleNotFoundError as exc:
        assert exc.name == "unexpected_dep"
    else:
        raise AssertionError("unrelated missing dependency must fail loudly")


def test_simulator_contract_keeps_provenance_when_ml_stack_is_missing(monkeypatch):
    monkeypatch.setattr(
        fidelity_reference.importlib_metadata,
        "version",
        _missing_distribution,
    )
    monkeypatch.setattr(fidelity_reference, "env_config", _missing_numpy_env)
    monkeypatch.setattr(fidelity_reference, "reference_fingerprints", lambda: {})
    monkeypatch.setattr(fidelity_reference, "load_reference_metadata", lambda: {})
    monkeypatch.setattr(fidelity_reference, "git_commit", lambda: "deadbeef")
    monkeypatch.setattr(fidelity_reference, "git_working_tree_clean", lambda: True)

    contract = fidelity_reference.build_simulator_v1_1_contract(
        evaluation_seed=14200,
        lobbies=8,
    )

    assert contract["code_commit"] == "deadbeef"
    assert contract["evaluation"]["base_seed"] == 14200
    assert contract["evaluation"]["lobbies"] == 8
    assert contract["runtime"]["torch_version"] is None
    assert contract["runtime"]["numpy_version"] is None
    assert contract["runtime"]["provenance_scope"] == "simulator_fidelity"
    assert contract["environment"]["field_size"] == 7
    assert contract["environment"]["max_decisions"] == 400
    assert contract["working_tree_clean"] is True
    assert "runtime_fingerprint_error" not in contract
