"""Wine-prefix discovery (linux) — regression for the 2026-08-20 ghost:
`setup` wrote log.config into an empty ~/.wine while the real Hearthstone
lived in a Lutris prefix the hardcoded list never checked."""

import os

import pytest

from hsbg_coach import config

pytestmark = pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux"), reason="linux-only")


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    # Empty plain-wine prefix (the trap) + a Lutris battlenet prefix with HS.
    (tmp_path / ".wine/drive_c/users/aidan/AppData/Local/Blizzard/Hearthstone"
     ).mkdir(parents=True)
    hs = tmp_path / "Games/battlenet/drive_c/Program Files (x86)/Hearthstone"
    logs = hs / "Logs/Hearthstone_2026_08_20_20_00_00"
    logs.mkdir(parents=True)
    (logs / "Power.log").write_text("x")
    (tmp_path / "Games/battlenet/drive_c/users/aidan").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HSBG_HS_DIR", raising=False)
    return tmp_path


def test_discovers_lutris_prefix_and_prefers_it(fake_home):
    dcs = config.wine_drive_cs()
    assert any("Games/battlenet" in d for d in dcs)
    assert "battlenet" in dcs[0]              # install-carrying prefix first


def test_finds_power_log_in_lutris_install(fake_home):
    power = config.newest_power_log(config.log_dir_candidates())
    assert power and "Games/battlenet" in power


def test_log_config_targets_the_real_prefix(fake_home):
    first = config.log_config_path_candidates()[0]
    assert "Games/battlenet" in first          # NOT the empty ~/.wine


def test_hs_dir_env_override(fake_home, monkeypatch):
    override = fake_home / "weird/Hearthstone"
    (override / "Logs").mkdir(parents=True)
    monkeypatch.setenv("HSBG_HS_DIR", str(override))
    assert str(override) in config.hearthstone_installs()
    assert str(override / "Logs") in config.log_dir_candidates()


def test_hearthstone_installs_lists_lutris(fake_home):
    installs = config.hearthstone_installs()
    assert len(installs) == 1 and "battlenet" in installs[0]
