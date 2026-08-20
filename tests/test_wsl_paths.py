"""WSL path discovery — the owner's real setup (2026-08-20): coach in WSL,
Hearthstone + HDT native on Windows, files under /mnt/c/."""

import pytest

from hsbg_coach import config, tier7


@pytest.fixture
def fake_mnt(tmp_path, monkeypatch):
    hs = tmp_path / "c/Program Files (x86)/Hearthstone"
    logs = hs / "Logs/Hearthstone_2026_08_20_18_00_00"
    logs.mkdir(parents=True)
    (logs / "Power.log").write_text("x")
    (tmp_path / "c/Users/aidan/AppData/Local/Blizzard/Hearthstone").mkdir(parents=True)
    (tmp_path / "c/Users/Public").mkdir(parents=True)
    hdt = tmp_path / "c/Users/aidan/AppData/Roaming/HearthstoneDeckTracker"
    hdt.mkdir(parents=True)
    (hdt / "hsreplay_oauth.json").write_text(
        '{"TokenData": {"access_token": "wsl-tok"}}')
    monkeypatch.setattr(config, "MNT_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "is_wsl", lambda: True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))   # no wine prefixes
    monkeypatch.delenv("HSBG_HS_DIR", raising=False)
    monkeypatch.delenv("HSREPLAY_TOKEN", raising=False)
    monkeypatch.delenv("HSREPLAY_OAUTH_FILE", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return tmp_path


def test_finds_windows_install_and_power_log(fake_mnt):
    assert any("Program Files (x86)" in i for i in config.hearthstone_installs())
    power = config.newest_power_log(config.log_dir_candidates())
    assert power and power.startswith(str(fake_mnt))


def test_log_config_targets_windows_appdata_not_public(fake_mnt):
    cands = config.log_config_path_candidates()
    assert "Users/aidan/AppData/Local/Blizzard/Hearthstone" in cands[0].replace("\\", "/")
    assert not any("Public" in c for c in cands)


def test_hdt_token_found_through_mnt(fake_mnt):
    assert tier7.find_token() == "wsl-tok"
