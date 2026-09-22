"""Log-path detection — macOS session folders + Linux Wine/Proton expansion."""

import os
import sys
import time

from hsbg_coach import config


def _make_log(d, name, age_s):
    sub = os.path.join(d, name)
    os.makedirs(sub)
    p = os.path.join(sub, "Power.log")
    with open(p, "w") as fh:
        fh.write("x")
    os.utime(p, (time.time() - age_s, time.time() - age_s))
    return p


def test_newest_power_log_picks_latest_session(tmp_path):
    d = str(tmp_path)
    _make_log(d, "Hearthstone_2026_06_22_21_44_17", 1000)
    newest = _make_log(d, "Hearthstone_2026_06_24_23_17_43", 1)
    assert config.newest_power_log([d]) == newest


def test_newest_power_log_handles_flat_layout(tmp_path):
    p = os.path.join(str(tmp_path), "Power.log")
    with open(p, "w") as fh:
        fh.write("x")
    assert config.newest_power_log([str(tmp_path)]) == p


def test_newest_power_log_none_when_absent(tmp_path):
    assert config.newest_power_log([str(tmp_path)]) is None


def test_detect_returns_paths_object():
    paths = config.Paths.detect()           # never raises, even with no game logs
    assert hasattr(paths, "power_log") and hasattr(paths, "log_config")


def _write(path: str, content: str = "x", age_s: float = 0) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    if age_s:
        os.utime(path, (time.time() - age_s, time.time() - age_s))
    return path


def test_linux_log_dir_candidates_expand_globs(tmp_path, monkeypatch):
    """Fake home with wine + bottles + steam layouts; candidates must expand."""
    if not sys.platform.startswith("linux"):
        # Still exercise the helper by forcing linux branch via monkeypatch.
        monkeypatch.setattr(config.sys, "platform", "linux")
    home = str(tmp_path)
    wine_logs = os.path.join(
        home, ".wine", "drive_c", "Program Files (x86)", "Hearthstone", "Logs")
    bottles_logs = os.path.join(
        home, ".local", "share", "bottles", "bottles", "HS", "drive_c",
        "users", "steamuser", "AppData", "Local", "Blizzard", "Hearthstone",
        "Logs")
    steam_logs = os.path.join(
        home, ".steam", "steam", "steamapps", "compatdata", "123", "pfx",
        "drive_c", "Program Files (x86)", "Hearthstone", "Logs")
    games_logs = os.path.join(
        home, "Games", "hearthstone-wine", "drive_c", "Program Files (x86)",
        "Hearthstone", "Logs")
    for d in (wine_logs, bottles_logs, steam_logs, games_logs):
        os.makedirs(d)

    cands = config.log_dir_candidates(home=home)
    assert wine_logs in cands
    assert bottles_logs in cands
    assert steam_logs in cands
    assert games_logs in cands


def test_discover_power_logs_prefers_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(config.sys, "platform", "linux")
    home = str(tmp_path)
    older = _write(os.path.join(
        home, ".wine", "drive_c", "Program Files (x86)", "Hearthstone", "Logs",
        "Power.log"), age_s=500)
    newer = _write(os.path.join(
        home, "Games", "hs", "drive_c", "users", "aidan", "AppData", "Local",
        "Blizzard", "Hearthstone", "Logs", "Hearthstone_2026_09_22_12_00_00",
        "Power.log"), age_s=1)
    # Noise outside discovery roots must be ignored
    _write(os.path.join(home, "Downloads", "Power.log"), age_s=0)

    hits = config.discover_power_logs(home=home)
    assert hits[0] == newer
    assert older in hits
    assert not any("Downloads" in h for h in hits)


def test_paths_detect_uses_discovery_and_nearby_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config.sys, "platform", "linux")
    home = str(tmp_path)
    power = _write(os.path.join(
        home, ".wine", "drive_c", "Program Files (x86)", "Hearthstone", "Logs",
        "Power.log"))
    cfg = _write(os.path.join(
        home, ".wine", "drive_c", "users", "aidan", "AppData", "Local",
        "Blizzard", "Hearthstone", "log.config"), content="[Power]\n")

    paths = config.Paths.detect(home=home)
    assert paths.power_log == power
    assert paths.log_config == cfg
    assert paths.log_dir == os.path.dirname(power)
