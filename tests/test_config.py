"""Log-path detection — especially the macOS per-launch session-folder layout
(/Applications/Hearthstone/Logs/Hearthstone_<timestamp>/Power.log)."""

import os
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
