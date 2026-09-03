"""Archived-log backfill: logs in logs/ without trajectories get parsed once."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "collect_power_logs", REPO / "scripts" / "collect_power_logs.py")
collector = importlib.util.module_from_spec(spec)
sys.modules["collect_power_logs"] = collector
spec.loader.exec_module(collector)


def test_parse_all_pending_backfills_and_marks(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "Power_a.log").write_text("x")
    (logs / "Power_b.log").write_text("x")
    manifest = {"version": 1, "files": {
        "aaa": {"name": "Power_a.log"},                    # pending
        "bbb": {"name": "Power_b.log", "parsed": True},    # already done
        "ccc": {"name": "Power_gone.log"},                 # file missing
    }}
    (logs / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(collector, "LOGS_DIR", logs)
    monkeypatch.setattr(collector, "MANIFEST", logs / "manifest.json")
    parsed = []
    monkeypatch.setattr(collector, "parse_one",
                        lambda log: parsed.append(log.name) or True)

    assert collector.parse_all_pending() == 1
    assert parsed == ["Power_a.log"]
    saved = json.loads((logs / "manifest.json").read_text())
    assert saved["files"]["aaa"]["parsed"] is True
    # idempotent: second run parses nothing
    assert collector.parse_all_pending() == 0
    assert parsed == ["Power_a.log"]


def test_parse_failure_not_marked(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "Power_a.log").write_text("x")
    (logs / "manifest.json").write_text(json.dumps(
        {"version": 1, "files": {"aaa": {"name": "Power_a.log"}}}))
    monkeypatch.setattr(collector, "LOGS_DIR", logs)
    monkeypatch.setattr(collector, "MANIFEST", logs / "manifest.json")
    monkeypatch.setattr(collector, "parse_one", lambda log: False)

    assert collector.parse_all_pending() == 0
    saved = json.loads((logs / "manifest.json").read_text())
    assert "parsed" not in saved["files"]["aaa"]   # retried next run


def test_looks_like_power_log(tmp_path):
    hs = b"D 10:00:00.0 GameState.DebugPrintPower() - CREATE_GAME\n"
    (tmp_path / "Power.log").write_bytes(b"anything")          # exact name
    (tmp_path / "Power_aug20.log").write_bytes(hs)             # renamed, real
    (tmp_path / "week2.txt").write_bytes(hs)                   # renamed, real
    (tmp_path / "powershell.log").write_bytes(b"PS boot ok\n") # decoy
    (tmp_path / "notes.txt").write_bytes(b"todo list\n")       # decoy
    (tmp_path / "video.mp4").write_bytes(hs)                   # wrong ext
    assert collector.looks_like_power_log(tmp_path / "Power.log")
    assert collector.looks_like_power_log(tmp_path / "Power_aug20.log")
    assert collector.looks_like_power_log(tmp_path / "week2.txt")
    assert not collector.looks_like_power_log(tmp_path / "powershell.log")
    assert not collector.looks_like_power_log(tmp_path / "notes.txt")
    assert not collector.looks_like_power_log(tmp_path / "video.mp4")


def test_deep_scan_finds_renamed_logs(tmp_path, monkeypatch):
    docs = tmp_path / "Documents" / "hs-backups"
    docs.mkdir(parents=True)
    hs = b"D 10:00:00.0 PowerTaskList.DebugPrintPower() - TAG_CHANGE\n" * 50
    (docs / "Power_2026_08_20.log").write_bytes(hs)
    (docs / "game_week1.txt").write_bytes(hs)
    (docs / "unrelated.log").write_bytes(b"app started\n" * 50)
    monkeypatch.setattr(collector, "LOGS_DIR", tmp_path / "repo-logs")
    found = {p.name for p in collector.deep_scan([tmp_path])}
    assert found == {"Power_2026_08_20.log", "game_week1.txt"}
