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
