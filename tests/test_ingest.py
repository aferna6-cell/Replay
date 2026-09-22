"""Post-session Power.log ingest (no overlay required)."""

import json
import tempfile
from pathlib import Path

from hsbg_coach.ingest import ingest_power_log, resolve_ingest_paths


FIXTURE = Path(__file__).parent / "fixtures" / "sample_bg.log"


def test_ingest_writes_jsonl_with_placement():
    assert FIXTURE.is_file()
    with tempfile.TemporaryDirectory() as td:
        result = ingest_power_log(str(FIXTURE), data_dir=td, quiet=True)
        assert result.games >= 1
        assert result.recorded
        assert result.placements == [3]
        rows = [json.loads(line) for line in Path(result.recorded[0]).read_text().splitlines()]
        assert rows
        assert all(r.get("placement") == 3 for r in rows)


def test_resolve_ingest_paths_explicit():
    paths = resolve_ingest_paths(path=str(FIXTURE))
    assert paths == [str(FIXTURE)]


def test_ingest_cli_help():
    from hsbg_coach.cli import build_parser
    p = build_parser()
    ns = p.parse_args(["ingest", "--path", str(FIXTURE), "--recent", "2"])
    assert ns.path == str(FIXTURE)
    assert ns.recent == 2
