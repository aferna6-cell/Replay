"""Replay Hearthstone Power.logs and report how closely games followed the
lobby playbook (and the coach its own rules), next to placement.

  python scripts/adherence_report.py                       # every log on this PC
  python scripts/adherence_report.py C:\\path\\to\\Logs      # a folder (searched)
  python scripts/adherence_report.py Power.log other.log   # specific files

Writes results/adherence/adherence_<timestamp>.json (per-game + per-turn detail)
and prints a summary table. See hsbg_coach/adherence.py for the metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg_coach.adherence import (audit_log, default_logs, find_logs,  # noqa: E402
                                  render_text, summarize)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="Power.log files, folders or globs "
                    "(default: every Power.log in Hearthstone's log folders)")
    ap.add_argument("-o", "--out", default=str(ROOT / "results" / "adherence"),
                    help="folder for the JSON report")
    ap.add_argument("--max-games", type=int, default=None,
                    help="stop each log after this many games")
    args = ap.parse_args(argv)

    logs = find_logs(args.paths) if args.paths else default_logs()
    if not logs:
        print("No Power.log found. Pass the Hearthstone Logs folder, e.g.\n"
              '  python scripts/adherence_report.py "C:\\Program Files (x86)\\Hearthstone\\Logs"')
        return 1
    games = []
    for i, log in enumerate(logs, 1):
        t0 = time.time()
        try:
            got = audit_log(log, max_games=args.max_games)
        except Exception as e:                       # one bad log never stops the run
            print(f"[{i}/{len(logs)}] {log}: skipped ({e})")
            continue
        games += got
        print(f"[{i}/{len(logs)}] {log}: {len(got)} game(s) in {time.time() - t0:.0f}s")
    summary = summarize(games)
    print()
    print(render_text(games, summary))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"adherence_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "games": [g.to_dict() for g in games]},
                               indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetail: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
