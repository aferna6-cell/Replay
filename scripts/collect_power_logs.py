#!/usr/bin/env python3
"""Collect Hearthstone Power.log files from this machine into the repo.

Scans the known Hearthstone log locations (via ``hsbg_coach.config``) plus a
deep scan of your home directory for any ``Power.log`` / ``Power_old.log``,
copies each *new* one (deduped by content hash) into ``logs/`` with a unique
name, parses it into training trajectories (``data/*.jsonl``), then commits and
pushes the logs to GitHub. Designed to run unattended from cron / Task
Scheduler — see ``scripts/schedule_collection.sh`` (mac/linux) and
``scripts/schedule_collection.ps1`` (Windows).

Usage:
    python scripts/collect_power_logs.py                 # collect + parse + push
    python scripts/collect_power_logs.py --train         # ...and retrain the eval net
    python scripts/collect_power_logs.py --dry-run       # show what would happen
    python scripts/collect_power_logs.py --no-deep-scan  # only known HS locations
    python scripts/collect_power_logs.py --scan-root D:\\  # extra scan root(s)

Every step after collection is best-effort: a parse or train failure never
blocks the commit/push of the raw logs — the raw log is the asset we must not
lose.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
MANIFEST = LOGS_DIR / "manifest.json"
LOG_NAMES = {"power.log", "power_old.log"}
MIN_SIZE = 1024  # bytes — anything smaller holds no games
# Directory names never worth descending into during the deep scan.
PRUNE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".cache", ".npm", ".Trash", "$RECYCLE.BIN", "Windows",
              "System Volume Information"}

sys.path.insert(0, str(REPO_ROOT))
try:
    from hsbg_coach import config as hs_config
except Exception:  # pragma: no cover — collector must survive a broken checkout
    hs_config = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.is_file():
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"version": 1, "files": {}}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


def known_location_candidates() -> list:
    """Power.log candidates from the same locations hsbg_coach itself watches."""
    if hs_config is None:
        return []
    import glob
    found = []
    for d in hs_config.log_dir_candidates():
        for pat in ("Power.log", "Power_old.log",
                    os.path.join("Hearthstone_*", "Power.log"),
                    os.path.join("Hearthstone_*", "Power_old.log")):
            found += glob.glob(os.path.join(d, pat))
    return [Path(p) for p in found]


def deep_scan(roots: list) -> list:
    """Walk each root for Power.log/Power_old.log, pruning junk dirs."""
    found = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                    onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames
                           if d not in PRUNE_DIRS and not d.startswith(".git")]
            # never re-collect our own archive
            if Path(dirpath).resolve() == LOGS_DIR.resolve():
                dirnames[:] = []
                continue
            for name in filenames:
                if name.lower() in LOG_NAMES:
                    found.append(Path(dirpath) / name)
    return found


def collect(args) -> list:
    """Copy new logs into logs/. Returns list of newly added Paths."""
    manifest = load_manifest()
    seen_hashes = set(manifest["files"])
    candidates = known_location_candidates()
    # Explicit --scan-root always scans those roots; --no-deep-scan only
    # disables the *default* home-directory sweep.
    roots = args.scan_root or ([] if args.no_deep_scan else [Path.home()])
    candidates += deep_scan(roots)

    added, examined = [], set()
    for path in candidates:
        try:
            path = path.resolve()
            if path in examined or not path.is_file():
                continue
            examined.add(path)
            if LOGS_DIR.resolve() in path.parents:
                continue
            if path.stat().st_size < MIN_SIZE:
                continue
            digest = sha256_of(path)
        except OSError:
            continue
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        stamp = datetime.fromtimestamp(path.stat().st_mtime)
        dest = LOGS_DIR / f"Power_{stamp:%Y%m%d-%H%M%S}_{digest[:8]}.log"
        print(f"  new: {path}  ->  {dest.name}")
        if not args.dry_run:
            LOGS_DIR.mkdir(exist_ok=True)
            shutil.copy2(path, dest)
            manifest["files"][digest] = {
                "name": dest.name,
                "size": path.stat().st_size,
                "source": str(path),
                "collected_at": datetime.now().isoformat(timespec="seconds"),
            }
        added.append(dest)

    if added and not args.dry_run:
        save_manifest(manifest)
    print(f"Examined {len(examined)} candidate file(s); {len(added)} new.")
    return added


def parse_one(log: Path) -> bool:
    """Parse a single log into (state, action, outcome) trajectories."""
    print(f"Parsing {log.name} -> trajectories…")
    res = subprocess.run(
        [sys.executable, "-m", "hsbg_coach", "parse-file", str(log)],
        cwd=REPO_ROOT)
    if res.returncode != 0:
        print(f"  WARN: parse failed for {log.name} (continuing)")
        return False
    return True


def parse_all_pending() -> int:
    """Parse every archived log not yet folded into data/ trajectories —
    both freshly collected logs and any backlog in logs/ (e.g. logs pulled
    from GitHub on a fresh clone, or archived before this flag existed).
    Tracked via a `parsed` flag per manifest entry, so this is idempotent."""
    manifest = load_manifest()
    parsed = 0
    for meta in manifest["files"].values():
        if meta.get("parsed"):
            continue
        log = LOGS_DIR / meta["name"]
        if not log.is_file():
            continue
        if parse_one(log):
            meta["parsed"] = True
            parsed += 1
    if parsed:
        save_manifest(manifest)
    return parsed


def train() -> None:
    """Fold the recorded trajectories into the board-evaluation net."""
    print("Retraining the eval net on data/ trajectories…")
    res = subprocess.run(
        [sys.executable, "-m", "ml.train_eval_net", "--epochs", "40",
         "--trajectories", "data/"],
        cwd=REPO_ROOT)
    if res.returncode != 0:
        print("  WARN: training failed (missing torch? run: "
              "pip install -r requirements-ml.txt). Logs were still collected.")


def git(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *cmd], cwd=REPO_ROOT,
                          capture_output=True, text=True)


def commit_and_push(count: int) -> None:
    git("add", "logs")
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("Nothing new staged — no commit.")
        return
    msg = f"Collect {count} Power.log file(s) ({datetime.now():%Y-%m-%d})"
    res = git("commit", "-m", msg)
    if res.returncode != 0:
        print("Commit failed:\n" + res.stderr)
        return
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    git("pull", "--rebase", "origin", branch)  # best effort; push retries below
    for attempt, wait in enumerate((0, 2, 4, 8, 16)):
        if wait:
            time.sleep(wait)
        res = git("push", "-u", "origin", branch)
        if res.returncode == 0:
            print(f"Pushed to origin/{branch}.")
            return
        print(f"Push attempt {attempt + 1} failed: {res.stderr.strip()}")
    print("Push failed after retries — logs are committed locally; "
          "they'll go up next run.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be collected; change nothing")
    ap.add_argument("--no-deep-scan", action="store_true",
                    help="skip the default home-directory sweep (known "
                         "Hearthstone locations and any --scan-root still run)")
    ap.add_argument("--scan-root", action="append", type=Path,
                    help="directory to scan (repeatable); "
                         "default: your home directory")
    ap.add_argument("--no-parse", action="store_true",
                    help="skip building trajectories from the new logs")
    ap.add_argument("--no-push", action="store_true",
                    help="collect and commit locally but don't push")
    ap.add_argument("--train", action="store_true",
                    help="retrain the eval net after parsing (needs torch)")
    args = ap.parse_args()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Collecting Power.log files…")
    added = collect(args)
    if args.dry_run:
        return 0
    if not args.no_parse:
        n = parse_all_pending()
        if n:
            print(f"Parsed {n} archived log(s) into trajectories.")
    if args.train:
        train()
    if added and not args.no_push:
        commit_and_push(len(added))
    elif added:
        git("add", "logs")
        git("commit", "-m",
            f"Collect {len(added)} Power.log file(s) ({datetime.now():%Y-%m-%d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
