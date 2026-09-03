# logs/ — collected Power.log archive

Every Hearthstone `Power.log` harvested from your machine lives here, one file
per unique log, named `Power_<mtime>_<sha256-prefix>.log`. These are the raw
material the training pipeline feeds on: each one is parsed into
`(state, action, outcome)` trajectories (`data/*.jsonl`, kept local) which
`ml/train_eval_net.py` then trains against.

- **Collector:** `python scripts/collect_power_logs.py` scans the known
  Hearthstone log locations plus your home directory, dedupes by content hash
  against `manifest.json`, copies new logs here, parses them, and pushes.
- **Schedule it:** `scripts/schedule_collection.sh` (mac/linux cron) or
  `scripts/schedule_collection.ps1` (Windows Task Scheduler) — every 48 hours
  by default, `--weekly` / `-Weekly` for weekly.
- **`manifest.json`** maps content hash → file, so the same log is never
  uploaded twice even if Hearthstone rotates or you copy it around.
- **`collect.log`** (gitignored) is the scheduled runs' output — check it when
  wondering whether collection ran.

Don't rename files here by hand: `manifest.json` and
`tests/test_real_log.py` refer to them by name.
