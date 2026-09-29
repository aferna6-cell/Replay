#!/usr/bin/env bash
# Daily Firestone pipeline (mac/linux). cron example, every day at 04:30:
#   30 4 * * *  /path/to/Replay/scripts/daily_firestone_pipeline.sh
# Extra arguments go to `python -m hsbg_coach.daily_pipeline` (e.g. --no-install).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m hsbg_coach.daily_pipeline --log "data/firestone/logs/daily-$(date +%F).log" "$@"
