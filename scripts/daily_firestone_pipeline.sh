#!/usr/bin/env bash
# Daily Firestone pipeline (mac/linux). By hand, from the Replay folder (output on screen):
#   scripts/daily_firestone_pipeline.sh [--no-install ...]
# cron example, every day at 04:30 (output to data/firestone/logs/):
#   30 4 * * *  /path/to/Replay/scripts/daily_firestone_pipeline.sh --scheduled
# Other arguments go to `python -m hsbg_coach.daily_pipeline`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
if [ "${1:-}" = "--scheduled" ]; then
  shift
  set -- --log "data/firestone/logs/daily-$(date +%F).log" "$@"
fi
exec "$PY" -m hsbg_coach.daily_pipeline "$@"
