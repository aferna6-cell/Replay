#!/usr/bin/env bash
# Install the weekly HSReplay refresh cron job (Sunday 20:00 by default).
#
#   ./scripts/schedule_hsreplay.sh             # install / update
#   ./scripts/schedule_hsreplay.sh --remove    # uninstall
#
# The job runs scripts/weekly_hsreplay_update.sh: autonomous stats fetch
# through your saved browser session -> import -> eval-net retrain -> push.
# Output lands in logs/hsreplay_weekly.log (gitignored). Requirements: you've
# run scripts/hsreplay_capture.py once (so the browser profile is logged in)
# and the laptop is on with your desktop session active at the scheduled time
# (the fetch opens a real browser window briefly).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
MARKER="# hsbg-hsreplay-weekly"
SCHEDULE="0 20 * * 0"    # Sunday 20:00

if [ "${1:-}" = "--remove" ]; then
  (crontab -l 2>/dev/null | grep -v "$MARKER") | crontab -
  echo "Removed the weekly HSReplay refresh cron job."
  exit 0
fi

LINE="$SCHEDULE cd \"$REPO\" && bash scripts/weekly_hsreplay_update.sh >> \"$REPO/logs/hsreplay_weekly.log\" 2>&1 $MARKER"
(crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$LINE") | crontab -
echo "Installed weekly HSReplay refresh:"
echo "  $LINE"
echo "Check runs at logs/hsreplay_weekly.log. Test now with:"
echo "  bash scripts/weekly_hsreplay_update.sh"
