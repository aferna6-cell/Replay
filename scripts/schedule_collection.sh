#!/usr/bin/env bash
# Install a cron job that runs the Power.log collector on a schedule.
#
#   ./scripts/schedule_collection.sh             # every 48 hours (21:00, every 2nd day)
#   ./scripts/schedule_collection.sh --weekly    # weekly (Sunday 21:00)
#   ./scripts/schedule_collection.sh --remove    # uninstall
#
# The job collects new Power.log files into logs/, parses them into training
# trajectories, retrains the eval net, and pushes the logs to GitHub. Output
# goes to logs/collect.log (gitignored). Re-running this script replaces the
# existing entry, so it's safe to switch cadence any time.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
# Prefer the project venv (it holds torch for the --train step; cron doesn't
# activate venvs on its own).
if [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON="$REPO/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi
MARKER="# hsbg-collect-power-logs"

SCHEDULE="0 21 */2 * *"   # 21:00 on every 2nd day of the month (~every 48h)
for arg in "$@"; do
  case "$arg" in
    --weekly) SCHEDULE="0 21 * * 0" ;;   # Sunday 21:00
    --remove)
      (crontab -l 2>/dev/null | grep -v "$MARKER") | crontab -
      echo "Removed the Power.log collection cron job."
      exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

LINE="$SCHEDULE cd \"$REPO\" && \"$PYTHON\" scripts/collect_power_logs.py --train >> \"$REPO/logs/collect.log\" 2>&1 $MARKER"
(crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$LINE") | crontab -

echo "Installed cron job:"
echo "  $LINE"
echo
echo "Note: cron's */2 is calendar-based (1st, 3rd, 5th… of the month), so at a"
echo "month boundary two runs can land a day apart — harmless, dedupe skips"
echo "already-collected logs. Check output at logs/collect.log."
