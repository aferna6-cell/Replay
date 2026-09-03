#!/usr/bin/env bash
# WSL-native weekly scheduling — run FROM your Ubuntu terminal, schedules VIA
# Windows Task Scheduler (which runs whenever Windows is on, and boots WSL on
# demand). This is the right scheduler for a Windows machine: cron inside WSL
# only fires while WSL happens to be running.
#
#   ./scripts/schedule_wsl.sh              # install both weekly tasks
#   ./scripts/schedule_wsl.sh --remove     # uninstall both
#   ./scripts/schedule_wsl.sh --status     # show the registered tasks
#
# Installs two Windows Scheduled Tasks (run as you, only when logged on, so
# the HSReplay browser window can appear):
#   HSBGCoach-HSReplayWeekly   Sunday 20:00  -> scripts/weekly_hsreplay_update.sh
#   HSBGCoach-CollectLogs      Sunday 21:00  -> scripts/collect_power_logs.py --train
# Output lands in logs/hsreplay_weekly.log and logs/collect.log as usual.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

SCHTASKS="$(command -v schtasks.exe || echo /mnt/c/Windows/System32/schtasks.exe)"
if [ ! -x "$SCHTASKS" ] && ! command -v schtasks.exe >/dev/null; then
  echo "schtasks.exe not reachable — is this WSL with Windows interop enabled?"
  echo "Fallback: use the cron installers (schedule_collection.sh --weekly,"
  echo "schedule_hsreplay.sh) and keep a WSL terminal open on Sundays."
  exit 1
fi

TASK_HS="HSBGCoach-HSReplayWeekly"
TASK_LOGS="HSBGCoach-CollectLogs"
TASK_VODS="HSBGCoach-Transcripts"

if [ "${1:-}" = "--remove" ]; then
  for t in "$TASK_HS" "$TASK_LOGS" "$TASK_VODS"; do
    "$SCHTASKS" /Delete /TN "$t" /F 2>/dev/null || true
  done
  echo "Removed the Windows scheduled tasks."
  exit 0
fi
if [ "${1:-}" = "--status" ]; then
  for t in "$TASK_HS" "$TASK_LOGS" "$TASK_VODS"; do
    "$SCHTASKS" /Query /TN "$t" 2>/dev/null || echo "$t: not installed"
  done
  exit 0
fi

# Windows-side .bat wrappers keep schtasks quoting sane. Find the Windows
# user profile via interop.
WINUSER="$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')"
WINHOME="/mnt/c/Users/$WINUSER"
if [ ! -d "$WINHOME" ]; then
  echo "Couldn't locate your Windows profile (got user '$WINUSER')."
  exit 1
fi
BATDIR="$WINHOME/hsbg-tasks"
mkdir -p "$BATDIR"

cat > "$BATDIR/hsbg_hsreplay.bat" <<EOF
@echo off
wsl.exe bash -lc "cd '$REPO' && bash scripts/weekly_hsreplay_update.sh >> logs/hsreplay_weekly.log 2>&1"
EOF
cat > "$BATDIR/hsbg_logs.bat" <<EOF
@echo off
wsl.exe bash -lc "cd '$REPO' && '$REPO/.venv/bin/python' scripts/collect_power_logs.py --train >> logs/collect.log 2>&1"
EOF
cat > "$BATDIR/hsbg_transcripts.bat" <<EOF
@echo off
wsl.exe bash -lc "cd '$REPO' && bash scripts/weekly_transcripts.sh >> logs/transcripts.log 2>&1"
EOF

WINBATDIR="C:\\Users\\$WINUSER\\hsbg-tasks"
"$SCHTASKS" /Create /F /TN "$TASK_HS" /TR "$WINBATDIR\\hsbg_hsreplay.bat" \
  /SC WEEKLY /D SUN /ST 20:00
"$SCHTASKS" /Create /F /TN "$TASK_LOGS" /TR "$WINBATDIR\\hsbg_logs.bat" \
  /SC WEEKLY /D SUN /ST 21:00
"$SCHTASKS" /Create /F /TN "$TASK_VODS" /TR "$WINBATDIR\\hsbg_transcripts.bat" \
  /SC WEEKLY /D SUN /ST 19:30

# Catch-up: if the laptop was off/asleep at the scheduled time, run the task
# as soon as it's next on and logged in instead of waiting a whole week.
# (schtasks can't set StartWhenAvailable; PowerShell can.)
PS="$(command -v powershell.exe || echo /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe)"
for task in "$TASK_HS" "$TASK_LOGS" "$TASK_VODS"; do
  "$PS" -NoProfile -Command \
    "Set-ScheduledTask -TaskName '$task' -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable)" \
    >/dev/null 2>&1 \
    || echo "note: couldn't enable missed-run catch-up for $task (runs still fire on schedule)"
done

echo
echo "Installed (Windows Task Scheduler, runs when you're logged on):"
echo "  $TASK_VODS  Sunday 19:30  streamer transcripts -> playbook (subscription)"
echo "  $TASK_HS    Sunday 20:00  HSReplay fetch -> retrain -> push"
echo "  $TASK_LOGS  Sunday 21:00  log sweep -> parse -> retrain -> push"
echo
echo "Test one now:    schtasks.exe /Run /TN $TASK_HS"
echo "Check output:    tail logs/hsreplay_weekly.log logs/collect.log"
echo "If you'd previously installed the cron versions, remove them to avoid"
echo "double runs:  ./scripts/schedule_collection.sh --remove ; ./scripts/schedule_hsreplay.sh --remove"
