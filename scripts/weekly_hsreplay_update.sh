#!/usr/bin/env bash
# Weekly autonomous HSReplay refresh: fetch fresh top-20% BG stats through the
# saved browser session, retrain the eval net on the updated priors, and push
# the normalized stats to GitHub. Installed on a schedule by
# scripts/schedule_hsreplay.sh; safe to run by hand any time.
set -uo pipefail
cd "$(dirname "$0")/.."

# Use the project venv when present (cron doesn't activate it).
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
  export PATH="$(pwd)/.venv/bin:$PATH"
else
  PY="$(command -v python3 || command -v python)"
fi
# A headed browser needs a display; under cron, borrow the desktop session's.
export DISPLAY="${DISPLAY:-:0}"

echo "==> 1/3  Fetch + import HSReplay BG stats ($(date))"
if ! "$PY" scripts/hsreplay_autofetch.py; then
  echo "Fetch failed (Cloudflare session expired?). Run"
  echo "  python scripts/hsreplay_capture.py"
  echo "once interactively to refresh the login, then re-run this script."
  exit 1
fi

echo "==> 2/3  Retrain the eval net on the refreshed priors"
"$PY" -m ml.train_eval_net --epochs 40 --trajectories data/ \
  || echo "WARN: retrain failed (missing torch?) — stats still refreshed."

echo "==> 3/3  Commit + push refreshed stats (only if they moved)"
git add data/stats/hsreplay_*.json 2>/dev/null || true
if git diff --cached --quiet; then
  echo "Stats unchanged — nothing to commit."
else
  git commit -m "Weekly HSReplay stats refresh ($(date +%Y-%m-%d))"
  git push || echo "WARN: push failed — commit is local; next run retries."
fi
echo "Done."
