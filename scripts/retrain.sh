#!/usr/bin/env bash
# Quick retrain on your own games — run after a session to fold the games you just
# played into the board-evaluation net. Lighter than weekly_update.sh: it reuses
# the cached meta and only retrains the eval net (adding data/*.jsonl trajectories).
set -euo pipefail
cd "$(dirname "$0")/.."

games=$(ls data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "Retraining the eval net on the meta + ${games} recorded game file(s)…"
python -m ml.train_eval_net --epochs 40 --trajectories data/
echo "Done."
