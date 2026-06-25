#!/usr/bin/env bash
# Weekly refresh: pull the latest cards + meta stats from Firestone/HearthstoneJSON
# and retrain the models on it (plus any games you've recorded).
#
# Run it by hand, or schedule it (see README "Keeping it fresh"):
#   cron (mac/linux):  0 9 * * 1  cd /path/to/Replay && ./scripts/weekly_update.sh
#   Windows Task Scheduler: weekly action -> `bash scripts/weekly_update.sh`
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/4  Refresh BG card knowledge (HearthstoneJSON)"
python -m hsbg_coach refresh-cards

echo "==> 2/4  Refresh hero/comp/card/trinket stats (Firestone, top-10% / past week)"
python -m hsbg_coach refresh-stats

echo "==> 3/4  Retrain card2vec on the latest winning boards"
python -m ml.train_card2vec --epochs 5

echo "==> 4/5  Retrain the board-evaluation net on the meta + your recorded games"
python -m ml.train_eval_net --epochs 40 --trajectories data/

echo "==> 5/6  Retrain the economy value net on self-play lobbies"
python -m ml.train_econ --lobbies 4000 --epochs 30

echo "==> 6/6  Retrain the self-play economy policy (REINFORCE)"
python -m ml.train_policy --iters 60 --lobbies 64

echo "Done. Models refreshed from this week's meta + your games."
