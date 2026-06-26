#!/usr/bin/env bash
# Weekly refresh: pull the latest cards + meta stats from Firestone/HearthstoneJSON
# and retrain the models on it (plus any games you've recorded).
#
# Run it by hand, or schedule it (see README "Keeping it fresh"):
#   cron (mac/linux):  0 9 * * 1  cd /path/to/Replay && ./scripts/weekly_update.sh
#   Windows Task Scheduler: weekly action -> `bash scripts/weekly_update.sh`
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/7  Refresh BG card knowledge (HearthstoneJSON)"
python -m hsbg_coach refresh-cards

echo "==> 2/7  Refresh hero/comp/card/trinket stats (Firestone, top-10% / past week)"
python -m hsbg_coach refresh-stats

# IMPORTANT: keep --dim at 48. The board-evaluation net (step 4) bakes the card2vec
# vector into its input features, so changing the dimension here without retraining
# the eval net breaks it (feature-shape mismatch). Retrain card2vec BEFORE the eval
# net so the net trains on the fresh vectors.
echo "==> 3/7  Retrain card2vec on the latest winning boards (dim 48)"
python -m ml.train_card2vec --dim 48 --epochs 8 --min-count 15

echo "==> 4/7  Retrain the board-evaluation net on the meta + your recorded games"
python -m ml.train_eval_net --epochs 40 --trajectories data/

echo "==> 5/7  Retrain the economy value net on self-play lobbies"
python -m ml.train_econ --lobbies 4000 --epochs 30

echo "==> 6/7  Retrain the self-play economy policy (REINFORCE)"
python -m ml.train_policy --iters 60 --lobbies 64

# Commit ONLY the tracked meta artifacts that actually moved. card2vec.json and the
# Firestone stats are deterministic from the data, so they change only when the meta
# does — no-op weeks produce no commit. (The .pt models are gitignored and stay
# local; they're non-deterministic and would churn every run.)
echo "==> 7/7  Commit refreshed meta (only if it moved)"
git add data/cards/card2vec.json data/cards data/stats 2>/dev/null || true
if git diff --cached --quiet; then
  echo "Meta unchanged since last refresh — nothing to commit. Synergy already sharp."
else
  git commit -m "Weekly meta refresh: stats + retrained card2vec ($(date +%Y-%m-%d))"
  echo "Committed refreshed meta. Push when ready:  git push"
fi

echo "Done. Models refreshed from this week's meta + your games."
