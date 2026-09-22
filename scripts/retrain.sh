#!/usr/bin/env bash
# Quick retrain on meta + recorded games (personal + hsreplay_expert).
# Expert rows (source=hsreplay_expert) are upweighted via --expert-weight.
set -euo pipefail
cd "$(dirname "$0")/.."

games=$(ls data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
expert=$(rg -l '"source":"hsreplay_expert"' data/*.jsonl 2>/dev/null | wc -l | tr -d ' ' || true)
echo "Retraining the eval net on the meta + ${games} recorded game file(s) (${expert} expert-tagged)…"
python -m ml.train_eval_net --epochs 40 --trajectories data/ --expert-weight "${EXPERT_WEIGHT:-3}"
echo "Done."
