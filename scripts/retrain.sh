#!/usr/bin/env bash
# Quick retrain on your own games — run after a play session to fold data/*.jsonl
# into the board-evaluation net. Adaptive population-vs-personal weighting is
# applied automatically (see hsbg_coach/config.py WEIGHTING).
#
# Usage:
#   ./scripts/retrain.sh              # full retrain (needs network/meta + torch)
#   ./scripts/retrain.sh --dry-run    # validate recordings + print mix (fast)
#   ./scripts/retrain.sh --status     # fingerprint / game counts only
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--status" ]]; then
  exec python -m hsbg_coach learn --status
fi
if [[ "${1:-}" == "--dry-run" ]] || [[ "${DRY_RUN:-}" == "1" ]]; then
  exec python -m hsbg_coach learn --dry-run
fi

games=$(ls data/game-*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "Retraining eval net on meta + ${games} recorded game file(s)…"
echo "(personal weight grows with game count — see: python -m hsbg_coach learn --status)"
python -m hsbg_coach learn --epochs "${EPOCHS:-40}"
echo "Done. Re-open or keep watching — live coach hot-swaps ml/eval_net.pt by mtime."
