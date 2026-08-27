#!/usr/bin/env bash
# Weekly streamer-transcript refresh: fetch the latest uploads from the
# configured channel(s), distill each new transcript into the strategy
# playbook via the Claude Code CLI (subscription — no API key), and rebuild
# data/vods/insights.md. Installed by scripts/schedule_wsl.sh; safe by hand.
#
# Add/replace channels by editing CHANNELS below, or set HSBG_CHANNELS
# (space-separated URLs) in the environment.
set -uo pipefail
cd "$(dirname "$0")/.."

CHANNELS="${HSBG_CHANNELS:-https://www.youtube.com/channel/UCG9RWeCIYuqKKk-HTQHZLGQ}"  # Rdu Hearthstone
LATEST="${HSBG_LATEST:-5}"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
  export PATH="$(pwd)/.venv/bin:$PATH"
else
  PY="$(command -v python3 || command -v python)"
fi

echo "==> 1/2  Fetch latest transcripts ($(date))"
for ch in $CHANNELS; do
  "$PY" scripts/fetch_vod_transcript.py --channel "$ch" --latest "$LATEST" \
    || echo "WARN: fetch failed for $ch (continuing)"
done

echo "==> 2/2  Distill new transcripts -> playbook (Claude Code subscription)"
"$PY" scripts/distill_transcripts.py --engine claude-code \
  || echo "WARN: distillation failed — transcripts are saved; re-run later:"\
          " python scripts/distill_transcripts.py --engine claude-code"

echo "Playbook: data/vods/insights.md"
echo "Done."
