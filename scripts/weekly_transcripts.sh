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

# Stream-VOD playlists, resolved by title on each channel. Format per entry:
# "<channel-url>|<playlist-title-substring>". Add more educational high-rank
# streamers as extra entries (space-separated) or via HSBG_SOURCES.
SOURCES="${HSBG_SOURCES:-https://www.youtube.com/channel/UCG9RWeCIYuqKKk-HTQHZLGQ|season 14}"  # Rdu stream VODs
LATEST="${HSBG_LATEST:-5}"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
  export PATH="$(pwd)/.venv/bin:$PATH"
else
  PY="$(command -v python3 || command -v python)"
fi

echo "==> 1/3  Fetch latest stream-VOD transcripts ($(date))"
for src in $SOURCES; do
  ch="${src%%|*}"; title="${src#*|}"
  "$PY" scripts/fetch_vod_transcript.py --channel "$ch" \
    --playlist-title "$title" --latest "$LATEST" \
    || echo "WARN: fetch failed for $ch (continuing)"
done

echo "==> 2/3  Distill new transcripts -> playbook + expert priors (subscription)"
"$PY" scripts/distill_transcripts.py --engine claude-code \
  || echo "WARN: distillation failed — transcripts are saved; re-run later:"\
          " python scripts/distill_transcripts.py --engine claude-code"

echo "==> 3/3  Commit + push expert priors (only if they moved)"
git add data/stats/expert_card_stats.json 2>/dev/null || true
if git diff --cached --quiet; then
  echo "Expert priors unchanged."
else
  git commit -m "Expert card priors refresh from streamer VODs ($(date +%Y-%m-%d))"
  git push || echo "WARN: push failed — commit is local; next run retries."
fi
echo "Playbook: data/vods/insights.md"
echo "Done."
