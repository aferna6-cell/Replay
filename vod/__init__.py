"""VOD ingestion — reconstruct BG trajectories from streamer videos.

Pipeline (specs/vod-ingestion_spec.md, rung 2):
  frames.py       download VOD + extract/dedupe frames (yt-dlp + ffmpeg + PIL)
  state_read.py   Claude vision: cheap phase classify, full recruit-state read
  reconstruct.py  frame states -> per-turn trajectory records (recorder schema)
  validate.py     accuracy gate vs a Power.log ground-truth trajectory

Orchestrated by scripts/ingest_vod.py. Output lands in data/vods/*.jsonl and
trains the eval net at the highest sample weight (ml/train_eval_net.py
--vod-weight).
"""
