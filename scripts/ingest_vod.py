#!/usr/bin/env python3
"""Ingest a streamer VOD into training trajectories (specs/vod-ingestion_spec.md).

    python scripts/ingest_vod.py <url-or-video-file>
    python scripts/ingest_vod.py <url> --section 00:10:00-00:40:00
    python scripts/ingest_vod.py <url> --cookies-from-browser chrome
    python scripts/ingest_vod.py <url> --dry-run       # no API calls, no cost

Stages: download (yt-dlp) -> frames (ffmpeg, 1 per 3s) -> dedupe (perceptual
hash) -> phase classify every frame (Claude Haiku, thumbnails) -> full state
read of recruit/endscreen frames (Claude Opus 5, structured output) ->
reconstruct per-turn trajectories -> data/vods/vod-<id>.jsonl.

Cost control: --max-reads caps the expensive reads (default 400); the run
prints its planned call counts before spending. Needs ANTHROPIC_API_KEY (or an
`ant auth login` profile), yt-dlp, ffmpeg, pillow, anthropic
(pip install -r requirements-vod.txt).

Train with the result (VOD data weighted highest by default):
    python -m ml.train_eval_net --trajectories data/ --epochs 40
    # data/vods/*.jsonl is folded in automatically at --vod-weight (default 3.0)
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vod import frames as F                      # noqa: E402
from vod.reconstruct import FrameRead, reconstruct  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "vods"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="VOD URL (YouTube/Twitch) or local video file")
    ap.add_argument("--section", help="time range like 00:10:00-00:40:00")
    ap.add_argument("--interval", type=float, default=F.DEFAULT_INTERVAL_S,
                    help="seconds between sampled frames (default 3)")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER",
                    help="pass your browser's cookies to yt-dlp "
                         "(chrome/firefox/…) if YouTube demands sign-in")
    ap.add_argument("--max-reads", type=int, default=400,
                    help="cap on full state reads (default 400)")
    ap.add_argument("--read-model", default=None,
                    help="override the state-read model")
    ap.add_argument("--dry-run", action="store_true",
                    help="download + frame stages only; no API calls")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the extracted frames cache for debugging")
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_file():
        video, vod_id = src, src.stem
    else:
        print("Downloading VOD…")
        video = F.download(args.source, OUT_DIR / "videos",
                           section=args.section,
                           cookies_browser=args.cookies_from_browser)
        vod_id = video.stem
        print(f"  -> {video}")

    frames_dir = OUT_DIR / "frames" / vod_id
    print(f"Extracting frames (1 per {args.interval:g}s)…")
    all_frames = F.extract_frames(video, frames_dir, interval_s=args.interval)
    kept = F.dedupe(all_frames)
    print(f"  {len(all_frames)} frames -> {len(kept)} after dedupe")

    if args.dry_run:
        print(f"Dry run: would classify {len(kept)} thumbnails (Haiku) and "
              f"full-read up to {min(len(kept), args.max_reads)} of them "
              f"(Opus 5). Frames kept at {frames_dir}")
        return 0

    from vod.state_read import FrameReader      # deferred: needs anthropic
    reader = FrameReader(**({"read_model": args.read_model}
                            if args.read_model else {}))

    print(f"Classifying {len(kept)} frames…")
    reads = []
    to_read = []
    for f in kept:
        phase = reader.classify_frame(f)
        ts = F.frame_timestamp(f, args.interval)
        if phase in ("recruit", "endscreen"):
            to_read.append((ts, phase, f))
        else:
            reads.append(FrameRead(ts=ts, phase=phase))

    if len(to_read) > args.max_reads:
        print(f"  {len(to_read)} readable frames > --max-reads {args.max_reads}"
              f" — sampling evenly (raise --max-reads for full coverage)")
        step = len(to_read) / args.max_reads
        to_read = [to_read[int(i * step)] for i in range(args.max_reads)]

    print(f"Reading {len(to_read)} recruit/endscreen frames…")
    for i, (ts, phase, f) in enumerate(to_read):
        state = None
        try:
            state = reader.read_state(f)
        except Exception as exc:            # one bad frame never kills a run
            print(f"  WARN read failed at {ts:.0f}s: {exc}")
        reads.append(FrameRead(ts=ts, phase=phase, state=state))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(to_read)}")

    reads.sort(key=lambda r: r.ts)
    games = reconstruct(reads, vod_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for records in games:
        if not records:
            continue
        path = OUT_DIR / f"{records[0]['game_id']}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        tag = (f"placement {records[0]['placement']}"
               if records[0]["placement"] else "no placement read")
        print(f"  {path.name}: {len(records)} turns ({tag})")
        written += 1

    if not args.keep_frames:
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"Done: {written} game(s), API calls {reader.calls}. "
          f"Fold into the net:  ./scripts/retrain.sh")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
