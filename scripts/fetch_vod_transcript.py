#!/usr/bin/env python3
"""Fetch a streamer VOD's transcript (YouTube/Twitch) into data/vods/.

First rung of the VOD-ingestion ladder (specs/vod-ingestion_spec.md):
transcripts capture the *reasoning* of high-level players — why they pivot,
what they scout for, when they level — which no aggregate stat carries. The
raw board states themselves need the vision rung (same spec, not built yet).

Requires yt-dlp (``pip install yt-dlp``). Uses the platform's subtitles when
present, else auto-generated captions. Output stays LOCAL (data/vods/ is
gitignored — transcripts of other people's content don't belong in the repo).

Usage:
    python scripts/fetch_vod_transcript.py <url> [<url> ...]
    python scripts/fetch_vod_transcript.py --lang en <url>

Output per VOD:
    data/vods/<video-id>.txt        cleaned transcript, [mm:ss] timestamped
    data/vods/<video-id>.info.json  title/channel/duration metadata
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "vods"


def _clean_vtt(vtt_text: str) -> str:
    """WebVTT -> '[mm:ss] line' text, deduping the rolling repeats that
    auto-captions emit (each cue re-shows the previous line)."""
    out, last = [], None
    stamp = None
    for raw in vtt_text.splitlines():
        line = raw.strip()
        m = re.match(r"(\d+):(\d+):(\d+)[.,]\d+\s+-->", line)
        if m:
            h, mnt, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            total = h * 3600 + mnt * 60 + s
            stamp = f"[{total // 60:d}:{total % 60:02d}]"
            continue
        if (not line or line == "WEBVTT" or line.isdigit()
                or line.startswith(("Kind:", "Language:", "NOTE"))):
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()   # strip cue tags
        if not text or text == last:
            continue
        # rolling-caption overlap: skip if the previous line ends with this one
        if last and (last.endswith(text) or text.startswith(last)):
            last = text if text.startswith(last) else last
            if text.startswith(last):
                continue
        out.append(f"{stamp or '[0:00]'} {text}")
        last = text
    return "\n".join(out) + "\n"


def fetch(url: str, lang: str) -> bool:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        print("yt-dlp not found — install it with:  pip install yt-dlp")
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run(
            [ytdlp, "--skip-download", "--write-info-json",
             "--write-subs", "--write-auto-subs",
             "--sub-langs", f"{lang}.*,{lang}", "--sub-format", "vtt",
             "-o", str(Path(tmp) / "%(id)s.%(ext)s"), url],
            capture_output=True, text=True)
        if res.returncode != 0:
            print(f"yt-dlp failed for {url}:\n{res.stderr.strip()[-500:]}")
            return False
        tmp_path = Path(tmp)
        infos = list(tmp_path.glob("*.info.json"))
        vtts = sorted(tmp_path.glob("*.vtt"))
        if not vtts:
            print(f"No subtitles/captions available for {url}")
            return False
        vid = infos[0].stem.replace(".info", "") if infos else vtts[0].stem.split(".")[0]
        transcript = _clean_vtt(vtts[0].read_text(encoding="utf-8",
                                                  errors="replace"))
        (OUT_DIR / f"{vid}.txt").write_text(transcript, encoding="utf-8")
        if infos:
            info = json.loads(infos[0].read_text(encoding="utf-8"))
            meta = {k: info.get(k) for k in
                    ("id", "title", "channel", "uploader", "duration",
                     "upload_date", "webpage_url")}
            (OUT_DIR / f"{vid}.info.json").write_text(
                json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        lines = transcript.count("\n")
        print(f"Saved {OUT_DIR / (vid + '.txt')}  ({lines} lines)")
    return True


def latest_video_ids(source_url: str, n: int) -> list:
    """Most recent video ids from a channel or a playlist (metadata only).
    n <= 0 means ALL entries (the first-pass backfill)."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        print("yt-dlp not found — install it with:  pip install yt-dlp")
        return []
    url = source_url.rstrip("/")
    is_playlist = "list=" in url or "/playlist" in url
    if not is_playlist and not url.endswith("/videos"):
        url += "/videos"
    cmd = [ytdlp, "--flat-playlist", "--print", "id"]
    if n > 0:
        cmd += ["-I", f"1:{n}"]
    res = subprocess.run(cmd + [url], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"listing failed:\n{res.stderr.strip()[-400:]}")
        return []
    return [v for v in res.stdout.split() if v]


def find_playlist(channel_url: str, title_pattern: str) -> str:
    """Resolve a channel playlist by (case-insensitive) title substring —
    e.g. --playlist-title 'season 14' on Rdu's channel."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        print("yt-dlp not found — install it with:  "
              "pip install -r requirements-vod.txt")
        return ""
    url = channel_url.rstrip("/")
    if not url.endswith("/playlists"):
        url += "/playlists"
    res = subprocess.run(
        [ytdlp, "--flat-playlist", "--print", "%(id)s|%(title)s", url],
        capture_output=True, text=True)
    if res.returncode != 0:
        print(f"playlist listing failed:\n{res.stderr.strip()[-400:]}")
        return ""
    want = title_pattern.lower()
    for line in res.stdout.splitlines():
        pid, _, title = line.partition("|")
        if want in title.lower():
            print(f"Playlist matched: {title.strip()}")
            return f"https://www.youtube.com/playlist?list={pid.strip()}"
    print(f"No playlist title containing '{title_pattern}' on {url}")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("urls", nargs="*", help="VOD URLs (YouTube or Twitch)")
    ap.add_argument("--channel",
                    help="a channel URL — fetch its latest uploads instead "
                         "of naming videos one by one")
    ap.add_argument("--playlist",
                    help="a playlist URL — fetch from this playlist")
    ap.add_argument("--playlist-title",
                    help="with --channel: resolve the playlist whose title "
                         "contains this text (e.g. 'season 14')")
    ap.add_argument("--latest", type=int, default=5,
                    help="how many recent videos (default 5)")
    ap.add_argument("--all", action="store_true",
                    help="fetch EVERY video in the source (first-pass backfill)")
    ap.add_argument("--lang", default="en", help="subtitle language (default en)")
    args = ap.parse_args()

    source = args.playlist
    if not source and args.channel and args.playlist_title:
        source = find_playlist(args.channel, args.playlist_title)
        if not source:
            return 1
    source = source or args.channel

    urls = list(args.urls)
    if source:
        ids = latest_video_ids(source, 0 if args.all else args.latest)
        skipped = [v for v in ids if (OUT_DIR / f"{v}.txt").exists()]
        fresh = [v for v in ids if v not in skipped]
        if skipped:
            print(f"{len(skipped)} of {len(ids)} already fetched.")
        urls += [f"https://www.youtube.com/watch?v={v}" for v in fresh]
    if not urls:
        print("Nothing new to fetch." if source
              else "Give VOD URLs, --channel, or --playlist.")
        return 0 if source else 2
    failures = sum(0 if fetch(u, args.lang) else 1 for u in urls)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
