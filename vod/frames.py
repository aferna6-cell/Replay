"""VOD download + frame extraction/dedupe.

Downloading needs yt-dlp; extraction needs ffmpeg on PATH; dedupe needs PIL.
YouTube blocks datacenter IPs, so this stage runs on a residential machine —
pass ``cookies_browser`` ("chrome"/"firefox"/…) if yt-dlp still asks you to
prove you're not a bot.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from PIL import Image

# Recruit turns run 60-90s; one frame every 3s never misses a whole turn and
# keeps a 30-minute game to ~600 raw frames (far fewer after dedupe).
DEFAULT_INTERVAL_S = 3.0


class VodError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise VodError(f"{binary} not found on PATH — install it first "
                       f"({'pip install yt-dlp' if binary == 'yt-dlp' else 'https://ffmpeg.org'})")
    return path


def download(url: str, out_dir: Path, max_height: int = 1080,
             section: Optional[str] = None,
             cookies_browser: Optional[str] = None) -> Path:
    """Download the VOD's video track. `section` like "00:10:00-00:35:00"
    grabs a time range (much faster for long VODs)."""
    ytdlp = _require("yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [ytdlp, "-f", f"bv*[height<={max_height}]/b[height<={max_height}]",
           "-o", str(out_dir / "%(id)s.%(ext)s"), "--no-part"]
    if section:
        cmd += ["--download-sections", f"*{section}"]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise VodError(f"yt-dlp failed:\n{res.stderr.strip()[-800:]}")
    vids = sorted(out_dir.glob("*.*"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    vids = [v for v in vids if v.suffix.lower() in
            (".mp4", ".webm", ".mkv", ".mov", ".ts")]
    if not vids:
        raise VodError("yt-dlp reported success but no video file found")
    return vids[0]


def extract_frames(video: Path, frames_dir: Path,
                   interval_s: float = DEFAULT_INTERVAL_S) -> List[Path]:
    """Sample one JPEG every `interval_s` seconds. Filenames carry the frame
    index so timestamps recover as index * interval_s."""
    ffmpeg = _require("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "f%06d.jpg")
    res = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"fps=1/{interval_s}", "-q:v", "3", pattern],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise VodError(f"ffmpeg failed:\n{res.stderr.strip()[-800:]}")
    return sorted(frames_dir.glob("f*.jpg"))


def _ahash(img: Image.Image, size: int = 8) -> int:
    """Average hash — tiny, dependency-free perceptual hash."""
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p > avg else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedupe(frames: List[Path], threshold: int = 4) -> List[Path]:
    """Drop frames nearly identical to the previous kept one (static screens,
    long thinks). Keeps the FIRST of each run; the reader later wants the LAST
    frame of each recruit turn, which reconstruct.py handles from turn numbers,
    so here we only need one representative per visual state."""
    kept: List[Path] = []
    prev = None
    for f in frames:
        try:
            h = _ahash(Image.open(f))
        except OSError:
            continue
        if prev is None or _hamming(h, prev) > threshold:
            kept.append(f)
            prev = h
    return kept


def frame_timestamp(frame: Path, interval_s: float = DEFAULT_INTERVAL_S) -> float:
    """Seconds into the video for a frame produced by extract_frames."""
    return (int(frame.stem[1:]) - 1) * interval_s
