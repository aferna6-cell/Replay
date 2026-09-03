"""Filesystem locations + project-wide knobs.

Hearthstone reads a ``log.config`` from a per-user config dir and, when
``FilePrinting=true``, writes each logger to its own file (e.g. ``Power.log``)
under a ``Logs`` dir. The exact locations differ Mac vs Windows.

IMPORTANT: the paths below are best-known defaults. They are NOT verified on a
real machine in this scaffold. ``detect`` (see cli.py) searches candidates and
reports what actually exists, so we never silently parse the wrong file.
"""

import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional


# --- Adaptive population-vs-personal weighting -----------------------------
# Population priors dominate early (larger, less noisy sample); personal weight
# grows as the personal dataset proves itself. This is a starting default and a
# dial, not a constant — see README "Data sourcing".
WEIGHTING = {
    "population_start": 0.80,   # weight on population priors at game 0
    "population_floor": 0.40,   # never let population drop below this
    "personal_full_at_games": 1500,  # personal weight maxes out around here
}


def _expand(*parts: str) -> str:
    return os.path.expanduser(os.path.join(*parts))


# Candidate log *base* dirs. Modern Hearthstone writes a NEW timestamped session
# folder per launch (e.g. .../Logs/Hearthstone_2026_06_24_23_17_43/Power.log), so
# detection globs these bases for both a flat Power.log and Hearthstone_*/Power.log
# and picks the most recently modified — verified on macOS (logs live inside the
# app install at /Applications/Hearthstone/Logs).
def log_dir_candidates() -> List[str]:
    if sys.platform == "darwin":
        return [
            "/Applications/Hearthstone/Logs",
            _expand("~/Applications/Hearthstone/Logs"),
            _expand("~/Library/Logs/Blizzard/Hearthstone"),
            _expand("~/Library/Preferences/Blizzard/Hearthstone/Logs"),
        ]
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA", _expand("~/AppData/Local"))
        return [
            r"C:\Program Files (x86)\Hearthstone\Logs",
            r"C:\Hearthstone\Logs",
            os.path.join(local, "Blizzard", "Hearthstone", "Logs"),
            os.path.join(local, "Blizzard", "Hearthstone"),
        ]
    # Linux (e.g. Lutris/Wine) — best effort.
    user = os.environ.get("USER", "user")
    return [
        _expand("~/Games/hearthstone/drive_c/Program Files (x86)/Hearthstone/Logs"),
        _expand("~/.wine/drive_c/Program Files (x86)/Hearthstone/Logs"),
        _expand("~/.wine/drive_c/users", user,
                "AppData/Local/Blizzard/Hearthstone/Logs"),
    ]


def newest_power_log(dirs: List[str]) -> Optional[str]:
    """The most recently modified Power.log across the candidate dirs, looking
    both flat and inside per-launch ``Hearthstone_*`` session folders."""
    matches: List[str] = []
    for d in dirs:
        matches += glob.glob(os.path.join(d, "Power.log"))
        matches += glob.glob(os.path.join(d, "Hearthstone_*", "Power.log"))
    matches = [m for m in matches if os.path.isfile(m)]
    return max(matches, key=os.path.getmtime) if matches else None


def log_config_path_candidates() -> List[str]:
    if sys.platform == "darwin":
        return [_expand("~/Library/Preferences/Blizzard/Hearthstone/log.config")]
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA", _expand("~/AppData/Local"))
        return [os.path.join(local, "Blizzard", "Hearthstone", "log.config")]
    return [_expand("~/.wine/drive_c/users", os.environ.get("USER", "user"),
                    "AppData/Local/Blizzard/Hearthstone/log.config")]


@dataclass
class Paths:
    log_dir: Optional[str]
    power_log: Optional[str]
    log_config: str

    @classmethod
    def detect(cls) -> "Paths":
        dirs = log_dir_candidates()
        # Pick the newest session's Power.log (handles per-launch folders).
        power_log = newest_power_log(dirs)
        if power_log:
            log_dir = os.path.dirname(power_log)
        else:
            log_dir = next((d for d in dirs if os.path.isdir(d)), None)
        # log.config we may need to *create*, so just take the first candidate.
        log_config = log_config_path_candidates()[0]
        return cls(log_dir=log_dir, power_log=power_log, log_config=log_config)


# Where we write recorded trajectories.
DATA_DIR = _expand(os.path.join(os.path.dirname(__file__), "..", "data"))
