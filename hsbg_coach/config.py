"""Filesystem locations + project-wide knobs.

Hearthstone reads a ``log.config`` from a per-user config dir and, when
``FilePrinting=true``, writes each logger to its own file (e.g. ``Power.log``)
under a ``Logs`` dir. The exact locations differ Mac vs Windows.

IMPORTANT: the paths below are best-known defaults. They are NOT verified on a
real machine in this scaffold. ``detect`` (see cli.py) searches candidates and
reports what actually exists, so we never silently parse the wrong file.
"""

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


# Candidate locations, most-likely first. ``detect`` picks the first that exists.
def log_dir_candidates() -> List[str]:
    if sys.platform == "darwin":
        return [
            _expand("~/Library/Logs/Blizzard/Hearthstone"),
            _expand("~/Library/Preferences/Blizzard/Hearthstone/Logs"),
        ]
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA", _expand("~/AppData/Local"))
        return [
            os.path.join(local, "Blizzard", "Hearthstone", "Logs"),
            os.path.join(local, "Blizzard", "Hearthstone"),
        ]
    # Linux (e.g. Lutris/Wine) — best effort.
    return [_expand("~/.wine/drive_c/users", os.environ.get("USER", "user"),
                    "AppData/Local/Blizzard/Hearthstone/Logs")]


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
        log_dir = next((d for d in log_dir_candidates() if os.path.isdir(d)), None)
        power_log = None
        if log_dir:
            candidate = os.path.join(log_dir, "Power.log")
            power_log = candidate if os.path.isfile(candidate) else None
        # log.config we may need to *create*, so just take the first candidate.
        log_config = log_config_path_candidates()[0]
        return cls(log_dir=log_dir, power_log=power_log, log_config=log_config)


# Where we write recorded trajectories.
DATA_DIR = _expand(os.path.join(os.path.dirname(__file__), "..", "data"))
