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
    # Linux: Hearthstone lives inside SOME Wine prefix (Lutris installs
    # under ~/Games/<slug> — usually "battlenet", not "hearthstone" —
    # Bottles/Flatpak elsewhere). A hardcoded prefix list ghost-pointed a
    # real machine at an empty ~/.wine (2026-08-20), so discover every
    # prefix and derive candidates from what actually exists.
    #
    # WSL: "linux" that is really a Windows machine — the game runs natively
    # on the Windows side and its files are reachable under /mnt/<drive>/.
    # (Owner's setup, 2026-08-20: coach in WSL, Hearthstone on Windows.)
    dirs: List[str] = []
    env = os.environ.get("HSBG_HS_DIR")        # explicit install-dir override
    if env:
        dirs += [os.path.join(env, "Logs"), env]
    if is_wsl():
        for hs in _wsl_windows_installs():
            dirs.append(os.path.join(hs, "Logs"))
        dirs += glob.glob(os.path.join(
            MNT_ROOT, "*", "Users", "*", "AppData", "Local", "Blizzard",
            "Hearthstone", "Logs"))
    for dc in wine_drive_cs():
        for pf in ("Program Files (x86)", "Program Files"):
            dirs.append(os.path.join(dc, pf, "Hearthstone", "Logs"))
        dirs += glob.glob(os.path.join(
            dc, "users", "*", "AppData", "Local", "Blizzard", "Hearthstone", "Logs"))
    return dirs


MNT_ROOT = "/mnt"                       # WSL's Windows-drive mount point


def is_wsl() -> bool:
    """Running inside Windows Subsystem for Linux (so the game and HDT are
    on the Windows side, under /mnt/<drive>/)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _wsl_windows_installs() -> List[str]:
    out = []
    for pf in ("Program Files (x86)", "Program Files"):
        out += glob.glob(os.path.join(MNT_ROOT, "*", pf, "Hearthstone"))
    return out


def wine_drive_cs() -> List[str]:
    """Every Wine-prefix drive_c on this machine (plain wine, Lutris,
    Bottles incl. Flatpak, Steam Proton compatdata) — install-carrying
    prefixes first so `setup` targets the prefix the game actually reads."""
    pats = [
        "~/.wine*/drive_c",
        "~/Games/*/drive_c",                                  # Lutris
        "~/.var/app/*/data/wine/drive_c",                     # Flatpak wine
        "~/.var/app/*/data/bottles/bottles/*/drive_c",        # Flatpak Bottles
        "~/.local/share/bottles/bottles/*/drive_c",           # Bottles
        "~/.local/share/lutris/prefixes/*/drive_c",
        "~/.steam/steam/steamapps/compatdata/*/pfx/drive_c",  # Proton
    ]
    found: List[str] = []
    for p in pats:
        found += glob.glob(_expand(p))
    found = sorted(set(found))
    found.sort(key=lambda dc: not _has_hearthstone(dc))
    return found


def _has_hearthstone(drive_c: str) -> bool:
    return any(os.path.isdir(os.path.join(drive_c, pf, "Hearthstone"))
               for pf in ("Program Files (x86)", "Program Files"))


def hearthstone_installs() -> List[str]:
    """Actual Hearthstone install dirs found (Wine prefixes + WSL /mnt)."""
    out = []
    env = os.environ.get("HSBG_HS_DIR")
    if env and os.path.isdir(env):
        out.append(env)
    if is_wsl():
        out += _wsl_windows_installs()
    for dc in wine_drive_cs():
        for pf in ("Program Files (x86)", "Program Files"):
            d = os.path.join(dc, pf, "Hearthstone")
            if os.path.isdir(d):
                out.append(d)
    return out


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
    # Linux: log.config must land in the SAME prefix the game reads.
    # wine_drive_cs() puts install-carrying prefixes first, so the first
    # candidate here is the right AppData for the real install — writing to
    # a prefix without Hearthstone configures nothing (the 2026-08-20 trap).
    cands: List[str] = []
    if is_wsl():
        # The game reads Windows-side AppData; write there via /mnt.
        for u in glob.glob(os.path.join(MNT_ROOT, "*", "Users", "*")):
            if os.path.basename(u).lower() in ("public", "default",
                                               "default user", "all users"):
                continue
            cands.append(os.path.join(
                u, "AppData", "Local", "Blizzard", "Hearthstone", "log.config"))
    for dc in wine_drive_cs():
        users = glob.glob(os.path.join(dc, "users", "*")) or \
            [os.path.join(dc, "users", os.environ.get("USER", "user"))]
        for u in users:
            if os.path.basename(u).lower() == "public":
                continue
            cands.append(os.path.join(
                u, "AppData", "Local", "Blizzard", "Hearthstone", "log.config"))
    return cands or [_expand("~/.wine/drive_c/users",
                             os.environ.get("USER", "user"),
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
