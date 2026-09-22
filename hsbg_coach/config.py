"""Filesystem locations + project-wide knobs.

Hearthstone reads a ``log.config`` from a per-user config dir and, when
``FilePrinting=true``, writes each logger to its own file (e.g. ``Power.log``)
under a ``Logs`` dir. The exact locations differ Mac vs Windows.

IMPORTANT: the paths below are best-known defaults. They are NOT verified on a
real machine in this scaffold. ``detect`` (see cli.py) searches candidates and
reports what actually exists, so we never silently parse the wrong file.

On Linux, Hearthstone usually runs under Wine / Lutris / Bottles / Steam Proton.
``log_dir_candidates`` and ``discover_power_logs`` therefore expand several
common prefixes via bounded globs (never a full home-directory walk).
"""

import glob
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


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


def _home() -> str:
    return os.path.expanduser("~")


def _unique_preserve(paths: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _glob_dirs(patterns: Sequence[str]) -> List[str]:
    """Expand glob patterns; return only existing directories, deduped."""
    hits: List[str] = []
    for pat in patterns:
        for match in glob.glob(os.path.expanduser(pat), recursive=True):
            if os.path.isdir(match):
                hits.append(match)
    return _unique_preserve(hits)


# Candidate log *base* dirs. Modern Hearthstone writes a NEW timestamped session
# folder per launch (e.g. .../Logs/Hearthstone_2026_06_24_23_17_43/Power.log), so
# detection globs these bases for both a flat Power.log and Hearthstone_*/Power.log
# and picks the most recently modified — verified on macOS (logs live inside the
# app install at /Applications/Hearthstone/Logs).
def log_dir_candidates(home: Optional[str] = None) -> List[str]:
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
    # Linux (Wine / Lutris / Bottles / Steam Proton) — bounded globs only.
    h = home if home is not None else _home()
    fixed = [
        os.path.join(h, "Games", "hearthstone", "drive_c",
                     "Program Files (x86)", "Hearthstone", "Logs"),
        os.path.join(h, ".wine", "drive_c", "Program Files (x86)",
                     "Hearthstone", "Logs"),
        os.path.join(h, ".wine", "drive_c", "users",
                     os.environ.get("USER", "user"),
                     "AppData", "Local", "Blizzard", "Hearthstone", "Logs"),
    ]
    patterns = [
        # Wine default prefix
        os.path.join(h, ".wine", "drive_c", "Program Files*", "Hearthstone", "Logs"),
        os.path.join(h, ".wine", "drive_c", "users", "*",
                     "AppData", "Local", "Blizzard", "Hearthstone", "Logs"),
        # Lutris / manual Wine prefixes under ~/Games (depth-limited: 1–2 game dirs)
        os.path.join(h, "Games", "*", "drive_c", "Program Files*", "Hearthstone", "Logs"),
        os.path.join(h, "Games", "*", "*", "drive_c", "Program Files*",
                     "Hearthstone", "Logs"),
        os.path.join(h, "Games", "*", "drive_c", "users", "*",
                     "AppData", "Local", "Blizzard", "Hearthstone", "Logs"),
        os.path.join(h, "Games", "*", "*", "drive_c", "users", "*",
                     "AppData", "Local", "Blizzard", "Hearthstone", "Logs"),
        # Bottles
        os.path.join(h, ".local", "share", "bottles", "bottles", "*", "drive_c",
                     "Program Files*", "Hearthstone", "Logs"),
        os.path.join(h, ".local", "share", "bottles", "bottles", "*", "drive_c",
                     "users", "*", "AppData", "Local", "Blizzard",
                     "Hearthstone", "Logs"),
        # Steam Proton (legacy + Flatpak-ish layouts)
        os.path.join(h, ".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                     "drive_c", "Program Files*", "Hearthstone", "Logs"),
        os.path.join(h, ".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                     "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                     "Hearthstone", "Logs"),
        os.path.join(h, ".local", "share", "Steam", "steamapps", "compatdata", "*",
                     "pfx", "drive_c", "Program Files*", "Hearthstone", "Logs"),
        os.path.join(h, ".local", "share", "Steam", "steamapps", "compatdata", "*",
                     "pfx", "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                     "Hearthstone", "Logs"),
    ]
    return _unique_preserve(fixed + _glob_dirs(patterns))


# Roots under which we may look for Power.log (NOT all of ~).
_DISCOVERY_ROOT_REL = (
    ".wine",
    "Games",
    os.path.join(".local", "share", "bottles"),
    os.path.join(".steam", "steam", "steamapps", "compatdata"),
    os.path.join(".local", "share", "Steam", "steamapps", "compatdata"),
)

# Bounded Power.log patterns relative to home (no recursive home walk).
_POWER_LOG_PATTERNS = (
    # Wine
    os.path.join(".wine", "drive_c", "Program Files*", "Hearthstone", "Logs",
                 "Power.log"),
    os.path.join(".wine", "drive_c", "Program Files*", "Hearthstone", "Logs",
                 "Hearthstone_*", "Power.log"),
    os.path.join(".wine", "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                 "Hearthstone", "Logs", "Power.log"),
    os.path.join(".wine", "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                 "Hearthstone", "Logs", "Hearthstone_*", "Power.log"),
    # Games / Lutris (1–2 levels before drive_c)
    os.path.join("Games", "*", "drive_c", "Program Files*", "Hearthstone", "Logs",
                 "Power.log"),
    os.path.join("Games", "*", "drive_c", "Program Files*", "Hearthstone", "Logs",
                 "Hearthstone_*", "Power.log"),
    os.path.join("Games", "*", "*", "drive_c", "Program Files*", "Hearthstone",
                 "Logs", "Power.log"),
    os.path.join("Games", "*", "*", "drive_c", "Program Files*", "Hearthstone",
                 "Logs", "Hearthstone_*", "Power.log"),
    os.path.join("Games", "*", "drive_c", "users", "*", "AppData", "Local",
                 "Blizzard", "Hearthstone", "Logs", "Power.log"),
    os.path.join("Games", "*", "drive_c", "users", "*", "AppData", "Local",
                 "Blizzard", "Hearthstone", "Logs", "Hearthstone_*", "Power.log"),
    os.path.join("Games", "*", "*", "drive_c", "users", "*", "AppData", "Local",
                 "Blizzard", "Hearthstone", "Logs", "Power.log"),
    os.path.join("Games", "*", "*", "drive_c", "users", "*", "AppData", "Local",
                 "Blizzard", "Hearthstone", "Logs", "Hearthstone_*", "Power.log"),
    # Bottles
    os.path.join(".local", "share", "bottles", "bottles", "*", "drive_c",
                 "Program Files*", "Hearthstone", "Logs", "Power.log"),
    os.path.join(".local", "share", "bottles", "bottles", "*", "drive_c",
                 "Program Files*", "Hearthstone", "Logs", "Hearthstone_*",
                 "Power.log"),
    os.path.join(".local", "share", "bottles", "bottles", "*", "drive_c", "users",
                 "*", "AppData", "Local", "Blizzard", "Hearthstone", "Logs",
                 "Power.log"),
    os.path.join(".local", "share", "bottles", "bottles", "*", "drive_c", "users",
                 "*", "AppData", "Local", "Blizzard", "Hearthstone", "Logs",
                 "Hearthstone_*", "Power.log"),
    # Steam Proton
    os.path.join(".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "Program Files*", "Hearthstone", "Logs", "Power.log"),
    os.path.join(".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "Program Files*", "Hearthstone", "Logs",
                 "Hearthstone_*", "Power.log"),
    os.path.join(".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                 "Hearthstone", "Logs", "Power.log"),
    os.path.join(".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                 "Hearthstone", "Logs", "Hearthstone_*", "Power.log"),
    os.path.join(".local", "share", "Steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "Program Files*", "Hearthstone", "Logs", "Power.log"),
    os.path.join(".local", "share", "Steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "Program Files*", "Hearthstone", "Logs",
                 "Hearthstone_*", "Power.log"),
    os.path.join(".local", "share", "Steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                 "Hearthstone", "Logs", "Power.log"),
    os.path.join(".local", "share", "Steam", "steamapps", "compatdata", "*", "pfx",
                 "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                 "Hearthstone", "Logs", "Hearthstone_*", "Power.log"),
)


def discovery_roots(home: Optional[str] = None) -> List[str]:
    """Likely Wine/Proton roots under ``home`` (existing dirs only)."""
    h = home if home is not None else _home()
    return [p for p in (os.path.join(h, rel) for rel in _DISCOVERY_ROOT_REL)
            if os.path.isdir(p)]


def discover_power_logs(home: Optional[str] = None) -> List[str]:
    """Bounded search for ``Power.log`` under Wine/Games/Bottles/Steam roots.

    Never walks all of ``~``. Results are sorted newest-mtime first.
    """
    h = home if home is not None else _home()
    matches: List[str] = []
    for rel in _POWER_LOG_PATTERNS:
        pat = os.path.join(h, rel)
        for match in glob.glob(pat):
            if os.path.isfile(match):
                matches.append(match)
    matches = _unique_preserve(matches)
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches


def newest_power_log(dirs: List[str]) -> Optional[str]:
    """The most recently modified Power.log across the candidate dirs, looking
    both flat and inside per-launch ``Hearthstone_*`` session folders."""
    matches: List[str] = []
    for d in dirs:
        matches += glob.glob(os.path.join(d, "Power.log"))
        matches += glob.glob(os.path.join(d, "Hearthstone_*", "Power.log"))
    matches = [m for m in matches if os.path.isfile(m)]
    return max(matches, key=os.path.getmtime) if matches else None


def find_command_hints(home: Optional[str] = None) -> List[str]:
    """Shell ``find`` commands a user can run to locate Power.log manually."""
    h = home if home is not None else _home()
    roots = [
        os.path.join(h, ".wine"),
        os.path.join(h, "Games"),
        os.path.join(h, ".local", "share", "bottles"),
        os.path.join(h, ".steam", "steam", "steamapps", "compatdata"),
        os.path.join(h, ".local", "share", "Steam", "steamapps", "compatdata"),
    ]
    cmds = []
    for root in roots:
        cmds.append(
            f'find "{root}" -name Power.log 2>/dev/null'
        )
    return cmds


def log_config_path_candidates(home: Optional[str] = None) -> List[str]:
    if sys.platform == "darwin":
        return [_expand("~/Library/Preferences/Blizzard/Hearthstone/log.config")]
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA", _expand("~/AppData/Local"))
        return [os.path.join(local, "Blizzard", "Hearthstone", "log.config")]
    h = home if home is not None else _home()
    fixed = [
        os.path.join(h, ".wine", "drive_c", "users",
                     os.environ.get("USER", "user"),
                     "AppData", "Local", "Blizzard", "Hearthstone", "log.config"),
    ]
    patterns = [
        os.path.join(h, ".wine", "drive_c", "users", "*",
                     "AppData", "Local", "Blizzard", "Hearthstone", "log.config"),
        os.path.join(h, "Games", "*", "drive_c", "users", "*",
                     "AppData", "Local", "Blizzard", "Hearthstone", "log.config"),
        os.path.join(h, "Games", "*", "*", "drive_c", "users", "*",
                     "AppData", "Local", "Blizzard", "Hearthstone", "log.config"),
        os.path.join(h, ".local", "share", "bottles", "bottles", "*", "drive_c",
                     "users", "*", "AppData", "Local", "Blizzard", "Hearthstone",
                     "log.config"),
        os.path.join(h, ".steam", "steam", "steamapps", "compatdata", "*", "pfx",
                     "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                     "Hearthstone", "log.config"),
        os.path.join(h, ".local", "share", "Steam", "steamapps", "compatdata", "*",
                     "pfx", "drive_c", "users", "*", "AppData", "Local", "Blizzard",
                     "Hearthstone", "log.config"),
    ]
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    return _unique_preserve(fixed + hits)


def _logs_base_dir(power_log: str) -> str:
    """Directory that directly contains Power.log or the Hearthstone_* session."""
    logs_dir = os.path.dirname(os.path.abspath(power_log))
    if os.path.basename(logs_dir).startswith("Hearthstone_"):
        logs_dir = os.path.dirname(logs_dir)
    return logs_dir


def _drive_c_root(path: str) -> Optional[str]:
    """Return the ``drive_c`` directory containing ``path``, if any."""
    parts = os.path.abspath(path).split(os.sep)
    for i, part in enumerate(parts):
        if part.lower() == "drive_c":
            root = os.sep.join(parts[: i + 1])
            if path.startswith(os.sep) and not root.startswith(os.sep):
                root = os.sep + root
            return root
    return None


def _config_near_power_log(power_log: str) -> Optional[str]:
    """Prefer log.config from the same Wine/pfx tree as the Power.log."""
    drive_c = _drive_c_root(power_log)
    if drive_c:
        matches = [
            m for m in glob.glob(os.path.join(
                drive_c, "users", "*", "AppData", "Local", "Blizzard",
                "Hearthstone", "log.config"))
            if os.path.isfile(m)
        ]
        if matches:
            return max(matches, key=os.path.getmtime)
    # Config may sit beside Logs under .../Blizzard/Hearthstone/
    sibling = os.path.join(os.path.dirname(_logs_base_dir(power_log)), "log.config")
    if os.path.isfile(sibling):
        return sibling
    return None


def _pick_log_config(power_log: Optional[str],
                     home: Optional[str] = None) -> str:
    """Pick log.config: near newest Power.log if found, else first existing,
    else first candidate (for creation by ``setup``)."""
    if power_log:
        near = _config_near_power_log(power_log)
        if near:
            return near
    cands = log_config_path_candidates(home=home)
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


@dataclass
class Paths:
    log_dir: Optional[str]
    power_log: Optional[str]
    log_config: str

    @classmethod
    def detect(cls, home: Optional[str] = None) -> "Paths":
        dirs = log_dir_candidates(home=home)
        # Prefer newest among candidate dirs; fall back to bounded discovery.
        power_log = newest_power_log(dirs)
        if not power_log:
            discovered = discover_power_logs(home=home)
            power_log = discovered[0] if discovered else None
        if power_log:
            log_dir = os.path.dirname(power_log)
        else:
            log_dir = next((d for d in dirs if os.path.isdir(d)), None)
        log_config = _pick_log_config(power_log, home=home)
        return cls(log_dir=log_dir, power_log=power_log, log_config=log_config)


# Where we write recorded trajectories.
DATA_DIR = _expand(os.path.join(os.path.dirname(__file__), "..", "data"))
