"""Write Hearthstone's ``log.config`` so the game emits the logs we parse.

Without this, ``Power.log`` either doesn't exist or lacks the GameState output.
Hearthstone reads ``log.config`` only at launch, so a restart is required after
writing it.

We enable:
  - Power        : the authoritative game-state stream (entities, tags, blocks)
  - Zone         : zone moves (hand/play/graveyard) — useful cross-checks
  - LoadingScreen: scene changes — how we detect a Battlegrounds game (BACON)
"""

import os
from typing import Iterable

REQUIRED_LOGGERS = ("Power", "Zone", "LoadingScreen")

_SECTION_TEMPLATE = """[{name}]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false
Verbose=true
"""


def desired_config(loggers: Iterable[str] = REQUIRED_LOGGERS) -> str:
    return "\n".join(_SECTION_TEMPLATE.format(name=name) for name in loggers)


def existing_sections(text: str) -> set:
    return {
        line.strip()[1:-1]
        for line in text.splitlines()
        if line.strip().startswith("[") and line.strip().endswith("]")
    }


def ensure_log_config(path: str, loggers: Iterable[str] = REQUIRED_LOGGERS) -> bool:
    """Ensure every required logger section exists in ``path``.

    Returns True if the file was created or modified (=> restart Hearthstone).
    Existing sections are left untouched; only missing ones are appended, so we
    never clobber a config another tool (e.g. HDT) already manages.
    """
    loggers = list(loggers)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    current = ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            current = fh.read()

    present = existing_sections(current)
    missing = [name for name in loggers if name not in present]
    if not missing:
        return False

    addition = "\n".join(_SECTION_TEMPLATE.format(name=name) for name in missing)
    new_text = current.rstrip() + ("\n\n" if current.strip() else "") + addition
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    return True
