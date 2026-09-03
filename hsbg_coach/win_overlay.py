"""Native Windows overlay bridge for WSL (owner's setup, 2026-08-20).

Tk-over-WSLg proved unreliable as a game overlay: the terminal drops behind
Hearthstone on click, and WSLg's window management doesn't guarantee
topmost over a game the way a native window does. This module gets the
HDT-grade experience without installing anything on Windows: the coach
(in WSL) writes the panel text to a file and launches
``scripts/windows_overlay.ps1`` with ``powershell.exe`` — WSL can start
Windows executables directly — which opens a real Windows WPF window:
borderless, semi-transparent, draggable, Topmost (re-asserted, HDT-style),
never focus-stealing.

The panel file lives on the Linux side; PowerShell reads it through the
``wsl.localhost`` UNC path that ``wslpath -w`` produces. If ``wslpath``
or ``powershell.exe`` is missing we report why and the CLI falls back to
Tk, then to the terminal panel.
"""

import os
import subprocess
from typing import Optional

_PS_CANDIDATES = (
    "powershell.exe",                       # usually on WSL's PATH
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
)

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                       "windows_overlay.ps1")


def powershell_exe() -> Optional[str]:
    import shutil
    for cand in _PS_CANDIDATES:
        found = shutil.which(cand) if os.sep not in cand else (
            cand if os.path.isfile(cand) else None)
        if found:
            return found
    return None


def to_windows_path(path: str) -> str:
    """Linux path -> a path Windows programs can open (\\\\wsl.localhost\\...)."""
    out = subprocess.run(["wslpath", "-w", path], capture_output=True,
                         text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"wslpath failed for {path}: {out.stderr.strip()}")
    return out.stdout.strip()


class WindowsOverlay:
    """Launch the native panel and stream text to it via the panel file."""

    def __init__(self, panel_file: Optional[str] = None):
        from . import config
        self.panel_file = panel_file or os.path.join(
            config.DATA_DIR, "overlay_panel.txt")
        self._proc: Optional[subprocess.Popen] = None
        self._last: Optional[str] = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.panel_file), exist_ok=True)
        self.update("HSBG Coach — waiting for the game…")
        ps = powershell_exe()
        if ps is None:
            raise RuntimeError(
                "powershell.exe not reachable from WSL (is Windows interop "
                "disabled?) — cannot open the native overlay")
        script = to_windows_path(os.path.abspath(_SCRIPT))
        panel = to_windows_path(os.path.abspath(self.panel_file))
        # Keep PowerShell's stderr in a log so a first-run failure is
        # diagnosable instead of silently vanishing.
        self.log_file = self.panel_file + ".ps.log"
        log = open(self.log_file, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", script, "-PanelFile", panel],
            stdout=log, stderr=log)

    def update(self, text: str) -> None:
        if text == self._last:
            return
        self._last = text
        tmp = self.panel_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, self.panel_file)      # atomic: PS never reads halves

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
