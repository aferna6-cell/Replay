"""Native Windows overlay bridge (WSL) — offline tests, no PowerShell."""

import os

from hsbg_coach.win_overlay import WindowsOverlay, powershell_exe


def test_update_writes_atomically_and_dedupes(tmp_path):
    ov = WindowsOverlay(panel_file=str(tmp_path / "panel.txt"))
    ov.update("hello")
    assert (tmp_path / "panel.txt").read_text() == "hello"
    mtime = os.path.getmtime(tmp_path / "panel.txt")
    ov.update("hello")                    # unchanged -> no rewrite
    assert os.path.getmtime(tmp_path / "panel.txt") == mtime
    ov.update("world")
    assert (tmp_path / "panel.txt").read_text() == "world"
    assert not (tmp_path / "panel.txt.tmp").exists()


def test_powershell_missing_in_plain_linux():
    assert powershell_exe() is None       # this container has no Windows side


def test_start_without_powershell_raises_actionably(tmp_path, monkeypatch):
    import pytest
    ov = WindowsOverlay(panel_file=str(tmp_path / "p.txt"))
    monkeypatch.setattr("hsbg_coach.win_overlay.powershell_exe", lambda: None)
    with pytest.raises(RuntimeError) as e:
        ov.start()
    assert "powershell.exe" in str(e.value)
