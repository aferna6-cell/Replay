"""Follow a log file like ``tail -f``, surviving truncation and rotation.

Hearthstone truncates/recreates ``Power.log`` on each launch, so a naive reader
that holds a file handle goes stale. We track the inode + size and reopen when
the file shrinks or is replaced. Stdlib only (polling) — no watchdog dependency,
so it runs anywhere Python does.
"""

import os
import time
from typing import Callable, Iterator, Optional


def _inode(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_ino
    except OSError:
        return None


def tail_lines(
    path: str,
    from_start: bool = False,
    poll_interval: float = 0.05,        # 20 Hz — pick up new log lines near-instantly
    stop_after_eof: bool = False,
) -> Iterator[str]:
    """Yield complete lines appended to ``path``.

    from_start:    read existing content first (default: start at EOF, live only)
    stop_after_eof: return at EOF instead of polling — used for offline parsing.
    """
    while True:
        try:
            fh = open(path, "r", encoding="utf-8", errors="replace")
        except OSError:
            if stop_after_eof:
                return
            time.sleep(poll_interval)
            continue

        with fh:
            cur_inode = _inode(path)
            if not from_start:
                fh.seek(0, os.SEEK_END)
            buffer = ""
            while True:
                chunk = fh.read()
                if chunk:
                    buffer += chunk
                    *lines, buffer = buffer.split("\n")
                    for line in lines:
                        yield line
                    continue

                if stop_after_eof:
                    if buffer:
                        yield buffer
                    return

                time.sleep(poll_interval)

                # Detect truncation (file shrank) or rotation (inode changed).
                try:
                    size = os.path.getsize(path)
                except OSError:
                    break  # file vanished -> reopen loop
                if size < fh.tell() or _inode(path) != cur_inode:
                    break  # reopen from the top


def tail_latest(
    resolve_path: Callable[[], Optional[str]],
    from_start: bool = False,
    poll_interval: float = 0.05,
    stop_event=None,
) -> Iterator[str]:
    """Follow whichever log ``resolve_path`` currently considers newest.

    Hearthstone creates a timestamped log directory on every launch.  Following
    one pathname forever therefore misses the next game when Hearthstone is
    restarted.  This variant re-runs discovery at EOF and immediately switches
    to a newly-created session, reading that new file from its beginning.
    """
    path: Optional[str] = None
    first_file = True
    while stop_event is None or not stop_event.is_set():
        newest = resolve_path()
        if not newest:
            if stop_event is not None:
                stop_event.wait(poll_interval)
            else:
                time.sleep(poll_interval)
            continue
        if newest != path:
            path = newest
            first_file_position = from_start if first_file else True
            first_file = False
        try:
            fh = open(path, "r", encoding="utf-8", errors="replace")
        except OSError:
            path = None
            continue
        with fh:
            inode = _inode(path)
            if not first_file_position:
                fh.seek(0, os.SEEK_END)
            buffer = ""
            while stop_event is None or not stop_event.is_set():
                chunk = fh.read()
                if chunk:
                    buffer += chunk
                    *lines, buffer = buffer.split("\n")
                    yield from lines
                    continue
                # Discovery is deliberately repeated while idle: a newer
                # Hearthstone_*/Power.log can appear without changing this file.
                discovered = resolve_path()
                if discovered and discovered != path:
                    break
                try:
                    changed = os.path.getsize(path) < fh.tell() or _inode(path) != inode
                except OSError:
                    changed = True
                if changed:
                    break
                if stop_event is not None:
                    stop_event.wait(poll_interval)
                else:
                    time.sleep(poll_interval)
        # Re-open rotations at the beginning, just like a newly-created session.
        first_file_position = True
