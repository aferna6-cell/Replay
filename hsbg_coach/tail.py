"""Follow a log file like ``tail -f``, surviving truncation and rotation.

Hearthstone truncates/recreates ``Power.log`` on each launch, so a naive reader
that holds a file handle goes stale. We track the inode + size and reopen when
the file shrinks or is replaced. Stdlib only (polling) — no watchdog dependency,
so it runs anywhere Python does.
"""

import os
import time
from typing import Iterator, Optional


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
