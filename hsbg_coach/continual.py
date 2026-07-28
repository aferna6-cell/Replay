"""Continual learning — always improving, never on the live path.

The model gets better from your recorded games, but training must NOT slow down the
in-game move recommendation. So:

  * Training runs BETWEEN games (triggered at game-over), in a separate, low-priority
    (`nice 19`) background process — it never shares the recommender's thread.
  * It's single-flight (one retrain at a time) and gated on having new recorded games,
    so it doesn't thrash.
  * The new model is written to a file; the live coach hot-swaps it by reloading when
    the file changes (a cheap mtime check off the hot path). The recommender only ever
    READS the current model — a swap is atomic.

If torch isn't installed (or `HSBG_NO_TRAIN=1`), this is a no-op and the coach runs on
the existing model. The recommendation path is unaffected either way.
"""

import glob
import os
import subprocess
import sys
from typing import Callable, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class BackgroundTrainer:
    """Retrain the eval net between games without touching the live recommender."""

    def __init__(self, data_dir: str, model_path: Optional[str] = None,
                 min_new_games: int = 1, enabled: Optional[bool] = None,
                 comp_source: Optional[str] = None,
                 spawn: Optional[Callable[[List[str]], object]] = None):
        self.data_dir = data_dir
        self.model_path = model_path or os.path.join(_REPO, "ml", "eval_net.pt")
        self.min_new_games = max(1, int(min_new_games))
        env_off = os.environ.get("HSBG_NO_TRAIN") == "1"
        self.enabled = (_torch_ok() and not env_off) if enabled is None else enabled
        self.comp_source = comp_source
        self._spawn_fn = spawn or self._spawn      # injectable for tests
        self._proc = None
        self._baseline = self._count_games()

    def _count_games(self) -> int:
        try:
            return len(glob.glob(os.path.join(self.data_dir, "game-*.jsonl")))
        except Exception:
            return 0

    def is_training(self) -> bool:
        p = self._proc
        return p is not None and getattr(p, "poll", lambda: 0)() is None

    def new_games(self) -> int:
        return self._count_games() - self._baseline

    def maybe_train(self) -> bool:
        """Kick off a background retrain if warranted. Returns True if one started.
        Non-blocking — returns immediately; the retrain runs in another process."""
        if not self.enabled or self.is_training():
            return False
        if self.new_games() < self.min_new_games:
            return False
        self._proc = self._spawn_fn(self._command())
        self._baseline = self._count_games()
        return True

    def _command(self) -> List[str]:
        # --with-context folds the ENTIRE recorded game state (board + economy +
        # survival + lobby) into the retrain, not just the board.
        cmd = [sys.executable, "-m", "ml.train_eval_net",
               "--trajectories", self.data_dir, "--out", self.model_path,
               "--with-context"]
        if self.comp_source:
            cmd += ["--comp-source", self.comp_source]
        return cmd

    def _spawn(self, cmd: List[str]):
        kwargs = {"cwd": _REPO,
                  "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if hasattr(os, "nice"):                    # deprioritize so live stays snappy
            kwargs["preexec_fn"] = lambda: os.nice(19)
        return subprocess.Popen(cmd, **kwargs)
