"""Clean-checkout helpers for optional ML deps and generated checkpoints.

The core extra is pytest-only. NumPy and Torch live in the ``ml`` extra and
must be declared — tests that need them skip via ``require_ml`` rather than
failing because an agent image happened to have the packages.

Generated ``*.pt`` checkpoints are gitignored on purpose. Tests that hash
live model bytes either build a deterministic temporary PolicyNet fixture or
skip when the local artifact is missing. Committed experiment JSON remains
the record of historical hashes. Do not commit trained weights.
"""

from __future__ import annotations

import os
from typing import Sequence

import pytest

GENERATED_CHECKPOINT_RATIONALE = (
    "generated model checkpoints (*.pt) are gitignored and are not present "
    "on a clean checkout; live model-byte comparison is skipped unless those "
    "files exist locally. Historical hashes stay in committed experiment JSON. "
    "Do not commit trained weights"
)


def require_numpy():
    """Fail as a skip when the core extra is installed without NumPy."""
    return pytest.importorskip("numpy")


def require_torch():
    """NumPy + Torch, the optional ``ml`` extra."""
    pytest.importorskip("numpy")
    return pytest.importorskip("torch")


def require_ml():
    """Alias for tests whose import graph reaches Torch (contracts, PPO)."""
    return require_torch()


def skip_unless_files(*paths: str, rationale: str = GENERATED_CHECKPOINT_RATIONALE):
    """Skip when generated checkpoint bytes are absent from the checkout."""
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        pytest.skip(f"{rationale}. Missing: {', '.join(missing)}")


def write_tiny_policy_checkpoint(path: str, seed: int = 0, meta=None):
    """Deterministic temporary PolicyNet bytes for fingerprint tests."""
    torch = require_torch()
    from hsbg_coach.synergy import load_embeddings
    from ml.policy_net import PolicyNet, save_policy
    from ml.tokens import token_dim

    torch.manual_seed(seed)
    net = PolicyNet(token_dim(load_embeddings()))
    save_policy(net, path, meta if meta is not None else {"kind": "test-fixture"})
    return net
