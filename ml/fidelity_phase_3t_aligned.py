"""Causal-aligned Phase 3T execution entrypoint.

Measurement-only wrapper around :mod:`ml.fidelity_phase_3t`.  It installs the
Phase 3T attribution adapter before execution so the first-divergence component
and first-event metadata come from one causal alignment result.

Use this entrypoint until the adapter is folded into the legacy runner:

    python -m ml.fidelity_phase_3t_aligned --lobbies 8 --seed 14200 --non-evaluative

No simulator behavior, RNG, seeds, alpha, scaling, 2Q, or hero-damage semantics
are changed by this module.
"""

from __future__ import annotations

import sys
from typing import Optional

from ml.phase_3t_alignment_adapter import install_into_legacy_module

# Install before importing the legacy runner.  The legacy diagnostic's compare
# function resolves its decomposition globals at call time, so this routes the
# measurement attribution without changing simulation execution.
install_into_legacy_module()

from ml.fidelity_phase_3t import main as _legacy_main  # noqa: E402
from ml.fidelity_phase_3t import run_phase_3t as _legacy_run_phase_3t  # noqa: E402


def run_phase_3t(**kwargs):
    """Run Phase 3T with causal alignment installed explicitly."""
    install_into_legacy_module()
    return _legacy_run_phase_3t(**kwargs)


def main(argv: Optional[list] = None) -> int:
    """CLI entrypoint preserving all legacy Phase 3T arguments and guards."""
    install_into_legacy_module()
    return _legacy_main(argv)


if __name__ == "__main__":
    sys.exit(main())
