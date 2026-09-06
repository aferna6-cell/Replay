"""Guard against bypassing the executable Phase 3U v5 admission gate.

The legacy v4 evaluator remains tested as a lower-level contract, but production
measurement code must not call/import it directly.  v5 is the canonical ranking
admission surface because it recomputes calibration/evaluation overlap from
source-bound observation manifests.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"
LEGACY_MODULE = "ml.phase_3u_admission"
LEGACY_NAME = "evaluate_ranking_admission"
CANONICAL_BRIDGE = ML_DIR / "phase_3u_admission_v5.py"


def _legacy_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    uses: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == LEGACY_MODULE:
            for alias in node.names:
                if alias.name == LEGACY_NAME:
                    uses.append(f"import:{alias.asname or alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == LEGACY_MODULE:
                    uses.append(f"module-import:{alias.asname or alias.name}")
    return uses


def test_only_v5_bridge_may_import_legacy_v4_ranking_evaluator():
    offenders: dict[str, list[str]] = {}
    for path in sorted(ML_DIR.rglob("*.py")):
        if path.name == "phase_3u_admission.py":
            continue
        uses = _legacy_uses(path)
        if uses and path != CANONICAL_BRIDGE:
            offenders[str(path.relative_to(ROOT))] = uses

    assert offenders == {}, (
        "Phase 3U ranking admission must go through "
        "ml.phase_3u_admission_v5.evaluate_ranking_admission_v5; "
        f"legacy v4 bypasses found: {offenders}"
    )


def test_v5_bridge_is_explicitly_the_single_legacy_importer():
    uses = _legacy_uses(CANONICAL_BRIDGE)
    assert uses == ["import:evaluate_v4"]
