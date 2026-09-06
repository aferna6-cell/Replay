"""Guard against bypassing the executable Phase 3U v6 admission gate.

The legacy v4/v5 evaluators remain tested as lower-level contracts, but
production measurement code must not call/import them directly. v6 is the
canonical ranking admission surface because it both recomputes manifest overlap
and verifies source SHA-256 values from the exact immutable artifact bytes.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"
V4_MODULE = "ml.phase_3u_admission"
V4_NAME = "evaluate_ranking_admission"
V5_MODULE = "ml.phase_3u_admission_v5"
V5_NAME = "evaluate_ranking_admission_v5"
V5_BRIDGE = ML_DIR / "phase_3u_admission_v5.py"
V6_BRIDGE = ML_DIR / "phase_3u_admission_v6.py"


def _uses(path: Path, module: str, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    uses: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                if alias.name == name:
                    uses.append(f"import:{alias.asname or alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    uses.append(f"module-import:{alias.asname or alias.name}")
    return uses


def _offenders(module: str, name: str, allowed: Path, implementation: str) -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for path in sorted(ML_DIR.rglob("*.py")):
        if path.name == implementation:
            continue
        uses = _uses(path, module, name)
        if uses and path != allowed:
            offenders[str(path.relative_to(ROOT))] = uses
    return offenders


def test_only_v5_bridge_may_import_legacy_v4_ranking_evaluator():
    offenders = _offenders(V4_MODULE, V4_NAME, V5_BRIDGE, "phase_3u_admission.py")
    assert offenders == {}, f"legacy v4 bypasses found: {offenders}"


def test_only_v6_bridge_may_import_v5_ranking_evaluator():
    offenders = _offenders(V5_MODULE, V5_NAME, V6_BRIDGE, "phase_3u_admission_v5.py")
    assert offenders == {}, (
        "Phase 3U ranking admission must go through "
        "ml.phase_3u_admission_v6.evaluate_ranking_admission_v6; "
        f"v5 bypasses found: {offenders}"
    )


def test_v5_bridge_is_explicitly_the_single_v4_importer():
    assert _uses(V5_BRIDGE, V4_MODULE, V4_NAME) == ["import:evaluate_v4"]


def test_v6_bridge_is_explicitly_the_single_v5_importer():
    assert _uses(V6_BRIDGE, V5_MODULE, V5_NAME) == ["import:evaluate_v5"]
