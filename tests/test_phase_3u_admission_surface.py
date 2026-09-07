"""Guard against bypassing the executable Phase 3U ranking/admission boundary.

Legacy evaluators remain tested as lower-level contracts, but production
measurement code must not call/import them directly. v7 is the canonical final
ranking surface because it composes v6's immutable plan/manifest/source-byte
checks with explicit parser execution-provenance admission for both calibration
and evaluation evidence.

The current parser reconciliation helper is deliberately *not* ranking-admissible:
it accepts an arbitrary Python callable after separately hash-checking parser
artifact/config bytes, so it cannot prove the executed callable originated from
those bytes. Until a real preregistered loader/entrypoint wrapper exists, production
``ml`` code must not import that helper at all. This makes the documented HOLD an
executable repository contract rather than a review convention.
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
V6_MODULE = "ml.phase_3u_admission_v6"
V6_NAME = "evaluate_ranking_admission_v6"
V5_BRIDGE = ML_DIR / "phase_3u_admission_v5.py"
V6_BRIDGE = ML_DIR / "phase_3u_admission_v6.py"
V7_BRIDGE = ML_DIR / "phase_3u_admission_v7.py"
PARSER_RECONCILIATION_MODULE = "ml.phase_3u_parser_reconciliation"
PARSER_RECONCILIATION_NAME = "reconcile_parser_output_to_manifest"
PARSER_RECONCILIATION_IMPLEMENTATION = "phase_3u_parser_reconciliation.py"


def _uses(path: Path, module: str, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    uses: list[str] = []
    module_parent, module_leaf = module.rsplit(".", 1)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_from = node.module
            # Production files live in the ``ml`` package. Resolve the relative
            # forms that can otherwise bypass an exact absolute-module check.
            if node.level:
                if node.module:
                    imported_from = f"ml.{node.module}"
                else:
                    imported_from = "ml"

            if imported_from == module:
                for alias in node.names:
                    if alias.name in (name, "*"):
                        uses.append(f"import:{alias.asname or alias.name}")
            elif imported_from == module_parent:
                # ``from ml import phase_3u_admission_v6`` (and the relative
                # ``from . import ...`` form) imports the whole forbidden module.
                for alias in node.names:
                    if alias.name == module_leaf:
                        uses.append(f"module-from:{alias.asname or alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    uses.append(f"module-import:{alias.asname or alias.name}")
    return uses


def _offenders(module: str, name: str, allowed: Path | None, implementation: str) -> dict[str, list[str]]:
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
        "Phase 3U ranking admission must go through the canonical bridge chain; "
        f"v5 bypasses found: {offenders}"
    )


def test_only_v7_bridge_may_import_v6_ranking_evaluator():
    offenders = _offenders(V6_MODULE, V6_NAME, V7_BRIDGE, "phase_3u_admission_v6.py")
    assert offenders == {}, (
        "Phase 3U final ranking admission must go through "
        "ml.phase_3u_admission_v7.evaluate_ranking_admission_v7; "
        f"v6 bypasses found: {offenders}"
    )


def test_untrusted_callable_reconciliation_is_not_imported_by_production_ml():
    offenders = _offenders(
        PARSER_RECONCILIATION_MODULE,
        PARSER_RECONCILIATION_NAME,
        None,
        PARSER_RECONCILIATION_IMPLEMENTATION,
    )
    assert offenders == {}, (
        "Phase 3U callable-based parser reconciliation is not ranking-admissible until "
        "a reviewed wrapper loads/executes the exact digest-bound parser artifact/config; "
        f"production imports found: {offenders}"
    )


def test_import_scanner_catches_whole_module_and_relative_bypass_forms(tmp_path):
    cases = {
        "absolute_from.py": "from ml import phase_3u_admission_v6 as old\n",
        "relative_from.py": "from . import phase_3u_admission_v6 as old\n",
        "relative_symbol.py": (
            "from .phase_3u_admission_v6 import "
            "evaluate_ranking_admission_v6 as old\n"
        ),
        "star.py": "from ml.phase_3u_admission_v6 import *\n",
        "module.py": "import ml.phase_3u_admission_v6 as old\n",
    }
    for filename, source in cases.items():
        path = tmp_path / filename
        path.write_text(source, encoding="utf-8")
        assert _uses(path, V6_MODULE, V6_NAME), f"scanner missed {filename}"


def test_v5_bridge_is_explicitly_the_single_v4_importer():
    assert _uses(V5_BRIDGE, V4_MODULE, V4_NAME) == ["import:evaluate_v4"]


def test_v6_bridge_is_explicitly_the_single_v5_importer():
    assert _uses(V6_BRIDGE, V5_MODULE, V5_NAME) == ["import:evaluate_v5"]


def test_v7_bridge_is_explicitly_the_single_v6_importer():
    assert _uses(V7_BRIDGE, V6_MODULE, V6_NAME) == ["import:evaluate_v6"]
