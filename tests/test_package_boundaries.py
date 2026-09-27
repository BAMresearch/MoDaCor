from __future__ import annotations

import ast
from pathlib import Path

import modacor

PACKAGE_ROOT = Path(modacor.__file__).resolve().parent


def _imports_below(root: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_names = (node.module or "",)
            elif isinstance(node, ast.Import):
                imported_names = tuple(alias.name for alias in node.names)
            else:
                continue
            for imported_name in imported_names:
                if any(
                    imported_name == prefix or imported_name.startswith(f"{prefix}.") for prefix in forbidden_prefixes
                ):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} imports {imported_name}")
    return violations


def test_geometry_has_no_higher_layer_dependencies() -> None:
    violations = _imports_below(
        PACKAGE_ROOT / "geometry",
        (
            "modacor.dataclasses",
            "modacor.io",
            "modacor.models",
            "modacor.modules",
            "modacor.runner",
            "modacor.server",
        ),
    )

    assert not violations, "Geometry boundary violations: " + ", ".join(violations)


def test_models_have_no_pipeline_or_io_dependencies() -> None:
    violations = _imports_below(
        PACKAGE_ROOT / "models",
        (
            "modacor.dataclasses",
            "modacor.io",
            "modacor.modules",
            "modacor.runner",
            "modacor.server",
        ),
    )

    assert not violations, "Model boundary violations: " + ", ".join(violations)
