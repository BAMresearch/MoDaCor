# SPDX-License-Identifier: BSD-3-Clause
"""Keep scientific unit conversions at BaseData boundaries."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src/modacor/modules"
ALLOW_LIST = {
    (
        "base_modules/plot_1d_visualization.py",
        "_convert_values",
    ): "Display-only conversion of plot values and their error bars.",
    ("helpers/quantities.py", "scalar_in_units"): "Configuration numbers only; BaseData is explicitly rejected.",
}


def conversions(source):
    """Yield (containing function, conversion kind, line) for forbidden syntax."""

    class Visitor(ast.NodeVisitor):
        function = "<module>"

        def visit_FunctionDef(self, node):
            previous = self.function
            self.function = node.name
            self.generic_visit(node)
            self.function = previous

        def visit_Attribute(self, node):
            if (
                node.attr == "magnitude"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "to"
            ):
                found.append((self.function, "to().magnitude", node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in {"m_as", "m_from"}:
                    found.append((self.function, func.attr, node.lineno))
                if func.attr == "Quantity" and isinstance(func.value, ast.Name) and func.value.id == "ureg":
                    found.append((self.function, "ureg.Quantity", node.lineno))
            self.generic_visit(node)

    found = []
    Visitor().visit(ast.parse(source))
    return found


def test_module_unit_conversions_use_central_api():
    violations = []
    used = set()
    for path in sorted(ROOT.rglob("*.py")):
        if "deprecated" in path.relative_to(ROOT).parts:
            continue
        for function, kind, line in conversions(path.read_text()):
            key = (path.relative_to(ROOT).as_posix(), function)
            if key in ALLOW_LIST:
                used.add(key)
            else:
                violations.append(f"{key[0]}:{line} ({function}): {kind}")
    assert not violations, "\n".join(violations)
    assert used == ALLOW_LIST.keys(), "Remove stale allow-list entries"
    assert all(ALLOW_LIST.values()), "Every exception needs a reason"


@pytest.mark.parametrize(
    "expression",
    [
        'data.to("m").magnitude',
        'data.m_as("m")',
        'unit.m_from("m")',
        'ureg.Quantity(1, "m")',
    ],
)
def test_guard_recognizes_prohibited_conversions(expression):
    assert conversions(expression)
