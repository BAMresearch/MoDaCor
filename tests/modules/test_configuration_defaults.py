"""Guard exported steps' schema reads and single source of configuration defaults.

Follow only self.configuration and aliases/parameters receiving that object.
Shared helpers are followed for key declarations, not their reusable fallbacks.
"""

import ast
import inspect
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

import modacor.modules as modules
from modacor.dataclasses.process_step import ProcessStep

_CONFIG = object()
_UNKNOWN = object()
ALLOWED_VIOLATIONS = set()


def _value(node, bindings):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self" and node.attr == "configuration":
            return _CONFIG
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # Common spelling: cfg = self.configuration or {}.
        return _value(node.values[0], bindings)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            value = _value(part.value, bindings) if isinstance(part, ast.FormattedValue) else part.value
            if not isinstance(value, str):
                return _UNKNOWN
            parts.append(value)
        return "".join(parts)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _UNKNOWN


@lru_cache(None)
def _function_source(function):
    lines, start = inspect.getsourcelines(function)
    source = textwrap.dedent("".join(lines))
    return ast.parse(source).body[0], source.splitlines(), start


def _aliases(nodes, initial):
    bindings = dict(initial)
    # Fixed point follows alias chains, never nested mapping values.
    for _ in range(len(nodes)):
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            if _value(node.value, bindings) is not _CONFIG:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and bindings.get(target.id) is not _CONFIG:
                    bindings[target.id] = _CONFIG
                    changed = True
        if not changed:
            break
    return bindings


def _read(node, bindings):
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        if _value(node.value, bindings) is _CONFIG:
            return _value(node.slice, bindings)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        if _value(node.func.value, bindings) is _CONFIG and node.args:
            return _value(node.args[0], bindings)
    return _UNKNOWN


def _literal_fallback(node, key, schema, lines):
    if not isinstance(node, ast.Call) or len(node.args) != 2:
        return None
    try:
        fallback = ast.literal_eval(node.args[1])
    except (ValueError, TypeError):
        return None
    # Exceptions require a None schema default and an inline reason.
    comment = lines[node.lineno - 1].partition("#")[2].strip()
    if key in schema and schema[key]["default"] is None and comment.startswith("config-default: "):
        return None
    return f"literal fallback {fallback!r}; schema default {schema.get(key, {}).get('default', '<undeclared>')!r}"


def _callee(node, function, step):
    if not isinstance(node, ast.Call):
        return None
    target = None
    if isinstance(node.func, ast.Name):
        target = function.__globals__.get(node.func.id)
    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id in {"self", "cls"}:
            target = getattr(step, node.func.attr, None)
    if (inspect.isfunction(target) or inspect.ismethod(target)) and target.__module__.startswith("modacor.modules."):
        return target
    return None


def _passed_arguments(node, target, bindings):
    parameters = list(inspect.signature(target).parameters)
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    arguments = dict(zip(parameters, node.args))
    arguments.update({kw.arg: kw.value for kw in node.keywords if kw.arg is not None})
    passed = {}
    for name, argument in arguments.items():
        value = _value(argument, bindings)
        if value is _CONFIG or isinstance(value, str):
            passed[name] = value
    return passed


def _configuration_violations(step):
    """Return one stable diagnostic per source occurrence and resolved key."""
    schema = step.effective_config_schema()
    found = set()
    visited = set()

    def scan(function, initial):
        function = inspect.unwrap(getattr(function, "__func__", function))
        context = (function, tuple(sorted((key, repr(value)) for key, value in initial.items())))
        if context in visited:
            return
        visited.add(context)
        tree, lines, start = _function_source(function)
        nodes = list(ast.walk(tree))
        bindings = _aliases(nodes, initial)
        shared = ".helpers." in function.__module__
        for node in nodes:
            key = _read(node, bindings)
            problems = []
            if isinstance(key, str):
                if key not in schema:
                    problems.append("undeclared configuration key")
                if not shared and (problem := _literal_fallback(node, key, schema, lines)):
                    problems.append(problem)
            for problem in problems:
                location = f"{Path(inspect.getfile(function)).name}:{start + node.lineno - 1}"
                found.add(f"{step.__name__}: {key}: {problem} ({location})")
            target = _callee(node, function, step)
            if target is not None:
                passed = _passed_arguments(node, target, bindings)
                method_call = (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"self", "cls"}
                )
                if method_call or any(value is _CONFIG for value in passed.values()):
                    scan(target, passed)

    for name in ("calculate", "dependency_contract", "prepare_execution"):
        method = getattr(step, name)
        if method.__module__.startswith("modacor.modules."):
            scan(method, {})
    return found


def _exported_steps():
    return [
        step
        for name in modules.__all__
        if inspect.isclass(step := getattr(modules, name))
        and issubclass(step, ProcessStep)
        and "deprecated" not in step.__module__.split(".")
    ]


def test_configuration_defaults():
    actual = set().union(*(_configuration_violations(step) for step in _exported_steps()))
    unexpected = actual - ALLOWED_VIOLATIONS
    stale = ALLOWED_VIOLATIONS - actual
    if unexpected or stale:
        pytest.fail(
            "\n".join(sorted(unexpected) + [f"Stale allowlist: {item}" for item in sorted(stale)]), pytrace=False
        )


@pytest.mark.parametrize(
    "key",
    [
        "chemical_formula",
        "chemical_formula_source",
        "density",
        "density_source",
        "density_units",
        "density_units_source",
    ],
)
def test_guard_follows_shared_helper_reads(monkeypatch, key):
    step = modules.AttenuatorPlateCorrection
    reduced_schema = dict(step.effective_config_schema())
    del reduced_schema[key]

    def effective_schema(cls):
        return reduced_schema

    monkeypatch.setattr(step, "effective_config_schema", classmethod(effective_schema))
    violations = _configuration_violations(step)
    assert any(
        f"{step.__name__}: {key}: undeclared configuration key (material_attenuation.py:" in item for item in violations
    )


def test_guard_scans_inherited_methods_against_subclass_schema():
    class Parent(ProcessStep):
        __module__ = "modacor.modules.guard_probe"

        def calculate(self):
            return self.configuration["parent_key"]

    class Child(Parent):
        __module__ = "modacor.modules.guard_probe"

        @classmethod
        def effective_config_schema(cls):
            return {}

    Parent.calculate.__module__ = Parent.__module__
    violations = _configuration_violations(Child)
    assert any("Child: parent_key: undeclared configuration key" in item for item in violations)


def test_guard_excludes_overridden_parent_methods():
    class Parent(ProcessStep):
        __module__ = "modacor.modules.guard_probe"

        def calculate(self):
            return self._read_value()

        def _read_value(self):
            return self.configuration["parent_key"]

    class Child(Parent):
        __module__ = "modacor.modules.guard_probe"

        def _read_value(self):
            return None

        @classmethod
        def effective_config_schema(cls):
            return {}

    Parent.calculate.__module__ = Parent.__module__
    Parent._read_value.__module__ = Parent.__module__
    Child._read_value.__module__ = Child.__module__
    assert _configuration_violations(Child) == set()


def test_guard_excludes_nested_dictionaries_and_unrelated_cfg_names():
    tree = ast.parse("""
def calculate(self):
    cfg = self.configuration
    alias = cfg
    nested = alias["uncertainty_estimation"]
    nested.get("method", "std")
    unrelated = {}
    unrelated.get("anything", 1)
    alias.get("reduction", "mean")
""")
    nodes = list(ast.walk(tree))
    bindings = _aliases(nodes, {})
    reads = {_read(node, bindings) for node in nodes} - {_UNKNOWN}
    assert reads == {"uncertainty_estimation", "reduction"}


@pytest.mark.parametrize(
    ("default", "comment", "allowed"),
    [
        (None, "", False),
        (None, "# config-default: None means use the computed value below.", True),
        (1, "# config-default: Invalid exception for a non-None default.", False),
    ],
)
def test_guard_requires_none_schema_default_and_reason(default, comment, allowed):
    source = f'cfg.get("key", None) {comment}'
    call = ast.parse(source).body[0].value
    problem = _literal_fallback(call, "key", {"key": {"default": default}}, [source])
    assert (problem is None) == allowed
