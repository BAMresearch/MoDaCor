from __future__ import annotations

import ast
import inspect
from textwrap import dedent
from typing import get_args, get_origin, get_type_hints

import pytest

import modacor.modules
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

PUBLIC_STEP_NAMES = tuple(modacor.modules.__all__)

# Multi-bundle and I/O contracts remain explicit; retain their existing
# hand-written default metadata expectations.
CUSTOM_DEFAULT_METADATA_CONTRACTS = {
    "ConcatenateDatabundles": (set(), {"configured data keys", "source_index"}),
    "DivideDatabundles": ({"signal"}, {"signal"}),
    "FindScaleFactor1D": ({"signal", "Q"}, {"scale_factor", "scale_background"}),
    "Integrate1D": ({"signal", "q"}, {"integral"}),
    "SubtractInterpolated1D": ({"signal", "Q"}, {"signal", "remap_mask"}),
}

# Extra metadata beyond declarations, with an explicit reason for each entry.
# These writes belong to the extension override, not the regular declarations.
DECLARATIVE_METADATA_EXCEPTIONS = {
    "FindCenterOfMass1D": {
        "required": set(),
        "modifies": {"centroid_count", "centroid_converged", "centroid_window_min", "centroid_window_max"},
        "reason": "The extension writes diagnostics derived from diagnostic_prefix.",
    },
}


def _uses_dependency_declarations(step_class) -> bool:
    method = step_class.dependency_contract
    if method is ProcessStep.dependency_contract:
        return True
    tree = ast.parse(dedent(inspect.getsource(method)))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dependency_contract"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        for node in ast.walk(tree)
    )


DECLARATIVE_STEP_NAMES = tuple(
    name for name in PUBLIC_STEP_NAMES if _uses_dependency_declarations(getattr(modacor.modules, name))
)


@pytest.mark.parametrize("step_name", PUBLIC_STEP_NAMES)
def test_public_process_step_metadata_contract(step_name: str) -> None:
    step_class = getattr(modacor.modules, step_name)
    assert issubclass(step_class, ProcessStep)
    assert step_class.__name__ == step_name

    documentation = step_class.documentation
    assert isinstance(documentation, ProcessStepDescriber)
    assert documentation.calling_id == step_name
    assert all(key.strip() for key in documentation.required_data_keys)
    assert all(key.strip() for key in documentation.modifies)
    assert all(
        isinstance(property_name, str) and property_name.strip()
        for property_names in documentation.modifies.values()
        for property_name in property_names
    )

    return_annotation = get_type_hints(step_class.calculate).get("return")
    assert get_origin(return_annotation) is dict
    assert get_args(return_annotation) == (str, DataBundle)


@pytest.mark.parametrize("step_name", CUSTOM_DEFAULT_METADATA_CONTRACTS)
def test_public_process_step_default_data_metadata(step_name: str) -> None:
    required_data_keys, modified_data_keys = CUSTOM_DEFAULT_METADATA_CONTRACTS[step_name]
    documentation = getattr(modacor.modules, step_name).documentation

    assert set(documentation.required_data_keys) == required_data_keys
    assert set(documentation.modifies) == modified_data_keys


def test_public_process_step_modules_do_not_use_assert_for_runtime_validation() -> None:
    assertions: list[str] = []
    module_paths = {
        getattr(modacor.modules, step_name).documentation.calling_module_path for step_name in PUBLIC_STEP_NAMES
    }
    for module_path in module_paths:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        assertions.extend(f"{module_path}:{node.lineno}" for node in ast.walk(tree) if isinstance(node, ast.Assert))

    assert not assertions, "Runtime assertions found: " + ", ".join(sorted(assertions))


@pytest.mark.parametrize("step_name", DECLARATIVE_STEP_NAMES)
def test_declarative_step_default_data_metadata(step_name: str) -> None:
    documentation = getattr(modacor.modules, step_name).documentation
    required = {key for key, access in documentation.fixed_data_keys.items() if access in {"read", "read_write"}}
    modified = {key for key, access in documentation.fixed_data_keys.items() if access in {"write", "read_write"}}
    defaults = documentation.initial_configuration()
    for name, spec in documentation.arguments.items():
        roles = spec.get("dependency_role", ())
        if not roles:
            continue
        if isinstance(roles, str):
            roles = (roles,)
        value = defaults[name]
        keys = {value} if isinstance(value, str) else set(value or ())
        for role in roles:
            if "read" in role:
                required.update(keys)
            if "write" in role:
                modified.update(keys)
    exception = DECLARATIVE_METADATA_EXCEPTIONS.get(step_name)
    if exception is not None:
        assert exception["reason"]
        required.update(exception["required"])
        modified.update(exception["modifies"])
    assert set(documentation.required_data_keys) == required
    assert set(documentation.modifies) == modified


def test_metadata_exception_table_covers_only_declarative_steps():
    assert DECLARATIVE_METADATA_EXCEPTIONS.keys() <= set(DECLARATIVE_STEP_NAMES)
    assert not CUSTOM_DEFAULT_METADATA_CONTRACTS.keys() & set(DECLARATIVE_STEP_NAMES)
