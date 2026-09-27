from __future__ import annotations

import ast
from typing import get_args, get_origin, get_type_hints

import pytest

import modacor.modules
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

PUBLIC_STEP_NAMES = tuple(modacor.modules.__all__)

DEFAULT_METADATA_CONTRACTS = {
    "DilateMask": ({"mask"}, {"mask"}),
    "DivideDatabundles": ({"signal"}, {"signal"}),
    "FindScaleFactor1D": ({"signal", "Q"}, {"scale_factor", "scale_background"}),
    "FlatPlateSelfAbsorptionCorrection": (
        {"signal", "CosAlpha"},
        {"signal", "flat_plate_self_absorption"},
    ),
    "IndexPixels": ({"signal", "Q", "Psi"}, {"pixel_index"}),
    "Integrate1D": ({"signal", "q"}, {"integral"}),
    "PoissonUncertainties": ({"signal"}, {"signal"}),
    "PolarizationCorrection": (
        {"signal", "TwoTheta", "Psi"},
        {"signal", "polarization_factor_map"},
    ),
    "ReduceMask": ({"mask"}, {"mask"}),
    "ThresholdMask": ({"signal"}, {"threshold_mask"}),
}


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


@pytest.mark.parametrize("step_name", DEFAULT_METADATA_CONTRACTS)
def test_public_process_step_default_data_metadata(step_name: str) -> None:
    required_data_keys, modified_data_keys = DEFAULT_METADATA_CONTRACTS[step_name]
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
