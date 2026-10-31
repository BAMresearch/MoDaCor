# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "28/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path

import pytest

from modacor.dataclasses.process_step_describer import ProcessStepDescriber


def _build_describer(**kwargs) -> ProcessStepDescriber:
    return ProcessStepDescriber(
        calling_name="Test",
        calling_id="test.step",
        calling_module_path=Path(__file__),
        calling_version="0",
        **kwargs,
    )


def test_arguments_must_be_mapping_of_dicts():
    with pytest.raises(TypeError):
        _build_describer(arguments=["not a mapping"])

    with pytest.raises(TypeError):
        _build_describer(arguments={"key": "not a dict"})


def test_arguments_required_flag_must_be_boolean():
    with pytest.raises(TypeError):
        _build_describer(arguments={"key": {"required": "yes"}})


def test_arguments_dependency_role_must_be_known():
    _build_describer(
        arguments={
            "source_key": {
                "default": "signal",
                "dependency_role": "processing_read_basedata_key",
            },
        }
    )

    with pytest.raises(ValueError, match="unknown role"):
        _build_describer(arguments={"source_key": {"dependency_role": "read_some_stuff"}})


def test_list_fields_allow_tuples_and_strip_whitespace():
    describer = _build_describer(
        required_data_keys=(" signal ", "units"),
        step_keywords=[" foo ", "bar"],
    )

    assert describer.required_data_keys == ["signal", "units"]
    assert describer.step_keywords == ["foo", "bar"]


def test_initial_configuration_is_isolated_from_defaults():
    describer = _build_describer(
        arguments={"nested": {"default": {"values": [1, 2]}}},
    )

    copied = describer.initial_configuration()
    copied["nested"]["values"].append(3)

    assert describer.arguments["nested"]["default"]["values"] == [1, 2]


def test_required_argument_names():
    describer = _build_describer(
        arguments={
            "needed": {"default": "", "required": True},
            "optional": {"default": 0},
        }
    )

    assert describer.required_argument_names() == ("needed",)


@pytest.mark.parametrize(
    "value, error, message",
    [
        ([], TypeError, "mapping"),
        ({1: "read"}, TypeError, "strings"),
        ({"": "read"}, ValueError, "non-empty"),
        ({"  ": "read"}, ValueError, "non-empty"),
        ({"signal": "invalid"}, ValueError, "unknown role"),
        ({"signal": []}, ValueError, "unknown role"),
    ],
)
def test_fixed_data_keys_validation_at_class_definition(value, error, message):
    with pytest.raises(error, match=message):

        class InvalidStep:
            documentation = _build_describer(fixed_data_keys=value)


def test_fixed_data_keys_normalize_and_survive_copy():
    describer = _build_describer(fixed_data_keys={" signal ": "read_write"})
    assert describer.fixed_data_keys == {"signal": "read_write"}
    assert describer.copy().fixed_data_keys == describer.fixed_data_keys
    assert _build_describer().fixed_data_keys == {}


@pytest.mark.parametrize("bundles", [["sample"], ["sample", "background"]])
def test_fixed_data_keys_expand_reads_and_writes(bundles):
    from modacor.dataclasses.process_step import ProcessStep

    class FixedStep(ProcessStep):
        documentation = _build_describer(fixed_data_keys={"input": "read", "output": "write", "signal": "read_write"})

        def calculate(self):
            return None

    step = FixedStep(configuration={"with_processing_keys": bundles})
    contract = step.dependency_contract()
    assert contract.source_refs == set()
    assert contract.processing_reads == {f"{bundle}.{key}" for bundle in bundles for key in ("input", "signal")}
    assert contract.processing_writes == {f"{bundle}.{key}" for bundle in bundles for key in ("output", "signal")}


def test_fixed_data_keys_combine_roles_mapping_keys_and_source_refs():
    from modacor.dataclasses.process_step import ProcessStep

    class MixedStep(ProcessStep):
        documentation = _build_describer(
            fixed_data_keys={"signal": "read_write"},
            arguments={
                "updates": {
                    "type": dict,
                    "default": {"Q": "1/nm", "angle": "rad"},
                    "dependency_role": "processing_read_write_basedata_key",
                },
                "mask_key": {"type": str, "default": "mask", "dependency_role": "processing_read_basedata_key"},
                "source": {"type": str, "default": "calibration::/value"},
            },
        )

        def calculate(self):
            return None

    step = MixedStep(configuration={"with_processing_keys": ["sample"]})
    contract = step.dependency_contract()
    assert contract.source_refs == {"calibration"}
    assert contract.processing_reads == {"sample.signal", "sample.Q", "sample.angle", "sample.mask"}
    assert contract.processing_writes == {"sample.signal", "sample.Q", "sample.angle"}
    step.modify_config_by_kwargs(with_processing_keys=None)
    contract = step.dependency_contract()
    assert contract.source_refs == {"calibration"}
    assert contract.processing_reads == contract.processing_writes == {"*"}


def test_fixed_data_keys_without_processing_keys_are_conservative():
    from modacor.dataclasses.process_step import ProcessStep

    class FixedStep(ProcessStep):
        documentation = _build_describer(fixed_data_keys={"signal": "read_write"})

        def calculate(self):
            return None

    contract = FixedStep().dependency_contract()
    assert contract.processing_reads == contract.processing_writes == {"*"}
