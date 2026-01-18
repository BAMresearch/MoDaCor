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


def test_required_arguments_must_be_list_of_strings():
    with pytest.raises(TypeError):
        _build_describer(required_arguments={})

    with pytest.raises(TypeError):
        _build_describer(required_arguments=[1])


def test_required_arguments_must_exist_in_default_configuration():
    with pytest.raises(ValueError):
        _build_describer(required_arguments=["needed"], default_configuration={})

    describer = _build_describer(required_arguments=["needed"], default_configuration={"needed": 1})
    assert describer.required_arguments == ["needed"]


def test_list_fields_allow_tuples_and_strip_whitespace():
    describer = _build_describer(
        required_data_keys=(" signal ", "units"),
        step_keywords=[" foo ", "bar"],
    )

    assert describer.required_data_keys == ["signal", "units"]
    assert describer.step_keywords == ["foo", "bar"]


def test_default_configuration_copy_is_isolated():
    describer = _build_describer(
        default_configuration={"nested": {"values": [1, 2]}},
    )

    copied = describer.default_configuration_copy()
    copied["nested"]["values"].append(3)

    assert describer.default_configuration["nested"]["values"] == [1, 2]


def test_argument_specs_requires_mapping_of_dicts():
    with pytest.raises(TypeError):
        _build_describer(argument_specs=["not a mapping"])

    with pytest.raises(TypeError):
        _build_describer(argument_specs={"key": "not a dict"})

    describer = _build_describer(argument_specs={"key": {"type": str, "required": True}})
    assert describer.argument_specs["key"]["required"] is True
