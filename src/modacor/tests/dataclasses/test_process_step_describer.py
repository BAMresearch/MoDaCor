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
