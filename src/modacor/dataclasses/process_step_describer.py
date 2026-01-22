# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Jérôme Kieffer", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from copy import deepcopy
from pathlib import Path
from typing import Any

from attrs import define, evolve, field
from attrs import validators as v

__all__ = ["ProcessStepDescriber"]


NXCite = str
ArgumentSpec = dict[str, Any]


def _normalize_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        raise TypeError(f"{field_name} must be a list of strings, got {type(value).__name__}.")
    if isinstance(value, (list, tuple, set)):
        return [item.strip() if isinstance(item, str) else item for item in value]
    raise TypeError(f"{field_name} must be a list of strings, got {type(value).__name__}.")


def _normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}.")
    return value


def validate_required_keys(instance, attribute, value):
    required_keys = [key.strip() for key in instance.required_arguments]
    available_keys = {str(key).strip() for key in value.keys()}
    missing = [key for key in required_keys if key not in available_keys]
    if missing:
        raise ValueError(f"Missing required argument keys in default_configuration: {missing}")


@define
class ProcessStepDescriber:
    calling_name: str = field()  # short name to identify the calling process for the UI
    calling_id: str = field()  # not sure what we were planning here. some UID perhaps? difference with calling_module
    calling_module_path: Path = field(
        validator=v.instance_of(Path)
    )  # partial path to the module from src/modacor/modules onwards
    calling_version: str = field()  # module version being executed
    required_data_keys: list[str] = field(
        factory=list,
        converter=lambda value: _normalize_str_list(value, "required_data_keys"),
        validator=v.deep_iterable(member_validator=v.instance_of(str), iterable_validator=v.instance_of(list)),
    )  # list of data keys required by the process
    required_arguments: list[str] = field(
        factory=list,
        converter=lambda value: _normalize_str_list(value, "required_arguments"),
        validator=v.deep_iterable(member_validator=v.instance_of(str), iterable_validator=v.instance_of(list)),
    )  # list of argument key-val combos required by the process
    default_configuration: dict[str, Any] = field(factory=dict, validator=validate_required_keys)
    argument_specs: dict[str, ArgumentSpec] = field(
        factory=dict,
        converter=lambda value: _normalize_mapping(value, "argument_specs"),
        validator=v.deep_mapping(key_validator=v.instance_of(str), value_validator=v.instance_of(dict)),
    )  # optional schema for arguments in default_configuration
    modifies: dict[str, list] = field(
        factory=dict, validator=v.instance_of(dict)
    )  # which aspects of BaseData are modified by this
    step_keywords: list[str] = field(
        factory=list,
        converter=lambda value: _normalize_str_list(value, "step_keywords"),
        validator=v.deep_iterable(member_validator=v.instance_of(str), iterable_validator=v.instance_of(list)),
    )  # list of keywords that can be used to identify the process (allowing for searches)
    step_doc: str = field(default="")  # documentation for the process
    step_reference: NXCite = field(default="")  # NXCite to the paper describing the process
    step_note: str | None = field(default=None)
    # use_frames_cache: list[str] = field(factory=list)
    # # for produced_values dictionary key names in this list, the produced_values are cached
    # # on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and
    # # one for per-execution.
    # use_overall_cache: list[str] = field(factory=list)
    # # for produced_values dictionary key names in this list, the produced_values are cached
    # # on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and
    # # one for per-execution.

    def copy(self) -> "ProcessStepDescriber":
        return evolve(self)

    def default_configuration_copy(self) -> dict[str, Any]:
        return deepcopy(self.default_configuration)
