# src/modacor/dataclasses/process_step.py
# # -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

from attrs import define, field
from attrs import validators as v

__all__ = ["ProcessStepDescriber"]


NXCite = str


def validate_required_keys(instance, attribute, value):
    # keys = [key.strip() for key in value.keys()]
    keys = [key.strip() for key in instance.required_arguments]
    missing = [key for key in keys if key not in instance.calling_arguments]
    if missing:
        raise ValueError(f"Missing required argument keys in calling_arguments: {missing}")


def validate_required_data_keys(instance, attribute, value):
    # keys = [key.strip() for key in value.keys()]
    keys = [key.strip() for key in instance.documentation.required_data_keys]
    missing = [key for key in keys if key not in instance.data.data]
    if missing:
        raise ValueError(f"Missing required data keys in instance.data: {missing}")


@define
class ProcessStepDescriber:
    calling_name: str = field()  # short name to identify the calling process for the UI
    calling_id: str = field()  # not sure what we were planning here. some UID perhaps? difference with calling_module
    calling_module_path: Path = field(
        validator=v.instance_of(Path)
    )  # partial path to the module from src/modacor/modules onwards
    calling_version: str = field()  # module version being executed
    required_data_keys: list[str] = field(factory=list)  # list of data keys required by the process
    required_arguments: list[str] = field(factory=list)  # list of argument key-val combos required by the process
    calling_arguments: dict[str, Any] = field(factory=dict, validator=validate_required_keys)
    works_on: dict[str, list] = field(
        factory=dict, validator=v.instance_of(dict)
    )  # which aspects of BaseData are modified by this
    step_keywords: list[str] = field(
        factory=list
    )  # list of keywords that can be used to identify the process (allowing for searches)
    step_doc: str = field(default="")  # documentation for the process
    step_reference: NXCite = field(default="")  # NXCite to the paper describing the process
    step_note: str | None = field(default=None)
    use_frames_cache: list[str] = field(factory=list)
    # for produced_values dictionary key names in this list, the produced_values are cached
    # on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and
    # one for per-execution.
    use_overall_cache: list[str] = field(factory=list)
    # for produced_values dictionary key names in this list, the produced_values are cached
    # on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and
    # one for per-execution.

    def copy(self) -> ProcessStepDescriber:
        raise NotImplementedError()
