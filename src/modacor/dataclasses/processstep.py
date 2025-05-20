# src/modacor/dataclasses/processstep.py
# # -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from attrs import define, field
from attrs import validators as v

from .messagehandler import MessageHandler
from .validators import is_list_of_ints

# import logging

# from modacor.dataclasses import DataBundle

# from modacor.dataclasses.basedata import BaseData
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
    calling_id: str = (
        field()
    )  # not sure what we were planning here. some UID perhaps? difference with calling_module
    calling_module_path: Path = field(
        validator=v.instance_of(Path)
    )  # partial path to the module from src/modacor/modules onwards
    calling_version: str = field()  # module version being executed
    required_data_keys: List[str] = field(factory=list)  # list of data keys required by the process
    required_arguments: List[str] = field(
        factory=list
    )  # list of argument key-val combos required by the process
    calling_arguments: Dict[str, Any] = field(factory=dict, validator=validate_required_keys)
    works_on: Dict[str, list] = field(
        factory=dict, validator=v.instance_of(dict)
    )  # which aspects of BaseData are modified by this
    step_keywords: List[str] = field(
        factory=list
    )  # list of keywords that can be used to identify the process (allowing for searches)
    step_doc: str = field(default="")  # documentation for the process
    step_reference: NXCite = field(default="")  # NXCite to the paper describing the process
    step_note: Optional[str] = field(default=None)
    use_frames_cache: List[str] = field(factory=list)
    # for produced_values dictionary key names in this list, the produced_values are cached
    # on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and
    # one for per-execution.
    use_overall_cache: List[str] = field(factory=list)
    # for produced_values dictionary key names in this list, the produced_values are cached
    # on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and
    # one for per-execution.


@define
class ProcessStep:
    """A base class defining a processing step"""

    # we recommend using kwargs for all arguments, and using the required_data_keys
    # to specify the required arguments
    documentation: ProcessStepDescriber = field(
        validator=v.instance_of(ProcessStepDescriber), init=False
    )
    # data = field(validator=validate_required_data_keys)
    #  can't import DataBundle here due to ciccular import
    configuration: dict = field(factory=dict, validator=v.instance_of(dict))
    previous_steps: list[int] = field(
        factory=list, validator=[v.instance_of(list), is_list_of_ints]
    )  # list of previous steps in the process
    # if the process produces intermediate arrays, they are stored here, optionally cached
    produced_outputs: dict[str, Any] = field(factory=dict)
    # a message handler, supporting logging, warnings, errors, etc.
    # emitted by the process during execution
    message_handler: MessageHandler = field(
        default=MessageHandler(), validator=v.instance_of(MessageHandler)
    )
    # the following will be in the configuration
    # saved: dict = field(factory=dict)  # dictionary to store any data that needs to be saved
    # in the output file. keys should be internal variable names, values should be HDF5 paths.

    @abstractmethod
    def can_apply(self) -> bool:
        pass

    @abstractmethod
    def apply(self, DataBundle, **kwargs):
        """return DataBundle"""
        pass
