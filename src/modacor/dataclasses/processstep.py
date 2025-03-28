# src/modacor/dataclasses/processstep.py
# # -*- coding: utf-8 -*-
from __future__ import annotations
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from attrs import define, field
from attrs import validators as v
# import logging

from modacor.dataclasses.messagehandler import MessageHandler

# from modacor.dataclasses.basedata import BaseData
NXCite = str


def validate_required_keys(instance, attribute, value):
    # keys = [key.strip() for key in value.keys()]
    keys = [key.strip() for key in instance.required_data_keys]
    missing = [key for key in keys if key not in instance.calling_arguments]
    if missing:
        raise ValueError(f"Missing required data keys in calling_arguments: {missing}")


@define
class ProcessStepExecutor:
    """A base class defining a processing step"""
    # we recommend using kwargs for all arguments, and using the required_data_keys to specify the required arguments
    kwargs: dict = field(factory=dict, validator=v.instance_of(dict))
    note: Optional[str] = field(default=None, validator=v.optional(v.instance_of(str)))
    start_time: Optional[datetime] = field(default=None)  # built-in profiling.... sort of. Will this do?
    stop_time: Optional[datetime] = field(default=None)  # built-in profiling.... sort of. Will this do?
    # if the process produces intermediate arrays, they are stored here, optionally cached
    produced_values: Dict[str, Any] = field(factory=dict)
    # a message handler, supporting logging, warnings, errors, etc. emitted by the process during execution
    message_handler: MessageHandler = field(
        default=MessageHandler(),
        validator=v.instance_of(MessageHandler)
    )
    saved: dict = field(factory=dict)  # dictionary to store any data that needs to be saved in the output file. keys should be internal variable names, values should be HDF5 paths. 

    def start(self):
        self.start_time = datetime.now(tz=timezone.utc)

    def stop(self):
        self.stop_time = datetime.now(tz=timezone.utc)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.stop_time:
            return (self.stop_time - self.start_time).total_seconds()
        return None

    @abstractmethod
    def can_apply(self) -> bool:
        pass

    @abstractmethod
    def apply(self):
        pass


@define
class ProcessStepDescriber:
    calling_name: str = field()  # short name to identify the calling process for the UI
    calling_id: str = field()  # not sure what we were planning here. some UID perhaps? difference with calling_module
    calling_module_path: Path = field(validator=v.instance_of(Path))  # partial path to the module from src/modacor/modules onwards
    calling_version: str = field()  # module version being executed
    required_data_keys: List[str] = field(factory=list)  # list of data keys required by the process
    calling_arguments: Dict[str, Any] = field(factory=dict, validator=validate_required_keys)
    step_keywords: List[str] = field(factory=list)  # list of keywords that can be used to identify the process (allowing for searches)
    step_doc: str = field(default="")  # documentation for the process
    step_reference: NXCite = field(default="")  # NXCite to the paper describing the process
    step_note: Optional[str] = field(default=None)
    use_frames_cache: List[str] = field(factory=list)  # for produced_values dictionary key names in this list, the produced_values are cached on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and one for per-execution. 
    use_overall_cache: List[str] = field(factory=list)  # for produced_values dictionary key names in this list, the produced_values are cached on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and one for per-execution. 
