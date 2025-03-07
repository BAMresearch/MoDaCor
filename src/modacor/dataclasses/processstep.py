# src/modacor/dataclasses/processstep.py
# # -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from attrs import define, field
from attrs import validators as v
import logging

# from modacor.dataclasses.basedata import BaseData

logger = logging.getLogger(__name__)

def validate_required_keys(instance, attribute, value):
    keys = [key.strip() for key in value.keys()]
    missing = [key for key in keys if key not in instance.calling_data]
    if missing:
        raise ValueError(f"Missing required data keys in calling_data: {missing}")

@define
class ProcessStep:
    calling_name: str = field() # short name to identify the calling process for the UI
    calling_id: str = field() # not sure what we were planning here. some UID perhaps? difference with calling_module
    # input variables go here: 
    calling_module_path: Path = field(validator=v.instance_of(Path)) # partial path to the module from src/modacor/modules onwards
    calling_version: str = field() # module version being executed
    required_data_keys: List[str] = field(factory=list) # list of data keys required by the process
    calling_arguments: Dict[str, Any] = field(factory=dict, validator=validate_required_keys)
    step_keywords: List[str] = field(factory=list) # list of keywords that can be used to identify the process (allowing for searches)
    step_doc: str = field(default="") # documentation for the process
    step_reference: NXCite = field(default="") # NXCite to the paper describing the process
    note: Optional[str] = field(default=None)
    # process_values: Dict[str, Any] = field(factory=dict)
    # if the process produces intermediate arrays, they are stored here, optionally cached
    produced_values: Dict[str, Any] = field(factory=dict)
    use_per_file_cache: List[str] = field(factory=list) # for produced_values dictionary key names in this list, the produced_values are cached on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and one for per-execution. 
    use_overall_cache: List[str] = field(factory=list) # for produced_values dictionary key names in this list, the produced_values are cached on first run, and reused on subsequent runs. Maybe two chaches, one for per-file and one for per-execution. 
    # status: str = field(default="pending") # can be "pending", "running", "completed", "failed"
    start_time: Optional[datetime] = field(default=None) # built-in profiling.... sort of. Will this do?
    stop_time: Optional[datetime] = field(default=None)
    message_handler: Any = field() # handler for the list of (logging?) messages emitted by the process during operation
    saved: dict = field(factory=dict) # dictionary to store any data that needs to be saved in the output file. keys should be internal variable names, values should be HDF5 paths. 

    def start(self):
        self.start_time = datetime.now(tz=timezone.utc)
    
    def stop(self):
        self.stop_time = datetime.now(tz=timezone.utc)
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.stop_time:
            return (self.stop_time - self.start_time).total_seconds()
        return None
    
    # # placeholders:
    # def can_execute(self) -> bool:
    #     return self.status == "pending" and all([key in self.calling_data for key in self.required_data_keys])

    # def execute(self, data_to_correct: "BaseData", **kwargs) -> "BaseData":
    #     if not self.can_execute():
    #         raise ValueError("Process step cannot be executed at this time.")