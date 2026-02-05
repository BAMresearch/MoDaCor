# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Tim Snow", "Brian R. Pauw", "Anja Hörmann"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports
__version__ = "20251121.1"

from abc import abstractmethod
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Type

from attrs import define, field
from attrs import validators as v

from ..io.io_sinks import IoSinks
from ..io.io_sources import IoSources
from .databundle import DataBundle
from .messagehandler import MessageHandler
from .process_step_describer import ProcessStepDescriber
from .processing_data import ProcessingData

# from .validators import is_list_of_ints


@define(eq=False)
class ProcessStep:
    """A base class defining a processing step"""

    # Class attributes for the process step
    CONFIG_KEYS = {
        "with_processing_keys": {
            "type": str,
            "allow_iterable": True,
            "allow_none": True,
            "default": None,
        },
        "output_processing_key": {
            "type": str,
            "allow_iterable": False,
            "allow_none": True,
            "default": None,
        },
    }

    # three input items for the process step. For backward compatibility, the first is io_sources
    # The configuration keys for the process step instantiation
    io_sources: IoSources | None = field(default=None, validator=v.optional(v.instance_of(IoSources)))
    # the processing data to work on
    processing_data: ProcessingData = field(default=None, validator=v.optional(v.instance_of(ProcessingData)))
    # optional IO sinks if needed
    io_sinks: IoSinks | None = field(default=None, validator=v.optional(v.instance_of(IoSinks)))

    # class attribute for a machine-readable description of the process step
    documentation = ProcessStepDescriber(
        calling_name="Generic Process step",
        calling_id="",  # to be filled in by the process
        calling_module_path=Path(__file__),
        calling_version=__version__,
    )

    # dynamic instance configuration
    configuration: dict = field(
        factory=dict,
        # validator=lambda inst, attrs, val: inst.is_process_step_dict,
        validator=lambda inst, attrs, val: ProcessStep.is_process_step_dict(inst, attrs.name if attrs else None, val),
    )

    # flags and attributes for running the pipeline
    requires_steps: list[str] = field(factory=list)
    step_id: int | str = field(default=-1, validator=v.instance_of((Integral, str)))
    executed: bool = field(default=False, validator=v.instance_of(bool))
    short_title: str | None = field(default=None, validator=v.optional(v.instance_of(str)))

    # if the process produces intermediate arrays, they are stored here, optionally cached
    produced_outputs: dict[str, Any] = field(factory=dict)
    # intermediate prepared data for the process step
    _prepared_data: dict[str, Any] = field(factory=dict)

    # a message handler, supporting logging, warnings, errors, etc. emitted by the process
    # during execution
    logger: MessageHandler = field(factory=MessageHandler, validator=v.instance_of(MessageHandler))

    # internal variables:
    __prepared: bool = field(default=False, validator=v.instance_of(bool))

    def __attrs_post_init__(self):
        """
        Post-initialization method to set up the process step.
        """
        self.configuration = self.default_config()
        self.configuration.update(self.documentation.initial_configuration())

    def __call__(self, processing_data: ProcessingData) -> None:
        """Allow the process step to be called like a function"""
        self.execute(processing_data)

    # add hash function. equality can be checked
    # def __hash__(self):
    #     return hash((self.documentation.__repr__(), self.configuration.__repr__(), self.step_id))
    def __hash__(self) -> int:
        return object.__hash__(self)

    def prepare_execution(self):
        """
        Prepare the execution of the ProcessStep

        This method can be used to run any costly setup code that is needed
        once before the process step can be executed.
        """
        pass

    def _normalised_processing_keys(self, cfg_key: str = "with_processing_keys") -> list[str]:
        """
        Normalize a ProcessingData key selection into a non-empty list of strings.

        Behavior:
        - None: if processing_data has exactly one key, use it; otherwise error.
        - str: wrap into a one-item list.
        - iterable: materialize into a list (must be non-empty).
        """
        if self.processing_data is None:
            raise RuntimeError(f"{self.__class__.__name__}: processing_data is None in _normalised_processing_keys.")

        cfg_value = self.configuration.get(cfg_key, None)

        if cfg_value is None:
            if len(self.processing_data) == 0:
                raise ValueError(f"{self.__class__.__name__}: {cfg_key} is None and processing_data is empty.")
            if len(self.processing_data) == 1:
                only_key = next(iter(self.processing_data.keys()))
                self.logger.info(
                    f"{self.__class__.__name__}: {cfg_key} not set; using the only key {only_key!r}."  # noqa: E702
                )
                return [only_key]
            raise ValueError(f"{self.__class__.__name__}: {cfg_key} is None but multiple databundles are present.")

        if isinstance(cfg_value, str):
            return [cfg_value]

        try:
            keys = list(cfg_value)
        except TypeError as exc:  # not iterable
            raise ValueError(
                f"{self.__class__.__name__}: {cfg_key} must be a string, an iterable of strings, or None."
            ) from exc

        if not keys:
            raise ValueError(f"{self.__class__.__name__}: {cfg_key} must not be an empty list.")
        return keys

    @abstractmethod
    def calculate(self) -> dict[str, DataBundle]:
        """Calculate the process step on the given data"""
        raise NotImplementedError("Subclasses must implement this method")

    def execute(self, data: ProcessingData) -> None:
        """Execute the process step on the given data"""
        self.processing_data = data
        if not self.__prepared:
            self.prepare_execution()
            self.__prepared = True
        self.produced_outputs = self.calculate()
        for _key, value in self.produced_outputs.items():
            if _key in data:
                data[_key].update(value)
            else:
                data[_key] = value
        self.executed = True

    def reset(self):
        """Reset the process step to its initial state"""
        self.__prepared = False
        self.executed = False
        self.produced_outputs = {}
        self._prepared_data = {}

    def modify_config_by_dict(self, by_dict: dict = {}) -> None:
        """Modify the configuration of the process step by a dictionary"""
        for key, value in by_dict.items():
            if key in self.configuration:
                self.configuration[key] = value
            elif key in self.documentation.arguments:
                # Allow setting documented arguments even if they were not part of the
                # current configuration snapshot yet.
                self.configuration[key] = value
            else:
                known_keys = ", ".join(sorted(self.configuration.keys()))
                raise KeyError(f"Key {key} not found in configuration. Known keys: {known_keys}")  # noqa
        # restart preparation after configuration change:
        self.__prepared = False

    def modify_config_by_kwargs(self, **kwargs) -> None:
        """Modify the configuration of the process step by keyword arguments"""
        if kwargs:
            self.modify_config_by_dict(kwargs)

    @classmethod
    def is_process_step_dict(cls, instance: Type | None, attribute: str | None, item: Any) -> bool:
        """
        Check if the value is a dictionary with the correct keys and types.
        """
        if not isinstance(item, dict):
            return False
        for _key, _value in item.items():
            if _key not in cls.CONFIG_KEYS:
                return False
            _config = cls.CONFIG_KEYS[_key]
            if _value is None:
                if _config["allow_none"]:
                    continue
                return False
            if isinstance(_value, Iterable) and not isinstance(_value, str):
                if not (_config["allow_iterable"] and all([isinstance(_i, _config["type"]) for _i in _value])):
                    return False
                continue
            if not isinstance(_value, _config["type"]):
                return False
        return True

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """
        Create an initial dictionary for the process step configuration.
        """
        return {_k: _v["default"] for _k, _v in cls.CONFIG_KEYS.items()}
