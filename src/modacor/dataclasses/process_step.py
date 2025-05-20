# SPDX-License-Identifier: BSD-3-Clause


__all__ = ["ProcessStep"]
__license__ = "BSD-3-Clause"


from __future__ import annotations
from abc import abstractmethod
from numbers import Integral
from typing import Any

from attrs import define, field
from attrs import validators as v

from modacor.dataclasses import ScatteringData

from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.dataclasses.validators import is_list_of_ints
from modacor.io.io_sources import IoSources


@define
class ProcessStep:
    """A base class defining a processing step"""

    io_sources : IoSources = field()

    # class attribute for a machine-readable description of the process step
    documentation: ProcessStepDescriber()

    # dynamic instance configuration
    configuration: dict = field(factory=dict, validator=v.instance_of(dict))

    # flags and attributes for running the pipeline
    requires_steps: list[int] = field(factory=list, validator=is_list_of_ints)
    step_id: int = field(default=-1, validator=v.instance_of(Integral))
    executed: bool = field(default=False, validator=v.instance_of(bool))

    # if the process produces intermediate arrays, they are stored here, optionally cached
    produced_outputs: dict[str, Any] = field(factory=dict)

    # a message handler, supporting logging, warnings, errors, etc. emitted by the process during execution
    message_handler: MessageHandler = field(
        default=MessageHandler(), validator=v.instance_of(MessageHandler)
    )

    # a list of data keys that are modified by this process
    def __attrs_post_init__(self):
        self.__prepared = False

    def prepare_execution(self):
        """
        Prepare the execution of the ProcessStep

        This method can be used to run any costly setup code that is needed
        once before the process step can be executed.
        """
        pass

    def can_execute(self, input_field_names: list[str]) -> bool:
        """
        Check if the process step can be executed

        The default implementation always returns True and any ProcessStep
        that has dependency checks should override this method.
        """
        return True

    @abstractmethod
    def calculate(self, data: ScatteringData, **kwargs: Any) -> dict[str, Any]:
        """Calculate the process step on the given data"""
        raise NotImplementedError("Subclasses must implement this method")

    def execute(self, data: ScatteringData, **kwargs: Any) -> ScatteringData:
        """Execute the process step on the given data"""
        if not self.__prepared:
            self.prepare_execution()
            self.__prepared = True
        self.produced_outputs = self.calculate(data, **kwargs)
        for _key, value in self.produced_outputs.items():
            data.data[_key] = value
        self.executed = True
        return data

    def reset(self):
        """Reset the process step to its initial state"""
        self.__prepared = False
        self.executed = False
        self.produced_outputs = {}

    def modify_config(self, key: str, value: Any):
        """Modify the configuration of the process step"""
        if key in self.configuration:
            self.configuration[key] = value
        else:
            raise KeyError(f"Key {key} not found in configuration")
        self.__prepared = False
