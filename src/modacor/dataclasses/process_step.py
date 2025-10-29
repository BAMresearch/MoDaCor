# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2025 MoDaCor Authors
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


__all__ = ["ProcessStep"]
__license__ = "BSD-3-Clause"
__version__ = "0.0.1"


from abc import abstractmethod
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Type

from attrs import define, field
from attrs import validators as v

from ..io.io_sources import IoSources
from .databundle import DataBundle
from .messagehandler import MessageHandler
from .process_step_describer import ProcessStepDescriber
from .processing_data import ProcessingData
from .validators import is_list_of_ints


@define
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

    # The configuration keys for the process step instantiation
    io_sources: IoSources = field()

    # class attribute for a machine-readable description of the process step
    documentation = ProcessStepDescriber(
        calling_name="Generic Process step",
        calling_id=None,
        calling_module_path=Path(__file__),
        calling_version=__version__,
    )

    # dynamic instance configuration
    configuration: dict = field(
        factory=dict,
        validator=lambda inst, attrs, val: inst.is_process_step_dict,
    )

    # flags and attributes for running the pipeline
    requires_steps: list[int] = field(factory=list, validator=is_list_of_ints)
    step_id: int = field(default=-1, validator=v.instance_of(Integral))
    executed: bool = field(default=False, validator=v.instance_of(bool))

    # if the process produces intermediate arrays, they are stored here, optionally cached
    produced_outputs: dict[str, Any] = field(factory=dict)

    # a message handler, supporting logging, warnings, errors, etc. emitted by the process
    # during execution
    message_handler: MessageHandler = field(default=MessageHandler(), validator=v.instance_of(MessageHandler))

    # internal variables:
    __prepared: bool = field(default=False, validator=v.instance_of(bool))
    processing_data: ProcessingData = field(default=None, validator=v.optional(v.instance_of(ProcessingData)))

    def __attrs_post_init__(self):
        """
        Post-initialization method to set up the process step.
        """
        self.configuration = self.default_config()

    # add hash function. equality can be checked
    def __hash__(self):
        return hash((self.documentation.__repr__(), self.configuration.__repr__(), self.step_id))

    def prepare_execution(self):
        """
        Prepare the execution of the ProcessStep

        This method can be used to run any costly setup code that is needed
        once before the process step can be executed.
        """
        pass

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

    def modify_config(self, by_dict: dict | None = None, **kwargs) -> None:
        """Modify the configuration of the process step"""
        if by_dict is not None:
            for key, value in by_dict.items():
                if key in self.configuration:
                    self.configuration[key] = value
                else:
                    raise KeyError(f"Key {key} not found in configuration")  # noqa
        for key, value in kwargs.items():
            if key in self.configuration:
                self.configuration[key] = value
            else:
                raise KeyError(f"Key {key} not found in configuration")  # noqa
        self.__prepared = False

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
