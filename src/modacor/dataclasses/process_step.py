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


def _as_frozen_str_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        values = [value]
    else:
        values = value
    return frozenset(str(item).strip() for item in values if str(item).strip())


@define(frozen=True, slots=True)
class ProcessStepDependencies:
    """Runtime dependency contract used for partial-rerun invalidation."""

    source_refs: frozenset[str] = field(factory=frozenset, converter=_as_frozen_str_set)
    processing_reads: frozenset[str] = field(factory=frozenset, converter=_as_frozen_str_set)
    processing_writes: frozenset[str] = field(factory=frozenset, converter=_as_frozen_str_set)


def normalize_processing_key_values(value: Any) -> list[str]:
    """Normalize a config value into ProcessingData key names for dependency contracts."""

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        return []


def processing_key_patterns(keys: Any, *, basedata_key: str | None = None) -> frozenset[str]:
    """
    Convert ProcessingData keys into dirty-matching patterns.

    Bundle-level dependencies are represented as ``sample.*``. BaseData-level
    dependencies are represented as ``sample.signal``.
    """

    patterns: set[str] = set()
    clean_basedata_key = str(basedata_key).strip() if basedata_key is not None else None
    for key in normalize_processing_key_values(keys):
        if key == "*":
            patterns.add("*")
        elif clean_basedata_key:
            patterns.add(f"{key}.{clean_basedata_key}")
        else:
            patterns.add(f"{key}.*")
    return frozenset(patterns)


def source_refs_from_references(*values: Any) -> frozenset[str]:
    """Extract source refs from IoSources references of the form ``ref::path``."""

    refs: set[str] = set()

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if "::" in value:
                ref = value.split("::", 1)[0].strip()
                if ref:
                    refs.add(ref)
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return frozenset(refs)


def matches_processing_pattern(changed_key: str, patterns: Iterable[str]) -> bool:
    changed = str(changed_key).strip()
    if not changed:
        return False
    for pattern in patterns:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if changed == prefix or changed.startswith(prefix + "."):
                return True
        if changed == pattern:
            return True
    return False


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

    def dependency_contract(self) -> ProcessStepDependencies:
        """
        Return the runtime dependencies used for partial-rerun invalidation.

        Subclasses with source/sink side effects or non-in-place outputs should
        override this method with an exact contract. The default covers ordinary
        in-place processing steps that use ``with_processing_keys`` and the
        optional ``output_processing_key`` convention.
        """

        cfg = self.configuration or {}
        source_refs = source_refs_from_references(cfg)
        read_patterns = processing_key_patterns(cfg.get("with_processing_keys"))

        output_key = cfg.get("output_processing_key")
        if isinstance(output_key, str) and output_key.strip():
            write_patterns = set(processing_key_patterns(output_key))
        else:
            write_patterns = set(read_patterns)

        processing_key = cfg.get("processing_key")
        if isinstance(processing_key, str) and processing_key.strip():
            databundle_output_key = cfg.get("databundle_output_key")
            if isinstance(databundle_output_key, str) and databundle_output_key.strip():
                write_patterns.update(processing_key_patterns(processing_key, basedata_key=databundle_output_key))
            else:
                write_patterns.update(processing_key_patterns(processing_key))

        if not source_refs and not read_patterns and not write_patterns:
            # Generic/custom steps without a contract are treated conservatively.
            read_patterns = frozenset({"*"})
            write_patterns = {"*"}

        return ProcessStepDependencies(
            source_refs=source_refs,
            processing_reads=read_patterns,
            processing_writes=write_patterns,
        )

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
