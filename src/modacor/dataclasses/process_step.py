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
from collections.abc import Iterable
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any, Type

from attrs import define, field
from attrs import validators as v

from ..io.io_sinks import IoSinks
from ..io.io_sources import IoSources
from .databundle import DataBundle
from .messagehandler import MessageHandler
from .process_step_describer import DEPENDENCY_ROLES, ProcessStepDescriber
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


_DEPENDENCY_ROLE_BEHAVIOR = {
    "processing_read_basedata_key": (True, False),
    "processing_write_basedata_key": (False, True),
    "processing_read_write_basedata_key": (True, True),
    "processing_read_basedata_key_list": (True, False),
    "processing_write_basedata_key_list": (False, True),
    "processing_read_write_basedata_key_list": (True, True),
}
assert set(_DEPENDENCY_ROLE_BEHAVIOR) == set(DEPENDENCY_ROLES)


def _dependency_roles_from_spec(config_spec: dict[str, Any]) -> tuple[str, ...]:
    role_value = config_spec.get("dependency_role")
    if role_value is None:
        return ()
    if isinstance(role_value, str):
        return (role_value,)
    try:
        return tuple(str(role) for role in role_value)
    except TypeError as exc:
        raise TypeError("Configuration schema 'dependency_role' must be a string or iterable of strings.") from exc


def _basedata_keys_from_dependency_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, dict):
        return []
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        text = str(value).strip()
        return [text] if text else []


def _config_type_tuple(type_spec: Any) -> tuple[type, ...] | None:
    if type_spec is None or type_spec is Any:
        return None
    if isinstance(type_spec, tuple):
        if not all(isinstance(item, type) for item in type_spec):
            raise TypeError("Configuration schema 'type' tuples must contain only type objects.")
        return type_spec
    if isinstance(type_spec, type):
        return (type_spec,)
    raise TypeError("Configuration schema 'type' must be a type, a tuple of types, or omitted.")


def _config_type_label(types: tuple[type, ...] | None) -> str:
    if types is None:
        return "any value"
    return " or ".join(item.__name__ for item in types) if types else "None"


def _normalize_config_value_for_spec(value: Any, config_spec: dict[str, Any]) -> Any:
    """Normalize transport-friendly config values into their declared runtime type."""

    expected_types = config_spec["type"]
    if value is None or expected_types is None:
        return value
    if (
        tuple in expected_types
        and list not in expected_types
        and not config_spec["allow_iterable"]
        and isinstance(value, list)
    ):
        return tuple(value)
    return value


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
    configuration: dict = field(factory=dict, validator=v.instance_of(dict))

    # flags and attributes for running the pipeline
    requires_steps: list[str] = field(factory=list)
    step_id: int | str = field(default=-1, validator=v.instance_of((Integral, str)))
    executed: bool = field(default=False, validator=v.instance_of(bool))
    short_title: str | None = field(default=None, validator=v.optional(v.instance_of(str)))

    # Optional bookkeeping returned by calculate(), such as touched DataBundle keys.
    # Authoritative data changes are made in-place on processing_data.
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
        provided_configuration = dict(self.configuration)
        self.configuration = self.default_config()
        if provided_configuration:
            self.modify_config_by_dict(provided_configuration)

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

        Argument specs may declare ``dependency_role`` to derive exact
        BaseData-level reads and writes from configuration values. Subclasses
        with source/sink side effects or unusual key relationships can still
        override this method with a custom contract. Without dependency roles,
        the default covers ordinary in-place processing steps that use
        ``with_processing_keys`` and the optional ``output_processing_key``
        convention.
        """

        cfg = self.configuration or {}
        source_refs = source_refs_from_references(cfg)
        schema = self.effective_config_schema()
        read_patterns: set[str] = set()
        write_patterns: set[str] = set()
        has_dependency_hints = False

        for config_key, config_spec in schema.items():
            dependency_roles = _dependency_roles_from_spec(config_spec)
            if not dependency_roles:
                continue
            has_dependency_hints = True
            basedata_keys = _basedata_keys_from_dependency_value(cfg.get(config_key, config_spec.get("default")))
            for role in dependency_roles:
                try:
                    reads, writes = _DEPENDENCY_ROLE_BEHAVIOR[role]
                except KeyError as exc:
                    known = ", ".join(sorted(_DEPENDENCY_ROLE_BEHAVIOR))
                    raise ValueError(f"Unknown dependency_role {role!r}. Known roles: {known}.") from exc
                for basedata_key in basedata_keys:
                    patterns = processing_key_patterns(cfg.get("with_processing_keys"), basedata_key=basedata_key)
                    if reads:
                        read_patterns.update(patterns)
                    if writes:
                        write_patterns.update(patterns)

        if not has_dependency_hints:
            read_patterns = set(processing_key_patterns(cfg.get("with_processing_keys")))

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

        if has_dependency_hints and not read_patterns and not write_patterns:
            # Hinted BaseData keys are only exact when paired with processing keys.
            # Without them, keep partial-rerun invalidation conservative.
            read_patterns = {"*"}
            write_patterns = {"*"}

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
        """
        Apply this process step to ``self.processing_data`` in-place.

        Implementations may return a mapping of touched ``ProcessingData`` keys
        to their current ``DataBundle`` values for diagnostics and compatibility.
        The return value is not merged by ``execute()``; the in-place mutation is
        the authoritative result.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def execute(self, data: ProcessingData) -> None:
        """Execute the process step on the given data."""
        self.processing_data = data
        if not self.__prepared:
            self.prepare_execution()
            self.__prepared = True
        produced_outputs = self.calculate()
        self.produced_outputs = {} if produced_outputs is None else produced_outputs
        if not isinstance(self.produced_outputs, dict):
            raise TypeError(
                f"{self.__class__.__name__}.calculate() must return a dict of touched outputs or None, "
                f"got {type(self.produced_outputs).__name__}."
            )
        self.executed = True

    def reset(self):
        """Reset the process step to its initial state"""
        self.__prepared = False
        self.executed = False
        self.produced_outputs = {}
        self._prepared_data = {}

    @classmethod
    def _normalize_config_spec(cls, spec: dict[str, Any], default: Any) -> dict[str, Any]:
        types = _config_type_tuple(spec.get("type", Any))
        type_allows_none = types is None or type(None) in types
        if types is not None:
            types = tuple(item for item in types if item is not type(None))

        allow_none = bool(spec.get("allow_none", type_allows_none or default is None))
        return {
            "type": types,
            "allow_iterable": bool(spec.get("allow_iterable", False)),
            "allow_none": allow_none,
            "required": bool(spec.get("required", False)),
            "default": default,
            "doc": spec.get("doc"),
            "dependency_role": spec.get("dependency_role"),
        }

    @classmethod
    def effective_config_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return the unified runtime configuration schema for this step class.

        `CONFIG_KEYS` remains the compatibility location for shared/base keys.
        Step-specific keys come from `documentation.arguments`.
        """
        schema: dict[str, dict[str, Any]] = {}

        for key, spec in cls.CONFIG_KEYS.items():
            schema[key] = cls._normalize_config_spec(spec, deepcopy(spec.get("default", None)))

        documentation = getattr(cls, "documentation", None)
        doc_arguments = getattr(documentation, "arguments", {}) or {}
        doc_defaults = documentation.initial_configuration() if hasattr(documentation, "initial_configuration") else {}

        for key, spec in doc_arguments.items():
            doc_spec = cls._normalize_config_spec(spec, doc_defaults.get(key))
            if key in schema:
                merged = dict(schema[key])
                merged["default"] = deepcopy(doc_spec["default"])
                merged["required"] = doc_spec["required"]
                merged["doc"] = doc_spec["doc"]
                schema[key] = merged
            else:
                schema[key] = doc_spec

        return schema

    @classmethod
    def normalize_config_value(cls, key: str, value: Any) -> Any:
        """Normalize one configuration value against the unified schema."""
        schema = cls.effective_config_schema()
        if key not in schema:
            known_keys = ", ".join(sorted(schema.keys()))
            raise KeyError(f"Key {key} not found in configuration. Known keys: {known_keys}")  # noqa
        return _normalize_config_value_for_spec(value, schema[key])

    @classmethod
    def validate_config_value(cls, key: str, value: Any) -> None:
        """Validate one configuration value against the unified schema."""
        schema = cls.effective_config_schema()
        if key not in schema:
            known_keys = ", ".join(sorted(schema.keys()))
            raise KeyError(f"Key {key} not found in configuration. Known keys: {known_keys}")  # noqa

        config_spec = schema[key]
        value = _normalize_config_value_for_spec(value, config_spec)
        if value is None:
            if config_spec["allow_none"]:
                return
            raise TypeError(f"Configuration key {key!r} does not allow None.")

        expected_types = config_spec["type"]
        if expected_types is None:
            return

        if config_spec["allow_iterable"] and isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            if all(isinstance(item, expected_types) for item in value):
                return

        if isinstance(value, expected_types):
            return

        expected = _config_type_label(expected_types)
        if config_spec["allow_iterable"]:
            expected = f"{expected} or an iterable of {expected}"
        raise TypeError(
            f"Configuration key {key!r} for {cls.__name__} must be {expected}, " f"got {type(value).__name__}."
        )

    def modify_config_by_dict(self, by_dict: dict | None = None) -> None:
        """Modify the configuration of the process step by a dictionary."""
        if by_dict is None:
            return
        if not isinstance(by_dict, dict):
            raise TypeError(f"Configuration update must be a dict, got {type(by_dict).__name__}.")

        for key, value in by_dict.items():
            value = self.__class__.normalize_config_value(key, value)
            self.__class__.validate_config_value(key, value)
            self.configuration[key] = value
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
            try:
                cls.validate_config_value(_key, _value)
            except (KeyError, TypeError, ValueError):
                return False
        return True

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """
        Create an initial dictionary for the process step configuration.
        """
        return {_k: deepcopy(_v["default"]) for _k, _v in cls.effective_config_schema().items()}
