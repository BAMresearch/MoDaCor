# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import importlib
import re
from typing import Dict, Type

from ..dataclasses.process_step import ProcessStep


def _pascal_to_snake(name: str) -> str:
    """Convert from PascalCase to snake_case module name."""
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


class ProcessStepRegistry:
    """
    Registry for ProcessStep subclasses.

    Responsibilities:
    - Map a logical name (usually the class name used in YAML, e.g. "Divide")
      to a ProcessStep subclass.
    - Optionally lazy-load modules from a base package if the class is not
      yet registered.
    """

    def __init__(self, base_package: str | None = None) -> None:
        """
        Parameters
        ----------
        base_package:
            Optional base package for lazy imports, e.g.
            "modacor.modules.base_modules". If provided, `get()` will try to
            import `<base_package>.<snake_case(name)>` on cache miss.
        """
        self._registry: Dict[str, Type[ProcessStep]] = {}
        self._base_package = base_package

    # ------------------------------------------------------------------ #
    # Registration / lookup API
    # ------------------------------------------------------------------ #

    def register(self, cls: Type[ProcessStep], name: str | None = None) -> None:
        """
        Register a ProcessStep subclass.

        Parameters
        ----------
        cls:
            The ProcessStep subclass to register.
        name:
            Optional explicit name. If omitted, `cls.__name__` is used.
        """
        if not issubclass(cls, ProcessStep):
            raise TypeError(f"Can only register ProcessStep subclasses, got {cls!r}.")
        key = name or cls.__name__
        self._registry[key] = cls

    def get(self, name: str) -> Type[ProcessStep]:
        """
        Retrieve a ProcessStep subclass by name.

        On cache miss, if `base_package` is set, it will attempt a lazy import
        of `<base_package>.<snake_case(name)>` and then look for `name` in
        that module.
        """
        if name in self._registry:
            return self._registry[name]

        if self._base_package is None:
            raise KeyError(
                f"ProcessStep {name!r} not found in registry and no base_package configured for lazy loading."  # noqa: E713
            )

        # Lazy-load from base package
        module_name = _pascal_to_snake(name)
        module_path = f"{self._base_package}.{module_name}"
        module = importlib.import_module(module_path)

        try:
            cls = getattr(module, name)
        except AttributeError as exc:
            raise KeyError(f"ProcessStep class {name!r} not found in module {module_path!r}.") from exc  # noqa: E713

        if not issubclass(cls, ProcessStep):
            raise TypeError(f"Object {name!r} in module {module_path!r} is not a ProcessStep subclass.")

        # Cache and return
        self._registry[name] = cls
        return cls

    def __contains__(self, name: str) -> bool:  # convenience
        return name in self._registry


# Default registry used by the Pipeline
DEFAULT_PROCESS_STEP_REGISTRY = ProcessStepRegistry(base_package="modacor.modules.base_modules")
