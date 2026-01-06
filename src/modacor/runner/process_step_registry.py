# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "25/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import importlib
import re
from pathlib import Path
from typing import Dict, Type

from ..dataclasses.process_step import ProcessStep

# ---------------------------------------------------------------------------
# Name / path helpers
# ---------------------------------------------------------------------------


def _pascal_to_snake(name: str) -> str:
    """
    Convert from PascalCase to snake_case module name.

    Examples
    --------
    XSGeometry  -> xs_geometry
    Divide      -> divide
    Q2Mapper    -> q2_mapper
    APIClient   -> api_client
    """
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", name).lower()


def _path_to_module_name(py_file: Path, package_root: Path) -> str:
    """
    Convert a file path under the given package_root into a dotted module name.

    Example
    -------
    py_file      = /.../modacor/modules/technique_modules/scattering/xs_geometry.py
    package_root = /.../modacor

    -> "modacor.modules.technique_modules.scattering.xs_geometry"
    """
    rel = py_file.with_suffix("").relative_to(package_root)
    return ".".join((package_root.name, *rel.parts))


def find_module(modules_root: Path, module_name: str) -> str:
    """
    Find the fully-qualified module path for a snake_case module name.

    Searches under:  modules/**/<module_name>.py

    Parameters
    ----------
    modules_root :
        Path to the 'modules' directory, e.g. .../modacor/modules
    module_name :
        Snake_case module name, e.g. "xs_geometry", "divide".

    Returns
    -------
    str
        Fully-qualified module path, e.g.
        "modacor.modules.technique_modules.scattering.xs_geometry".

    Raises
    ------
    ModuleNotFoundError
        If no matching file is found.
    RuntimeError
        If multiple matching files are found (ambiguous).
    """
    package_root = modules_root.parent  # e.g. .../modacor

    candidates = [p for p in modules_root.rglob(f"{module_name}.py") if p.is_file()]

    if not candidates:
        raise ModuleNotFoundError(f"No module file '{module_name}.py' found under {modules_root}.")

    if len(candidates) > 1:
        details = ", ".join(str(c) for c in candidates)
        raise RuntimeError(f"Ambiguous module name '{module_name}': found multiple candidates: {details}")

    return _path_to_module_name(candidates[0], package_root)


# Default locations
_DEFAULT_MODULES_ROOT = (Path(__file__).resolve().parents[1] / "modules").resolve()
_DEFAULT_CURATED_MODULE_NAME = "modacor.modules"


class ProcessStepRegistry:
    """
    Registry for ProcessStep subclasses.

    Resolution strategy
    -------------------
    1. Check internal registry cache.
    2. Try to resolve the class from the curated root module
       (by default: ``modacor.modules``).
    3. If still not found, search the filesystem under
       ``modules_root/**/<snake_case(name)>.py``, import it, and grab the class.
    """

    def __init__(
        self,
        modules_root: Path | None = None,
        curated_module: str | None = _DEFAULT_CURATED_MODULE_NAME,
    ) -> None:
        """
        Parameters
        ----------
        modules_root :
            Root directory for filesystem discovery. Defaults to
            the 'modules' directory alongside this file, i.e. ".../modacor/modules".
        curated_module :
            Dotted module path for the curated/explicit process steps module.
            Defaults to "modacor.modules". Set to None to disable the curated
            lookup step.
        """
        self._registry: Dict[str, Type[ProcessStep]] = {}
        self._modules_root = (modules_root or _DEFAULT_MODULES_ROOT).resolve()

        self._curated_module_name: str | None = curated_module
        self._curated_module = None
        if curated_module is not None:
            try:
                self._curated_module = importlib.import_module(curated_module)
            except ModuleNotFoundError:
                # If it doesn't exist, we silently disable the curated step.
                self._curated_module = None

    # ------------------------------------------------------------------ #
    # Registration / lookup API
    # ------------------------------------------------------------------ #

    def register(self, cls: Type[ProcessStep], name: str | None = None) -> None:
        """
        Register a ProcessStep subclass.

        Parameters
        ----------
        cls :
            The ProcessStep subclass to register.
        name :
            Optional explicit name. If omitted, `cls.__name__` is used.
        """
        if not issubclass(cls, ProcessStep):
            raise TypeError(f"Can only register ProcessStep subclasses, got {cls!r}.")
        key = name or cls.__name__
        self._registry[key] = cls

    def get(self, name: str) -> Type[ProcessStep]:
        """
        Retrieve a ProcessStep subclass by name.

        Resolution order
        ----------------
        1. If `name` has been explicitly registered, return it.
        2. If the curated module (e.g. modacor.modules) exposes an attribute `name`,
           validate it as a ProcessStep subclass, cache and return it.
        3. Else, convert `name` from PascalCase to snake_case and search for a
           module named `<snake_case(name)>.py` under `modules_root/**`.
           Import that module, fetch `name`, validate it, cache and return.
        """
        # 1. Already registered?
        if name in self._registry:
            return self._registry[name]

        # 2. Curated module lookup (modacor.modules by default)
        if self._curated_module is not None:
            try:
                cls = getattr(self._curated_module, name)
            except AttributeError:
                cls = None
            else:
                if not issubclass(cls, ProcessStep):
                    raise TypeError(
                        f"Object {name!r} in curated module "
                        f"{self._curated_module_name!r} is not a ProcessStep subclass."
                    )
                self._registry[name] = cls
                return cls

        # 3. Filesystem-based discovery under modules_root/**/<snake_case(name)>.py
        module_name = _pascal_to_snake(name)

        try:
            module_path = find_module(self._modules_root, module_name)
        except (ModuleNotFoundError, RuntimeError) as exc:
            raise KeyError(
                f"ProcessStep {name!r} not found in registry, not exported from "  # noqa: E713
                f"{self._curated_module_name!r}, and no module file '{module_name}.py' "
                f"could be resolved under {self._modules_root}."
            ) from exc

        module = importlib.import_module(module_path)
        try:
            cls = getattr(module, name)
        except AttributeError as exc:
            raise KeyError(f"ProcessStep class {name!r} not found in module {module_path!r}.") from exc  # noqa: E713

        if not issubclass(cls, ProcessStep):
            raise TypeError(f"Object {name!r} in module {module_path!r} is not a ProcessStep subclass.")

        self._registry[name] = cls
        return cls

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Default registry used by the Pipeline:
# - curated lookups via modacor.modules
# - filesystem discovery under src/modacor/modules/**/snake_case.py
DEFAULT_PROCESS_STEP_REGISTRY = ProcessStepRegistry()
