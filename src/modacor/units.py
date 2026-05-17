# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/12/2025"
__status__ = "Development"  # "Development", "Production"
__version__ = "20251213.2"

from collections import ChainMap
from typing import Iterable

__all__ = ["configure_detector_pixel_units", "remove_pixel_units"]


def _delete_from_mapping(mapping, key: str) -> bool:
    """
    Delete key from dict-like or ChainMap-like mappings.
    Returns True if something was deleted.
    """
    deleted = False

    if mapping is None:
        return False

    if isinstance(mapping, ChainMap):
        for m in mapping.maps:
            if key in m:
                del m[key]
                deleted = True
    else:
        try:
            if key in mapping:
                del mapping[key]
                deleted = True
        except TypeError:
            # Not a normal mapping; fall back to pop if available
            pop = getattr(mapping, "pop", None)
            if callable(pop):
                _ = pop(key, None)
                deleted = True

    return deleted


def _delete_unit_names(ureg, names: Iterable[str]) -> None:
    """
    Remove unit definitions / aliases from the registry internals.

    Note: Pint does not currently expose a stable public "undefine" API,
    so we do controlled internal deletions and then rebuild the cache.
    """
    unit_tables = [
        getattr(ureg, "_units", None),
        getattr(ureg, "_units_casei", None),  # present in some pint versions
    ]

    for name in names:
        for table in unit_tables:
            _delete_from_mapping(table, name)

    # Rebuild Pint caches after modifications (important for correct parsing/dimensionality)
    build_cache = getattr(ureg, "_build_cache", None)
    if callable(build_cache):
        build_cache()


PIXEL_UNIT_NAMES = (
    "pixel",
    "pixels",
    "px",
    "css_pixel",
    "dot",
    "pel",
    "picture_element",
)


def remove_pixel_units(ureg) -> None:
    """
    Remove Pint's display-pixel unit definitions from a registry.

    This is a low-level cleanup helper used before MoDaCor installs its own
    dimensionless detector-coordinate pixel aliases.
    """
    # Pint's pixel-related aliases vary by version, so delete a small superset.
    _delete_unit_names(ureg, names=PIXEL_UNIT_NAMES)


_DETECTOR_PIXEL_DEFINITION = "pixel = 1 = px = pixels"
_DETECTOR_PIXEL_ALIASES = "@alias pixel = css_pixel = dot = pel = picture_element"


def configure_detector_pixel_units(ureg) -> None:
    """
    Configure detector element pixel names as dimensionless units.

    Pint ships display/CSS pixel definitions with physical display semantics.
    MoDaCor detector element indices are array coordinates instead, so we first
    remove Pint's defaults and then redefine common pixel spellings as aliases
    of a named unit with scale factor 1.
    """
    remove_pixel_units(ureg)
    ureg.define(_DETECTOR_PIXEL_DEFINITION)
    ureg.define(_DETECTOR_PIXEL_ALIASES)
