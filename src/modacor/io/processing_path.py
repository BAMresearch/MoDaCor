# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["ProcessingPath", "parse_processing_path", "resolve_processing_path", "infer_units_for_path"]

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.processing_data import ProcessingData


@dataclass(frozen=True, slots=True)
class ProcessingPath:
    original: str
    databundle_key: str
    basedata_name: str
    subpath: tuple[str, ...]


def parse_processing_path(path: str) -> ProcessingPath:
    if not isinstance(path, str) or not path.strip():
        raise TypeError("ProcessingData path must be a non-empty string.")

    original = path.strip()
    # normalize like posix path; allow leading '/'
    pp = PurePosixPath(original)
    parts = tuple(p for p in pp.parts if p != "/")

    if len(parts) < 2:
        raise ValueError(f"ProcessingData path must be at least '<bundle>/<basedata>' (got: {original}).")

    return ProcessingPath(
        original=original,
        databundle_key=parts[0],
        basedata_name=parts[1],
        subpath=parts[2:],
    )


def resolve_processing_path(processing_data: ProcessingData, path: str) -> Any:
    pp = parse_processing_path(path)
    bd = processing_data[pp.databundle_key][pp.basedata_name]
    if not isinstance(bd, BaseData):
        raise TypeError(f"Path {pp.original} did not resolve to a BaseData at the root.")

    obj: Any = bd
    for key in pp.subpath:
        if isinstance(obj, BaseData):
            if key == "signal":
                obj = obj.signal
                continue
            if key == "weights":
                obj = obj.weights
                continue
            if key == "uncertainties":
                obj = obj.uncertainties
                continue
            if key == "variances":
                obj = obj.variances
                continue

        if isinstance(obj, dict):
            obj = obj[key]
        else:
            if not hasattr(obj, key):
                raise AttributeError(
                    f"Object of type {type(obj).__name__} has no attribute '{key}' (path: {pp.original})"
                )
            obj = getattr(obj, key)

    return obj


def infer_units_for_path(processing_data: ProcessingData, path: str) -> str:
    """
    Stable unit inference for CSV header row 2.

    - .../signal -> bd.units
    - .../uncertainties/<k> -> bd.units
    - .../variances/<k> -> bd.units**2
    - .../weights -> dimensionless
    - otherwise -> ""
    """
    pp = parse_processing_path(path)
    bd: BaseData = processing_data[pp.databundle_key][pp.basedata_name]

    if len(pp.subpath) == 0:
        return ""  # no guessing

    first = pp.subpath[0]
    if first == "signal":
        return str(bd.units)
    if first == "uncertainties":
        return str(bd.units)
    if first == "variances":
        return str(bd.units**2)
    if first == "weights":
        return str(ureg.dimensionless)

    return ""
