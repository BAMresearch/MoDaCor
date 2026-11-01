# SPDX-License-Identifier: BSD-3-Clause
"""Nominal scalar conversion at configuration and numerical-kernel boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData

__all__ = ["scalar_in_units"]


def _decode_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape != ():
        flattened = arr.reshape(-1)
        if flattened.size != 1:
            first = flattened[0]
            if np.issubdtype(arr.dtype, np.number):
                is_constant = np.allclose(flattened, first, rtol=0.0, atol=0.0, equal_nan=True)
            else:
                is_constant = bool(np.all(flattened == first))
            if not is_constant:
                raise ValueError(f"Expected a scalar or constant-valued array, got shape {arr.shape}.")
        value = flattened[0]
    else:
        value = arr.item() if hasattr(arr, "item") else value
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return value


def scalar_in_units(
    value: BaseData | float | int,
    target_units: str | pint.Unit,
    *,
    name: str,
    units: str | pint.Unit | None = None,
) -> float:
    """Convert a nominal scalar, accepting exactly constant BaseData arrays.

    Numeric configuration values without ``units`` already use target units.
    BaseData carries its own units and is converted on a copy. This function
    returns only a nominal value; uncertainty-aware arithmetic must retain the
    BaseData or adapt its uncertainty components at the kernel boundary.
    """
    try:
        target = ureg.Unit(target_units)
        if isinstance(value, BaseData):
            if units is not None:
                raise ValueError("units must be None for BaseData inputs")
            converted = value.copy(with_axes=False)
            converted.to_units(target)
            return float(_decode_scalar(converted.signal))
        return float(ureg.Quantity(float(value), target if units is None else units).to(target).magnitude)
    except (pint.PintError, ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"{name}: {exc}") from exc
