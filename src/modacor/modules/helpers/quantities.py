# SPDX-License-Identifier: BSD-3-Clause
"""Nominal scalar conversion at configuration and numerical-kernel boundaries."""

from __future__ import annotations

import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData

__all__ = ["scalar_in_units"]


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
            return value.scalar_components(target)[0]
        return float(ureg.Quantity(float(value), target if units is None else units).to(target).magnitude)
    except (pint.PintError, ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"{name}: {exc}") from exc
