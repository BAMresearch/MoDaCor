# SPDX-License-Identifier: BSD-3-Clause
"""Nominal scalar conversion at configuration and numerical-kernel boundaries."""

from __future__ import annotations

import numbers

import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData

__all__ = ["scalar_in_units"]


def scalar_in_units(
    value: float | int,
    target_units: str | pint.Unit,
    *,
    name: str,
    units: str | pint.Unit | None = None,
) -> float:
    """Convert a configuration number; BaseData must use its central accessors.

    Without ``units``, the number already uses the requested target units.
    """
    if isinstance(value, BaseData) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name}: expected a configuration number, got {type(value).__name__}.")
    try:
        target = ureg.Unit(target_units)
        return float(ureg.Quantity(float(value), target if units is None else units).to(target).magnitude)
    except (pint.PintError, ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"{name}: {exc}") from exc
