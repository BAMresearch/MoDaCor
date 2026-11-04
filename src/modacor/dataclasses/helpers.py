# SPDX-License-Identifier: BSD-3-Clause
"""Load BaseData and uncertainty components from registered IO sources."""

from __future__ import annotations

__all__ = ["basedata_from_sources", "uncertainties_from_sources"]
__authors__ = ["Brian R. Pauw", "Armin Moser"]

import logging

import numpy as np
import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.io.io_sources import IoSources

logger = logging.getLogger(__name__)


def uncertainties_from_sources(
    io_sources: IoSources,
    uncertainty_sources: dict[str, str],
    signal_units: pint.Unit,
    *,
    read_units: bool = True,
) -> dict[str, np.ndarray]:
    """Load sigmas in signal units through BaseData's difference-conversion path.

    Missing/empty units, or an explicitly unsupported attributes capability,
    mean the values already use signal units. With read_units=False, do not
    request attributes at all. Reject offset/non-multiplicative units rather
    than interpret an uncertainty as an absolute quantity.
    """
    uncertainties = {}
    for name, reference in uncertainty_sources.items():
        values = io_sources.get_data(reference)
        if not read_units:
            uncertainties[name] = values
            continue
        try:
            attributes = io_sources.get_data_attributes(reference)
        except NotImplementedError:
            # Attribute-free sources (including YAML) may signal this capability explicitly.
            attributes = {}
        raw_units = attributes.get("units")
        try:
            source_units = raw_units.decode("utf-8") if isinstance(raw_units, bytes) else raw_units
            if source_units is None or (isinstance(source_units, str) and not source_units.strip()):
                uncertainties[name] = values
                continue
            source_units = ureg.Unit(source_units)
        except Exception as exc:
            # Unit-attribute parser boundary: Pint can raise Python grammar errors; retain the cause.
            raise ValueError(
                f"Uncertainty {name!r} from {reference!r}: cannot convert units {raw_units!r} "
                f"to signal units {signal_units!s}: {exc}"
            ) from exc
        try:
            target_units = ureg.Unit(signal_units)
            # A difference must keep zero at zero; Pint supplies this offset check.
            if any(ureg.Quantity(0.0, unit).to_base_units().magnitude != 0 for unit in (source_units, target_units)):
                raise ValueError("Offset/non-multiplicative units cannot represent uncertainty differences.")
            if source_units == target_units:
                uncertainties[name] = values
                continue
            converted = BaseData(
                signal=np.zeros_like(values), units=source_units, uncertainties={name: values}
            ).magnitudes(target_units)[1][name]
        except (pint.PintError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Uncertainty {name!r} from {reference!r}: cannot convert units {raw_units!r} "
                f"to signal units {signal_units!s}: {exc}"
            ) from exc
        uncertainties[name] = converted
        if source_units != target_units:
            logger.debug("Converted uncertainty %r from %s (%s) to %s", name, reference, source_units, target_units)
    return uncertainties


def basedata_from_sources(
    io_sources: IoSources,
    signal_source: str | None,
    units_source: str | None = None,
    uncertainty_sources: dict[str, str] = {},
    weights_source: str | None = None,
    *,
    units_override: str | pint.Unit | None = None,
) -> BaseData:
    """Helper function to build a BaseData object from IoSources

    Parameters
    ----------
    io_sources : IoSources
        The IoSources object to load data from.
    signal_source : str | None
        Required non-blank source key for the signal data. None raises ValueError.
    units_source : str | None, optional
        The source key for the units data, by default None.
        for iosources that support attributes, the units can also be stored as an attribute.
        In that case, it can be specified by 'key to the dataset@[units_attribute_name]'
    uncertainty_sources : dict[str, str], optional
        Map uncertainty names to source keys. Each dataset's units attribute is
        converted to signal units as a difference. Missing/empty attributes use
        signal units; incompatible, unreadable and offset units raise ValueError.
    weights_source : str | None, optional
        Optional source key for BaseData weights, by default None.
    units_override : str | pint.Unit | None, optional
        Label the signal and all sigmas with these units without converting
        their values. Skip units_source and uncertainty-unit attributes entirely.
        Scalar overrides retain their labels and values without scalar reduction.
    """
    if signal_source is None or not signal_source.strip():
        raise ValueError("signal_source is required")
    signal = io_sources.get_data(signal_source)
    if units_override is not None:
        units = ureg.Unit(units_override)
    else:
        units = (
            ureg.Unit(io_sources.get_static_metadata(units_source)) if units_source is not None else ureg.dimensionless
        )
    uncertainties = uncertainties_from_sources(
        io_sources, uncertainty_sources, units, read_units=units_override is None
    )
    kwargs = {"signal": signal, "units": units, "uncertainties": uncertainties}
    if weights_source is not None:
        kwargs["weights"] = io_sources.get_data(weights_source)
    if units_override is None and np.size(signal) == 1:
        return BaseData.from_scalar(**kwargs)
    return BaseData(**kwargs)
