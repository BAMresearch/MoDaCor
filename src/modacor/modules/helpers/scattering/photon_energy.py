# SPDX-License-Identifier: BSD-3-Clause
"""Uncertainty-aware photon energy and wavelength conversion."""

from __future__ import annotations

__all__ = [
    "as_photon_wavelength",
    "photon_energy_from_wavelength",
    "photon_wavelength_from_energy",
]

import numpy as np
import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData


def _positive_finite(value: BaseData, *, quantity_name: str, reference_units: pint.Unit) -> None:
    converted = value.copy(with_axes=False)
    try:
        converted.to_units(reference_units)
    except (ValueError, pint.DimensionalityError) as exc:
        raise ValueError(f"Photon {quantity_name} must have {quantity_name} units, got {value.units}.") from exc
    signal = np.asarray(converted.signal, dtype=float)
    if np.any(~np.isfinite(signal)) or np.any(signal <= 0.0):
        raise ValueError(f"Photon {quantity_name} must be finite and positive.")


def _photon_energy_wavelength_reciprocal(
    value: BaseData,
    *,
    quantity_name: str,
    reference_units: pint.Unit,
    output_units: str | pint.Unit,
) -> BaseData:
    _positive_finite(value, quantity_name=quantity_name, reference_units=reference_units)
    photon_energy_length = BaseData(
        signal=np.asarray(1.0),
        units=ureg.planck_constant * ureg.speed_of_light,
    )
    converted = photon_energy_length / value
    try:
        converted.to_units(ureg.Unit(output_units))
    except (ValueError, pint.DimensionalityError) as exc:
        target_name = "wavelength" if quantity_name == "energy" else "energy"
        raise ValueError(f"Photon {target_name} output units are incompatible: {output_units}.") from exc
    return converted


def photon_wavelength_from_energy(
    energy: BaseData,
    *,
    output_units: str | pint.Unit = "m",
) -> BaseData:
    """Return ``h*c/energy`` while preserving BaseData uncertainties and metadata."""

    return _photon_energy_wavelength_reciprocal(
        energy,
        quantity_name="energy",
        reference_units=ureg.joule,
        output_units=output_units,
    )


def as_photon_wavelength(
    value: BaseData,
    *,
    output_units: str | pint.Unit = "m",
) -> BaseData:
    """Return photon energy or wavelength input as wavelength.

    The input representation is inferred from its Pint dimensionality. Energy
    inputs are converted through ``h*c/E``; length inputs are unit-converted
    directly. Both paths preserve BaseData uncertainties and metadata.
    """

    dimensionality = value.units.dimensionality
    if dimensionality == ureg.joule.dimensionality:
        return photon_wavelength_from_energy(value, output_units=output_units)
    if dimensionality != ureg.meter.dimensionality:
        raise ValueError(f"Photon metadata must have energy or wavelength units, got {value.units}.")

    _positive_finite(value, quantity_name="wavelength", reference_units=ureg.meter)
    wavelength = value.copy(with_axes=True)
    try:
        wavelength.to_units(ureg.Unit(output_units))
    except (ValueError, pint.DimensionalityError) as exc:
        raise ValueError(f"Photon wavelength output units are incompatible: {output_units}.") from exc
    return wavelength


def photon_energy_from_wavelength(
    wavelength: BaseData,
    *,
    output_units: str | pint.Unit = "J",
) -> BaseData:
    """Return ``h*c/wavelength`` while preserving BaseData uncertainties and metadata."""

    return _photon_energy_wavelength_reciprocal(
        wavelength,
        quantity_name="wavelength",
        reference_units=ureg.meter,
        output_units=output_units,
    )
