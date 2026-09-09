# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.io.io_sources import IoSources

HC_KEV_ANGSTROM = 12.398419843320026


@dataclass(frozen=True)
class MaterialAttenuation:
    linear_attenuation_coefficient_m_inv: float
    material: str | None = None
    density_g_cm3: float | None = None
    energy_kev: float | None = None


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


def scalar_quantity_from_config_or_source(
    io_sources: IoSources,
    cfg: dict[str, Any],
    name: str,
    *,
    default_units: str,
    required: bool = False,
    uncertainty_sources: dict[str, str] | None = None,
) -> BaseData | None:
    uncertainty_sources = {} if uncertainty_sources is None else uncertainty_sources
    source = cfg.get(f"{name}_source")
    if source is not None:
        return basedata_from_sources(
            io_sources=io_sources,
            signal_source=str(source),
            units_source=cfg.get(f"{name}_units_source"),
            uncertainty_sources=uncertainty_sources,
        )

    if name in cfg and cfg[name] is not None:
        return BaseData(
            signal=np.asarray(cfg[name], dtype=float),
            units=ureg.Unit(str(cfg.get(f"{name}_units", default_units))),
            uncertainties={
                uncertainty_name: io_sources.get_data(reference)
                for uncertainty_name, reference in uncertainty_sources.items()
            },
            rank_of_data=0,
        )

    if required:
        raise ValueError(f"Missing required material attenuation parameter: {name}.")
    return None


def scalar_in_units(value: BaseData, units: str, *, name: str) -> float:
    converted = value.copy(with_axes=False)
    converted.to_units(ureg.Unit(units))
    return float(_decode_scalar(converted.signal))


def text_from_config_or_source(io_sources: IoSources, cfg: dict[str, Any], *names: str) -> str | None:
    for name in names:
        if name in cfg and cfg[name] is not None:
            return str(_decode_scalar(cfg[name]))
        source = cfg.get(f"{name}_source")
        if source is None:
            continue
        try:
            return str(_decode_scalar(io_sources.get_static_metadata(str(source))))
        except Exception:
            return str(_decode_scalar(io_sources.get_data(str(source))))
    return None


def energy_kev_from_config_or_wavelength(io_sources: IoSources, cfg: dict[str, Any]) -> float:
    energy = scalar_quantity_from_config_or_source(
        io_sources,
        cfg,
        "beam_energy",
        default_units="keV",
        required=False,
    )
    if energy is not None:
        return scalar_in_units(energy, "keV", name="beam_energy")

    wavelength = scalar_quantity_from_config_or_source(
        io_sources,
        cfg,
        "wavelength",
        default_units="angstrom",
        required=True,
    )
    wavelength_angstrom = scalar_in_units(wavelength, "angstrom", name="wavelength")
    if wavelength_angstrom <= 0:
        raise ValueError("wavelength must be positive.")
    return HC_KEV_ANGSTROM / wavelength_angstrom


def _linear_attenuation_from_xraylib(material: str, density_g_cm3: float, energy_kev: float) -> float:
    try:
        import xraylib
    except ImportError as exc:
        raise ImportError(
            "Material attenuation lookup requires xraylib. Install MoDaCor[attenuation], "
            "or provide linear_attenuation_coefficient directly."
        ) from exc

    try:
        mass_attenuation_cm2_g = float(xraylib.CS_Total_CP(material, energy_kev))
    except Exception:
        z = xraylib.SymbolToAtomicNumber(material)
        mass_attenuation_cm2_g = float(xraylib.CS_Total(z, energy_kev))

    linear_cm_inv = mass_attenuation_cm2_g * density_g_cm3
    return linear_cm_inv * 100.0


def material_attenuation_from_config(io_sources: IoSources, cfg: dict[str, Any]) -> MaterialAttenuation:
    linear = scalar_quantity_from_config_or_source(
        io_sources,
        cfg,
        "linear_attenuation_coefficient",
        default_units="1/m",
        required=False,
    )
    if linear is not None:
        return MaterialAttenuation(
            linear_attenuation_coefficient_m_inv=scalar_in_units(linear, "1/m", name="linear_attenuation_coefficient")
        )

    material = text_from_config_or_source(io_sources, cfg, "chemical_composition", "composition", "material")
    if material is None:
        raise ValueError("Material attenuation requires material, composition, or chemical_composition.")

    density = scalar_quantity_from_config_or_source(io_sources, cfg, "density", default_units="g/cm^3", required=True)
    density_g_cm3 = scalar_in_units(density, "g/cm^3", name="density")
    if density_g_cm3 <= 0:
        raise ValueError("density must be positive.")

    energy_kev = energy_kev_from_config_or_wavelength(io_sources, cfg)
    if energy_kev <= 0:
        raise ValueError("beam energy must be positive.")

    return MaterialAttenuation(
        linear_attenuation_coefficient_m_inv=_linear_attenuation_from_xraylib(material, density_g_cm3, energy_kev),
        material=material,
        density_g_cm3=density_g_cm3,
        energy_kev=energy_kev,
    )


def thickness_m_from_config(io_sources: IoSources, cfg: dict[str, Any]) -> float:
    thickness = scalar_quantity_from_config_or_source(io_sources, cfg, "thickness", default_units="m", required=True)
    thickness_m = scalar_in_units(thickness, "m", name="thickness")
    if thickness_m < 0:
        raise ValueError("thickness must be >= 0.")
    return thickness_m


def positive_cos_alpha(cos_alpha: BaseData, *, minimum_cos_alpha: float) -> np.ndarray:
    cos_signal = np.asarray(cos_alpha.signal, dtype=float)
    if not np.all(np.isfinite(cos_signal)):
        raise ValueError("cos_alpha contains non-finite values.")
    if np.any(cos_signal <= 0):
        raise ValueError("cos_alpha must be positive for material attenuation corrections.")
    return np.clip(cos_signal, minimum_cos_alpha, None)
