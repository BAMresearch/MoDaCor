# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["CapillarySelfAbsorptionCorrection"]
__version__ = "20260906.3"

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step import (
    ProcessStep,
    ProcessStepDependencies,
    processing_key_patterns,
    source_refs_from_references,
)
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.geometry import ConcentricCylinderGeometry
from modacor.models.attenuation import (
    adaptive_attenuation_factors_on_grid,
    attenuation_factors_at_detectors,
    beam_chord_quadrature,
    direct_beam_transmission,
    gaussian_beam_profile,
    image_beam_profile,
    trapezoid_beam_profile,
)
from modacor.modules.helpers import get_first_present
from modacor.modules.technique_modules.scattering.material_attenuation import (
    scalar_in_units,
    scalar_quantity_from_config_or_source,
)


def _length_array(value, units: str, *, name: str, shape=None) -> np.ndarray:
    try:
        converted = ureg.Quantity(np.asarray(value, dtype=float), ureg.Unit(str(units))).to("m").magnitude
    except Exception as error:
        raise ValueError(f"{name} must use units compatible with length.") from error
    result = np.asarray(converted, dtype=float)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _rotation_radians(profile_config: dict[str, Any]) -> float:
    value = float(profile_config.get("rotation", 0.0))
    units = str(profile_config.get("rotation_units", "degree"))
    try:
        return float(ureg.Quantity(value, ureg.Unit(units)).to("radian").magnitude)
    except Exception as error:
        raise ValueError("beam_profile rotation must use angular units.") from error


@dataclass(frozen=True, slots=True)
class _ResolvedSampleMu:
    value: float
    uncertainties: dict[str, float]


def _add_uncertainty_component(target: dict[str, np.ndarray], name: str, values) -> None:
    component = np.asarray(values, dtype=float)
    if name in target:
        component = np.hypot(target[name], component)
    target[name] = component


def _nominal_and_derivative(
    values: np.ndarray, delta: float | None, central: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    if delta is None:
        return values, None
    nominal = values[0]
    if central:
        return nominal, (values[1] - values[2]) / (2.0 * delta)
    return nominal, (values[1] - nominal) / delta


def _uncertainties_from_derivative(
    derivative: np.ndarray | float | None, parameter_uncertainties: dict[str, float]
) -> dict[str, np.ndarray]:
    if derivative is None:
        return {}
    return {name: np.abs(derivative) * uncertainty for name, uncertainty in parameter_uncertainties.items()}


class CapillarySelfAbsorptionCorrection(ProcessStep):
    """Correct sample-origin attenuation in a centred concentric capillary."""

    documentation = ProcessStepDescriber(
        calling_name="Capillary sample self-absorption correction",
        calling_id="CapillarySelfAbsorptionCorrection",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "coord_x", "coord_y", "coord_z"],
        modifies={
            "signal": ["signal", "uncertainties"],
            "capillary_sample_attenuation": ["signal", "uncertainties"],
            "capillary_self_absorption": ["signal", "uncertainties"],
            "capillary_calculated_transmission": ["signal", "uncertainties"],
            "capillary_effective_sample_mu": ["signal", "uncertainties"],
            "capillary_beam_profile_retained_fraction": ["signal"],
            "capillary_attenuation_evaluated": ["signal"],
        },
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": ["sample"],
                "doc": "ProcessingData keys whose detector signal should be corrected.",
            },
            "coord_x_key": {"type": str, "default": "coord_x", "doc": "Detector lab x-coordinate key."},
            "coord_y_key": {"type": str, "default": "coord_y", "doc": "Detector lab y-coordinate key."},
            "coord_z_key": {"type": str, "default": "coord_z", "doc": "Detector lab z-coordinate key."},
            "mask_key": {
                "type": (str, type(None)),
                "default": "mask",
                "doc": "Optional boolean mask key; True pixels are inactive and retain identity factors.",
            },
            "capillary_axis": {
                "type": tuple,
                "default": (1.0, 0.0, 0.0),
                "doc": "Capillary centreline direction in the lab frame; horizontal by default.",
            },
            "capillary_centre": {
                "type": tuple,
                "default": (0.0, 0.0, 0.0),
                "doc": "Point on the capillary centreline in capillary_centre_units.",
            },
            "capillary_centre_units": {"type": str, "default": "m", "doc": "Capillary-centre length units."},
            "incident_direction": {
                "type": tuple,
                "default": (0.0, 0.0, 1.0),
                "doc": "Incident beam propagation direction in the lab frame.",
            },
            "sample_radius": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Inner/sample radius; alternatively use sample_radius_source.",
            },
            "sample_radius_units": {"type": str, "default": "m", "doc": "Sample-radius units."},
            "sample_radius_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for sample radius.",
            },
            "sample_radius_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for sample-radius units.",
            },
            "wall_thickness": {
                "type": (int, float, type(None)),
                "default": 0.0,
                "doc": "Capillary wall thickness, separate from sample radius.",
            },
            "wall_thickness_units": {"type": str, "default": "m", "doc": "Wall-thickness units."},
            "wall_thickness_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for wall thickness.",
            },
            "wall_thickness_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for wall-thickness units.",
            },
            "sample_mu": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Sample linear attenuation coefficient; alternatively use its source or phase-factor inputs.",
            },
            "sample_mu_units": {"type": str, "default": "1/m", "doc": "Sample-mu reciprocal-length units."},
            "sample_mu_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for sample linear attenuation coefficient.",
            },
            "sample_mu_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for sample-mu units.",
            },
            "sample_phase_absorption": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Sample-only absorbed fraction A=1-T; use with sample_phase_thickness instead of sample_mu.",
            },
            "sample_phase_absorption_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for the dimensionless sample-only absorbed fraction.",
            },
            "sample_phase_absorption_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty-name to IoSources-reference mapping for sample-only absorption.",
            },
            "sample_phase_transmission": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Sample-only transmission T; use with sample_phase_thickness instead of sample_mu.",
            },
            "sample_phase_transmission_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for the dimensionless sample-only transmission.",
            },
            "sample_phase_transmission_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty-name to IoSources-reference mapping for sample-only transmission.",
            },
            "sample_phase_thickness": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Transmission path thickness used to derive effective sample mu.",
            },
            "sample_phase_thickness_units": {
                "type": str,
                "default": "m",
                "doc": "Sample-phase transmission path-thickness units.",
            },
            "sample_phase_thickness_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for the sample-phase transmission path thickness.",
            },
            "sample_phase_thickness_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for sample-phase thickness units.",
            },
            "sample_phase_thickness_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty-name to IoSources-reference mapping for sample-phase thickness.",
            },
            "wall_mu": {
                "type": (int, float, type(None)),
                "default": 0.0,
                "doc": "Wall linear attenuation coefficient; zero is allowed for limit checks.",
            },
            "wall_mu_units": {"type": str, "default": "1/m", "doc": "Wall-mu reciprocal-length units."},
            "wall_mu_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for wall linear attenuation coefficient.",
            },
            "wall_mu_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for wall-mu units.",
            },
            "beam_profile": {
                "type": dict,
                "required": True,
                "default": None,
                "doc": "Measured-image, gaussian_2d, or trapezoid_2d beam-profile configuration.",
            },
            "input_state": {
                "type": str,
                "default": "transmission_normalized",
                "doc": "One of raw, flux_normalized, or transmission_normalized.",
            },
            "transmission_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Measured effective whole-beam transmission for transmission_normalized input.",
            },
            "transmission_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for measured-transmission units.",
            },
            "transmission_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty-name to IoSources-reference mapping for measured transmission.",
            },
            "evaluation_mode": {
                "type": str,
                "default": "adaptive",
                "doc": "Adaptive detector-grid interpolation or exact active-pixel point rays.",
            },
            "relative_tolerance": {"type": (int, float), "default": 1e-3, "doc": "Adaptive relative tolerance."},
            "absolute_tolerance": {"type": (int, float), "default": 1e-12, "doc": "Adaptive absolute tolerance."},
            "max_depth": {"type": int, "default": 10, "doc": "Maximum adaptive subdivision depth."},
            "chord_order": {"type": int, "default": 12, "doc": "Gauss--Legendre nodes per occupied chord."},
            "sample_mu_sensitivity_relative_step": {
                "type": (int, float),
                "default": 1e-2,
                "doc": "Relative finite-difference step for derived-sample-mu uncertainty propagation.",
            },
            "detector_chunk_size": {"type": int, "default": 256, "doc": "Expert detector chunk-size override."},
            "minimum_attenuation_factor": {
                "type": (int, float),
                "default": 1e-12,
                "doc": "Minimum allowed active-pixel attenuation divisor.",
            },
            "attenuation_key": {
                "type": str,
                "default": "capillary_sample_attenuation",
                "doc": "Absolute A_s,sc map key.",
            },
            "correction_key": {
                "type": str,
                "default": "capillary_self_absorption",
                "doc": "Applied residual-divisor map key.",
            },
            "calculated_transmission_key": {
                "type": str,
                "default": "capillary_calculated_transmission",
                "doc": "Calculated whole-beam transmission diagnostic key.",
            },
            "effective_sample_mu_key": {
                "type": str,
                "default": "capillary_effective_sample_mu",
                "doc": "Resolved or derived effective sample attenuation-coefficient key.",
            },
            "profile_retained_fraction_key": {
                "type": str,
                "default": "capillary_beam_profile_retained_fraction",
                "doc": "Retained incident-weight fraction after beam-profile truncation or thresholding.",
            },
            "evaluated_mask_key": {
                "type": str,
                "default": "capillary_attenuation_evaluated",
                "doc": "Boolean map identifying exactly evaluated detector pixels.",
            },
        },
        step_keywords=["sample", "self absorption", "capillary", "cylinder", "transmission"],
        step_doc="Correct detector-resolved sample self-absorption in a centred cylindrical capillary.",
        step_reference="https://doi.org/10.1021/acs.cgd.5c00551",
        step_note=(
            "The default capillary axis is horizontal; tilted straight capillaries are supported. "
            "Measured transmission is an effective whole-beam value, not a centreline estimate of mu. "
            "This step is only for sample-origin scattering after consistent container removal or when wall "
            "scattering is negligible; filled/empty attenuation-aware container subtraction requires the "
            "separate composite correction."
        ),
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        keys = cfg.get("with_processing_keys")
        read_keys = [
            "signal",
            cfg.get("coord_x_key", "coord_x"),
            cfg.get("coord_y_key", "coord_y"),
            cfg.get("coord_z_key", "coord_z"),
        ]
        mask_key = cfg.get("mask_key", "mask")
        if mask_key:
            read_keys.append(mask_key)
            if mask_key in {"mask", "Mask"}:
                read_keys.append("Mask" if mask_key == "mask" else "mask")
        write_keys = [
            "signal",
            cfg.get("attenuation_key", "capillary_sample_attenuation"),
            cfg.get("correction_key", "capillary_self_absorption"),
            cfg.get("calculated_transmission_key", "capillary_calculated_transmission"),
            cfg.get("effective_sample_mu_key", "capillary_effective_sample_mu"),
            cfg.get("profile_retained_fraction_key", "capillary_beam_profile_retained_fraction"),
            cfg.get("evaluated_mask_key", "capillary_attenuation_evaluated"),
        ]
        return ProcessStepDependencies(
            source_refs=source_refs_from_references(cfg),
            processing_reads=frozenset().union(
                *(processing_key_patterns(keys, basedata_key=name) for name in read_keys)
            ),
            processing_writes=frozenset().union(
                *(processing_key_patterns(keys, basedata_key=name) for name in write_keys)
            ),
        )

    def _scalar(self, name: str, units: str, *, required: bool) -> float:
        value, _uncertainties = self._scalar_with_uncertainties(name, units, required=required)
        return value

    def _scalar_with_uncertainties(self, name: str, units: str, *, required: bool) -> tuple[float, dict[str, float]]:
        data = scalar_quantity_from_config_or_source(
            self.io_sources,
            self.configuration,
            name,
            default_units=units,
            required=required,
            uncertainty_sources=self.configuration.get(f"{name}_uncertainties_sources", {}),
        )
        if data is None:
            return 0.0, {}
        value = scalar_in_units(data, units, name=name)
        converted = data.copy(with_axes=False)
        converted.to_units(ureg.Unit(units))
        uncertainties = {}
        for uncertainty_name, uncertainty_values in converted.uncertainties.items():
            values = np.asarray(uncertainty_values, dtype=float).reshape(-1)
            if values.size != 1 and not np.allclose(values, values[0], rtol=0.0, atol=0.0, equal_nan=True):
                raise ValueError(f"{name} uncertainty {uncertainty_name!r} must be scalar or constant-valued.")
            uncertainty = float(values[0])
            if not np.isfinite(uncertainty) or uncertainty < 0.0:
                raise ValueError(f"{name} uncertainty {uncertainty_name!r} must be finite and non-negative.")
            uncertainties[uncertainty_name] = uncertainty
        return value, uncertainties

    def _is_configured(self, name: str) -> bool:
        return self.configuration.get(f"{name}_source") is not None or self.configuration.get(name) is not None

    def _sample_mu(self) -> _ResolvedSampleMu:
        has_mu = self._is_configured("sample_mu")
        has_absorption = self._is_configured("sample_phase_absorption")
        has_transmission = self._is_configured("sample_phase_transmission")
        derived_count = int(has_absorption) + int(has_transmission)
        if has_mu and derived_count:
            raise ValueError("Configure sample_mu or a sample-phase absorption/transmission with thickness, not both.")
        if derived_count > 1:
            raise ValueError("Configure only one of sample_phase_absorption and sample_phase_transmission.")
        if has_mu:
            return _ResolvedSampleMu(self._scalar("sample_mu", "1/m", required=True), {})
        if not derived_count:
            raise ValueError("Configure sample_mu, sample_phase_absorption, or sample_phase_transmission.")
        if not self._is_configured("sample_phase_thickness"):
            raise ValueError(
                "sample_phase_thickness is required when deriving sample_mu from absorption or transmission."
            )

        thickness, thickness_uncertainties = self._scalar_with_uncertainties(
            "sample_phase_thickness", "m", required=True
        )
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("sample_phase_thickness must be finite and positive.")
        if has_absorption:
            absorption, factor_uncertainties = self._scalar_with_uncertainties(
                "sample_phase_absorption", "dimensionless", required=True
            )
            if not np.isfinite(absorption) or not 0.0 <= absorption < 1.0:
                raise ValueError("sample_phase_absorption must be finite and in [0, 1).")
            transmission = 1.0 - absorption
            factor_derivative = 1.0 / (thickness * transmission)
        else:
            transmission, factor_uncertainties = self._scalar_with_uncertainties(
                "sample_phase_transmission", "dimensionless", required=True
            )
            if not np.isfinite(transmission) or not 0.0 < transmission <= 1.0:
                raise ValueError("sample_phase_transmission must be finite and in (0, 1].")
            factor_derivative = -1.0 / (thickness * transmission)
        sample_mu = float(-np.log(transmission) / thickness)
        thickness_derivative = -sample_mu / thickness
        uncertainties = {}
        for uncertainty_name in factor_uncertainties.keys() | thickness_uncertainties.keys():
            factor_component = factor_derivative * factor_uncertainties.get(uncertainty_name, 0.0)
            thickness_component = thickness_derivative * thickness_uncertainties.get(uncertainty_name, 0.0)
            uncertainties[uncertainty_name] = float(np.hypot(factor_component, thickness_component))
        return _ResolvedSampleMu(sample_mu, uncertainties)

    def _geometry_and_coefficients(
        self,
    ) -> tuple[ConcentricCylinderGeometry, np.ndarray, _ResolvedSampleMu]:
        radius = self._scalar("sample_radius", "m", required=True)
        thickness = self._scalar("wall_thickness", "m", required=False)
        resolved_sample_mu = self._sample_mu()
        sample_mu = resolved_sample_mu.value
        wall_mu = self._scalar("wall_mu", "1/m", required=False)
        if radius <= 0.0:
            raise ValueError("sample_radius must be positive.")
        if thickness < 0.0:
            raise ValueError("wall_thickness must be non-negative.")
        if not np.isfinite(sample_mu) or not np.isfinite(wall_mu) or sample_mu < 0.0 or wall_mu < 0.0:
            raise ValueError("sample_mu and wall_mu must be finite and non-negative.")
        if thickness > 0.0 and wall_mu == 0.0:
            self.logger.warning("Capillary wall thickness is positive but wall_mu is zero.")

        radii = [radius]
        coefficients = [sample_mu]
        if thickness > 0.0:
            radii.append(radius + thickness)
            coefficients.append(wall_mu)
        centre = _length_array(
            self.configuration.get("capillary_centre", (0.0, 0.0, 0.0)),
            self.configuration.get("capillary_centre_units", "m"),
            name="capillary_centre",
            shape=(3,),
        )
        geometry = ConcentricCylinderGeometry(
            radii=np.asarray(radii),
            axis=self.configuration.get("capillary_axis", (1.0, 0.0, 0.0)),
            centre=centre,
        )
        return geometry, np.asarray(coefficients), resolved_sample_mu

    def _sample_mu_coefficient_sets(
        self,
        geometry: ConcentricCylinderGeometry,
        coefficients: np.ndarray,
        sample_mu_uncertainties: dict[str, float],
    ) -> tuple[np.ndarray, float | None, bool]:
        if not sample_mu_uncertainties:
            return coefficients, None, False
        relative_step = float(self.configuration.get("sample_mu_sensitivity_relative_step", 1e-2))
        if not np.isfinite(relative_step) or not 0.0 < relative_step < 1.0:
            raise ValueError("sample_mu_sensitivity_relative_step must be finite and in (0, 1).")
        sample_mu = float(coefficients[0])
        delta = relative_step * max(sample_mu, 1.0 / float(geometry.radii[0]))
        increased = coefficients.copy()
        increased[0] = sample_mu + delta
        if sample_mu <= delta:
            return np.stack((coefficients, increased)), delta, False
        decreased = coefficients.copy()
        decreased[0] = sample_mu - delta
        return np.stack((coefficients, increased, decreased)), delta, True

    def _beam_profile(self, centre: np.ndarray, incident_direction):
        profile = self.configuration.get("beam_profile")
        if not isinstance(profile, dict):
            raise ValueError("beam_profile must be configured as a mapping.")
        profile_type = str(profile.get("type", "")).strip().lower()
        rotation = _rotation_radians(profile)
        if profile_type == "image":
            if profile.get("signal_source") is not None:
                image = self.io_sources.get_data(str(profile["signal_source"]))
            elif profile.get("signal") is not None:
                image = profile["signal"]
            else:
                raise ValueError("An image beam_profile requires signal or signal_source.")
            pitch = _length_array(
                profile.get("pixel_pitch"),
                profile.get("pixel_pitch_units", "m"),
                name="beam_profile pixel_pitch",
                shape=(2,),
            )
            return image_beam_profile(
                image,
                pixel_pitch=pitch,
                image_centre=profile.get("image_centre"),
                centre=centre,
                incident_direction=incident_direction,
                rotation=rotation,
                downsample=profile.get("downsample", 1),
                relative_weight_cutoff=profile.get("relative_weight_cutoff", 0.0),
            )
        if profile_type in {"gaussian", "gaussian_2d"}:
            sigma = _length_array(
                profile.get("standard_deviations"),
                profile.get("width_units", "m"),
                name="beam_profile standard_deviations",
                shape=(2,),
            )
            return gaussian_beam_profile(
                standard_deviations=sigma,
                quadrature_order=profile.get("quadrature_order", 12),
                truncation_sigma=profile.get("truncation_sigma", (4.0, 4.0)),
                centre=centre,
                incident_direction=incident_direction,
                rotation=rotation,
            )
        if profile_type in {"trapezoid", "trapezoid_2d"}:
            units = profile.get("width_units", "m")
            plateau = _length_array(profile.get("plateau_width"), units, name="beam_profile plateau_width", shape=(2,))
            ramps = _length_array(profile.get("ramp_widths"), units, name="beam_profile ramp_widths", shape=(2, 2))
            return trapezoid_beam_profile(
                plateau_width=plateau,
                ramp_widths=ramps,
                quadrature_order_per_region=profile.get("quadrature_order_per_region", 6),
                centre=centre,
                incident_direction=incident_direction,
                rotation=rotation,
            )
        raise ValueError(f"Unsupported beam_profile type: {profile_type!r}.")

    @staticmethod
    def _coordinate_grid(databundle: DataBundle, keys: tuple[str, str, str]) -> np.ndarray:
        converted = []
        for key in keys:
            if key not in databundle:
                raise KeyError(f"CapillarySelfAbsorptionCorrection requires BaseData key {key!r}.")
            coordinate = databundle[key].copy(with_axes=False)
            coordinate.to_units(ureg.m)
            converted.append(np.asarray(coordinate.signal, dtype=float))
        x, y, z = np.broadcast_arrays(*converted)
        if x.ndim != 2:
            raise ValueError(f"Detector coordinates must broadcast to a two-dimensional grid, got {x.shape}.")
        return np.stack((x, y, z), axis=-1)

    def _active_mask(self, databundle: DataBundle, detector_shape: tuple[int, int]) -> np.ndarray:
        mask_key = self.configuration.get("mask_key", "mask")
        if not mask_key:
            return np.ones(detector_shape, dtype=bool)
        aliases = (str(mask_key),)
        if mask_key in {"mask", "Mask"}:
            aliases += ("Mask" if mask_key == "mask" else "mask",)
        mask_data = get_first_present(databundle, *aliases)
        if mask_data is None:
            return np.ones(detector_shape, dtype=bool)
        raw_mask = np.asarray(mask_data.signal, dtype=bool)
        if raw_mask.ndim > 2 and raw_mask.shape[-2:] == detector_shape:
            # A detector pixel is globally inactive only when it is masked in
            # every leading scan/frame position. The same static correction
            # map can then be broadcast over all frames without evaluating
            # pixels that never contribute.
            raw_mask = np.all(raw_mask, axis=tuple(range(raw_mask.ndim - 2)))
        try:
            mask = np.broadcast_to(raw_mask, detector_shape)
        except ValueError as error:
            raise ValueError(
                f"Detector mask shape {np.shape(mask_data.signal)} cannot broadcast to {detector_shape}."
            ) from error
        return ~mask

    def _measured_transmission(self) -> BaseData | None:
        state = str(self.configuration.get("input_state", "transmission_normalized")).strip().lower()
        if state not in {"raw", "flux_normalized", "transmission_normalized"}:
            raise ValueError("input_state must be raw, flux_normalized, or transmission_normalized.")
        if state != "transmission_normalized":
            return None
        source = self.configuration.get("transmission_source")
        if source is None:
            raise ValueError("transmission_source is required for transmission_normalized input.")
        transmission = basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=str(source),
            units_source=self.configuration.get("transmission_units_source"),
            uncertainty_sources=self.configuration.get("transmission_uncertainties_sources", {}),
        )
        value = scalar_in_units(transmission, "dimensionless", name="transmission")
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("transmission must be finite and in (0, 1].")
        return transmission

    def calculate(self) -> dict[str, DataBundle]:
        geometry, coefficients, resolved_sample_mu = self._geometry_and_coefficients()
        evaluation_coefficients, mu_delta, central_mu_difference = self._sample_mu_coefficient_sets(
            geometry, coefficients, resolved_sample_mu.uncertainties
        )
        incident_direction = self.configuration.get("incident_direction", (0.0, 0.0, 1.0))
        profile = self._beam_profile(geometry.centre, incident_direction)
        scattering_points, volume_weights = beam_chord_quadrature(
            geometry=geometry,
            beam_points=profile.points,
            beam_weights=profile.weights,
            phase_index=0,
            incident_direction=incident_direction,
            chord_order=self.configuration.get("chord_order", 12),
        )
        coefficient_rows = (
            evaluation_coefficients[None, :] if evaluation_coefficients.ndim == 1 else evaluation_coefficients
        )
        calculated_transmission_values = np.asarray(
            [
                direct_beam_transmission(
                    geometry=geometry,
                    attenuation_coefficients=coefficient_row,
                    beam_points=profile.points,
                    beam_weights=profile.weights,
                    incident_direction=incident_direction,
                )
                for coefficient_row in coefficient_rows
            ]
        )
        if mu_delta is None:
            calculated_transmission = calculated_transmission_values[0]
            transmission_mu_derivative = None
        else:
            calculated_transmission, transmission_mu_derivative = _nominal_and_derivative(
                calculated_transmission_values, mu_delta, central_mu_difference
            )
        measured_transmission = self._measured_transmission()
        measured_value = (
            1.0
            if measured_transmission is None
            else scalar_in_units(measured_transmission, "dimensionless", name="transmission")
        )
        evaluation_mode = str(self.configuration.get("evaluation_mode", "adaptive")).strip().lower()
        if evaluation_mode not in {"adaptive", "exact"}:
            raise ValueError("evaluation_mode must be adaptive or exact.")

        coordinate_keys = (
            self.configuration.get("coord_x_key", "coord_x"),
            self.configuration.get("coord_y_key", "coord_y"),
            self.configuration.get("coord_z_key", "coord_z"),
        )
        attenuation_key = self.configuration.get("attenuation_key", "capillary_sample_attenuation")
        correction_key = self.configuration.get("correction_key", "capillary_self_absorption")
        transmission_key = self.configuration.get("calculated_transmission_key", "capillary_calculated_transmission")
        effective_mu_key = self.configuration.get("effective_sample_mu_key", "capillary_effective_sample_mu")
        evaluated_key = self.configuration.get("evaluated_mask_key", "capillary_attenuation_evaluated")
        retained_fraction_key = self.configuration.get(
            "profile_retained_fraction_key", "capillary_beam_profile_retained_fraction"
        )
        minimum_factor = float(self.configuration.get("minimum_attenuation_factor", 1e-12))
        if not np.isfinite(minimum_factor) or minimum_factor <= 0.0:
            raise ValueError("minimum_attenuation_factor must be finite and positive.")

        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            databundle = self.processing_data[key]
            detector_grid = self._coordinate_grid(databundle, coordinate_keys)
            active = self._active_mask(databundle, detector_grid.shape[:2])
            if evaluation_mode == "adaptive":
                adaptive = adaptive_attenuation_factors_on_grid(
                    geometry=geometry,
                    attenuation_coefficients=evaluation_coefficients,
                    scattering_points=scattering_points,
                    volume_weights=volume_weights,
                    detector_position_grid=detector_grid,
                    active_mask=active,
                    incident_direction=incident_direction,
                    relative_tolerance=self.configuration.get("relative_tolerance", 1e-3),
                    absolute_tolerance=self.configuration.get("absolute_tolerance", 1e-12),
                    max_depth=self.configuration.get("max_depth", 10),
                    detector_chunk_size=self.configuration.get("detector_chunk_size", 256),
                )
                attenuation_values = adaptive.factors
                evaluated = adaptive.evaluated_mask
            else:
                active_values = attenuation_factors_at_detectors(
                    geometry=geometry,
                    attenuation_coefficients=evaluation_coefficients,
                    scattering_points=scattering_points,
                    volume_weights=volume_weights,
                    detector_positions=detector_grid[active],
                    incident_direction=incident_direction,
                    detector_chunk_size=self.configuration.get("detector_chunk_size", 256),
                )
                if evaluation_coefficients.ndim == 1:
                    attenuation_values = np.ones(detector_grid.shape[:2], dtype=float)
                    attenuation_values[active] = active_values
                else:
                    attenuation_values = np.ones(
                        (evaluation_coefficients.shape[0], *detector_grid.shape[:2]), dtype=float
                    )
                    attenuation_values[:, active] = active_values
                evaluated = active.copy()

            attenuation, attenuation_mu_derivative = _nominal_and_derivative(
                attenuation_values, mu_delta, central_mu_difference
            )

            if not np.all(np.isfinite(attenuation[active])) or np.any(attenuation[active] < minimum_factor):
                raise ValueError("Capillary attenuation factor is too small or non-finite at active pixels.")
            residual = attenuation / measured_value
            residual[~active] = 1.0
            attenuation_uncertainties = {}
            if attenuation_mu_derivative is not None:
                attenuation_uncertainties = {
                    name: np.abs(attenuation_mu_derivative) * uncertainty
                    for name, uncertainty in resolved_sample_mu.uncertainties.items()
                }
                for values in attenuation_uncertainties.values():
                    values[~active] = 0.0
            correction_uncertainties = {
                name: values / measured_value for name, values in attenuation_uncertainties.items()
            }
            if measured_transmission is not None:
                for name, values in measured_transmission.uncertainties.items():
                    component = attenuation * np.asarray(values, dtype=float) / measured_value**2
                    component[~active] = 0.0
                    _add_uncertainty_component(correction_uncertainties, name, component)

            transmission_uncertainties = {}
            if transmission_mu_derivative is not None:
                transmission_uncertainties = {
                    name: np.asarray(abs(transmission_mu_derivative) * uncertainty)
                    for name, uncertainty in resolved_sample_mu.uncertainties.items()
                }

            rank = min(databundle["signal"].rank_of_data, attenuation.ndim)
            databundle[attenuation_key] = BaseData(
                signal=attenuation,
                units=ureg.dimensionless,
                uncertainties=attenuation_uncertainties,
                rank_of_data=rank,
            )
            databundle[correction_key] = BaseData(
                signal=residual,
                units=ureg.dimensionless,
                uncertainties=correction_uncertainties,
                rank_of_data=rank,
            )
            databundle[transmission_key] = BaseData(
                signal=np.asarray(calculated_transmission),
                units=ureg.dimensionless,
                uncertainties=transmission_uncertainties,
                rank_of_data=0,
            )
            databundle[effective_mu_key] = BaseData(
                signal=np.asarray(resolved_sample_mu.value),
                units=ureg.Unit("1/m"),
                uncertainties={
                    name: np.asarray(uncertainty) for name, uncertainty in resolved_sample_mu.uncertainties.items()
                },
                rank_of_data=0,
            )
            databundle[evaluated_key] = BaseData(
                signal=evaluated,
                units=ureg.dimensionless,
                rank_of_data=rank,
            )
            databundle[retained_fraction_key] = BaseData(
                signal=np.asarray(profile.retained_weight_fraction),
                units=ureg.dimensionless,
                rank_of_data=0,
            )
            databundle["signal"] /= databundle[correction_key]
            output[key] = databundle
        return output
