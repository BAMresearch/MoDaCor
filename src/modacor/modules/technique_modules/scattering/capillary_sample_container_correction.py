# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["CapillarySampleContainerCorrection"]
__version__ = "20260906.2"

from copy import deepcopy
from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import (
    ProcessStepDependencies,
    processing_key_patterns,
    source_refs_from_references,
)
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.attenuation import (
    adaptive_attenuation_factors_on_grid,
    attenuation_factors_at_detectors,
    beam_chord_quadrature,
    direct_beam_transmission,
)
from modacor.modules.helpers import get_first_present
from modacor.modules.technique_modules.scattering.capillary_self_absorption_correction import (
    CapillarySelfAbsorptionCorrection,
)

_ARGUMENTS = deepcopy(CapillarySelfAbsorptionCorrection.documentation.arguments)
for _name in (
    "with_processing_keys",
    "input_state",
    "transmission_source",
    "transmission_units_source",
    "transmission_uncertainties_sources",
    "attenuation_key",
    "correction_key",
    "calculated_transmission_key",
    "evaluated_mask_key",
    "profile_retained_fraction_key",
):
    _ARGUMENTS.pop(_name)
_ARGUMENTS.update(
    {
        "filled_processing_key": {
            "type": str,
            "required": True,
            "default": "sample",
            "doc": "DataBundle containing the filled-capillary measurement and output signal.",
        },
        "empty_processing_key": {
            "type": str,
            "required": True,
            "default": "background",
            "doc": "DataBundle containing the matching empty-capillary measurement.",
        },
        "empty_mask_key": {
            "type": (str, type(None)),
            "default": None,
            "doc": "Empty-capillary mask key; defaults to mask_key.",
        },
        "empty_centre_mu": {
            "type": (int, float, type(None)),
            "default": 0.0,
            "doc": "Linear attenuation coefficient inside the nominally empty capillary.",
        },
        "empty_centre_mu_units": {
            "type": str,
            "default": "1/m",
            "doc": "Empty-centre-mu reciprocal-length units.",
        },
        "empty_centre_mu_source": {
            "type": (str, type(None)),
            "default": None,
            "doc": "Optional IoSources reference for empty-centre attenuation.",
        },
        "empty_centre_mu_units_source": {
            "type": (str, type(None)),
            "default": None,
            "doc": "Optional IoSources reference for empty-centre-mu units.",
        },
        "wall_chord_order": {
            "type": (int, type(None)),
            "default": None,
            "doc": "Wall-origin nodes per occupied chord; defaults to chord_order.",
        },
        "sample_attenuation_key": {
            "type": str,
            "default": "capillary_sample_attenuation",
            "doc": "Output key for A_s,sc.",
        },
        "wall_filled_attenuation_key": {
            "type": str,
            "default": "capillary_wall_attenuation_filled",
            "doc": "Output key for A_c,sc.",
        },
        "wall_empty_attenuation_key": {
            "type": str,
            "default": "capillary_wall_attenuation_empty",
            "doc": "Output key for A_c,c.",
        },
        "wall_subtraction_scale_key": {
            "type": str,
            "default": "capillary_wall_subtraction_scale",
            "doc": "Output key for A_c,sc/A_c,c.",
        },
        "filled_calculated_transmission_key": {
            "type": str,
            "default": "capillary_filled_calculated_transmission",
            "doc": "Output key for calculated filled-capillary transmission.",
        },
        "empty_calculated_transmission_key": {
            "type": str,
            "default": "capillary_empty_calculated_transmission",
            "doc": "Output key for calculated empty-capillary transmission.",
        },
        "profile_retained_fraction_key": {
            "type": str,
            "default": "capillary_beam_profile_retained_fraction",
            "doc": "Output key for retained beam-profile incident weight.",
        },
        "sample_evaluated_mask_key": {
            "type": str,
            "default": "capillary_sample_attenuation_evaluated",
            "doc": "Exact-evaluation mask for A_s,sc.",
        },
        "wall_filled_evaluated_mask_key": {
            "type": str,
            "default": "capillary_wall_filled_attenuation_evaluated",
            "doc": "Exact-evaluation mask for A_c,sc.",
        },
        "wall_empty_evaluated_mask_key": {
            "type": str,
            "default": "capillary_wall_empty_attenuation_evaluated",
            "doc": "Exact-evaluation mask for A_c,c.",
        },
    }
)


class CapillarySampleContainerCorrection(CapillarySelfAbsorptionCorrection):
    """Recover sample scattering from matched filled and empty capillaries."""

    documentation = ProcessStepDescriber(
        calling_name="Capillary sample and container attenuation correction",
        calling_id="CapillarySampleContainerCorrection",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "coord_x", "coord_y", "coord_z"],
        modifies={
            "signal": ["signal", "uncertainties"],
            "mask": ["signal"],
            "capillary_sample_attenuation": ["signal"],
            "capillary_wall_attenuation_filled": ["signal"],
            "capillary_wall_attenuation_empty": ["signal"],
            "capillary_wall_subtraction_scale": ["signal"],
            "capillary_filled_calculated_transmission": ["signal"],
            "capillary_empty_calculated_transmission": ["signal"],
            "capillary_beam_profile_retained_fraction": ["signal"],
            "capillary_sample_attenuation_evaluated": ["signal"],
            "capillary_wall_filled_attenuation_evaluated": ["signal"],
            "capillary_wall_empty_attenuation_evaluated": ["signal"],
        },
        arguments=_ARGUMENTS,
        step_keywords=["sample", "container", "capillary", "absorption", "subtraction"],
        step_doc=(
            "Subtract empty-capillary wall scattering on its filled-capillary attenuation scale, then correct "
            "sample-origin attenuation."
        ),
        step_reference="https://doi.org/10.1107/S0021889810021114",
        step_note=(
            "The filled and empty signals must already have comparable exposure and incident-flux normalization. "
            "Do not include a separate normalization by measured sample or empty-capillary transmission in the "
            "same correction pipeline. The operation is [F - (A_c,sc/A_c,c) E] / A_s,sc."
        ),
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        filled_key = cfg.get("filled_processing_key", "sample")
        empty_key = cfg.get("empty_processing_key", "background")
        coordinate_names = (
            cfg.get("coord_x_key", "coord_x"),
            cfg.get("coord_y_key", "coord_y"),
            cfg.get("coord_z_key", "coord_z"),
        )
        mask_key = cfg.get("mask_key", "mask")
        empty_mask_key = cfg.get("empty_mask_key") or mask_key
        filled_reads = {"signal", *coordinate_names}
        if mask_key:
            filled_reads.add(mask_key)
            if mask_key in {"mask", "Mask"}:
                filled_reads.add("Mask" if mask_key == "mask" else "mask")
        empty_reads = {"signal"}
        if empty_mask_key:
            empty_reads.add(empty_mask_key)
            if empty_mask_key in {"mask", "Mask"}:
                empty_reads.add("Mask" if empty_mask_key == "mask" else "mask")
        output_names = {
            "signal",
            mask_key,
            cfg.get("sample_attenuation_key", "capillary_sample_attenuation"),
            cfg.get("wall_filled_attenuation_key", "capillary_wall_attenuation_filled"),
            cfg.get("wall_empty_attenuation_key", "capillary_wall_attenuation_empty"),
            cfg.get("wall_subtraction_scale_key", "capillary_wall_subtraction_scale"),
            cfg.get("filled_calculated_transmission_key", "capillary_filled_calculated_transmission"),
            cfg.get("empty_calculated_transmission_key", "capillary_empty_calculated_transmission"),
            cfg.get("profile_retained_fraction_key", "capillary_beam_profile_retained_fraction"),
            cfg.get("sample_evaluated_mask_key", "capillary_sample_attenuation_evaluated"),
            cfg.get("wall_filled_evaluated_mask_key", "capillary_wall_filled_attenuation_evaluated"),
            cfg.get("wall_empty_evaluated_mask_key", "capillary_wall_empty_attenuation_evaluated"),
        }
        return ProcessStepDependencies(
            source_refs=source_refs_from_references(cfg),
            processing_reads=(
                frozenset().union(*(processing_key_patterns(filled_key, basedata_key=name) for name in filled_reads))
                | frozenset().union(*(processing_key_patterns(empty_key, basedata_key=name) for name in empty_reads))
            ),
            processing_writes=frozenset().union(
                *(processing_key_patterns(filled_key, basedata_key=name) for name in output_names if name)
            ),
        )

    @staticmethod
    def _mask_array(databundle: DataBundle, mask_key: str | None, target_shape) -> np.ndarray:
        if not mask_key:
            return np.zeros(target_shape, dtype=bool)
        aliases = (str(mask_key),)
        if mask_key in {"mask", "Mask"}:
            aliases += ("Mask" if mask_key == "mask" else "mask",)
        mask_data = get_first_present(databundle, *aliases)
        if mask_data is None:
            return np.zeros(target_shape, dtype=bool)
        try:
            return np.broadcast_to(np.asarray(mask_data.signal, dtype=bool), target_shape).copy()
        except ValueError as error:
            raise ValueError(
                f"Detector mask shape {np.shape(mask_data.signal)} cannot broadcast to signal shape {target_shape}."
            ) from error

    def _factor_map(
        self,
        *,
        geometry,
        coefficients,
        scattering_points,
        volume_weights,
        detector_grid,
        active,
        incident_direction,
    ) -> tuple[np.ndarray, np.ndarray]:
        evaluation_mode = str(self.configuration.get("evaluation_mode", "adaptive")).strip().lower()
        if evaluation_mode == "adaptive":
            result = adaptive_attenuation_factors_on_grid(
                geometry=geometry,
                attenuation_coefficients=coefficients,
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
            return result.factors, result.evaluated_mask
        if evaluation_mode != "exact":
            raise ValueError("evaluation_mode must be adaptive or exact.")
        coefficients_array = np.asarray(coefficients)
        coefficient_set_count = 1 if coefficients_array.ndim == 1 else coefficients_array.shape[0]
        exact = attenuation_factors_at_detectors(
            geometry=geometry,
            attenuation_coefficients=coefficients,
            scattering_points=scattering_points,
            volume_weights=volume_weights,
            detector_positions=detector_grid[active],
            incident_direction=incident_direction,
            detector_chunk_size=self.configuration.get("detector_chunk_size", 256),
        )
        if coefficients_array.ndim == 1:
            factors = np.ones(detector_grid.shape[:2], dtype=float)
            factors[active] = exact
        else:
            factors = np.ones((coefficient_set_count, *detector_grid.shape[:2]), dtype=float)
            factors[:, active] = exact
        return factors, active.copy()

    @staticmethod
    def _restore_masked_values(corrected: BaseData, original: BaseData, mask: np.ndarray) -> None:
        corrected.signal = np.where(mask, original.signal, corrected.signal)
        for name, values in corrected.uncertainties.items():
            original_values = original.uncertainties.get(name)
            replacement = 0.0 if original_values is None else original_values
            corrected.uncertainties[name] = np.where(mask, replacement, values)

    def calculate(self) -> dict[str, DataBundle]:
        filled_key = self.configuration.get("filled_processing_key", "sample")
        empty_key = self.configuration.get("empty_processing_key", "background")
        if filled_key == empty_key:
            raise ValueError("filled_processing_key and empty_processing_key must be different.")
        if filled_key not in self.processing_data or empty_key not in self.processing_data:
            raise KeyError("Both filled and empty processing keys must exist.")
        filled = self.processing_data[filled_key]
        empty = self.processing_data[empty_key]
        filled_signal = filled["signal"].copy()
        empty_signal = empty["signal"].copy()
        if filled_signal.shape != empty_signal.shape:
            raise ValueError(
                f"Filled and empty signal shapes must match, got {filled_signal.shape} and {empty_signal.shape}."
            )

        geometry, filled_coefficients = self._geometry_and_coefficients()
        if geometry.phase_count != 2:
            raise ValueError("Composite sample/container correction requires a positive wall_thickness.")
        empty_mu = self._scalar("empty_centre_mu", "1/m", required=False)
        if empty_mu < 0.0:
            raise ValueError("empty_centre_mu must be non-negative.")
        empty_coefficients = np.asarray([empty_mu, filled_coefficients[1]])
        incident_direction = self.configuration.get("incident_direction", (0.0, 0.0, 1.0))
        profile = self._beam_profile(geometry.centre, incident_direction)
        chord_order = self.configuration.get("chord_order", 12)
        wall_chord_order = self.configuration.get("wall_chord_order")
        if wall_chord_order is None:
            wall_chord_order = chord_order
        sample_points, sample_weights = beam_chord_quadrature(
            geometry=geometry,
            beam_points=profile.points,
            beam_weights=profile.weights,
            phase_index=0,
            incident_direction=incident_direction,
            chord_order=chord_order,
        )
        wall_points, wall_weights = beam_chord_quadrature(
            geometry=geometry,
            beam_points=profile.points,
            beam_weights=profile.weights,
            phase_index=1,
            incident_direction=incident_direction,
            chord_order=wall_chord_order,
        )

        coordinate_keys = (
            self.configuration.get("coord_x_key", "coord_x"),
            self.configuration.get("coord_y_key", "coord_y"),
            self.configuration.get("coord_z_key", "coord_z"),
        )
        detector_grid = self._coordinate_grid(filled, coordinate_keys)
        if filled_signal.signal.shape[-2:] != detector_grid.shape[:2]:
            raise ValueError(
                "The last two filled-signal axes must match the detector-coordinate grid; "
                f"got {filled_signal.signal.shape} and {detector_grid.shape[:2]}."
            )
        mask_key = self.configuration.get("mask_key", "mask")
        empty_mask_key = self.configuration.get("empty_mask_key") or mask_key
        filled_mask = self._mask_array(filled, mask_key, filled_signal.shape)
        empty_mask = self._mask_array(empty, empty_mask_key, empty_signal.shape)
        combined_mask = filled_mask | empty_mask
        active = (
            ~np.all(combined_mask, axis=tuple(range(combined_mask.ndim - 2)))
            if combined_mask.ndim > 2
            else ~combined_mask
        )

        sample_factor, sample_evaluated = self._factor_map(
            geometry=geometry,
            coefficients=filled_coefficients,
            scattering_points=sample_points,
            volume_weights=sample_weights,
            detector_grid=detector_grid,
            active=active,
            incident_direction=incident_direction,
        )
        wall_factors, wall_evaluated = self._factor_map(
            geometry=geometry,
            coefficients=np.stack((filled_coefficients, empty_coefficients)),
            scattering_points=wall_points,
            volume_weights=wall_weights,
            detector_grid=detector_grid,
            active=active,
            incident_direction=incident_direction,
        )
        wall_filled_factor, wall_empty_factor = wall_factors
        wall_filled_evaluated = wall_evaluated
        wall_empty_evaluated = wall_evaluated
        minimum_factor = float(self.configuration.get("minimum_attenuation_factor", 1e-12))
        if not np.isfinite(minimum_factor) or minimum_factor <= 0.0:
            raise ValueError("minimum_attenuation_factor must be finite and positive.")
        for name, factor in (
            ("sample", sample_factor),
            ("filled-wall", wall_filled_factor),
            ("empty-wall", wall_empty_factor),
        ):
            if not np.all(np.isfinite(factor[active])) or np.any(factor[active] < minimum_factor):
                raise ValueError(f"The {name} attenuation factor is too small or non-finite.")

        rank = min(filled_signal.rank_of_data, sample_factor.ndim)

        def factor_data(values):
            return BaseData(signal=values, units=ureg.dimensionless, rank_of_data=rank)

        sample_factor_data = factor_data(sample_factor)
        wall_filled_data = factor_data(wall_filled_factor)
        wall_empty_data = factor_data(wall_empty_factor)
        wall_scale_data = wall_filled_data / wall_empty_data
        corrected = (filled_signal - wall_scale_data * empty_signal) / sample_factor_data
        self._restore_masked_values(corrected, filled_signal, combined_mask)

        filled["signal"] = corrected
        if mask_key:
            filled[mask_key] = BaseData(
                signal=combined_mask,
                units=ureg.dimensionless,
                rank_of_data=filled_signal.rank_of_data,
            )
        output_values = {
            self.configuration.get("sample_attenuation_key", "capillary_sample_attenuation"): sample_factor_data,
            self.configuration.get(
                "wall_filled_attenuation_key", "capillary_wall_attenuation_filled"
            ): wall_filled_data,
            self.configuration.get("wall_empty_attenuation_key", "capillary_wall_attenuation_empty"): wall_empty_data,
            self.configuration.get("wall_subtraction_scale_key", "capillary_wall_subtraction_scale"): wall_scale_data,
            self.configuration.get("sample_evaluated_mask_key", "capillary_sample_attenuation_evaluated"): BaseData(
                signal=sample_evaluated, units=ureg.dimensionless, rank_of_data=rank
            ),
            self.configuration.get(
                "wall_filled_evaluated_mask_key", "capillary_wall_filled_attenuation_evaluated"
            ): BaseData(signal=wall_filled_evaluated, units=ureg.dimensionless, rank_of_data=rank),
            self.configuration.get(
                "wall_empty_evaluated_mask_key", "capillary_wall_empty_attenuation_evaluated"
            ): BaseData(signal=wall_empty_evaluated, units=ureg.dimensionless, rank_of_data=rank),
            self.configuration.get(
                "filled_calculated_transmission_key", "capillary_filled_calculated_transmission"
            ): BaseData(
                signal=np.asarray(
                    direct_beam_transmission(
                        geometry=geometry,
                        attenuation_coefficients=filled_coefficients,
                        beam_points=profile.points,
                        beam_weights=profile.weights,
                        incident_direction=incident_direction,
                    )
                ),
                units=ureg.dimensionless,
                rank_of_data=0,
            ),
            self.configuration.get(
                "empty_calculated_transmission_key", "capillary_empty_calculated_transmission"
            ): BaseData(
                signal=np.asarray(
                    direct_beam_transmission(
                        geometry=geometry,
                        attenuation_coefficients=empty_coefficients,
                        beam_points=profile.points,
                        beam_weights=profile.weights,
                        incident_direction=incident_direction,
                    )
                ),
                units=ureg.dimensionless,
                rank_of_data=0,
            ),
            self.configuration.get(
                "profile_retained_fraction_key", "capillary_beam_profile_retained_fraction"
            ): BaseData(
                signal=np.asarray(profile.retained_weight_fraction),
                units=ureg.dimensionless,
                rank_of_data=0,
            ),
        }
        filled.update(output_values)
        return {filled_key: filled}
