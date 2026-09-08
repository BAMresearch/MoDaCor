# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["FlatPlateSelfAbsorptionCorrection"]
__version__ = "20260905.1"

from pathlib import Path

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
from modacor.modules.technique_modules.scattering.material_attenuation import positive_cos_alpha, scalar_in_units


class FlatPlateSelfAbsorptionCorrection(ProcessStep):
    """Correct self-absorption in a uniform flat sample normal to the incident beam."""

    documentation = ProcessStepDescriber(
        calling_name="Flat-plate sample self-absorption correction",
        calling_id="FlatPlateSelfAbsorptionCorrection",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "CosAlpha"],
        modifies={"signal": ["signal", "uncertainties"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": ["sample"],
                "doc": "ProcessingData keys whose signal should be corrected.",
            },
            "cos_alpha_key": {
                "type": str,
                "default": "CosAlpha",
                "doc": "BaseData key containing cos(2theta) for the sample exit angle.",
            },
            "correction_key": {
                "type": str,
                "default": "flat_plate_self_absorption",
                "doc": "BaseData key used to store the relative attenuation factor.",
            },
            "transmission_source": {
                "type": str,
                "required": True,
                "default": None,
                "doc": "IoSources key for the sample transmission factor.",
            },
            "transmission_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for transmission units.",
            },
            "transmission_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Mapping of uncertainty names to IoSources keys for sample transmission.",
            },
            "minimum_cos_alpha": {
                "type": (int, float),
                "default": 1e-12,
                "doc": "Lower numerical clip for positive exit-angle cosines.",
            },
            "minimum_attenuation_factor": {
                "type": (int, float),
                "default": 1e-12,
                "doc": "Lower allowed relative attenuation factor before division.",
            },
        },
        step_keywords=["sample", "self absorption", "flat plate", "transmission"],
        step_doc="Correct angle-dependent self-absorption for a transmission-normalized flat sample.",
        step_note=(
            "For s=1/cos(2theta), x=(s-1) ln(T), the relative attenuation is expm1(x)/x. "
            "The expression is the depth integral for scattering generated uniformly through a plate and is not "
            "the same as transmission through a separate downstream attenuator."
        ),
    )

    @staticmethod
    def _relative_attenuation(transmission: float, cos_alpha: np.ndarray) -> np.ndarray:
        x = ((1.0 / cos_alpha) - 1.0) * np.log(transmission)
        result = np.ones_like(x, dtype=float)
        regular = np.abs(x) >= 1e-6
        result[regular] = np.expm1(x[regular]) / x[regular]
        small = ~regular
        result[small] = 1.0 + x[small] / 2.0 + x[small] ** 2 / 6.0
        return result

    @staticmethod
    def _relative_attenuation_derivative(transmission: float, cos_alpha: np.ndarray) -> np.ndarray:
        """Derivative of the relative attenuation factor with respect to transmission."""
        k = (1.0 / cos_alpha) - 1.0
        x = k * np.log(transmission)
        derivative_x = np.empty_like(x, dtype=float)
        regular = np.abs(x) >= 1e-6
        derivative_x[regular] = (np.exp(x[regular]) * x[regular] - np.expm1(x[regular])) / (x[regular] ** 2)
        small = ~regular
        derivative_x[small] = 0.5 + x[small] / 3.0 + x[small] ** 2 / 8.0
        return derivative_x * k / transmission

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        keys = cfg.get("with_processing_keys")
        cos_alpha_key = cfg.get("cos_alpha_key", "CosAlpha")
        correction_key = cfg.get("correction_key", "flat_plate_self_absorption")
        sources = [cfg.get("transmission_source"), cfg.get("transmission_units_source")]
        return ProcessStepDependencies(
            source_refs=source_refs_from_references(sources),
            processing_reads=(
                processing_key_patterns(keys, basedata_key="signal")
                | processing_key_patterns(keys, basedata_key=cos_alpha_key)
            ),
            processing_writes=(
                processing_key_patterns(keys, basedata_key="signal")
                | processing_key_patterns(keys, basedata_key=correction_key)
            ),
        )

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        transmission_data = basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=cfg.get("transmission_source"),
            units_source=cfg.get("transmission_units_source"),
            uncertainty_sources=cfg.get("transmission_uncertainties_sources", {}),
        )
        transmission = scalar_in_units(transmission_data, "dimensionless", name="transmission")
        if not np.isfinite(transmission) or transmission <= 0.0 or transmission > 1.0:
            raise ValueError("transmission must be finite and in the interval (0, 1].")

        minimum_cos_alpha = float(cfg.get("minimum_cos_alpha", 1e-12))
        minimum_factor = float(cfg.get("minimum_attenuation_factor", 1e-12))
        if minimum_factor <= 0.0:
            raise ValueError("minimum_attenuation_factor must be positive.")
        cos_alpha_key = cfg.get("cos_alpha_key", "CosAlpha")
        correction_key = cfg.get("correction_key", "flat_plate_self_absorption")

        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            databundle = self.processing_data[key]
            cos_alpha_bd = databundle[cos_alpha_key]
            cos_alpha = positive_cos_alpha(cos_alpha_bd, minimum_cos_alpha=minimum_cos_alpha)
            attenuation = self._relative_attenuation(transmission, cos_alpha)
            if np.any(attenuation < minimum_factor) or not np.all(np.isfinite(attenuation)):
                raise ValueError("Flat-plate self-absorption factor is too small or non-finite.")

            derivative = self._relative_attenuation_derivative(transmission, cos_alpha)
            correction_uncertainties = {
                name: np.abs(derivative) * float(np.asarray(values).reshape(-1)[0])
                for name, values in transmission_data.uncertainties.items()
            }
            correction = BaseData(
                signal=attenuation,
                units=ureg.dimensionless,
                uncertainties=correction_uncertainties,
                rank_of_data=min(cos_alpha_bd.rank_of_data, np.ndim(attenuation)),
            )
            databundle[correction_key] = correction
            databundle["signal"] /= correction
            output[key] = databundle
        return output
