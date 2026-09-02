# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["AttenuatorPlateCorrection"]
__version__ = "20260902.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import (
    ProcessStep,
    ProcessStepDependencies,
    processing_key_patterns,
    source_refs_from_references,
)
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.technique_modules.scattering.material_attenuation import (
    material_attenuation_from_config,
    positive_cos_alpha,
    thickness_m_from_config,
)


class AttenuatorPlateCorrection(ProcessStep):
    """Correct absorption through a flat attenuator plate in front of a detector."""

    documentation = ProcessStepDescriber(
        calling_name="Attenuator plate correction",
        calling_id="AttenuatorPlateCorrection",
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
                "doc": "BaseData key containing the incidence cosine n dot rhat.",
            },
            "correction_key": {
                "type": str,
                "default": "attenuator_transmission",
                "doc": "BaseData key used to store the applied transmission map.",
            },
            "normalize_to_normal_incidence": {
                "type": bool,
                "default": False,
                "doc": "If true, divide by transmission(cos_alpha) / transmission(cos_alpha=1).",
            },
            "linear_attenuation_coefficient": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Optional direct linear attenuation coefficient.",
            },
            "linear_attenuation_coefficient_units": {
                "type": str,
                "default": "1/m",
                "doc": "Units for linear_attenuation_coefficient.",
            },
            "linear_attenuation_coefficient_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for a direct linear attenuation coefficient.",
            },
            "linear_attenuation_coefficient_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for linear attenuation coefficient units.",
            },
            "material": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Attenuator material formula or element symbol, used with xraylib lookup.",
            },
            "material_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for attenuator material formula or element symbol.",
            },
            "chemical_composition": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Alias for material.",
            },
            "chemical_composition_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for chemical_composition.",
            },
            "density": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Attenuator material density.",
            },
            "density_units": {
                "type": str,
                "default": "g/cm^3",
                "doc": "Units for density.",
            },
            "density_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for attenuator density.",
            },
            "density_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for density units.",
            },
            "thickness": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Attenuator thickness.",
            },
            "thickness_units": {
                "type": str,
                "default": "m",
                "doc": "Units for thickness.",
            },
            "thickness_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for attenuator thickness.",
            },
            "thickness_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for thickness units.",
            },
            "beam_energy": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Incident beam energy.",
            },
            "beam_energy_units": {
                "type": str,
                "default": "keV",
                "doc": "Units for beam_energy.",
            },
            "beam_energy_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for incident beam energy.",
            },
            "beam_energy_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for beam energy units.",
            },
            "wavelength": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Incident wavelength, used to derive beam energy if beam_energy is not configured.",
            },
            "wavelength_units": {
                "type": str,
                "default": "angstrom",
                "doc": "Units for wavelength.",
            },
            "wavelength_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for incident wavelength.",
            },
            "wavelength_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources key for wavelength units.",
            },
            "minimum_cos_alpha": {
                "type": (int, float),
                "default": 1e-12,
                "doc": "Lower numerical clip for positive cos_alpha values.",
            },
        },
        step_keywords=["attenuator", "plate", "attenuation", "absorption", "transmission"],
        step_doc="Divide signal by angle-dependent attenuator plate transmission.",
        step_note=(
            "Material parameters can be provided as direct values or *_source paths. "
            "xraylib is used for material lookup unless linear_attenuation_coefficient is configured directly."
        ),
    )

    @staticmethod
    def _transmission(mu_m_inv: float, thickness_m: float, cos_alpha: np.ndarray) -> np.ndarray:
        return np.exp((-mu_m_inv * thickness_m) / cos_alpha)

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        keys = cfg.get("with_processing_keys")
        cos_alpha_key = cfg.get("cos_alpha_key", "CosAlpha")
        source_values = [
            value for key, value in cfg.items() if key.endswith("_source") or key.endswith("_units_source")
        ]
        return ProcessStepDependencies(
            source_refs=source_refs_from_references(source_values),
            processing_reads=processing_key_patterns(keys) | processing_key_patterns(keys, basedata_key=cos_alpha_key),
            processing_writes=processing_key_patterns(keys),
        )

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        attenuation = material_attenuation_from_config(self.io_sources, cfg)
        thickness_m = thickness_m_from_config(self.io_sources, cfg)
        minimum_cos_alpha = float(cfg.get("minimum_cos_alpha", 1e-12))
        normalize = bool(cfg.get("normalize_to_normal_incidence", False))
        cos_alpha_key = cfg.get("cos_alpha_key", "CosAlpha")
        correction_key = cfg.get("correction_key", "attenuator_transmission")

        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            databundle = self.processing_data[key]
            cos_alpha_bd = databundle[cos_alpha_key]
            cos_alpha = positive_cos_alpha(cos_alpha_bd, minimum_cos_alpha=minimum_cos_alpha)

            transmission = self._transmission(
                attenuation.linear_attenuation_coefficient_m_inv,
                thickness_m,
                cos_alpha,
            )
            if normalize:
                normal_transmission = float(
                    self._transmission(attenuation.linear_attenuation_coefficient_m_inv, thickness_m, np.asarray(1.0))
                )
                if normal_transmission <= 0:
                    raise ValueError("Normal-incidence attenuator transmission is zero.")
                divisor = transmission / normal_transmission
            else:
                divisor = transmission
            if np.any(divisor <= 0) or not np.all(np.isfinite(divisor)):
                raise ValueError("Attenuator plate correction contains non-positive or non-finite values.")

            correction = BaseData(
                signal=np.asarray(divisor, dtype=float),
                units=ureg.dimensionless,
                rank_of_data=cos_alpha_bd.rank_of_data,
            )
            databundle[correction_key] = correction
            databundle["signal"] /= correction
            output[key] = databundle
        return output
