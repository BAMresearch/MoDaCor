# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["PolarizationCorrection"]
__version__ = "20260902.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, processing_key_patterns
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class PolarizationCorrection(ProcessStep):
    """Correct linear beam polarization using TwoTheta and detector azimuth Psi."""

    documentation = ProcessStepDescriber(
        calling_name="Polarization correction",
        calling_id="PolarizationCorrection",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "TwoTheta", "Psi"],
        modifies={"signal": ["signal", "uncertainties"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": ["sample"],
                "doc": "ProcessingData keys whose signal should be corrected.",
            },
            "mode": {
                "type": str,
                "default": "linear_fraction",
                "doc": "Polarization model. Currently supports 'linear_fraction'; 'stokes' is reserved.",
            },
            "two_theta_key": {
                "type": str,
                "default": "TwoTheta",
                "doc": "BaseData key containing the scattering angle 2theta.",
            },
            "psi_key": {
                "type": str,
                "default": "Psi",
                "doc": "BaseData key containing the detector azimuth.",
            },
            "correction_key": {
                "type": str,
                "default": "polarization_factor_map",
                "doc": "BaseData key used to store the applied polarization factor.",
            },
            "polarization_factor": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Fraction of horizontally polarized intensity. 0.5 is unpolarized.",
            },
            "polarisation_factor": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "British-spelling alias for polarization_factor.",
            },
            "polarization_angular_offset": {
                "type": (int, float),
                "default": 0.0,
                "doc": "Angular offset between Psi=0 and the horizontal polarization axis.",
            },
            "polarisation_angular_offset": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "British-spelling alias for polarization_angular_offset.",
            },
            "polarization_angular_offset_units": {
                "type": str,
                "default": "degree",
                "doc": "Units for polarization_angular_offset.",
            },
            "polarisation_angular_offset_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "British-spelling alias for polarization_angular_offset_units.",
            },
            "minimum_polarization_factor": {
                "type": (int, float),
                "default": 1e-12,
                "doc": "Lower allowed polarization factor before division.",
            },
        },
        step_keywords=["polarization", "polarisation", "scattering", "correction"],
        step_doc="Divide signal by the linear-polarization intensity factor.",
        step_note="""
            The implemented mode is DAWN/pyFAI-style linear_fraction:

              P = f * (1 - sin(2theta)^2 * cos(Psi - offset)^2)
                + (1 - f) * (1 - sin(2theta)^2 * sin(Psi - offset)^2)

            mode='stokes' is reserved for a future implementation once the
            Stokes reference-frame convention has been fixed.
        """,
    )

    @staticmethod
    def _angle_signal_radian(value: BaseData, *, name: str) -> np.ndarray:
        angle = value.copy(with_axes=False)
        try:
            angle.to_units(ureg.radian)
        except Exception as exc:
            raise ValueError(f"{name} must have angular units compatible with radians.") from exc
        return np.asarray(angle.signal, dtype=float)

    @staticmethod
    def _linear_fraction_factor(two_theta: np.ndarray, psi: np.ndarray, fraction: float, offset_radian: float):
        phi = psi - offset_radian
        sin2 = np.sin(two_theta) ** 2
        cos_phi2 = np.cos(phi) ** 2
        sin_phi2 = np.sin(phi) ** 2
        return fraction * (1.0 - sin2 * cos_phi2) + (1.0 - fraction) * (1.0 - sin2 * sin_phi2)

    @staticmethod
    def _fraction_from_config(cfg: dict[str, Any]) -> float:
        value = cfg.get("polarization_factor")
        if value is None:
            value = cfg.get("polarisation_factor", 0.5)
        fraction = float(value)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("polarization_factor must be between 0 and 1.")
        return fraction

    @staticmethod
    def _offset_from_config(cfg: dict[str, Any]) -> float:
        value = cfg.get("polarization_angular_offset")
        if cfg.get("polarisation_angular_offset") is not None:
            value = cfg["polarisation_angular_offset"]
        units = cfg.get("polarization_angular_offset_units", "degree")
        if cfg.get("polarisation_angular_offset_units") is not None:
            units = cfg["polarisation_angular_offset_units"]
        return float((float(value) * ureg.Unit(str(units))).to(ureg.radian).magnitude)

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        keys = cfg.get("with_processing_keys")
        two_theta_key = cfg.get("two_theta_key", "TwoTheta")
        psi_key = cfg.get("psi_key", "Psi")
        correction_key = cfg.get("correction_key", "polarization_factor_map")
        return ProcessStepDependencies(
            source_refs=(),
            processing_reads=(
                processing_key_patterns(keys, basedata_key="signal")
                | processing_key_patterns(keys, basedata_key=two_theta_key)
                | processing_key_patterns(keys, basedata_key=psi_key)
            ),
            processing_writes=(
                processing_key_patterns(keys, basedata_key="signal")
                | processing_key_patterns(keys, basedata_key=correction_key)
            ),
        )

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        mode = str(cfg.get("mode", "linear_fraction")).strip().lower()
        if mode == "stokes":
            raise NotImplementedError(
                "PolarizationCorrection mode='stokes' is reserved until the Stokes reference-frame convention is set."
            )
        if mode not in {"linear_fraction", "linear"}:
            raise ValueError("PolarizationCorrection mode must be 'linear_fraction' or reserved mode 'stokes'.")

        fraction = self._fraction_from_config(cfg)
        offset_radian = self._offset_from_config(cfg)
        minimum = float(cfg.get("minimum_polarization_factor", 1e-12))
        if minimum <= 0:
            raise ValueError("minimum_polarization_factor must be positive.")

        two_theta_key = cfg.get("two_theta_key", "TwoTheta")
        psi_key = cfg.get("psi_key", "Psi")
        correction_key = cfg.get("correction_key", "polarization_factor_map")

        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            databundle = self.processing_data[key]
            two_theta_bd = databundle[two_theta_key]
            psi_bd = databundle[psi_key]
            two_theta = self._angle_signal_radian(two_theta_bd, name=f"{key}::{two_theta_key}")
            psi = self._angle_signal_radian(psi_bd, name=f"{key}::{psi_key}")

            factor = self._linear_fraction_factor(two_theta, psi, fraction, offset_radian)
            if np.any(factor < minimum) or not np.all(np.isfinite(factor)):
                raise ValueError("Polarization correction contains too-small or non-finite factors.")

            correction = BaseData(
                signal=np.asarray(factor, dtype=float),
                units=ureg.dimensionless,
                rank_of_data=min(two_theta_bd.rank_of_data, np.ndim(factor)),
            )
            databundle[correction_key] = correction
            databundle["signal"] /= correction
            output[key] = databundle
        return output
