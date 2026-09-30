# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["AngleToQ"]
__version__ = "20260930.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.geometry.scattering_angle import q_angle_factor
from modacor.modules.helpers.scattering.photon_energy import photon_wavelength_from_energy


class AngleToQ(ProcessStep):
    """Convert a centred scattering or Bragg angle to signed momentum transfer."""

    documentation = ProcessStepDescriber(
        calling_name="Convert angle to signed Q",
        calling_id="AngleToQ",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["angle", "energy"],
        modifies={"Q": ["signal", "uncertainties", "units", "axes"]},
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "DataBundle key or keys whose angular coordinates are converted.",
            },
            "angle_key": {
                "type": str,
                "default": "angle",
                "doc": "Angular-coordinate BaseData key.",
            },
            "incident_key": {
                "type": str,
                "default": "energy",
                "doc": "Scalar or angle-shaped photon-energy or wavelength BaseData key.",
            },
            "incident_quantity": {
                "type": str,
                "default": "energy",
                "doc": "Quantity stored under incident_key: 'energy' or 'wavelength'.",
            },
            "center_key": {
                "type": (str, type(None)),
                "default": "beam_center",
                "doc": "Angle-zero BaseData key used when angle_zero is None; None means zero angle.",
            },
            "angle_zero": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Optional configured angle zero, overriding center_key.",
            },
            "angle_zero_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for angle_zero; defaults to the input angle units.",
            },
            "angle_convention": {
                "type": str,
                "default": "scattering_angle",
                "doc": (
                    "Angle convention: scattering_angle/two_theta uses sin(angle/2); "
                    "bragg_angle/theta uses sin(angle)."
                ),
            },
            "output_key": {
                "type": str,
                "default": "Q",
                "doc": "Output BaseData key.",
            },
            "output_units": {
                "type": str,
                "default": "1/nm",
                "doc": "Momentum-transfer output units.",
            },
        },
        step_keywords=["scattering angle", "two theta", "Bragg angle", "Q", "momentum transfer", "geometry"],
        step_doc="Convert an angle relative to its direct-beam or diffraction centre into signed Q.",
        step_note=(
            "Uses Q = 4*pi/lambda*sin(angle/2) for scattering_angle/two_theta and "
            "Q = 4*pi/lambda*sin(angle) for bragg_angle/theta. Photon energy is converted to wavelength "
            "with uncertainty-aware BaseData arithmetic. The angle sign is retained."
        ),
    )

    @staticmethod
    def _optional_key(value: object) -> str | None:
        if value is None:
            return None
        key = str(value).strip()
        return key or None

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if not processing_keys:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})
        angle_key = str(self.configuration.get("angle_key", "angle"))
        incident_key = str(self.configuration.get("incident_key", "energy"))
        read_keys = {angle_key, incident_key}
        if self.configuration.get("angle_zero") is None:
            center_key = self._optional_key(self.configuration.get("center_key", "beam_center"))
            if center_key is not None:
                read_keys.add(center_key)
        output_key = str(self.configuration.get("output_key", "Q"))
        return ProcessStepDependencies(
            processing_reads={f"{processing_key}.{key}" for processing_key in processing_keys for key in read_keys},
            processing_writes={f"{processing_key}.{output_key}" for processing_key in processing_keys},
        )

    def _angle_zero(self, bundle: DataBundle, angle: BaseData) -> BaseData:
        configured = self.configuration.get("angle_zero")
        if configured is not None:
            units = ureg.Unit(self.configuration.get("angle_zero_units") or angle.units)
            zero = BaseData(np.asarray(float(configured)), units)
            zero.to_units(angle.units)
            return zero
        center_key = self._optional_key(self.configuration.get("center_key", "beam_center"))
        if center_key is None:
            return BaseData(np.asarray(0.0), angle.units)
        if center_key not in bundle:
            raise KeyError(f"AngleToQ is missing configured center key {center_key!r}.")
        zero = bundle[center_key].copy(with_axes=False)
        zero.to_units(angle.units)
        return zero

    def _wavelength(self, incident: BaseData) -> BaseData:
        quantity = str(self.configuration.get("incident_quantity", "energy")).strip().lower()
        if quantity == "energy":
            return photon_wavelength_from_energy(incident, output_units="m")
        if quantity == "wavelength":
            wavelength = incident.copy(with_axes=False)
            wavelength.to_units(ureg.meter)
            signal = np.asarray(wavelength.signal, dtype=float)
            if np.any(~np.isfinite(signal)) or np.any(signal <= 0.0):
                raise ValueError("AngleToQ photon wavelength must be finite and positive.")
            return wavelength
        raise ValueError("AngleToQ incident_quantity must be 'energy' or 'wavelength'.")

    def calculate(self) -> dict[str, DataBundle]:
        angle_key = str(self.configuration.get("angle_key", "angle"))
        incident_key = str(self.configuration.get("incident_key", "energy"))
        output_key = str(self.configuration.get("output_key", "Q")).strip()
        if not output_key:
            raise ValueError("AngleToQ output_key must not be empty.")
        output_units = ureg.Unit(self.configuration.get("output_units", "1/nm"))
        factor = q_angle_factor(str(self.configuration.get("angle_convention", "scattering_angle")))
        four_pi = BaseData(signal=np.asarray(4.0 * np.pi), units=ureg.dimensionless)

        output: dict[str, DataBundle] = {}
        for processing_key in self._normalised_processing_keys():
            bundle = self.processing_data[processing_key]
            angle = bundle[angle_key].copy(with_axes=True)
            incident = bundle[incident_key].copy(with_axes=False)
            relative_angle = angle - self._angle_zero(bundle, angle)
            q = four_pi * (relative_angle * factor).sin() / self._wavelength(incident)
            q.to_units(output_units)
            bundle[output_key] = q
            output[processing_key] = bundle
        return output
