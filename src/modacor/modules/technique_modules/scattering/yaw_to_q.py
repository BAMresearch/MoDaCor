# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["YawToQ"]
__version__ = "20260929.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class YawToQ(ProcessStep):
    """Convert analyser yaw to signed momentum transfer."""

    documentation = ProcessStepDescriber(
        calling_name="Convert analyser yaw to signed Q",
        calling_id="YawToQ",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["yaw", "energy"],
        modifies={"Q": ["signal", "uncertainties", "units", "axes"]},
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "DataBundle key or keys whose yaw coordinates are converted.",
            },
            "yaw_key": {
                "type": str,
                "default": "yaw",
                "doc": "Analyser-yaw BaseData key.",
            },
            "energy_key": {
                "type": str,
                "default": "energy",
                "doc": "Scalar or yaw-shaped photon-energy BaseData key.",
            },
            "center_key": {
                "type": (str, type(None)),
                "default": "beam_center",
                "doc": "Yaw-zero BaseData key used when yaw_zero is None; None means zero yaw.",
            },
            "yaw_zero": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Optional configured yaw zero, overriding center_key.",
            },
            "yaw_zero_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for yaw_zero; defaults to yaw units.",
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
        step_keywords=["USAXS", "yaw", "Q", "momentum transfer", "geometry"],
        step_doc="Convert analyser yaw relative to its direct-beam centre into signed Q.",
        step_note=(
            "Uses Q = 4*pi/lambda*sin((yaw-yaw_zero)/2), with wavelength derived from measured photon energy. "
            "The sign is retained so the negative and positive analyser wings remain distinguishable."
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
        yaw_key = str(self.configuration.get("yaw_key", "yaw"))
        energy_key = str(self.configuration.get("energy_key", "energy"))
        read_keys = {yaw_key, energy_key}
        if self.configuration.get("yaw_zero") is None:
            center_key = self._optional_key(self.configuration.get("center_key", "beam_center"))
            if center_key is not None:
                read_keys.add(center_key)
        output_key = str(self.configuration.get("output_key", "Q"))
        return ProcessStepDependencies(
            processing_reads={f"{processing_key}.{key}" for processing_key in processing_keys for key in read_keys},
            processing_writes={f"{processing_key}.{output_key}" for processing_key in processing_keys},
        )

    def _yaw_zero(self, bundle: DataBundle, yaw: BaseData) -> BaseData:
        configured = self.configuration.get("yaw_zero")
        if configured is not None:
            units = ureg.Unit(self.configuration.get("yaw_zero_units") or yaw.units)
            zero = BaseData(np.asarray(float(configured)), units)
            zero.to_units(yaw.units)
            return zero
        center_key = self._optional_key(self.configuration.get("center_key", "beam_center"))
        if center_key is None:
            return BaseData(np.asarray(0.0), yaw.units)
        if center_key not in bundle:
            raise KeyError(f"YawToQ is missing configured center key {center_key!r}.")
        zero = bundle[center_key].copy(with_axes=False)
        zero.to_units(yaw.units)
        return zero

    def calculate(self) -> dict[str, DataBundle]:
        yaw_key = str(self.configuration.get("yaw_key", "yaw"))
        energy_key = str(self.configuration.get("energy_key", "energy"))
        output_key = str(self.configuration.get("output_key", "Q")).strip()
        if not output_key:
            raise ValueError("YawToQ output_key must not be empty.")
        output_units = ureg.Unit(self.configuration.get("output_units", "1/nm"))
        four_pi = BaseData(
            signal=np.asarray(4.0 * np.pi),
            units=ureg.dimensionless,
        )
        photon_energy_length = BaseData(
            signal=np.asarray(1.0),
            units=ureg.planck_constant * ureg.speed_of_light,
        )

        output: dict[str, DataBundle] = {}
        for processing_key in self._normalised_processing_keys():
            bundle = self.processing_data[processing_key]
            yaw = bundle[yaw_key].copy(with_axes=True)
            energy = bundle[energy_key].copy(with_axes=False)
            if np.any(np.asarray(energy.signal) <= 0.0):
                raise ValueError("YawToQ photon energy must be positive.")
            half_angle = (yaw - self._yaw_zero(bundle, yaw)) / 2.0
            q = four_pi * half_angle.sin() * energy / photon_energy_length
            q.to_units(output_units)
            bundle[output_key] = q
            output[processing_key] = bundle
        return output
