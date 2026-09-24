# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["B21FrameQualityFilter"]
__version__ = "20260916.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, processing_key_patterns
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class B21FrameQualityFilter(ProcessStep):
    """Flag B21 frames from low- and high-q intensity excursions.

    Bit 0 (value 1) marks a high-q total below the configured fraction of the
    largest valid high-q total. Bit 1 (value 2) marks a low-q total above the
    configured factor times the smallest valid low-q total. Missing/non-finite
    regional data set the corresponding bit.
    """

    HIGH_Q_LOW_INTENSITY = np.uint32(1)
    LOW_Q_HIGH_INTENSITY = np.uint32(2)

    documentation = ProcessStepDescriber(
        calling_name="B21 frame quality filter",
        calling_id="B21FrameQualityFilter",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Q"],
        modifies={},
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "Frame-wise integrated B21 DataBundle key or keys.",
            },
            "signal_key": {
                "type": str,
                "default": "signal",
                "doc": "Frame-wise I(q) BaseData key.",
                "dependency_role": "processing_read_basedata_key",
            },
            "q_key": {
                "type": str,
                "default": "Q",
                "doc": "Q-coordinate BaseData key.",
                "dependency_role": "processing_read_basedata_key",
            },
            "low_q_max": {
                "type": (int, float),
                "required": True,
                "default": 0.02,
                "doc": "Inclusive upper bound of the low-q assessment region.",
            },
            "high_q_min": {
                "type": (int, float),
                "required": True,
                "default": 0.15,
                "doc": "Inclusive lower bound of the high-q assessment region.",
            },
            "q_limits_unit": {
                "type": str,
                "default": "1/angstrom",
                "doc": "Units of low_q_max and high_q_min.",
            },
            "high_q_min_fraction": {
                "type": (int, float),
                "default": 0.95,
                "doc": "Minimum accepted high-q total divided by the maximum reference total.",
            },
            "low_q_max_factor": {
                "type": (int, float),
                "default": 1.05,
                "doc": "Maximum accepted low-q total divided by the minimum reference total.",
            },
            "flags_key": {
                "type": str,
                "default": "frame_quality_flags",
                "doc": "Output uint32 bitfield BaseData key.",
                "dependency_role": "processing_write_basedata_key",
            },
        },
        step_keywords=["DLS", "B21", "frame", "quality", "bubble", "radiation damage", "flags"],
        step_doc="Flag B21 frames whose coarse I(q) totals depart from batch-relative references.",
        step_reference="",
        step_note=(
            "Flags are batch-relative: 0=accepted, 1=high-q intensity too low, "
            "2=low-q intensity too high, and 3=both conditions."
        ),
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        keys = self.configuration.get("with_processing_keys")
        reads = set()
        writes = set()
        for data_key in (self.configuration.get("signal_key"), self.configuration.get("q_key")):
            reads.update(processing_key_patterns(keys, basedata_key=str(data_key)))
        for data_key in (
            self.configuration.get("flags_key"),
            "high_q_total",
            "low_q_total",
            "high_q_reference",
            "low_q_reference",
        ):
            writes.update(processing_key_patterns(keys, basedata_key=str(data_key)))
        return ProcessStepDependencies(processing_reads=reads, processing_writes=writes)

    def _evaluate_bundle(self, bundle: DataBundle) -> DataBundle:
        signal_key = str(self.configuration.get("signal_key", "signal"))
        q_key = str(self.configuration.get("q_key", "Q"))
        flags_key = str(self.configuration.get("flags_key", "frame_quality_flags"))
        signal_bd = bundle[signal_key]
        q_bd = bundle[q_key]

        signal = np.asarray(signal_bd.signal, dtype=float)
        if signal.ndim < 1:
            raise ValueError("B21FrameQualityFilter requires at least one Q-bin dimension.")
        q_values = np.asarray(q_bd.signal, dtype=float)
        try:
            q_values = np.broadcast_to(q_values, signal.shape)
        except ValueError as exc:
            raise ValueError(
                f"B21FrameQualityFilter: Q shape {q_values.shape} cannot broadcast to I(q) shape {signal.shape}."
            ) from exc

        q_unit = ureg.Unit(self.configuration.get("q_limits_unit", "1/angstrom"))
        low_q_max = (float(self.configuration["low_q_max"]) * q_unit).to(q_bd.units).magnitude
        high_q_min = (float(self.configuration["high_q_min"]) * q_unit).to(q_bd.units).magnitude
        if not np.isfinite(low_q_max) or not np.isfinite(high_q_min) or low_q_max >= high_q_min:
            raise ValueError("B21FrameQualityFilter requires finite low_q_max < high_q_min.")

        high_fraction = float(self.configuration.get("high_q_min_fraction", 0.95))
        low_factor = float(self.configuration.get("low_q_max_factor", 1.05))
        if not 0.0 < high_fraction <= 1.0:
            raise ValueError("high_q_min_fraction must satisfy 0 < value <= 1.")
        if low_factor < 1.0:
            raise ValueError("low_q_max_factor must be at least 1.")

        frame_shape = signal.shape[:-1]
        curves = signal.reshape((-1, signal.shape[-1]))
        q_curves = q_values.reshape(curves.shape)
        finite = np.isfinite(curves) & np.isfinite(q_curves)
        high_region = q_curves >= high_q_min
        low_region = q_curves <= low_q_max
        high_counts = np.sum(finite & high_region, axis=-1)
        low_counts = np.sum(finite & low_region, axis=-1)
        high_totals = np.sum(np.where(finite & high_region, curves, 0.0), axis=-1)
        low_totals = np.sum(np.where(finite & low_region, curves, 0.0), axis=-1)

        valid_high = high_counts > 0
        valid_low = low_counts > 0
        if not np.any(valid_high):
            raise ValueError("B21FrameQualityFilter found no finite bins in the high-q region.")
        if not np.any(valid_low):
            raise ValueError("B21FrameQualityFilter found no finite bins in the low-q region.")
        high_reference = float(np.max(high_totals[valid_high]))
        low_reference = float(np.min(low_totals[valid_low]))
        if high_reference <= 0.0 or low_reference <= 0.0:
            raise ValueError("B21 frame-quality reference totals must be positive.")

        flags = np.zeros(curves.shape[0], dtype=np.uint32)
        flags[(~valid_high) | (high_totals < high_fraction * high_reference)] |= self.HIGH_Q_LOW_INTENSITY
        flags[(~valid_low) | (low_totals > low_factor * low_reference)] |= self.LOW_Q_HIGH_INTENSITY

        metric_rank = 0
        bundle[flags_key] = BaseData(
            signal=flags.reshape(frame_shape),
            units=ureg.dimensionless,
            rank_of_data=metric_rank,
        )
        bundle["high_q_total"] = BaseData(
            signal=high_totals.reshape(frame_shape),
            units=signal_bd.units,
            rank_of_data=metric_rank,
        )
        bundle["low_q_total"] = BaseData(
            signal=low_totals.reshape(frame_shape),
            units=signal_bd.units,
            rank_of_data=metric_rank,
        )
        bundle["high_q_reference"] = BaseData(
            signal=np.asarray(high_reference),
            units=signal_bd.units,
            rank_of_data=0,
        )
        bundle["low_q_reference"] = BaseData(
            signal=np.asarray(low_reference),
            units=signal_bd.units,
            rank_of_data=0,
        )
        return bundle

    def calculate(self) -> dict[str, DataBundle]:
        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            if key not in self.processing_data:
                raise KeyError(f"B21FrameQualityFilter DataBundle not found: {key!r}")
            output[key] = self._evaluate_bundle(self.processing_data[key])
        return output
