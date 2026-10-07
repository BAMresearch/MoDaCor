# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["SubtractInterpolated1D"]
__version__ = "20260929.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.interpolation import remap_coefficients_1d


class SubtractInterpolated1D(ProcessStep):
    """Remap a background curve to a sample axis and subtract it."""

    documentation = ProcessStepDescriber(
        calling_name="Subtract remapped one-dimensional background",
        calling_id="SubtractInterpolated1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Q"],
        modifies={
            "signal": ["signal", "uncertainties", "weights", "units"],
            "remap_mask": ["signal"],
        },
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "Two processing keys: sample/minuend then background/subtrahend.",
            },
            "signal_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData signal key in both bundles.",
            },
            "axis_key": {
                "type": str,
                "default": "Q",
                "doc": "One-dimensional coordinate key in both bundles.",
            },
            "mode": {
                "type": str,
                "default": "nearest",
                "doc": "Remapping mode: nearest or linear.",
            },
            "mask_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional input mask key used in both bundles; nonzero means invalid.",
            },
            "max_gap": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Optional maximum linear-interpolation bracket width; unsupported in nearest mode.",
            },
            "max_gap_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units of max_gap; defaults to the sample axis units.",
            },
            "output_signal_key": {
                "type": str,
                "default": "signal",
                "doc": "Sample-bundle key receiving the subtracted signal.",
            },
            "remapped_background_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional sample-bundle key receiving the remapped background.",
            },
            "output_mask_key": {
                "type": (str, type(None)),
                "default": "remap_mask",
                "doc": "Optional output key marking targets invalid for subtraction.",
            },
        },
        step_keywords=["subtract", "background", "nearest", "linear", "remap", "1D"],
        step_doc="Remap a 1D background by nearest neighbour or linear interpolation, then subtract it.",
        step_note=(
            "Nearest mode always chooses the nearest valid background coordinate and has no tolerance setting. "
            "Neither mode extrapolates: sample points outside the background domain are invalidated. max_gap "
            "applies only to linear bracket width. Linear uncertainties are propagated from interpolation "
            "coefficients before BaseData subtraction."
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
        if len(processing_keys) != 2:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})
        sample_key, background_key = processing_keys
        signal_key = str(self.configuration.get("signal_key", "signal"))
        axis_key = str(self.configuration.get("axis_key", "Q"))
        mask_key = self._optional_key(self.configuration.get("mask_key"))
        read_data_keys = {signal_key, axis_key}
        if mask_key is not None:
            read_data_keys.add(mask_key)
        reads = {
            f"{processing_key}.{data_key}"
            for processing_key in (sample_key, background_key)
            for data_key in read_data_keys
        }
        writes = {f"{sample_key}.{self.configuration.get('output_signal_key', 'signal')}"}
        for configured_key in ("remapped_background_key", "output_mask_key"):
            data_key = self._optional_key(self.configuration.get(configured_key))
            if data_key is not None:
                writes.add(f"{sample_key}.{data_key}")
        return ProcessStepDependencies(processing_reads=reads, processing_writes=writes)

    @staticmethod
    def _one_dimensional(bundle: DataBundle, key: str, role: str) -> BaseData:
        if key not in bundle:
            raise KeyError(f"SubtractInterpolated1D {role} is missing {key!r}.")
        data = bundle[key]
        if data.signal.ndim != 1:
            raise ValueError(f"SubtractInterpolated1D {role} {key!r} must be one-dimensional.")
        return data

    @staticmethod
    def _valid_points(
        signal: BaseData,
        axis: BaseData,
        mask: BaseData | None,
    ) -> np.ndarray:
        if signal.shape != axis.shape:
            raise ValueError("SubtractInterpolated1D signal and axis shapes must match within each bundle.")
        valid = np.isfinite(signal.signal) & np.isfinite(axis.signal)
        weights = np.broadcast_to(np.asarray(signal.weights, dtype=float), signal.shape)
        valid &= np.isfinite(weights) & (weights > 0.0)
        for uncertainty in signal.uncertainties.values():
            valid &= np.isfinite(np.broadcast_to(uncertainty, signal.shape))
        if mask is not None:
            if mask.shape != signal.shape:
                raise ValueError("SubtractInterpolated1D mask shape must match its signal.")
            valid &= np.asarray(mask.signal) == 0
        return valid

    def calculate(self) -> dict[str, DataBundle]:  # noqa: C901 - validation and propagation are kept explicit
        processing_keys = self._normalised_processing_keys()
        if len(processing_keys) != 2:
            raise ValueError("SubtractInterpolated1D requires two processing keys: sample then background.")
        sample_processing_key, background_processing_key = processing_keys
        sample = self.processing_data[sample_processing_key]
        background = self.processing_data[background_processing_key]

        signal_key = str(self.configuration.get("signal_key", "signal"))
        axis_key = str(self.configuration.get("axis_key", "Q"))
        output_signal_key = str(self.configuration.get("output_signal_key", "signal")).strip()
        if not output_signal_key:
            raise ValueError("SubtractInterpolated1D output_signal_key must not be empty.")
        mode = str(self.configuration.get("mode", "nearest")).strip().lower()
        mask_key = self._optional_key(self.configuration.get("mask_key"))

        sample_signal = self._one_dimensional(sample, signal_key, "sample")
        sample_axis = self._one_dimensional(sample, axis_key, "sample")
        background_signal = self._one_dimensional(background, signal_key, "background").copy(with_axes=False)
        background_axis = self._one_dimensional(background, axis_key, "background").copy(with_axes=False)
        background_signal.signal = np.asarray(background_signal.signal, dtype=float)
        background_axis.signal = np.asarray(background_axis.signal, dtype=float)
        background_signal.to_units(sample_signal.units)
        background_axis.to_units(sample_axis.units)

        sample_mask = sample.get(mask_key) if mask_key is not None else None
        background_mask = background.get(mask_key) if mask_key is not None else None
        sample_valid = self._valid_points(sample_signal, sample_axis, sample_mask)
        background_valid = self._valid_points(background_signal, background_axis, background_mask)
        if not np.any(background_valid):
            raise ValueError("SubtractInterpolated1D background has no valid points.")

        background_coordinates = np.asarray(background_axis.signal, dtype=float)[background_valid]
        order = np.argsort(background_coordinates, kind="stable")
        background_coordinates = background_coordinates[order]
        if background_coordinates.size > 1 and np.any(np.diff(background_coordinates) <= 0.0):
            raise ValueError("SubtractInterpolated1D valid background coordinates must be unique.")
        source_indices = np.flatnonzero(background_valid)[order]

        max_gap = self.configuration.get("max_gap")
        if max_gap is not None:
            max_gap_units = self.configuration.get("max_gap_units")
            gap_units = sample_axis.units if max_gap_units is None else ureg.Unit(max_gap_units)
            max_gap = (float(max_gap) * gap_units).to(sample_axis.units).magnitude

        left, right, left_weight, right_weight, remap_valid = remap_coefficients_1d(
            background_coordinates,
            np.asarray(sample_axis.signal, dtype=float),
            mode=mode,
            max_gap=max_gap,
        )
        valid = sample_valid & remap_valid
        left_source = source_indices[left]
        right_source = source_indices[right]

        remapped_signal = (
            left_weight * np.asarray(background_signal.signal)[left_source]
            + right_weight * np.asarray(background_signal.signal)[right_source]
        )
        remapped_uncertainties = {}
        for name, uncertainty in background_signal.uncertainties.items():
            uncertainty = np.broadcast_to(np.asarray(uncertainty, dtype=float), background_signal.shape)
            remapped_uncertainties[name] = np.sqrt(
                (left_weight * uncertainty[left_source]) ** 2 + (right_weight * uncertainty[right_source]) ** 2
            )

        remapped_signal = np.asarray(remapped_signal, dtype=float)
        remapped_signal[~valid] = np.nan
        for uncertainty in remapped_uncertainties.values():
            uncertainty[~valid] = np.nan
        remapped = BaseData(
            signal=remapped_signal,
            units=sample_signal.units,
            uncertainties=remapped_uncertainties,
            rank_of_data=1,
        )
        result = sample_signal - remapped
        result.weights = np.broadcast_to(np.asarray(sample_signal.weights, dtype=float), sample_signal.shape).copy()
        result.weights[~valid] = 0.0
        sample[output_signal_key] = result

        remapped_background_key = self._optional_key(self.configuration.get("remapped_background_key"))
        if remapped_background_key is not None:
            sample[remapped_background_key] = remapped
        output_mask_key = self._optional_key(self.configuration.get("output_mask_key", "remap_mask"))
        if output_mask_key is not None:
            sample[output_mask_key] = BaseData(
                signal=(~valid).astype(np.uint32),
                units=ureg.dimensionless,
                rank_of_data=1,
            )

        return {sample_processing_key: sample}
