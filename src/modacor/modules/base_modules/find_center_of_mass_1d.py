# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["FindCenterOfMass1D"]
__version__ = "20260929.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.centroid import CentroidResult, intensity_centroid_1d


class FindCenterOfMass1D(ProcessStep):
    """Determine an iterative intensity centroid around a one-dimensional peak."""

    documentation = ProcessStepDescriber(
        calling_name="Find one-dimensional intensity centre of mass",
        calling_id="FindCenterOfMass1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={
            "beam_center": ["signal", "uncertainties", "units"],
            "centroid_count": ["signal"],
            "centroid_converged": ["signal"],
            "centroid_window_min": ["signal", "units"],
            "centroid_window_max": ["signal", "units"],
        },
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "DataBundle key or keys containing peaks to centre.",
            },
            "signal_key": {
                "type": str,
                "default": "signal",
                "doc": "Intensity BaseData key.",
            },
            "axis_key": {
                "type": str,
                "default": "yaw",
                "doc": "One-dimensional coordinate BaseData key.",
            },
            "mask_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional mask key; nonzero values are excluded.",
            },
            "half_width": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Optional physical half-width around the iterated centre.",
            },
            "width_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for half_width and convergence_tolerance; defaults to axis units.",
            },
            "baseline": {
                "type": (float, int),
                "default": 0.0,
                "doc": "Scalar baseline in baseline_units, subtracted before clipping negative weights.",
            },
            "baseline_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for baseline; defaults to signal units.",
            },
            "maximum_iterations": {
                "type": int,
                "default": 5,
                "doc": "Maximum number of window-recentring iterations.",
            },
            "convergence_tolerance": {
                "type": (float, int),
                "default": 0.0,
                "doc": "Absolute centre-change tolerance in width_units.",
            },
            "output_key": {
                "type": str,
                "default": "beam_center",
                "doc": "Scalar centroid output key.",
            },
            "diagnostic_prefix": {
                "type": str,
                "default": "centroid",
                "doc": "Prefix for count, convergence, and window-bound diagnostic keys.",
            },
        },
        step_keywords=["centroid", "center of mass", "peak", "beam centre", "1D"],
        step_doc="Find a masked, windowed intensity centroid without assuming a peak shape.",
        step_note=(
            "The coarse maximum selects a contiguous valid peak region. After optional baseline subtraction, "
            "negative intensities are clipped to zero. Signal and coordinate uncertainty components are "
            "propagated separately to the scalar centre."
        ),
    )

    def _output_keys(self) -> tuple[str, str, str, str, str]:
        output_key = str(self.configuration.get("output_key", "beam_center")).strip()
        prefix = str(self.configuration.get("diagnostic_prefix", "centroid")).strip()
        if not output_key or not prefix:
            raise ValueError("FindCenterOfMass1D output_key and diagnostic_prefix must not be empty.")
        return output_key, f"{prefix}_count", f"{prefix}_converged", f"{prefix}_window_min", f"{prefix}_window_max"

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if not processing_keys:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})
        signal_key = str(self.configuration.get("signal_key", "signal"))
        axis_key = str(self.configuration.get("axis_key", "yaw"))
        mask_key = self.configuration.get("mask_key")
        read_keys = {signal_key, axis_key}
        if mask_key is not None and str(mask_key).strip():
            read_keys.add(str(mask_key).strip())
        output_keys = self._output_keys()
        return ProcessStepDependencies(
            processing_reads={f"{processing_key}.{key}" for processing_key in processing_keys for key in read_keys},
            processing_writes={f"{processing_key}.{key}" for processing_key in processing_keys for key in output_keys},
        )

    @staticmethod
    def _contiguous_component(mask: np.ndarray, anchor: int) -> np.ndarray:
        component = np.zeros(mask.shape, dtype=bool)
        if not mask[anchor]:
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                return component
            anchor = int(candidates[np.argmin(np.abs(candidates - anchor))])
        start = anchor
        stop = anchor + 1
        while start > 0 and mask[start - 1]:
            start -= 1
        while stop < mask.size and mask[stop]:
            stop += 1
        component[start:stop] = True
        return component

    @staticmethod
    def _propagated_uncertainties(
        result: CentroidResult,
        signal: BaseData,
        axis: BaseData,
    ) -> dict[str, np.ndarray]:
        propagated: dict[str, np.ndarray] = {}
        for name, uncertainty in signal.uncertainties.items():
            sigma = np.broadcast_to(np.asarray(uncertainty, dtype=float), signal.shape)
            propagated[f"signal:{name}"] = np.asarray(np.sqrt(np.sum((result.signal_sensitivity * sigma) ** 2)))
        for name, uncertainty in axis.uncertainties.items():
            sigma = np.broadcast_to(np.asarray(uncertainty, dtype=float), axis.shape)
            propagated[f"axis:{name}"] = np.asarray(np.sqrt(np.sum((result.axis_sensitivity * sigma) ** 2)))
        return propagated

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        signal_key = str(cfg.get("signal_key", "signal"))
        axis_key = str(cfg.get("axis_key", "yaw"))
        mask_key = cfg.get("mask_key")
        mask_key = str(mask_key).strip() if mask_key is not None else None
        output_key, count_key, converged_key, window_min_key, window_max_key = self._output_keys()
        maximum_iterations = int(cfg.get("maximum_iterations", 5))
        if maximum_iterations < 1:
            raise ValueError("FindCenterOfMass1D maximum_iterations must be positive.")

        output: dict[str, DataBundle] = {}
        for processing_key in self._normalised_processing_keys():
            bundle = self.processing_data[processing_key]
            signal = bundle[signal_key]
            axis = bundle[axis_key]
            if signal.signal.ndim != 1 or axis.shape != signal.shape:
                raise ValueError("FindCenterOfMass1D requires matching one-dimensional signal and axis arrays.")

            valid = np.isfinite(signal.signal) & np.isfinite(axis.signal)
            signal_weights = np.broadcast_to(np.asarray(signal.weights, dtype=float), signal.shape)
            valid &= np.isfinite(signal_weights) & (signal_weights > 0.0)
            if mask_key is not None and mask_key in bundle:
                if bundle[mask_key].shape != signal.shape:
                    raise ValueError("FindCenterOfMass1D mask shape must match signal shape.")
                valid &= np.asarray(bundle[mask_key].signal) == 0
            if not np.any(valid):
                raise ValueError(f"FindCenterOfMass1D {processing_key!r} has no valid points.")

            width_units = ureg.Unit(cfg.get("width_units") or axis.units)
            half_width_cfg = cfg.get("half_width")
            half_width = None
            if half_width_cfg is not None:
                half_width = (float(half_width_cfg) * width_units).to(axis.units).magnitude
                if not np.isfinite(half_width) or half_width <= 0.0:
                    raise ValueError("FindCenterOfMass1D half_width must be positive and finite.")
            tolerance = (float(cfg.get("convergence_tolerance", 0.0)) * width_units).to(axis.units).magnitude
            if tolerance < 0.0 or not np.isfinite(tolerance):
                raise ValueError("FindCenterOfMass1D convergence_tolerance must be finite and non-negative.")
            baseline_units = ureg.Unit(cfg.get("baseline_units") or signal.units)
            baseline = (float(cfg.get("baseline", 0.0)) * baseline_units).to(signal.units).magnitude

            valid_indices = np.flatnonzero(valid)
            anchor = int(valid_indices[np.argmax(np.asarray(signal.signal)[valid])])
            center = float(np.asarray(axis.signal)[anchor])
            converged = False
            result: CentroidResult | None = None
            for _ in range(maximum_iterations):
                in_window = valid.copy()
                if half_width is not None:
                    in_window &= np.abs(np.asarray(axis.signal, dtype=float) - center) <= half_width
                anchor = int(np.argmin(np.abs(np.asarray(axis.signal, dtype=float) - center)))
                component = self._contiguous_component(in_window, anchor)
                result = intensity_centroid_1d(
                    axis.signal,
                    signal.signal,
                    valid=component,
                    baseline=baseline,
                )
                change = abs(result.center - center)
                center = result.center
                if change <= tolerance:
                    converged = True
                    break
            if result is None:
                raise RuntimeError("FindCenterOfMass1D internal error: no centroid iteration was performed.")

            contributors = np.flatnonzero(result.contributors)
            bundle[output_key] = BaseData(
                signal=np.asarray(result.center),
                units=axis.units,
                uncertainties=self._propagated_uncertainties(result, signal, axis),
                rank_of_data=0,
            )
            bundle[count_key] = BaseData(np.asarray(result.contributor_count), ureg.dimensionless)
            bundle[converged_key] = BaseData(np.asarray(converged), ureg.dimensionless)
            bundle[window_min_key] = BaseData(np.asarray(np.min(np.asarray(axis.signal)[contributors])), axis.units)
            bundle[window_max_key] = BaseData(np.asarray(np.max(np.asarray(axis.signal)[contributors])), axis.units)
            output[processing_key] = bundle
        return output
