# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "OpenAI ChatGPT"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "17/02/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["LakeDesmearing"]
__version__ = "20260217.3"

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class LakeDesmearing(ProcessStep):
    """Iterative Lake-style desmearing for 1D slit-smeared scattering data."""

    documentation = ProcessStepDescriber(
        calling_name="Lake desmearing (1D)",
        calling_id="LakeDesmearing1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={
            "desmeared_signal": ["signal", "uncertainties", "units"],
        },
        default_configuration={
            "signal_key": "signal",
            "axis_key": "Q",
            "resolution_top_halfwidth_key": None,
            "resolution_top_halfwidth_value": None,
            "resolution_halfwidth_key": None,
            "resolution_halfwidth_value": None,
            "secondary_resolution_top_halfwidth_key": None,
            "secondary_resolution_top_halfwidth_value": None,
            "secondary_resolution_halfwidth_key": None,
            "secondary_resolution_halfwidth_value": None,
            "regularization_lambda": 0.0,
            "clip_non_negative": True,
            "output_signal_key": "desmeared_signal",
            "store_debug_outputs": False,
        },
        argument_specs={
            "signal_key": {
                "type": str,
                "required": False,
                "doc": "BaseData key containing the smeared intensity to correct.",
            },
            "axis_key": {
                "type": str,
                "required": False,
                "doc": "BaseData key providing the independent axis (e.g. Q).",
            },
            "resolution_top_halfwidth_key": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Optional BaseData key with per-point top half-widths of the trapezoidal resolution function.",
            },
            "resolution_top_halfwidth_value": {
                "type": (float, int, str, Mapping, type(None)),
                "required": False,
                "doc": (
                    "Constant top half-width (in axis units) for the trapezoid plateau if no BaseData key is provided."
                    " Accepts numbers (interpreted in axis units) or pint-compatible strings (e.g. '0.005 1/nm')."
                ),
            },
            "resolution_halfwidth_key": {
                "type": (str, type(None)),
                "required": False,
                "doc": (
                    "Optional BaseData key with per-point bottom half-widths of the trapezoidal resolution function."
                ),
            },
            "resolution_halfwidth_value": {
                "type": (float, int, str, Mapping, type(None)),
                "required": False,
                "doc": (
                    "Constant bottom half-width for the resolution (in axis units) if no BaseData key is provided."
                    " Accepts numbers (interpreted in axis units) or pint-compatible strings (e.g. '0.01 1/nm')."
                ),
            },
            "secondary_resolution_top_halfwidth_key": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Optional BaseData key with per-point top half-widths for the orthogonal slit dimension.",
            },
            "secondary_resolution_top_halfwidth_value": {
                "type": (float, int, str, Mapping, type(None)),
                "required": False,
                "doc": (
                    "Constant top half-width for the orthogonal slit dimension (in axis units) if no BaseData key is"
                    " provided. Accepts numbers (interpreted in axis units) or pint-compatible strings (e.g. '0.005"
                    " 1/nm')."
                ),
            },
            "secondary_resolution_halfwidth_key": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Optional BaseData key with per-point bottom half-widths for the orthogonal slit dimension.",
            },
            "secondary_resolution_halfwidth_value": {
                "type": (float, int, str, Mapping, type(None)),
                "required": False,
                "doc": (
                    "Constant bottom half-width for the orthogonal slit dimension (in axis units) if no BaseData key is"
                    " provided. Accepts numbers (interpreted in axis units) or pint-compatible strings (e.g. '0.01"
                    " 1/nm')."
                ),
            },
            "regularization_lambda": {
                "type": (float, int),
                "required": False,
                "doc": "Tikhonov regularisation strength applied when inverting the smearing kernel.",
            },
            "clip_non_negative": {
                "type": bool,
                "required": False,
                "doc": "Whether to clip the desmeared signal at zero to avoid negative intensities.",
            },
            "output_signal_key": {
                "type": str,
                "required": False,
                "doc": "Name of the BaseData entry that will receive the desmeared signal.",
            },
            "store_debug_outputs": {
                "type": bool,
                "required": False,
                "doc": "Store kernel/operator in produced_outputs for inspection (may increase memory usage).",
            },
        },
        step_keywords=["Lake", "desmearing", "slit", "deconvolution", "1D"],
        step_doc=(
            "Implements an iterative Lake-style desmearing of a 1D scattering curve using trapezoidal slit-smearing"
            " kernels. The method constructs primary (and optionally orthogonal secondary) slit kernels from the"
            " provided axis and top/bottom half-widths, applies a Tikhonov-regularised pseudo-inverse to recover the"
            " best least-squares estimate of the desmeared signal, and propagates uncorrelated uncertainties."
        ),
        step_reference="Lake, J. Appl. Phys. 38, 3895 (1967)",
        step_note=(
            "Provide either 'resolution_halfwidth_key'/'resolution_halfwidth_value' (bottom half-widths) and optionally"
            " 'resolution_top_halfwidth_*' (top half-widths) for the primary slit dimension. Supply"
            " 'secondary_resolution_halfwidth_*' and 'secondary_resolution_top_halfwidth_*' for the orthogonal"
            " dimension if required. Setting top and bottom half-widths equal reproduces rectangular kernels."
        ),
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        processing_keys = self._normalised_processing_keys()

        output: dict[str, DataBundle] = {}
        debug_outputs: dict[str, Any] = {}

        for key in processing_keys:
            bundle = self.processing_data[key]

            signal_key = cfg.get("signal_key", "signal")
            axis_key = cfg.get("axis_key", "Q")
            output_key = cfg.get("output_signal_key", "desmeared_signal")

            if signal_key not in bundle:
                raise KeyError(f"LakeDesmearing: DataBundle '{key}' does not contain the signal key '{signal_key}'.")
            if axis_key not in bundle:
                raise KeyError(f"LakeDesmearing: DataBundle '{key}' does not contain the axis key '{axis_key}'.")

            signal_bd = bundle[signal_key]
            axis_bd = bundle[axis_key]

            axis_vals, signal_vals, original_shape = self._prepare_axis_and_signal(axis_bd, signal_bd)
            primary_top, primary_bottom = self._resolve_halfwidth_pair(
                bundle,
                axis_bd,
                len(axis_vals),
                top_key_cfg_name="resolution_top_halfwidth_key",
                top_value_cfg_name="resolution_top_halfwidth_value",
                bottom_key_cfg_name="resolution_halfwidth_key",
                bottom_value_cfg_name="resolution_halfwidth_value",
                required=True,
            )
            secondary_pair = self._resolve_halfwidth_pair(
                bundle,
                axis_bd,
                len(axis_vals),
                top_key_cfg_name="secondary_resolution_top_halfwidth_key",
                top_value_cfg_name="secondary_resolution_top_halfwidth_value",
                bottom_key_cfg_name="secondary_resolution_halfwidth_key",
                bottom_value_cfg_name="secondary_resolution_halfwidth_value",
                required=False,
            )
            if secondary_pair is None:
                secondary_top = None
                secondary_bottom = None
            else:
                secondary_top, secondary_bottom = secondary_pair

            kernel, primary_kernel, secondary_kernel = self._build_resolution_kernel(
                axis_vals,
                primary_top,
                primary_bottom,
                secondary_top,
                secondary_bottom,
            )

            reg_lambda = float(cfg.get("regularization_lambda", 0.0) or 0.0)
            desmear_operator = self._compute_desmearing_operator(kernel, reg_lambda)

            smeared_flat = signal_vals.reshape(-1)
            desmeared_flat = desmear_operator @ smeared_flat

            if cfg.get("clip_non_negative", True):
                desmeared_flat = np.maximum(desmeared_flat, 0.0)

            desmeared_signal = desmeared_flat.reshape(original_shape)
            uncertainties = self._propagate_uncertainties(signal_bd, desmear_operator, original_shape)

            weights_copy = np.array(signal_bd.weights, copy=True)
            axes_copy = list(signal_bd.axes) if signal_bd.axes else [axis_bd]

            bundle[output_key] = BaseData(
                signal=desmeared_signal,
                units=signal_bd.units,
                uncertainties=uncertainties,
                weights=weights_copy,
                axes=axes_copy,
                rank_of_data=signal_bd.rank_of_data,
            )

            output[key] = bundle

            if cfg.get("store_debug_outputs", False):
                debug_outputs[key] = {
                    "kernel": kernel,
                    "primary_kernel": primary_kernel,
                    "secondary_kernel": secondary_kernel,
                    "desmear_operator": desmear_operator,
                    "residual": smeared_flat - (kernel @ desmeared_flat),
                    "primary_top_halfwidths": primary_top.copy(),
                    "primary_bottom_halfwidths": primary_bottom.copy(),
                    "secondary_top_halfwidths": None if secondary_top is None else secondary_top.copy(),
                    "secondary_bottom_halfwidths": None if secondary_bottom is None else secondary_bottom.copy(),
                }

        if debug_outputs:
            self.produced_outputs.update(debug_outputs)

        return output

    # ------------------------------------------------------------------
    # Helper implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_axis_and_signal(
        axis_bd: BaseData, signal_bd: BaseData
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        signal_array = np.asarray(signal_bd.signal, dtype=float)
        if signal_array.ndim != 1:
            raise ValueError("LakeDesmearing only supports 1D signals (rank-1 BaseData).")

        axis_array = np.asarray(axis_bd.signal, dtype=float)
        if axis_array.ndim != 1:
            raise ValueError("LakeDesmearing requires a 1D axis BaseData.")

        if signal_array.shape != axis_array.shape:
            raise ValueError("Signal and axis must have the same shape for Lake desmearing.")

        if not np.all(np.diff(axis_array) > 0):
            raise ValueError("The axis values must be strictly increasing for Lake desmearing.")

        return axis_array.copy(), signal_array.copy(), signal_array.shape

    def _resolve_halfwidths(
        self,
        bundle: DataBundle,
        axis_bd: BaseData,
        size: int,
        *,
        key_cfg_name: str = "resolution_halfwidth_key",
        value_cfg_name: str = "resolution_halfwidth_value",
        required: bool = True,
    ) -> np.ndarray | None:
        cfg = self.configuration
        halfwidth_key = cfg.get(key_cfg_name)
        halfwidth_value = cfg.get(value_cfg_name)

        if halfwidth_key:
            if halfwidth_key not in bundle:
                raise KeyError(
                    f"LakeDesmearing: DataBundle missing resolution half-width key '{halfwidth_key}' (configured via"
                    f" '{key_cfg_name}')."
                )
            half_bd = bundle[halfwidth_key].copy(with_axes=False)
            if half_bd.units != axis_bd.units:
                half_bd.to_units(axis_bd.units)
            halfwidths = np.asarray(half_bd.signal, dtype=float).reshape(-1)
        elif halfwidth_value is not None:
            halfwidths = self._halfwidth_from_value(halfwidth_value, axis_bd.units, size)
        elif required:
            raise ValueError(
                f"LakeDesmearing requires either '{key_cfg_name}' or '{value_cfg_name}' in the configuration."
            )
        else:
            return None

        if halfwidths.size == 1:
            halfwidths = np.full(size, float(halfwidths[0]), dtype=float)
        elif halfwidths.size != size:
            raise ValueError(
                "Resolution half-width array must match the length of the axis."
                " Received size"
                f" {halfwidths.size} but expected {size}."
            )

        if np.any(halfwidths < 0.0):
            raise ValueError("Resolution half-widths must be non-negative.")

        return halfwidths.astype(float, copy=False)

    def _resolve_halfwidth_pair(
        self,
        bundle: DataBundle,
        axis_bd: BaseData,
        size: int,
        *,
        top_key_cfg_name: str,
        top_value_cfg_name: str,
        bottom_key_cfg_name: str,
        bottom_value_cfg_name: str,
        required: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        bottom = self._resolve_halfwidths(
            bundle,
            axis_bd,
            size,
            key_cfg_name=bottom_key_cfg_name,
            value_cfg_name=bottom_value_cfg_name,
            required=required,
        )
        if bottom is None:
            return None

        top = self._resolve_halfwidths(
            bundle,
            axis_bd,
            size,
            key_cfg_name=top_key_cfg_name,
            value_cfg_name=top_value_cfg_name,
            required=False,
        )
        if top is None:
            top = bottom.copy()
        else:
            if top.size == 1:
                top = np.full(size, float(top[0]), dtype=float)
            elif top.size != size:
                raise ValueError(
                    "Resolution top half-width array must match the length of the axis."
                    f" Received size {top.size} but expected {size}."
                )
            if np.any(top < 0.0):
                raise ValueError("Resolution top half-widths must be non-negative.")

        if np.any(top - bottom > 1e-12):
            raise ValueError("Top half-widths must not exceed bottom half-widths.")

        return top.astype(float, copy=False), bottom.astype(float, copy=False)

    @staticmethod
    def _halfwidth_from_value(value: Any, axis_units, size: int) -> np.ndarray:
        if isinstance(value, (int, float)):
            quantity = ureg.Quantity(float(value), axis_units)
        elif isinstance(value, str):
            quantity = ureg.Quantity(value).to(axis_units)
        elif isinstance(value, Mapping):
            if "value" not in value:
                raise ValueError("Mapping for resolution_halfwidth_value must contain a 'value' key.")
            units = value.get("units", axis_units)
            quantity = ureg.Quantity(value["value"], units).to(axis_units)
        else:
            raise TypeError(
                "resolution_halfwidth_value must be a float, int, str, or mapping with 'value' (and optional 'units')."
            )

        magnitude = float(quantity.to(axis_units).magnitude)
        if magnitude < 0.0:
            raise ValueError("resolution_halfwidth_value must be non-negative.")
        return np.full(size, magnitude, dtype=float)

    @staticmethod
    def _bin_edges(axis: np.ndarray) -> np.ndarray:
        n = axis.size
        if n == 0:
            raise ValueError("Axis must contain at least one point.")
        if n == 1:
            width = 1.0
            return np.array([axis[0] - width * 0.5, axis[0] + width * 0.5], dtype=float)

        diffs = np.diff(axis)
        edges = np.empty(n + 1, dtype=float)
        edges[1:-1] = axis[:-1] + 0.5 * diffs
        edges[0] = axis[0] - 0.5 * diffs[0]
        edges[-1] = axis[-1] + 0.5 * diffs[-1]
        return edges

    @staticmethod
    def _build_trapezoid_kernel(
        axis: np.ndarray,
        top_halfwidths: np.ndarray,
        bottom_halfwidths: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        n = axis.size
        kernel = np.zeros((n, n), dtype=float)
        eps = 1e-12

        for idx in range(n):
            hb = float(bottom_halfwidths[idx])
            ht = float(top_halfwidths[idx])
            if hb <= eps:
                kernel[idx, idx] = 1.0
                continue

            ht = min(max(ht, 0.0), hb)
            area = ht + hb
            if area <= eps:
                kernel[idx, idx] = 1.0
                continue

            centre = axis[idx]
            row = np.zeros(n, dtype=float)
            for col in range(n):
                r_lo = edges[col] - centre
                r_hi = edges[col + 1] - centre
                overlap = LakeDesmearing._trapezoid_overlap_integral(r_lo, r_hi, ht, hb)
                if overlap > 0.0:
                    row[col] = overlap / area

            row_sum = row.sum()
            if row_sum <= eps:
                kernel[idx, idx] = 1.0
            else:
                kernel[idx, :] = row / row_sum

        return kernel

    @classmethod
    def _build_resolution_kernel(
        cls,
        axis: np.ndarray,
        primary_top: np.ndarray,
        primary_bottom: np.ndarray,
        secondary_top: np.ndarray | None = None,
        secondary_bottom: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        edges = cls._bin_edges(axis)
        primary_kernel = cls._build_trapezoid_kernel(axis, primary_top, primary_bottom, edges)
        if secondary_top is None or secondary_bottom is None:
            return primary_kernel, primary_kernel, None

        secondary_kernel = cls._build_trapezoid_kernel(axis, secondary_top, secondary_bottom, edges)
        combined_kernel = secondary_kernel @ primary_kernel
        return combined_kernel, primary_kernel, secondary_kernel

    @staticmethod
    def _trapezoid_overlap_integral(r_lo: float, r_hi: float, top_hw: float, bottom_hw: float) -> float:
        if r_hi < r_lo:
            r_lo, r_hi = r_hi, r_lo
        if bottom_hw <= 0.0 or r_lo == r_hi:
            return 0.0

        support_lo = -bottom_hw
        support_hi = bottom_hw
        if r_hi <= support_lo or r_lo >= support_hi:
            return 0.0

        lo = max(r_lo, support_lo)
        hi = min(r_hi, support_hi)
        if lo >= hi:
            return 0.0

        if lo < 0.0 < hi:
            return LakeDesmearing._integrate_positive_interval(
                0.0, hi, top_hw, bottom_hw
            ) + LakeDesmearing._integrate_positive_interval(0.0, -lo, top_hw, bottom_hw)
        if hi <= 0.0:
            return LakeDesmearing._integrate_positive_interval(-hi, -lo, top_hw, bottom_hw)
        return LakeDesmearing._integrate_positive_interval(lo, hi, top_hw, bottom_hw)

    @staticmethod
    def _integrate_positive_interval(a: float, b: float, top_hw: float, bottom_hw: float) -> float:
        eps = 1e-12
        if b <= a or bottom_hw <= eps:
            return 0.0

        a = max(a, 0.0)
        b = min(b, bottom_hw)
        if b <= a:
            return 0.0

        top_hw = min(max(top_hw, 0.0), bottom_hw)
        if b <= top_hw or bottom_hw <= top_hw + eps:
            return b - a

        result = 0.0
        if a < top_hw:
            result += top_hw - a
            a = top_hw

        if bottom_hw <= top_hw + eps:
            return result

        factor = bottom_hw - top_hw
        result += (bottom_hw * (b - a) - 0.5 * (b**2 - a**2)) / factor
        return result

    @staticmethod
    def _compute_desmearing_operator(kernel: np.ndarray, reg_lambda: float) -> np.ndarray:
        normal_matrix = kernel.T @ kernel
        n = normal_matrix.shape[0]
        if reg_lambda > 0.0:
            normal_matrix = normal_matrix + (reg_lambda * np.eye(n, dtype=float))

        try:
            pseudo_inverse = np.linalg.solve(normal_matrix, kernel.T)
        except np.linalg.LinAlgError:
            pseudo_inverse = np.linalg.pinv(normal_matrix) @ kernel.T
        return pseudo_inverse

    @staticmethod
    def _propagate_uncertainties(
        signal_bd: BaseData, desmear_operator: np.ndarray, reshape_to: tuple[int, ...]
    ) -> dict[str, np.ndarray]:
        if not signal_bd.uncertainties:
            return {}

        combined_variance = None
        signal_shape = signal_bd.signal.shape
        for key, values in signal_bd.uncertainties.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 1:
                arr = np.full(signal_shape, float(arr), dtype=float)
            else:
                arr = np.broadcast_to(arr, signal_shape)
            var = arr.reshape(-1) ** 2
            combined_variance = var if combined_variance is None else combined_variance + var

        if combined_variance is None:
            return {}

        sigma_squared = combined_variance
        propagated_variance = (desmear_operator**2) @ sigma_squared
        propagated_sigma = np.sqrt(np.maximum(propagated_variance, 0.0)).reshape(reshape_to)

        return {"desmeared": propagated_sigma}

    # ------------------------------------------------------------------
    # Utility helpers for external use (tests, diagnostics)
    # ------------------------------------------------------------------

    @staticmethod
    def smear_signal(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply the smearing kernel to a 1D signal (useful for testing)."""
        return kernel @ signal.reshape(-1)
