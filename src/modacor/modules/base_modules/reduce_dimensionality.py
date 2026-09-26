# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "24/09/2026"
__status__ = "Development"  # "Development", "Production"

__all__ = ["ReduceDimensionality"]
__version__ = "20260924.2"

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, processing_key_patterns
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.helpers import finalize_weighted_scatter, leading_non_data_axes

# Facility-pluggable logger; by default this uses std logging
logger = MessageHandler(name=__name__)

_COLLISION_POLICIES = frozenset({"error", "overwrite_existing", "keep_existing", "propagate"})
_ESTIMATOR_REDUCTIONS = {
    "standard_deviation": frozenset({"mean", "sum"}),
    "standard_error_mean": frozenset({"mean"}),
    "standard_error_sum": frozenset({"sum"}),
}


@dataclass(frozen=True)
class _EstimatorSpec:
    output_key: str
    method: str
    ddof: int
    collision_policy: str


class ReduceDimensionality(ProcessStep):
    """
    Compute a (possibly weighted) mean or sum over one or more axes, propagate
    existing uncertainties, and optionally estimate uncertainties from scatter.

    For each uncertainty key `k`, assumes uncorrelated errors:

        μ = Σ w_i x_i / Σ w_i
        σ_μ^2 = Σ (w_i^2 σ_i^2) / (Σ w_i)^2

    NaN handling:
        - If nan_policy == 'omit', NaNs in `signal` (and their σ) are ignored.
        - If nan_policy == 'propagate', NaNs behave like in plain numpy: if any NaN is
          present along the reduced axes, the result becomes NaN.
    """

    documentation = ProcessStepDescriber(
        calling_name="average or sum, weighted or unweighted, over axes",
        calling_id="ReduceDimensionality",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["signal", "uncertainties", "units", "weights"]},
        arguments={
            "axes": {
                "type": (int, list, tuple, str, type(None)),
                "default": None,
                "doc": (
                    "Axis or axes to reduce (int, list/tuple, or None for all). "
                    "Use 'non_data' to reduce every leading axis before the final rank_of_data axes."
                ),
            },
            "use_weights": {
                "type": bool,
                "default": True,
                "doc": "Use BaseData weights for weighted reduction.",
            },
            "nan_policy": {
                "type": str,
                "default": "omit",
                "doc": "NaN handling policy: 'omit' or 'propagate'.",
            },
            "reduction": {
                "type": str,
                "default": "mean",
                "doc": "Reduction method: 'mean' or 'sum'.",
            },
            "mask_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": (
                    "Optional BaseData mask key in each selected DataBundle. Nonzero mask values are excluded "
                    "without modifying the input signal."
                ),
            },
            "mask_bits": {
                "type": (int, list, tuple, type(None)),
                "default": None,
                "doc": (
                    "Optional uint32 bit value or iterable of bit values to exclude. None excludes any nonzero mask."
                ),
            },
            "uncertainty_estimation": {
                "type": (dict, type(None)),
                "default": None,
                "doc": (
                    "Optional mapping containing estimator definitions and a collision policy for "
                    "scatter-derived uncertainty components."
                ),
            },
        },
        step_keywords=["average", "mean", "weighted", "nanmean", "reduce", "axis", "sum"],
        step_doc=(
            "Compute a weighted or unweighted mean/sum over the given axes, propagate existing "
            "uncertainties, and optionally add scatter-derived uncertainty estimates."
        ),
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "This step reduces the dimensionality of the signal by averaging over one or more axes. "
            "With axes='non_data', it automatically reduces leading acquisition axes until signal.ndim "
            "equals rank_of_data. "
            "An optional integer mask can exclude selected reason bits without modifying the input signal. "
            "Units are preserved; complete axes metadata is reduced along the same axes."
        ),
    )

    # ---------------------------- helpers ---------------------------------

    @staticmethod
    def _normalize_mask_configuration(mask_key: Any, mask_bits: Any) -> tuple[str | None, int | None]:
        """Validate mask configuration and combine selected reason bits."""

        if mask_key is None:
            if mask_bits is not None:
                raise ValueError("ReduceDimensionality mask_bits requires mask_key to be configured.")
            return None, None
        if not isinstance(mask_key, str) or not mask_key.strip():
            raise ValueError("ReduceDimensionality mask_key must be a non-empty string or None.")

        if mask_bits is None:
            return mask_key, None
        if isinstance(mask_bits, (int, np.integer)) and not isinstance(mask_bits, bool):
            bit_values = (int(mask_bits),)
        elif isinstance(mask_bits, (list, tuple)):
            bit_values = tuple(mask_bits)
            if not bit_values:
                raise ValueError("ReduceDimensionality mask_bits must not be empty.")
        else:
            raise TypeError("ReduceDimensionality mask_bits must be an integer, a list/tuple, or None.")

        combined = 0
        uint32_max = int(np.iinfo(np.uint32).max)
        for bit_value in bit_values:
            if isinstance(bit_value, bool) or not isinstance(bit_value, (int, np.integer)):
                raise TypeError("ReduceDimensionality mask_bits values must be integers.")
            bit_value = int(bit_value)
            if not 0 < bit_value <= uint32_max:
                raise ValueError(f"ReduceDimensionality mask_bits values must be between 1 and {uint32_max}.")
            combined |= bit_value
        return mask_key, combined

    @staticmethod
    def _mask_exclusion(
        *,
        databundle: DataBundle,
        signal_shape: tuple[int, ...],
        mask_key: str | None,
        mask_bits: int | None,
    ) -> np.ndarray | None:
        """Return a signal-shaped boolean exclusion mask without mutating the source mask."""

        if mask_key is None:
            return None
        if mask_key not in databundle:
            raise KeyError(f"ReduceDimensionality mask_key {mask_key!r} is not present in the DataBundle.")

        mask = np.asarray(databundle[mask_key].signal)
        if not np.issubdtype(mask.dtype, np.integer):
            raise TypeError(f"ReduceDimensionality mask {mask_key!r} must have an integer dtype, got {mask.dtype}.")
        try:
            mask = np.broadcast_to(mask, signal_shape)
        except ValueError as exc:
            raise ValueError(
                f"ReduceDimensionality mask {mask_key!r} with shape {mask.shape} cannot broadcast "
                f"to signal shape {signal_shape}."
            ) from exc

        mask_u32 = mask.astype(np.uint32, copy=False)
        if mask_bits is None:
            return mask_u32 != 0
        return np.bitwise_and(mask_u32, np.uint32(mask_bits)) != 0

    @staticmethod
    def _effective_values_and_weights(
        *,
        bd: BaseData,
        use_weights: bool,
        nan_policy: str,
        exclusion_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply explicit-mask and NaN selection to values and effective weights."""

        x = np.asarray(bd.signal, dtype=float)
        if use_weights:
            weights = np.broadcast_to(np.asarray(bd.weights, dtype=float), x.shape)
        else:
            weights = np.broadcast_to(np.array(1.0), x.shape)

        if exclusion_mask is None:
            effective_mask = np.zeros(x.shape, dtype=bool)
        else:
            effective_mask = np.asarray(exclusion_mask, dtype=bool)

        if nan_policy == "omit":
            effective_mask = effective_mask | np.isnan(x) | np.isnan(weights)
        elif nan_policy != "propagate":
            raise ValueError(f"Invalid nan_policy: {nan_policy!r}. Use 'omit' or 'propagate'.")

        x_eff = np.where(effective_mask, 0.0, x)
        w_eff = np.where(effective_mask, 0.0, weights)
        return x_eff, w_eff, effective_mask

    @staticmethod
    def _normalize_uncertainty_estimation(
        configuration: Mapping[str, Any] | None,
        *,
        reduction: str,
    ) -> tuple[_EstimatorSpec, ...]:
        """Validate and normalize the nested uncertainty-estimation contract."""

        if configuration is None:
            return ()
        if not isinstance(configuration, Mapping):
            raise TypeError("ReduceDimensionality uncertainty_estimation must be a mapping or None.")

        unknown_keys = set(configuration) - {"collision_policy", "estimators"}
        if unknown_keys:
            raise ValueError(
                "ReduceDimensionality uncertainty_estimation contains unknown key(s): "
                f"{', '.join(sorted(map(str, unknown_keys)))}."
            )

        default_policy = configuration.get("collision_policy", "error")
        if not isinstance(default_policy, str) or default_policy not in _COLLISION_POLICIES:
            allowed = ", ".join(sorted(_COLLISION_POLICIES))
            raise ValueError(
                f"Invalid uncertainty-estimation collision_policy {default_policy!r}. Use one of: {allowed}."
            )

        estimators = configuration.get("estimators", {})
        if estimators is None:
            estimators = {}
        if not isinstance(estimators, Mapping):
            raise TypeError("ReduceDimensionality uncertainty_estimation.estimators must be a mapping.")

        normalized: list[_EstimatorSpec] = []
        for output_key, raw_spec in estimators.items():
            if not isinstance(output_key, str) or not output_key.strip():
                raise ValueError("Uncertainty-estimator destination keys must be non-empty strings.")
            if not isinstance(raw_spec, Mapping):
                raise TypeError(f"Uncertainty estimator {output_key!r} must be configured by a mapping.")

            unknown_spec_keys = set(raw_spec) - {"method", "ddof", "collision_policy"}
            if unknown_spec_keys:
                raise ValueError(
                    f"Uncertainty estimator {output_key!r} contains unknown key(s): "
                    f"{', '.join(sorted(map(str, unknown_spec_keys)))}."
                )

            method = raw_spec.get("method")
            if not isinstance(method, str) or method not in _ESTIMATOR_REDUCTIONS:
                allowed = ", ".join(sorted(_ESTIMATOR_REDUCTIONS))
                raise ValueError(
                    f"Unknown uncertainty estimator method {method!r} for {output_key!r}. Use one of: {allowed}."
                )
            if reduction not in _ESTIMATOR_REDUCTIONS[method]:
                raise ValueError(f"Uncertainty estimator method {method!r} is not valid for reduction={reduction!r}.")

            ddof = raw_spec.get("ddof", 1)
            if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
                raise ValueError(f"Uncertainty estimator {output_key!r} ddof must be a non-negative integer.")

            collision_policy = raw_spec.get("collision_policy", default_policy)
            if not isinstance(collision_policy, str) or collision_policy not in _COLLISION_POLICIES:
                allowed = ", ".join(sorted(_COLLISION_POLICIES))
                raise ValueError(
                    f"Invalid collision_policy {collision_policy!r} for uncertainty estimator "
                    f"{output_key!r}. Use one of: {allowed}."
                )

            normalized.append(
                _EstimatorSpec(
                    output_key=output_key,
                    method=method,
                    ddof=int(ddof),
                    collision_policy=collision_policy,
                )
            )

        return tuple(normalized)

    @staticmethod
    def _estimate_uncertainties_from_scatter(
        *,
        bd: BaseData,
        axis: int | tuple[int, ...] | None,
        use_weights: bool,
        nan_policy: str,
        exclusion_mask: np.ndarray | None,
        estimator_specs: tuple[_EstimatorSpec, ...],
        uncertainties_out: dict[str, np.ndarray],
    ) -> None:
        """Calculate requested estimates and merge them into ``uncertainties_out``."""

        active_specs: list[_EstimatorSpec] = []
        for spec in estimator_specs:
            if spec.output_key not in uncertainties_out:
                active_specs.append(spec)
                continue
            if spec.collision_policy == "error":
                raise ValueError(
                    f"Uncertainty estimator output key {spec.output_key!r} already exists after propagation. "
                    "Choose overwrite_existing, keep_existing, or propagate to resolve the collision."
                )
            if spec.collision_policy == "keep_existing":
                logger.warning(
                    f"ReduceDimensionality: keeping existing propagated uncertainty {spec.output_key!r}; "
                    "discarding the configured estimator result."
                )
                continue
            active_specs.append(spec)

        if not active_specs:
            return

        x_eff, w_eff, _ = ReduceDimensionality._effective_values_and_weights(
            bd=bd,
            use_weights=use_weights,
            nan_policy=nan_policy,
            exclusion_mask=exclusion_mask,
        )

        if np.any(np.isfinite(w_eff) & (w_eff < 0.0)):
            raise ValueError("Scatter-derived uncertainty estimators require non-negative effective weights.")

        sum_w = np.sum(w_eff, axis=axis)
        sum_w2 = np.sum(w_eff**2, axis=axis)
        sum_w_keepdims = np.sum(w_eff, axis=axis, keepdims=True)
        sum_wx_keepdims = np.sum(w_eff * x_eff, axis=axis, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_keepdims = sum_wx_keepdims / sum_w_keepdims
        deviations = x_eff - mean_keepdims
        weighted_squared_deviations = np.where(w_eff == 0.0, 0.0, w_eff * deviations**2)
        sum_w_squared_deviations = np.sum(weighted_squared_deviations, axis=axis)

        estimates_by_ddof = {}
        for spec in active_specs:
            estimates = estimates_by_ddof.get(spec.ddof)
            if estimates is None:
                estimates = finalize_weighted_scatter(
                    sum_w=sum_w,
                    sum_w2=sum_w2,
                    sum_w_squared_deviations=sum_w_squared_deviations,
                    ddof=spec.ddof,
                )
                estimates_by_ddof[spec.ddof] = estimates

            estimate = np.asarray(getattr(estimates, spec.method))
            existing = uncertainties_out.get(spec.output_key)
            if existing is None:
                uncertainties_out[spec.output_key] = estimate
            elif spec.collision_policy == "overwrite_existing":
                logger.warning(
                    f"ReduceDimensionality: overwriting existing propagated uncertainty {spec.output_key!r}."
                )
                uncertainties_out[spec.output_key] = estimate
            else:  # propagate; error and keep_existing were handled above
                logger.warning(
                    f"ReduceDimensionality: combining existing and estimated uncertainty "
                    f"{spec.output_key!r} in quadrature."
                )
                uncertainties_out[spec.output_key] = np.hypot(np.asarray(existing, dtype=float), estimate)

    @staticmethod
    def _normalize_axes(axes: Any) -> int | tuple[int, ...] | None:
        """
        Normalize configuration 'axes' into a numpy-compatible axis argument.
        Allowed:
          - None           → reduce over all axes
          - int            → single axis
          - list/tuple[int] → tuple of axes
        """
        if axes is None:
            logger.debug("ReduceDimensionality: axes=None → reducing over all axes.")
            return None
        if isinstance(axes, int):
            logger.debug(f"ReduceDimensionality: single axis requested: axes={axes}.")
            return axes
        # list/tuple of ints
        normalized = tuple(int(a) for a in axes)
        logger.debug(f"ReduceDimensionality: multiple axes requested: axes={normalized}.")
        return normalized

    @staticmethod
    def _resolve_axes(axes: Any, bd: BaseData) -> int | tuple[int, ...] | None:
        """Resolve explicit axes or the leading ``non_data`` axis mode."""

        if isinstance(axes, str):
            mode = axes.strip().lower()
            if mode != "non_data":
                raise ValueError("ReduceDimensionality axes string must be 'non_data'.")
            resolved = leading_non_data_axes(bd.signal.ndim, bd.rank_of_data)
            logger.debug(
                f"ReduceDimensionality: axes='non_data' resolved to {resolved} for "
                f"ndim={bd.signal.ndim} and rank_of_data={bd.rank_of_data}."
            )
            return resolved
        return ReduceDimensionality._normalize_axes(axes)

    @staticmethod
    def _axis_count(shape: tuple[int, ...], axis: int | tuple[int, ...] | None) -> int:
        if axis is None:
            return int(np.prod(shape))
        if isinstance(axis, tuple):
            return int(np.prod([shape[a if a >= 0 else len(shape) + a] for a in axis]))
        return int(shape[axis if axis >= 0 else len(shape) + axis])

    @staticmethod
    def _scalar_weight(weights: np.ndarray) -> float | None:
        weights = np.asarray(weights, dtype=float)
        if weights.size != 1:
            return None
        return float(weights.reshape(-1)[0])

    @staticmethod
    def _weighted_mean_with_uncertainty(
        bd: BaseData,
        axis: int | tuple[int, ...] | None,
        use_weights: bool,
        nan_policy: str,
        reduction: str = "mean",
        estimator_specs: tuple[_EstimatorSpec, ...] = (),
        exclusion_mask: np.ndarray | None = None,
    ) -> BaseData:
        """
        Compute weighted reduction ('mean' or 'sum') of a BaseData over axis,
        with uncertainty propagation.

        reduction:
            'mean' → μ = Σ w x / Σ w
            'sum'  → S = Σ w x
        """
        x = bd.signal
        scalar_weight = ReduceDimensionality._scalar_weight(bd.weights) if use_weights else 1.0

        if scalar_weight is not None and exclusion_mask is None:
            result = ReduceDimensionality._scalar_weight_reduction(
                bd=bd,
                axis=axis,
                nan_policy=nan_policy,
                reduction=reduction,
                scalar_weight=scalar_weight,
            )
            if estimator_specs:
                ReduceDimensionality._estimate_uncertainties_from_scatter(
                    bd=bd,
                    axis=axis,
                    use_weights=use_weights,
                    nan_policy=nan_policy,
                    exclusion_mask=None,
                    estimator_specs=estimator_specs,
                    uncertainties_out=result.uncertainties,
                )
            return result

        x_eff, w_eff, effective_mask = ReduceDimensionality._effective_values_and_weights(
            bd=bd,
            use_weights=use_weights,
            nan_policy=nan_policy,
            exclusion_mask=exclusion_mask,
        )

        # Weighted sums
        w_sum = np.sum(w_eff, axis=axis)
        wx_sum = np.sum(w_eff * x_eff, axis=axis)

        # Σ w_i^2 σ_i^2 for each key
        uncertainties_out: dict[str, np.ndarray] = {}

        # Precompute denom for mean case
        if reduction == "mean":
            denom = np.where(w_sum == 0, np.nan, w_sum)
            signal_out = wx_sum / denom
        elif reduction == "sum":
            # For sum, just take Σ w x (or Σ x when use_weights=False)
            signal_out = wx_sum
        else:
            raise ValueError(f"Invalid reduction: {reduction!r}. Use 'mean' or 'sum'.")

        for key, err in bd.uncertainties.items():
            err_arr = np.asarray(err, dtype=float)
            err_arr = np.broadcast_to(err_arr, x.shape)

            err_arr_eff = np.where(effective_mask, 0.0, err_arr)

            var_sum = np.sum((w_eff**2) * (err_arr_eff**2), axis=axis)

            if reduction == "mean":
                sigma = np.sqrt(var_sum) / denom
            else:  # 'sum'
                sigma = np.sqrt(var_sum)

            uncertainties_out[key] = sigma

        if estimator_specs:
            ReduceDimensionality._estimate_uncertainties_from_scatter(
                bd=bd,
                axis=axis,
                use_weights=use_weights,
                nan_policy=nan_policy,
                exclusion_mask=exclusion_mask,
                estimator_specs=estimator_specs,
                uncertainties_out=uncertainties_out,
            )

        # --- build result BaseData (numeric content) ---
        result = BaseData(
            signal=signal_out,
            units=bd.units,
            uncertainties=uncertainties_out,
            weights=np.array(1.0) if reduction == "mean" else w_sum,
        )

        # --- metadata: axes + rank_of_data ---

        # New dimensionality after reduction
        new_ndim = result.signal.ndim

        # Determine which axes were reduced, in normalized (non-negative) form
        if axis is None:
            # reducing over all axes
            reduced_axes_tuple: tuple[int, ...] = tuple(range(x.ndim))
        elif isinstance(axis, tuple):
            reduced_axes_tuple = axis
        else:
            reduced_axes_tuple = (axis,)

        reduced_axes_norm: set[int] = set()
        for a in reduced_axes_tuple:
            a_norm = a if a >= 0 else x.ndim + a
            reduced_axes_norm.add(a_norm)

        # Reduce axes metadata if we have a full set (one entry per dimension).
        old_axes = bd.axes
        if len(old_axes) == x.ndim:
            # Keep only axes that were NOT reduced
            new_axes = [ax for i, ax in enumerate(old_axes) if i not in reduced_axes_norm]
        else:
            # If metadata length does not match ndim, fall back to empty list
            new_axes = []

        result.axes = new_axes

        # Rank of data: cannot exceed new ndim, and should not exceed original rank
        result.rank_of_data = min(bd.rank_of_data, new_ndim)

        return result

    @staticmethod
    def _scalar_weight_reduction(
        *,
        bd: BaseData,
        axis: int | tuple[int, ...] | None,
        nan_policy: str,
        reduction: str,
        scalar_weight: float,
    ) -> BaseData:
        x = bd.signal

        if nan_policy == "propagate":
            if reduction == "mean":
                signal_out = np.mean(x, axis=axis)
                denom = ReduceDimensionality._axis_count(x.shape, axis)
            elif reduction == "sum":
                signal_out = np.sum(x, axis=axis) * scalar_weight
                denom = None
            else:
                raise ValueError(f"Invalid reduction: {reduction!r}. Use 'mean' or 'sum'.")

            uncertainties_out = {}
            for key, err in bd.uncertainties.items():
                err_arr = np.broadcast_to(np.asarray(err, dtype=float), x.shape)
                sigma = np.sqrt(np.sum(err_arr * err_arr, axis=axis))
                if reduction == "mean":
                    sigma = sigma / denom
                else:
                    sigma = sigma * abs(scalar_weight)
                uncertainties_out[key] = sigma
        elif nan_policy == "omit":
            if np.isnan(scalar_weight):
                signal_out = np.sum(np.full_like(x, np.nan), axis=axis)
                uncertainties_out = {key: np.sum(np.full_like(x, np.nan), axis=axis) for key in bd.uncertainties}
            else:
                mask = np.isnan(x)
                finite_count = np.sum(~mask, axis=axis)
                x_eff = np.where(mask, 0.0, x)
                if reduction == "mean":
                    denom = np.where(finite_count == 0, np.nan, finite_count)
                    signal_out = np.sum(x_eff, axis=axis) / denom
                elif reduction == "sum":
                    signal_out = np.sum(x_eff, axis=axis) * scalar_weight
                    denom = None
                else:
                    raise ValueError(f"Invalid reduction: {reduction!r}. Use 'mean' or 'sum'.")

                uncertainties_out = {}
                for key, err in bd.uncertainties.items():
                    err_arr = np.broadcast_to(np.asarray(err, dtype=float), x.shape)
                    err_eff = np.where(mask, 0.0, err_arr)
                    sigma = np.sqrt(np.sum(err_eff * err_eff, axis=axis))
                    if reduction == "mean":
                        sigma = sigma / denom
                    else:
                        sigma = sigma * abs(scalar_weight)
                    uncertainties_out[key] = sigma
        else:
            raise ValueError(f"Invalid nan_policy: {nan_policy!r}. Use 'omit' or 'propagate'.")

        result = BaseData(
            signal=signal_out,
            units=bd.units,
            uncertainties=uncertainties_out,
            weights=(
                np.array(1.0)
                if reduction == "mean"
                else np.full(signal_out.shape, scalar_weight * ReduceDimensionality._axis_count(x.shape, axis))
            ),
        )

        new_ndim = result.signal.ndim
        if axis is None:
            reduced_axes_tuple: tuple[int, ...] = tuple(range(x.ndim))
        elif isinstance(axis, tuple):
            reduced_axes_tuple = axis
        else:
            reduced_axes_tuple = (axis,)

        reduced_axes_norm = {a if a >= 0 else x.ndim + a for a in reduced_axes_tuple}
        old_axes = bd.axes
        result.axes = (
            [ax for i, ax in enumerate(old_axes) if i not in reduced_axes_norm] if len(old_axes) == x.ndim else []
        )
        result.rank_of_data = min(bd.rank_of_data, new_ndim)
        return result

    # ---------------------------- main API ---------------------------------

    def dependency_contract(self) -> ProcessStepDependencies:
        keys = self.configuration.get("with_processing_keys")
        reads = set(processing_key_patterns(keys, basedata_key="signal"))
        writes = set(reads)
        mask_key = self.configuration.get("mask_key")
        if isinstance(mask_key, str) and mask_key.strip():
            reads.update(processing_key_patterns(keys, basedata_key=mask_key))
        return ProcessStepDependencies(
            source_refs=(),
            processing_reads=reads,
            processing_writes=writes,
        )

    def prepare_execution(self) -> None:
        """Validate estimator configuration before processing data."""

        reduction = self.configuration.get("reduction", "mean")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction!r}. Use 'mean' or 'sum'.")
        self._normalize_uncertainty_estimation(
            self.configuration.get("uncertainty_estimation"),
            reduction=reduction,
        )
        self._normalize_mask_configuration(
            self.configuration.get("mask_key"),
            self.configuration.get("mask_bits"),
        )

    def calculate(self) -> dict[str, DataBundle]:
        axes_spec = self.configuration.get("axes")
        use_weights = bool(self.configuration.get("use_weights", True))
        nan_policy = self.configuration.get("nan_policy", "omit")
        reduction = self.configuration.get("reduction", "mean")
        if nan_policy not in {"omit", "propagate"}:
            raise ValueError(f"Invalid nan_policy: {nan_policy!r}. Use 'omit' or 'propagate'.")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction!r}. Use 'mean' or 'sum'.")
        estimator_specs = self._normalize_uncertainty_estimation(
            self.configuration.get("uncertainty_estimation"),
            reduction=reduction,
        )
        mask_key, mask_bits = self._normalize_mask_configuration(
            self.configuration.get("mask_key"),
            self.configuration.get("mask_bits"),
        )

        output: dict[str, DataBundle] = {}

        for key in self._normalised_processing_keys():
            databundle: DataBundle = self.processing_data.get(key)
            bd: BaseData = databundle["signal"]
            axis = self._resolve_axes(axes_spec, bd)

            if axis == ():
                logger.debug(
                    f"ReduceDimensionality: {key}::signal already has ndim={bd.signal.ndim} "
                    f"matching rank_of_data={bd.rank_of_data}; skipping."
                )
                output[key] = databundle
                continue

            exclusion_mask = self._mask_exclusion(
                databundle=databundle,
                signal_shape=bd.signal.shape,
                mask_key=mask_key,
                mask_bits=mask_bits,
            )

            averaged = self._weighted_mean_with_uncertainty(
                bd=bd,
                axis=axis,
                use_weights=use_weights,
                nan_policy=nan_policy,
                reduction=reduction,
                estimator_specs=estimator_specs,
                exclusion_mask=exclusion_mask,
            )

            databundle["signal"] = averaged
            output[key] = databundle

        logger.info(f"ReduceDimensionality: calculation finished for {len(output)} keys.")

        return output
