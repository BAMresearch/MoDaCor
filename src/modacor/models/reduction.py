# SPDX-License-Identifier: BSD-3-Clause
"""Array-only axis and grouped reduction kernels with existing statistical policies."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from attrs import define

from modacor.models.statistics import finalize_weighted_scatter

__all__ = [
    "effective_values_and_weights",
    "axis_scatter_moments",
    "axis_count",
    "weighted_axis_reduction",
    "scalar_weight_reduction",
    "GroupedObservations",
    "grouped_scatter_statistics",
    "grouped_uncertainties",
    "group_observations",
    "grouped_weights",
    "grouped_mean",
]

ReductionAxis = int | tuple[int, ...] | None


def effective_values_and_weights(
    *,
    signal: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    nan_policy: str,
    exclusion_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply explicit-mask and NaN selection to values and effective weights."""

    x = np.asarray(signal, dtype=float)
    if use_weights:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), x.shape)
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


def axis_scatter_moments(
    x_eff: np.ndarray, w_eff: np.ndarray, axis: ReductionAxis
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate weighted population-scatter moments over regular axes."""
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

    return sum_w, sum_w2, sum_w_squared_deviations


def axis_count(shape: tuple[int, ...], axis: int | tuple[int, ...] | None) -> int:
    if axis is None:
        return int(np.prod(shape))
    if isinstance(axis, tuple):
        return int(np.prod([shape[a if a >= 0 else len(shape) + a] for a in axis]))
    return int(shape[axis if axis >= 0 else len(shape) + axis])


def weighted_axis_reduction(
    signal: np.ndarray,
    uncertainties: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    axis: ReductionAxis,
    use_weights: bool,
    nan_policy: str,
    reduction: str,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Reduce values and independent sigmas using broadcast array weights."""
    x_eff, w_eff, effective_mask = effective_values_and_weights(
        signal=signal,
        weights=weights,
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

    for key, err in uncertainties.items():
        err_arr = np.asarray(err, dtype=float)
        err_arr = np.broadcast_to(err_arr, signal.shape)

        err_arr_eff = np.where(effective_mask, 0.0, err_arr)

        var_sum = np.sum((w_eff**2) * (err_arr_eff**2), axis=axis)

        if reduction == "mean":
            sigma = np.sqrt(var_sum) / denom
        else:  # 'sum'
            sigma = np.sqrt(var_sum)

        uncertainties_out[key] = sigma

    return signal_out, uncertainties_out, np.array(1.0) if reduction == "mean" else w_sum


def scalar_weight_reduction(
    signal: np.ndarray,
    uncertainties: Mapping[str, np.ndarray],
    *,
    axis: ReductionAxis,
    nan_policy: str,
    reduction: str,
    scalar_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Reduce with the original optimized scalar-weight and NaN conventions."""
    x = signal

    if nan_policy == "propagate":
        if reduction == "mean":
            signal_out = np.mean(x, axis=axis)
            denom = axis_count(x.shape, axis)
        elif reduction == "sum":
            signal_out = np.sum(x, axis=axis) * scalar_weight
            denom = None
        else:
            raise ValueError(f"Invalid reduction: {reduction!r}. Use 'mean' or 'sum'.")

        uncertainties_out = {}
        for key, err in uncertainties.items():
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
            uncertainties_out = {key: np.sum(np.full_like(x, np.nan), axis=axis) for key in uncertainties}
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
            for key, err in uncertainties.items():
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

    output_weights = (
        np.array(1.0) if reduction == "mean" else np.full(signal_out.shape, scalar_weight * axis_count(x.shape, axis))
    )
    return signal_out, uncertainties_out, output_weights


@define(frozen=True, slots=True)
class GroupedObservations:
    bin_ids: np.ndarray
    selection: np.ndarray
    dense_indices: np.ndarray
    weights: np.ndarray
    values: np.ndarray
    axis_values: np.ndarray | None
    sum_weights: np.ndarray
    sum_squared_weights: np.ndarray


def grouped_scatter_statistics(
    values: np.ndarray,
    means: np.ndarray,
    dense_indices: np.ndarray,
    weights: np.ndarray,
    sum_weights: np.ndarray,
    sum_squared_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    deviations = values - means[dense_indices]
    squared_deviation_sum = np.bincount(
        dense_indices,
        weights=weights * deviations**2,
        minlength=sum_weights.size,
    )
    estimates = finalize_weighted_scatter(
        sum_w=sum_weights,
        sum_w2=sum_squared_weights,
        sum_w_squared_deviations=squared_deviation_sum,
        ddof=0,
    )
    enough_observations = estimates.effective_sample_size > 1.0
    sem = np.where(enough_observations, estimates.standard_error_mean, np.nan)
    std = np.where(enough_observations, estimates.standard_deviation, np.nan)
    return sem, std


def grouped_uncertainties(
    components: Mapping[str, np.ndarray], *, dense_indices: np.ndarray, weights: np.ndarray, sum_weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Propagate selected independent sigmas through grouped weighted means."""
    result: dict[str, np.ndarray] = {}
    for name, selected in components.items():
        variance_sum = np.bincount(
            dense_indices,
            weights=(weights**2) * (selected**2),
            minlength=sum_weights.size,
        )
        result[name] = np.sqrt(variance_sum) / sum_weights
    return result


def group_observations(
    indices: np.ndarray, values: np.ndarray, axis_values: np.ndarray | None, weights: np.ndarray, valid: np.ndarray
) -> GroupedObservations:
    """Select populated bins and accumulate their first weight moments."""
    contributing = valid & (weights > 0.0)
    if not np.any(contributing):
        raise ValueError("IndexedAverager: no valid observations with positive weight remain.")

    populated_bin_ids = np.unique(indices[contributing])
    selection = valid & np.isin(indices, populated_bin_ids)
    selected_indices = indices[selection]
    dense_indices = np.searchsorted(populated_bin_ids, selected_indices)
    selected_weights = weights[selection]
    selected_values = values[selection]
    n_bins = populated_bin_ids.size

    sum_weights = np.bincount(dense_indices, weights=selected_weights, minlength=n_bins)
    sum_squared_weights = np.bincount(dense_indices, weights=selected_weights**2, minlength=n_bins)
    return GroupedObservations(
        bin_ids=populated_bin_ids,
        selection=selection,
        dense_indices=dense_indices,
        weights=selected_weights,
        values=selected_values,
        axis_values=None if axis_values is None else axis_values[selection],
        sum_weights=sum_weights,
        sum_squared_weights=sum_squared_weights,
    )


def grouped_weights(
    shape: tuple[int, ...],
    valid: np.ndarray,
    *,
    value_weights: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply value and inverse-variance weights with existing validity rules."""
    valid = valid.copy()
    weights = np.ones(shape, dtype=float)
    if value_weights is not None:
        weights *= value_weights
    if sigma is not None:
        usable_sigma = np.isfinite(sigma) & (sigma > 0.0)
        valid &= usable_sigma
        weights[usable_sigma] *= 1.0 / sigma[usable_sigma] ** 2
    if np.any(valid & np.isfinite(weights) & (weights < 0.0)):
        raise ValueError("IndexedAverager: weights must not be negative.")
    valid &= np.isfinite(weights)
    return weights, valid


def grouped_mean(
    values: np.ndarray, dense_indices: np.ndarray, weights: np.ndarray, sum_weights: np.ndarray
) -> np.ndarray:
    """Return a weighted mean for each populated dense bin."""
    return np.bincount(dense_indices, weights=weights * values, minlength=sum_weights.size) / sum_weights
