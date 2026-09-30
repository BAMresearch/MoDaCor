# SPDX-License-Identifier: BSD-3-Clause
"""Format-independent curve-alignment and scaling kernels."""

from __future__ import annotations

__all__ = [
    "DependentData1D",
    "FitData1D",
    "ScaleFitResult",
    "fit_scale_factor_1d",
    "fit_lognormal_scale_factor_1d",
    "prepare_scale_fit_data",
]

import numpy as np
from attrs import define
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


@define(slots=True)
class DependentData1D:
    """Numerical dependent values and their matching uncertainty and weight arrays."""

    y: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray


@define(slots=True)
class FitData1D:
    """Aligned arrays used by the robust scale fit."""

    x: np.ndarray
    y_ref: np.ndarray
    y_work: np.ndarray
    sigma_ref: np.ndarray
    sigma_work: np.ndarray
    weights: np.ndarray


@define(frozen=True, slots=True)
class ScaleFitResult:
    """Scale-fit parameters and their estimated standard uncertainties."""

    scale: float
    scale_sigma: float
    background: float | None = None
    background_sigma: float | None = None


def _overlap_range(x1: np.ndarray, x2: np.ndarray) -> tuple[float, float]:
    return float(max(np.nanmin(x1), np.nanmin(x2))), float(min(np.nanmax(x1), np.nanmax(x2)))


def _prepare_unique_dependent(
    x: np.ndarray,
    dependent: DependentData1D,
    *,
    require_positive_weights: bool,
) -> tuple[np.ndarray, DependentData1D]:
    """Sort, filter, and uncertainty-average repeated coordinates."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(dependent.y, dtype=float)
    sigma = np.asarray(dependent.sigma, dtype=float)
    weights = np.asarray(dependent.weights, dtype=float)
    if not (x.shape == y.shape == sigma.shape == weights.shape) or x.ndim != 1:
        raise ValueError("Scale-fit coordinates, values, uncertainties, and weights must be matching 1D arrays.")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
    if require_positive_weights:
        valid &= np.isfinite(weights) & (weights > 0.0)
    x = x[valid]
    y = y[valid]
    sigma = sigma[valid]
    weights = weights[valid]
    if x.size < 2:
        raise ValueError("Not enough valid points for scale fitting.")

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    sigma = sigma[order]
    weights = weights[order]
    unique_x, group = np.unique(x, return_inverse=True)
    if unique_x.size == x.size:
        return x, DependentData1D(y=y, sigma=sigma, weights=weights)

    precision = 1.0 / sigma**2
    summed_precision = np.bincount(group, weights=precision)
    averaged_y = np.bincount(group, weights=precision * y) / summed_precision
    averaged_sigma = 1.0 / np.sqrt(summed_precision)
    averaged_weights = np.bincount(group, weights=precision * weights) / summed_precision
    return unique_x, DependentData1D(
        y=averaged_y,
        sigma=averaged_sigma,
        weights=averaged_weights,
    )


def prepare_scale_fit_data(
    *,
    x_work: np.ndarray,
    dep_work: DependentData1D,
    x_ref: np.ndarray,
    dep_ref: DependentData1D,
    require_overlap: bool,
    interpolation_kind: str,
    fit_min: float,
    fit_max: float,
    use_weights: bool,
) -> FitData1D:
    """Align work data to the selected reference-axis fit window."""

    x_work, dep_work = _prepare_unique_dependent(
        x_work,
        dep_work,
        require_positive_weights=use_weights,
    )
    x_ref, dep_ref = _prepare_unique_dependent(
        x_ref,
        dep_ref,
        require_positive_weights=use_weights,
    )

    ov_min, ov_max = _overlap_range(x_ref, x_work)
    if require_overlap and not (ov_min < ov_max):
        raise ValueError("No overlap between working and reference x-axes.")

    lo = max(fit_min, ov_min) if require_overlap else fit_min
    hi = min(fit_max, ov_max) if require_overlap else fit_max
    if not lo < hi:
        raise ValueError("Empty fit range after overlap constraints.")

    mask = (x_ref >= lo) & (x_ref <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough points in fit window.")

    x_fit = x_ref[mask]
    y_ref = dep_ref.y[mask]
    sigma_ref = dep_ref.sigma[mask]
    weights_ref = dep_ref.weights[mask]

    bounds_error = require_overlap
    fill_value = None if bounds_error else "extrapolate"
    interp_y = interp1d(
        x_work,
        dep_work.y,
        kind=interpolation_kind,
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=True,
    )
    interp_sigma = interp1d(
        x_work,
        dep_work.sigma,
        kind="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=True,
    )
    interp_weights = interp1d(
        x_work,
        dep_work.weights,
        kind="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=True,
    )

    y_work_interpolated = interp_y(x_fit)
    sigma_work_interpolated = interp_sigma(x_fit)
    weights_work_interpolated = interp_weights(x_fit)
    weights = weights_ref * weights_work_interpolated if use_weights else np.ones_like(y_ref)
    valid = (
        np.isfinite(y_ref)
        & np.isfinite(y_work_interpolated)
        & np.isfinite(sigma_ref)
        & (sigma_ref > 0)
        & np.isfinite(sigma_work_interpolated)
        & (sigma_work_interpolated >= 0)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("Not enough valid points after masking.")

    return FitData1D(
        x=x_fit[valid],
        y_ref=y_ref[valid],
        y_work=y_work_interpolated[valid],
        sigma_ref=sigma_ref[valid],
        sigma_work=sigma_work_interpolated[valid],
        weights=weights[valid],
    )


def fit_scale_factor_1d(
    fit_data: FitData1D,
    *,
    fit_background: bool,
    robust_loss: str,
    robust_fscale: float,
) -> ScaleFitResult:
    """Fit a robust multiplicative scale and optional constant background."""

    def residuals(parameters: np.ndarray) -> np.ndarray:
        scale = parameters[0]
        background = parameters[1] if fit_background else 0.0
        model = scale * fit_data.y_work + background
        sigma = np.sqrt(fit_data.sigma_ref**2 + (scale * fit_data.sigma_work) ** 2)
        residual = (fit_data.y_ref - model) / sigma
        return np.sqrt(fit_data.weights) * residual

    if fit_background:
        design = np.column_stack([fit_data.y_work, np.ones_like(fit_data.y_work)])
        initial, *_ = np.linalg.lstsq(design, fit_data.y_ref, rcond=None)
    else:
        denominator = np.dot(fit_data.y_work, fit_data.y_work) or 1.0
        initial = np.array([np.dot(fit_data.y_ref, fit_data.y_work) / denominator])

    fitted = least_squares(
        residuals,
        x0=initial,
        loss=robust_loss,
        f_scale=float(robust_fscale),
    )
    degrees_of_freedom = max(1, len(fitted.fun) - len(fitted.x))
    residual_variance = np.sum(fitted.fun**2) / degrees_of_freedom
    covariance = residual_variance * np.linalg.pinv(fitted.jac.T @ fitted.jac)
    parameter_sigmas = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))

    return ScaleFitResult(
        scale=float(fitted.x[0]),
        scale_sigma=float(parameter_sigmas[0]),
        background=float(fitted.x[1]) if fit_background else None,
        background_sigma=float(parameter_sigmas[1]) if fit_background else None,
    )


def fit_lognormal_scale_factor_1d(fit_data: FitData1D) -> ScaleFitResult:
    """Estimate a positive scale from an uncertainty-weighted mean log ratio.

    The diagonal log-ratio variance is obtained by first-order propagation of
    the selected uncertainty component on the reference and working signals.
    ``fit_data.weights`` supplies any additional BaseData quality weights.
    """

    valid = (
        np.isfinite(fit_data.y_ref)
        & (fit_data.y_ref > 0.0)
        & np.isfinite(fit_data.y_work)
        & (fit_data.y_work > 0.0)
        & np.isfinite(fit_data.sigma_ref)
        & (fit_data.sigma_ref > 0.0)
        & np.isfinite(fit_data.sigma_work)
        & (fit_data.sigma_work > 0.0)
        & np.isfinite(fit_data.weights)
        & (fit_data.weights > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("Lognormal scaling requires at least two positive valid overlap points.")

    y_ref = fit_data.y_ref[valid]
    y_work = fit_data.y_work[valid]
    log_ratio = np.log(y_ref / y_work)
    log_ratio_variance = (fit_data.sigma_ref[valid] / y_ref) ** 2 + (fit_data.sigma_work[valid] / y_work) ** 2
    weights = fit_data.weights[valid] / log_ratio_variance
    finite_weight = np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(finite_weight) < 2:
        raise ValueError("Lognormal scaling has fewer than two finite positive statistical weights.")
    weights = weights[finite_weight]
    log_ratio = log_ratio[finite_weight]
    sum_weights = float(np.sum(weights))
    mean_log_scale = float(np.sum(weights * log_ratio) / sum_weights)
    log_scale_sigma = float(np.sqrt(1.0 / sum_weights))
    scale = float(np.exp(mean_log_scale))
    return ScaleFitResult(scale=scale, scale_sigma=scale * log_scale_sigma)
