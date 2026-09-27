# SPDX-License-Identifier: BSD-3-Clause
"""Format-independent curve-alignment and scaling kernels."""

from __future__ import annotations

__all__ = [
    "DependentData1D",
    "FitData1D",
    "ScaleFitResult",
    "fit_scale_factor_1d",
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

    order = np.argsort(x_work)
    x_work = x_work[order]
    y_work = dep_work.y[order]
    sigma_work = dep_work.sigma[order]
    weights_work = dep_work.weights[order]

    bounds_error = require_overlap
    fill_value = None if bounds_error else "extrapolate"
    interp_y = interp1d(
        x_work,
        y_work,
        kind=interpolation_kind,
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=True,
    )
    interp_sigma = interp1d(
        x_work,
        sigma_work,
        kind="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=True,
    )
    interp_weights = interp1d(
        x_work,
        weights_work,
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
