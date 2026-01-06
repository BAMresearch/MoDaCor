# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"

__all__ = ["FindScaleFactor1D"]
__version__ = "20251212.2"

from pathlib import Path
from typing import Dict

import numpy as np
from attrs import define
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# -------------------------------------------------------------------------
# Small data containers (attrs, not namedtuple)
# -------------------------------------------------------------------------


@define(slots=True)
class DependentData1D:
    y: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray


@define(slots=True)
class FitData1D:
    x: np.ndarray
    y_ref: np.ndarray
    y_work: np.ndarray
    sigma_ref: np.ndarray
    sigma_work: np.ndarray
    weights: np.ndarray


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _combined_sigma(bd: BaseData) -> np.ndarray:
    if not bd.uncertainties:
        return np.asarray(1.0)

    sig2 = None
    for u in bd.uncertainties.values():
        arr = np.asarray(u, dtype=float)
        sig2 = arr * arr if sig2 is None else sig2 + arr * arr
    return np.sqrt(sig2)


def _extract_dependent(bd: BaseData) -> DependentData1D:
    if bd.rank_of_data != 1:
        raise ValueError("Dependent BaseData must be rank-1.")

    y = np.asarray(bd.signal, dtype=float).squeeze()
    if y.ndim != 1:
        raise ValueError("Dependent signal must be 1D.")

    sigma = np.asarray(_combined_sigma(bd), dtype=float)
    weights = np.asarray(bd.weights, dtype=float)

    if sigma.size == 1:
        sigma = np.full_like(y, float(sigma))
    else:
        sigma = sigma.squeeze()

    if weights.size == 1:
        weights = np.full_like(y, float(weights))
    else:
        weights = weights.squeeze()

    if sigma.shape != y.shape or weights.shape != y.shape:
        raise ValueError("Uncertainties and weights must match dependent signal shape.")

    sigma = np.where(sigma <= 0.0, np.nan, sigma)

    return DependentData1D(y=y, sigma=sigma, weights=weights)


def _overlap_range(x1: np.ndarray, x2: np.ndarray) -> tuple[float, float]:
    return float(max(np.nanmin(x1), np.nanmin(x2))), float(min(np.nanmax(x1), np.nanmax(x2)))


def _prepare_fit_data(
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

    # sort working data
    order = np.argsort(x_work)
    x_work = x_work[order]
    y_work = dep_work.y[order]
    sigma_work = dep_work.sigma[order]
    weights_work = dep_work.weights[order]

    bounds_error = require_overlap
    fill_value = None if bounds_error else "extrapolate"

    interp_y = interp1d(
        x_work, y_work, kind=interpolation_kind, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True
    )
    interp_sigma = interp1d(
        x_work, sigma_work, kind="linear", bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True
    )
    interp_w = interp1d(
        x_work, weights_work, kind="linear", bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True
    )

    y_work_i = interp_y(x_fit)
    sigma_work_i = interp_sigma(x_fit)
    weights_work_i = interp_w(x_fit)

    weights = (weights_ref * weights_work_i) if use_weights else np.ones_like(y_ref)

    valid = (
        np.isfinite(y_ref)
        & np.isfinite(y_work_i)
        & np.isfinite(sigma_ref)
        & (sigma_ref > 0)
        & np.isfinite(sigma_work_i)
        & (sigma_work_i >= 0)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError("Not enough valid points after masking.")

    return FitData1D(
        x=x_fit[valid],
        y_ref=y_ref[valid],
        y_work=y_work_i[valid],
        sigma_ref=sigma_ref[valid],
        sigma_work=sigma_work_i[valid],
        weights=weights[valid],
    )


# -------------------------------------------------------------------------
# Main ProcessStep
# -------------------------------------------------------------------------


class FindScaleFactor1D(ProcessStep):
    documentation = ProcessStepDescriber(
        calling_name="Scale 1D curve to reference (compute-only)",
        calling_id="FindScaleFactor1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={
            "scale_factor": ["signal", "uncertainties", "units"],
            "scale_background": ["signal", "uncertainties", "units"],
        },
        default_configuration={
            "signal_key": "signal",
            "independent_axis_key": "Q",
            "scale_output_key": "scale_factor",
            "background_output_key": "scale_background",
            "fit_background": False,
            "fit_min_val": None,
            "fit_max_val": None,
            "fit_val_units": None,
            "require_overlap": True,
            "interpolation_kind": "linear",
            "robust_loss": "huber",
            "robust_fscale": 1.0,
            "use_basedata_weights": True,
        },
        step_keywords=["scale", "calibration", "1D"],
        step_doc="Compute scale factor between two 1D curves using robust least squares.",
    )

    def calculate(self) -> Dict[str, DataBundle]:
        cfg = self.configuration
        work_key, ref_key = cfg["with_processing_keys"]

        sig_key = cfg.get("signal_key", "signal")
        axis_key = cfg.get("independent_axis_key", "Q")

        work_db = self.processing_data[work_key]
        ref_db = self.processing_data[ref_key]

        y_work_bd = work_db[sig_key].copy(with_axes=True)
        y_ref_bd = ref_db[sig_key].copy(with_axes=True)

        x_work_bd = work_db[axis_key].copy(with_axes=False)
        x_ref_bd = ref_db[axis_key].copy(with_axes=False)

        if x_work_bd.units != x_ref_bd.units:
            x_work_bd.to_units(x_ref_bd.units)

        x_work = np.asarray(x_work_bd.signal, dtype=float).squeeze()
        x_ref = np.asarray(x_ref_bd.signal, dtype=float).squeeze()

        dep_work = _extract_dependent(y_work_bd)
        dep_ref = _extract_dependent(y_ref_bd)

        fit_min = cfg.get("fit_min_val")
        fit_max = cfg.get("fit_max_val")

        fit_units = cfg.get("fit_val_units") or x_ref_bd.units
        if fit_min is not None:
            fit_min = ureg.Quantity(fit_min, fit_units).to(x_ref_bd.units).magnitude
        else:
            fit_min = np.nanmin(x_ref)

        if fit_max is not None:
            fit_max = ureg.Quantity(fit_max, fit_units).to(x_ref_bd.units).magnitude
        else:
            fit_max = np.nanmax(x_ref)

        fit_data = _prepare_fit_data(
            x_work=x_work,
            dep_work=dep_work,
            x_ref=x_ref,
            dep_ref=dep_ref,
            require_overlap=cfg.get("require_overlap", True),
            interpolation_kind=cfg.get("interpolation_kind", "linear"),
            fit_min=float(fit_min),
            fit_max=float(fit_max),
            use_weights=cfg.get("use_basedata_weights", True),
        )

        fit_background = bool(cfg.get("fit_background", False))

        def residuals(p: np.ndarray) -> np.ndarray:
            scale = p[0]
            background = p[1] if fit_background else 0.0
            model = scale * fit_data.y_work + background
            sigma = np.sqrt(fit_data.sigma_ref**2 + (scale * fit_data.sigma_work) ** 2)
            r = (fit_data.y_ref - model) / sigma
            return np.sqrt(fit_data.weights) * r

        if fit_background:
            X = np.column_stack([fit_data.y_work, np.ones_like(fit_data.y_work)])
            x0, *_ = np.linalg.lstsq(X, fit_data.y_ref, rcond=None)
        else:
            denom = np.dot(fit_data.y_work, fit_data.y_work) or 1.0
            x0 = np.array([np.dot(fit_data.y_ref, fit_data.y_work) / denom])

        res = least_squares(
            residuals,
            x0=x0,
            loss=cfg.get("robust_loss", "huber"),
            f_scale=float(cfg.get("robust_fscale", 1.0)),
        )

        J = res.jac
        dof = max(1, len(res.fun) - len(res.x))
        s_sq = np.sum(res.fun**2) / dof

        cov = s_sq * np.linalg.pinv(J.T @ J)
        sig_params = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        scale = float(res.x[0])
        scale_sigma = float(sig_params[0])

        out_key = cfg.get("scale_output_key", "scale_factor")
        work_db[out_key] = BaseData(
            signal=np.array([scale]),
            units="dimensionless",
            uncertainties={"propagate_to_all": np.array([scale_sigma])},
            rank_of_data=0,
        )

        if fit_background:
            bg_key = cfg.get("background_output_key", "scale_background")
            work_db[bg_key] = BaseData(
                signal=np.array([float(res.x[1])]),
                units=y_ref_bd.units,
                uncertainties={"propagate_to_all": np.array([sig_params[1]])},
                rank_of_data=0,
            )

        return {work_key: work_db}
