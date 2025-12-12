# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["FindScaleFactor1D"]
__version__ = "20251212.1"

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


def _combined_sigma(bd: BaseData) -> np.ndarray:
    """
    Combine all uncertainty contributions in quadrature.
    BaseData.uncertainties stores 1-sigma arrays.
    Returns an array broadcastable to bd.signal.
    """
    if not bd.uncertainties:
        return np.asarray(1.0)

    sig2 = None
    for v in bd.uncertainties.values():
        vv = np.asarray(v, dtype=float)
        sig2 = (vv * vv) if sig2 is None else (sig2 + vv * vv)
    return np.sqrt(sig2)  # type: ignore[arg-type]


def _as_1d_y_sig_w(bd: BaseData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract y, sigma_y, weights as 1D arrays from a rank-1 BaseData.
    Independent variable is handled separately via independent_axis_key.
    """
    if bd.rank_of_data != 1:
        raise ValueError("FindScaleFactor1D expects rank-1 BaseData for the dependent signal.")

    y = np.asarray(bd.signal, dtype=float).squeeze()
    if y.ndim != 1:
        raise ValueError("FindScaleFactor1D expects 1D dependent signal arrays.")

    sig = np.asarray(_combined_sigma(bd), dtype=float)
    w = np.asarray(bd.weights, dtype=float)

    # Broadcast sigma and weights to y
    if sig.size == 1:
        sig = np.full_like(y, float(sig))
    else:
        sig = np.asarray(sig, dtype=float).squeeze()
    if w.size == 1:
        w = np.full_like(y, float(w))
    else:
        w = np.asarray(w, dtype=float).squeeze()

    if sig.shape != y.shape or w.shape != y.shape:
        raise ValueError("Uncertainties/weights must be broadcastable to the dependent signal shape for 1D fitting.")

    sig = np.where(sig <= 0.0, np.nan, sig)
    return y, sig, w


def _overlap_range(x1: np.ndarray, x2: np.ndarray) -> tuple[float, float]:
    """
    Determine the overlapping range between two 1D arrays.
    Returns (min, max) of the overlap. If no overlap, min >= max.
    """
    return float(max(np.nanmin(x1), np.nanmin(x2))), float(min(np.nanmax(x1), np.nanmax(x2)))


class FindScaleFactor1D(ProcessStep):
    """
    Compute a robust scaling factor between two 1D curves, with optional flat background.
    This step computes and stores the factor(s) but does not apply them to the working data.
    """

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
        calling_arguments={
            "signal_key": "signal",  # rename to dependent_axis_key?
            "independent_axis_key": "Q",  # should be used rather than axes[0]
            "scale_output_key": "scale_factor",
            "background_output_key": "scale_background",
            "fit_background": False,
            "fit_min_val": None,
            "fit_max_val": None,
            "fit_val_units": None,  # units for fit_min_val and fit_max_val
            "require_overlap": True,
            "interpolation_kind": "linear",
            "robust_loss": "huber",  # huber|soft_l1|cauchy|linear
            "robust_fscale": 1.0,
            "use_basedata_weights": True,
        },
        step_keywords=["scale", "calibration", "absolute_units", "scaling", "factor", "1D"],
        step_doc="Robust least-squares scale (and optional flat background) between two 1D curves with interpolation.",
        step_reference="DOI 10.1107/S0021889897001157",
        step_note="""
            Fits y_ref(x) ≈ s*y_work(x) + b (optional), on a user-specified range and within axis overlap.
            Uses scipy.optimize.least_squares with robust loss. Stores scale (and background) as one-element BaseData
            with uncertainty under key 'propagate_to_all'. Does not modify the working signal.
        """,
    )

    def calculate(self) -> Dict[str, DataBundle]:  # noqa: C901 -- complexity to address later
        def _residuals(p: np.ndarray) -> np.ndarray:
            """
            Residuals function for least_squares.
            p[0] = scale factor
            p[1] = background (if fit_background)
            Returns weighted residuals array.
            """
            s = float(p[0])
            b = float(p[1]) if fit_background else 0.0

            model = s * y_work + b

            # total sigma includes working contribution through scale
            sig_tot = np.sqrt(sig_ref * sig_ref + (s * sig_work) * (s * sig_work))
            sig_tot = np.where(np.isfinite(sig_tot) & (sig_tot > 0.0), sig_tot, np.nan)

            # residuals: sqrt(weight) * (data-model)/sigma
            r = (y_ref - model) / sig_tot
            r = np.where(np.isfinite(r), r, 0.0)
            return np.sqrt(np.clip(w_extra, 0.0, np.inf)) * r

        cfg = self.configuration
        if not cfg.get("with_processing_keys") or len(cfg["with_processing_keys"]) != 2:
            raise ValueError(
                "FindScaleFactor1D requires exactly two processing keys in 'with_processing_keys': "
                "[working_key, reference_key]."
            )

        working_key, reference_key = cfg["with_processing_keys"]
        signal_key = cfg.get("signal_key", "signal")

        working_db = self.processing_data.get(working_key)
        reference_db = self.processing_data.get(reference_key)
        if working_db is None or reference_db is None:
            raise KeyError("Could not find working/reference DataBundle in processing_data.")

        independent_axis_key = str(cfg.get("independent_axis_key", "Q"))

        # Clone so we can do to_units() safely without side-effects
        working_y_bd = working_db[signal_key].copy(with_axes=True)
        reference_y_bd = reference_db[signal_key].copy(with_axes=True)

        # For the independent axis, axes metadata is irrelevant; copy without axes is slightly cleaner
        working_x_bd = working_db[independent_axis_key].copy(with_axes=False)
        reference_x_bd = reference_db[independent_axis_key].copy(with_axes=False)

        # Sanity checks for axis BD
        x_w = np.asarray(working_x_bd.signal, dtype=float).squeeze()
        x_r = np.asarray(reference_x_bd.signal, dtype=float).squeeze()
        if x_w.ndim != 1 or x_r.ndim != 1:
            raise ValueError(f"Independent axis '{independent_axis_key}' must be 1D for FindScaleFactor1D.")

        # Unify x-axis units: convert working axis to reference axis units (in-place on clone)
        if working_x_bd.units != reference_x_bd.units:
            working_x_bd.to_units(reference_x_bd.units)
            x_w = np.asarray(working_x_bd.signal, dtype=float).squeeze()

        # Extract dependent y, sigma, weights (x handled separately)
        y_w, sig_w, w_w = _as_1d_y_sig_w(working_y_bd)
        y_r, sig_r, w_r = _as_1d_y_sig_w(reference_y_bd)

        # ensure shapes match:
        if x_w.shape[0] != y_w.shape[0]:
            raise ValueError(f"Working axis '{independent_axis_key}' length does not match '{signal_key}' length.")
        if x_r.shape[0] != y_r.shape[0]:
            raise ValueError(f"Reference axis '{independent_axis_key}' length does not match '{signal_key}' length.")

        # Fit range + overlap
        require_overlap = bool(cfg.get("require_overlap", True))
        ov_min, ov_max = _overlap_range(x_r, x_w)
        if require_overlap and not (ov_min < ov_max):
            raise ValueError("No overlap between working and reference x-axes.")

        fit_min_val = cfg.get("fit_min_val", None)
        fit_max_val = cfg.get("fit_max_val", None)

        fit_min_val = float(fit_min_val) if fit_min_val is not None else None
        fit_max_val = float(fit_max_val) if fit_max_val is not None else None

        fit_val_units = cfg.get("fit_val_units", None)
        if fit_val_units is None:
            fit_val_units = reference_x_bd.units  # default: same units as reference axis

        # Convert fit window into reference axis units if needed
        if fit_val_units is not None and fit_val_units != reference_x_bd.units:
            if fit_min_val is not None:
                fit_min_val = float(ureg.Quantity(fit_min_val, fit_val_units).to(reference_x_bd.units).magnitude)
            if fit_max_val is not None:
                fit_max_val = float(ureg.Quantity(fit_max_val, fit_val_units).to(reference_x_bd.units).magnitude)

        fit_min_eff = fit_min_val if fit_min_val is not None else (ov_min if require_overlap else float(np.nanmin(x_r)))
        fit_max_eff = fit_max_val if fit_max_val is not None else (ov_max if require_overlap else float(np.nanmax(x_r)))

        if require_overlap:
            fit_min_eff = max(fit_min_eff, ov_min)
            fit_max_eff = min(fit_max_eff, ov_max)

        if not (fit_min_eff < fit_max_eff):
            raise ValueError("Fit range is empty after applying overlap and fit_min_val/fit_max_val.")

        m = (x_r >= fit_min_eff) & (x_r <= fit_max_eff)
        if m.sum() < (3 if bool(cfg.get("fit_background", False)) else 2):
            raise ValueError("Not enough points in fit range to determine parameters.")

        x_fit = x_r[m]
        y_ref = y_r[m]
        sig_ref = sig_r[m]
        w_ref = w_r[m]

        # ensure sorted working data:
        iw = np.argsort(x_w)
        x_w = x_w[iw]
        y_w = y_w[iw]
        sig_w = sig_w[iw]
        w_w = w_w[iw]

        # Interpolate working curve (and optionally its sigma/weights) onto reference x grid
        kind = str(cfg.get("interpolation_kind", "linear"))
        bounds_error = bool(require_overlap)
        fill_value = "extrapolate" if not bounds_error else None

        f_y = interp1d(x_w, y_w, kind=kind, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True)
        y_work = np.asarray(f_y(x_fit), dtype=float)

        # interpolate working sigma and weights too (keeps code short and consistent)
        f_sigw = interp1d(
            x_w, sig_w, kind="linear", bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True
        )
        sig_work = np.asarray(f_sigw(x_fit), dtype=float)

        f_ww = interp1d(x_w, w_w, kind="linear", bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True)
        w_work = np.asarray(f_ww(x_fit), dtype=float)

        # Combine uncertainties (reference + working propagated through scale inside residual)
        # Also incorporate BaseData.weights as an optional extra weighting factor.
        use_weights = bool(cfg.get("use_basedata_weights", True))
        w_extra = (w_ref * w_work) if use_weights else np.ones_like(y_ref)

        fit_background = bool(cfg.get("fit_background", False))

        # Initial guess: scale via simple weighted dot-product, background 0
        # (kept simple; robust solver will refine)
        if fit_background:
            # Solve y_ref ≈ s*y_work + b (unweighted seed)
            X = np.column_stack([y_work, np.ones_like(y_work)])
            beta0, *_ = np.linalg.lstsq(X, y_ref, rcond=None)
            x0 = beta0.astype(float)
        else:
            denom = float(np.dot(y_work, y_work)) or 1.0
            s0 = float(np.dot(y_ref, y_work) / denom)
            x0 = np.array([s0], dtype=float)

        res = least_squares(
            _residuals,
            x0=x0,
            loss=str(cfg.get("robust_loss", "huber")),
            f_scale=float(cfg.get("robust_fscale", 1.0)),
        )

        p = res.x
        scale = float(p[0])
        background = float(p[1]) if fit_background else 0.0

        # Parameter covariance estimate from Jacobian
        # cov ≈ s_sq * inv(J^T J), s_sq = RSS/dof
        J = res.jac
        n = J.shape[0]
        k = J.shape[1]
        dof = max(1, n - k)
        rss = float(np.sum(res.fun * res.fun))
        s_sq = rss / dof

        try:
            JTJ = J.T @ J
            cov = s_sq * np.linalg.inv(JTJ)
            sig_params = np.sqrt(np.diag(cov))
            scale_sigma = float(sig_params[0])
            bg_sigma = float(sig_params[1]) if fit_background else float("nan")
        except np.linalg.LinAlgError:
            scale_sigma = float("nan")
            bg_sigma = float("nan")

        # Store outputs on WORKING databundle (compute-only)
        scale_out_key = str(cfg.get("scale_output_key", "scale_factor"))
        bg_out_key = str(cfg.get("background_output_key", "scale_background"))

        working_db[scale_out_key] = BaseData(
            signal=np.array([scale], dtype=float),
            units="dimensionless",
            uncertainties={"propagate_to_all": np.array([scale_sigma], dtype=float)},
            rank_of_data=0,
        )

        if fit_background:
            working_db[bg_out_key] = BaseData(
                signal=np.array([background], dtype=float),
                units=reference_y_bd.units,
                uncertainties={"propagate_to_all": np.array([bg_sigma], dtype=float)},
                rank_of_data=0,
            )

        return {working_key: working_db}
