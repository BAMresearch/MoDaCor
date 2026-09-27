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
__version__ = "20260927.3"

from pathlib import Path
from typing import Dict

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.scaling import DependentData1D, fit_scale_factor_1d, prepare_scale_fit_data

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

    y = bd.signal.squeeze()
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


# -------------------------------------------------------------------------
# Main ProcessStep
# -------------------------------------------------------------------------


class FindScaleFactor1D(ProcessStep):
    documentation = ProcessStepDescriber(
        calling_name="Scale 1D curve to reference (compute-only)",
        calling_id="FindScaleFactor1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Q"],
        modifies={
            "scale_factor": ["signal", "uncertainties", "units"],
            "scale_background": ["signal", "uncertainties", "units"],
        },
        arguments={
            "signal_key": {
                "type": str,
                "default": "signal",
                "doc": (
                    "BaseData key for the dependent variable signal. Working and reference "
                    "signals must have compatible units; fitting uses the reference units."
                ),
            },
            "independent_axis_key": {
                "type": str,
                "default": "Q",
                "doc": "BaseData key for the independent axis.",
            },
            "scale_output_key": {
                "type": str,
                "default": "scale_factor",
                "doc": "BaseData key to store the scale factor output.",
            },
            "background_output_key": {
                "type": str,
                "default": "scale_background",
                "doc": "BaseData key to store the fitted background output.",
            },
            "fit_background": {
                "type": bool,
                "default": False,
                "doc": "Whether to fit a constant background offset.",
            },
            "fit_min_val": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Minimum x-value for the fit (in fit_val_units).",
            },
            "fit_max_val": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Maximum x-value for the fit (in fit_val_units).",
            },
            "fit_val_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for fit_min_val/fit_max_val if provided.",
            },
            "require_overlap": {
                "type": bool,
                "default": True,
                "doc": "Require overlapping x-range between reference and work data.",
            },
            "interpolation_kind": {
                "type": str,
                "default": "linear",
                "doc": "Interpolation kind passed to scipy/numpy interpolation.",
            },
            "robust_loss": {
                "type": str,
                "default": "huber",
                "doc": "Robust loss function name for the fit.",
            },
            "robust_fscale": {
                "type": (float, int),
                "default": 1.0,
                "doc": "Robust loss scale parameter.",
            },
            "use_basedata_weights": {
                "type": bool,
                "default": True,
                "doc": "Use BaseData weights when fitting.",
            },
        },
        step_keywords=["scale", "calibration", "1D"],
        step_doc="Compute scale factor between two 1D curves using robust least squares.",
    )

    def calculate(self) -> Dict[str, DataBundle]:
        cfg = self.configuration
        keys = self._normalised_processing_keys()
        if len(keys) != 2:
            raise ValueError("FindScaleFactor1D requires exactly two processing keys in 'with_processing_keys'.")
        work_key, ref_key = keys

        sig_key = cfg.get("signal_key", "signal")
        axis_key = cfg.get("independent_axis_key", "Q")

        work_db = self.processing_data[work_key]
        ref_db = self.processing_data[ref_key]

        y_work_bd = work_db[sig_key].copy(with_axes=True)
        y_ref_bd = ref_db[sig_key].copy(with_axes=True)

        if y_work_bd.units != y_ref_bd.units:
            y_work_bd.to_units(y_ref_bd.units)

        x_work_bd = work_db[axis_key].copy(with_axes=False)
        x_ref_bd = ref_db[axis_key].copy(with_axes=False)

        if x_work_bd.units != x_ref_bd.units:
            x_work_bd.to_units(x_ref_bd.units)

        x_work = x_work_bd.signal.squeeze()
        x_ref = x_ref_bd.signal.squeeze()

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

        fit_data = prepare_scale_fit_data(
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
        fit_result = fit_scale_factor_1d(
            fit_data,
            fit_background=fit_background,
            robust_loss=cfg.get("robust_loss", "huber"),
            robust_fscale=float(cfg.get("robust_fscale", 1.0)),
        )

        out_key = cfg.get("scale_output_key", "scale_factor")
        work_db[out_key] = BaseData(
            signal=np.array([fit_result.scale]),
            units="dimensionless",
            uncertainties={"propagate_to_all": np.array([fit_result.scale_sigma])},
            rank_of_data=0,
        )

        if fit_background:
            bg_key = cfg.get("background_output_key", "scale_background")
            work_db[bg_key] = BaseData(
                signal=np.array([fit_result.background]),
                units=y_ref_bd.units,
                uncertainties={"propagate_to_all": np.array([fit_result.background_sigma])},
                rank_of_data=0,
            )

        return {work_key: work_db}
