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
__version__ = "20260929.1"

from pathlib import Path
from typing import Dict

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.scaling import (
    DependentData1D,
    fit_lognormal_scale_factor_1d,
    fit_scale_factor_1d,
    prepare_scale_fit_data,
)

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


def _extract_dependent(bd: BaseData, uncertainty_key: str | None = None) -> DependentData1D:
    if bd.rank_of_data != 1:
        raise ValueError("Dependent BaseData must be rank-1.")

    y = bd.signal.squeeze()
    if y.ndim != 1:
        raise ValueError("Dependent signal must be 1D.")

    if uncertainty_key is None:
        sigma = np.asarray(_combined_sigma(bd), dtype=float)
    else:
        if uncertainty_key not in bd.uncertainties:
            raise KeyError(f"Uncertainty key {uncertainty_key!r} is not present in the dependent BaseData.")
        sigma = np.asarray(bd.uncertainties[uncertainty_key], dtype=float)
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
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "Two processing keys: working curve then reference curve.",
            },
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
            "fit_model": {
                "type": str,
                "default": "normal",
                "doc": "Scale estimator: normal or lognormal.",
            },
            "uncertainty_weight_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": (
                    "Named propagated uncertainty component used for weighting on both curves. "
                    "Required for lognormal fitting; normal fitting combines components only when this is None."
                ),
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
        step_keywords=["scale", "calibration", "lognormal", "1D"],
        step_doc="Compute a normal robust-fit or uncertainty-weighted lognormal scale between two 1D curves.",
        step_reference="DOI 10.1107/S1600577513030117",
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if len(processing_keys) != 2:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})
        work_key, reference_key = processing_keys
        signal_key = str(self.configuration.get("signal_key", "signal"))
        axis_key = str(self.configuration.get("independent_axis_key", "Q"))
        scale_key = str(self.configuration.get("scale_output_key", "scale_factor"))
        writes = {f"{work_key}.{scale_key}"}
        if bool(self.configuration.get("fit_background", False)):
            background_key = str(self.configuration.get("background_output_key", "scale_background"))
            writes.add(f"{work_key}.{background_key}")
        return ProcessStepDependencies(
            processing_reads={
                f"{work_key}.{signal_key}",
                f"{work_key}.{axis_key}",
                f"{reference_key}.{signal_key}",
                f"{reference_key}.{axis_key}",
            },
            processing_writes=writes,
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

        fit_model = str(cfg.get("fit_model", "normal")).strip().lower()
        if fit_model not in {"normal", "lognormal"}:
            raise ValueError("FindScaleFactor1D fit_model must be 'normal' or 'lognormal'.")
        uncertainty_weight_key = cfg.get("uncertainty_weight_key")
        if uncertainty_weight_key is not None:
            uncertainty_weight_key = str(uncertainty_weight_key)
        if fit_model == "lognormal" and not uncertainty_weight_key:
            raise ValueError("FindScaleFactor1D lognormal fitting requires uncertainty_weight_key.")

        dep_work = _extract_dependent(y_work_bd, uncertainty_weight_key)
        dep_ref = _extract_dependent(y_ref_bd, uncertainty_weight_key)

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
        if fit_model == "lognormal":
            if fit_background:
                raise ValueError("FindScaleFactor1D lognormal fitting does not support fit_background.")
            fit_result = fit_lognormal_scale_factor_1d(fit_data)
        else:
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
