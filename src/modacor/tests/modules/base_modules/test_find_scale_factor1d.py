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

import numpy as np
import pytest

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.find_scale_factor1d import FindScaleFactor1D


def _make_1d_bd(
    arr: np.ndarray,
    *,
    units: str,
    sigma: float | np.ndarray = 0.02,
    weights: float | np.ndarray = 1.0,
    rank_of_data: int = 1,
) -> BaseData:
    arr = np.asarray(arr, dtype=float)

    if np.isscalar(sigma):
        sig_arr = np.full_like(arr, float(sigma), dtype=float)
    else:
        sig_arr = np.asarray(sigma, dtype=float)
        if sig_arr.size == 1:
            sig_arr = np.full_like(arr, float(sig_arr.ravel()[0]), dtype=float)

    if np.isscalar(weights):
        w_arr = np.full_like(arr, float(weights), dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.size == 1:
            w_arr = np.full_like(arr, float(w_arr.ravel()[0]), dtype=float)

    return BaseData(
        signal=arr,
        units=units,
        uncertainties={"propagate_to_all": sig_arr} if rank_of_data == 1 else {},
        weights=w_arr,
        axes=[],
        rank_of_data=rank_of_data,
    )


def _make_curve_bundle(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_units: str = "1/nm",
    y_units: str = "dimensionless",
    sigma_y: float | np.ndarray = 0.02,
    weights_y: float | np.ndarray = 1.0,
) -> DataBundle:
    """
    Build a DataBundle compatible with FindScaleFactor1D's new contract:
      - independent axis as databundle key "Q"
      - dependent as databundle key "signal"
    """
    x_bd = BaseData(signal=np.asarray(x, dtype=float), units=x_units, rank_of_data=1)
    y_bd = BaseData(
        signal=np.asarray(y, dtype=float),
        units=y_units,
        uncertainties=(
            {"propagate_to_all": np.full_like(y, float(sigma_y))}
            if np.isscalar(sigma_y)
            else {"propagate_to_all": np.asarray(sigma_y, dtype=float)}
        ),
        weights=np.full_like(y, float(weights_y)) if np.isscalar(weights_y) else np.asarray(weights_y, dtype=float),
        rank_of_data=1,
    )
    return DataBundle({"signal": y_bd, "Q": x_bd})


def _run_step(pd: ProcessingData, cfg: dict) -> None:
    step = FindScaleFactor1D(io_sources=IoSources())
    step.modify_config_by_dict(cfg)
    step.execute(pd)


def test_find_scale_factor_scale_only_perfect_overlap():
    x = np.linspace(0.0, 10.0, 500)
    y_work = np.sin(x) + 0.2
    true_scale = 2.5
    y_ref = true_scale * y_work

    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x, y_work, sigma_y=0.01)
    pd["ref"] = _make_curve_bundle(x, y_ref, sigma_y=0.01)

    _run_step(
        pd,
        {
            "with_processing_keys": ["work", "ref"],
            "fit_background": False,
            "fit_min_val": 2.0,
            "fit_max_val": 8.0,
            "fit_val_units": "1/nm",
            "require_overlap": True,
            "robust_loss": "linear",
            "use_basedata_weights": True,
            "independent_axis_key": "Q",
            "signal_key": "signal",
        },
    )

    sf_bd = pd["work"]["scale_factor"]
    sf = float(sf_bd.signal.item())
    assert sf == pytest.approx(true_scale, rel=1e-3, abs=1e-3)

    assert "propagate_to_all" in sf_bd.uncertainties
    assert sf_bd.uncertainties["propagate_to_all"].size == 1


def test_find_scale_factor_scale_and_background_mismatched_axes_robust():
    x_w = np.linspace(0.0, 10.0, 700)
    x_r = np.linspace(1.0, 9.0, 400)

    base = np.exp(-0.2 * x_w) + 0.1 * np.cos(3 * x_w)
    true_scale = 1.7
    true_bg = 0.35

    y_work_on_ref = np.interp(x_r, x_w, base)
    y_ref = true_scale * y_work_on_ref + true_bg

    y_ref_noisy = y_ref.copy()
    y_ref_noisy[50] += 5.0
    y_ref_noisy[120] -= 4.0

    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x_w, base, sigma_y=0.02)
    pd["ref"] = _make_curve_bundle(x_r, y_ref_noisy, sigma_y=0.02)

    _run_step(
        pd,
        {
            "with_processing_keys": ["work", "ref"],
            "fit_background": True,
            "fit_min_val": 2.0,
            "fit_max_val": 8.0,
            "fit_val_units": "1/nm",
            "require_overlap": True,
            "interpolation_kind": "linear",
            "robust_loss": "huber",
            "robust_fscale": 1.0,
            "use_basedata_weights": True,
            "independent_axis_key": "Q",
            "signal_key": "signal",
        },
    )

    sf = float(pd["work"]["scale_factor"].signal.item())
    bg = float(pd["work"]["scale_background"].signal.item())

    assert sf == pytest.approx(true_scale, rel=3e-2, abs=3e-2)
    assert bg == pytest.approx(true_bg, rel=3e-2, abs=3e-2)


def test_find_scale_factor_raises_on_no_overlap_when_required():
    x_w = np.linspace(0.0, 1.0, 200)
    x_r = np.linspace(2.0, 3.0, 200)

    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x_w, np.ones_like(x_w), sigma_y=0.01)
    pd["ref"] = _make_curve_bundle(x_r, np.ones_like(x_r) * 2.0, sigma_y=0.01)

    with pytest.raises(ValueError, match="No overlap"):
        _run_step(
            pd,
            {
                "with_processing_keys": ["work", "ref"],
                "require_overlap": True,
                "independent_axis_key": "Q",
                "signal_key": "signal",
            },
        )


def test_find_scale_factor_weights_have_effect():
    x = np.linspace(0.0, 10.0, 600)
    y_work = 0.5 + np.sin(x)
    true_scale = 3.0
    y_ref = true_scale * y_work

    weights_ref = np.ones_like(x)
    weights_ref[x > 5.0] = 0.1

    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x, y_work, sigma_y=0.01, weights_y=1.0)
    pd["ref"] = _make_curve_bundle(x, y_ref, sigma_y=0.01, weights_y=weights_ref)

    _run_step(
        pd,
        {
            "with_processing_keys": ["work", "ref"],
            "fit_background": False,
            "fit_min_val": 0.5,
            "fit_max_val": 9.5,
            "fit_val_units": "1/nm",
            "robust_loss": "linear",
            "use_basedata_weights": True,
            "independent_axis_key": "Q",
            "signal_key": "signal",
        },
    )

    sf = float(pd["work"]["scale_factor"].signal.item())
    assert sf == pytest.approx(true_scale, rel=1e-3, abs=1e-3)
