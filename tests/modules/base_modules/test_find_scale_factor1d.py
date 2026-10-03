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
from modacor.dataclasses.process_step import ProcessStepDependencies
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


def test_find_scale_factor_converts_dependent_data_and_uncertainties_to_reference_units():
    x = np.linspace(1.0, 10.0, 200)
    physical_work_m = 0.3 + np.exp(-x / 4.0)
    true_scale = 2.5

    work_signal_cm = physical_work_m * 100.0
    work_sigma_cm = np.full_like(x, 1.0)
    reference_signal_m = true_scale * physical_work_m

    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(
        x,
        work_signal_cm,
        y_units="cm",
        sigma_y=work_sigma_cm,
    )
    pd["ref"] = _make_curve_bundle(
        x,
        reference_signal_m,
        y_units="m",
        sigma_y=0.01,
    )

    _run_step(
        pd,
        {
            "with_processing_keys": ["work", "ref"],
            "fit_background": False,
            "robust_loss": "linear",
            "use_basedata_weights": True,
        },
    )

    assert float(pd["work"]["scale_factor"].signal.item()) == pytest.approx(true_scale, rel=1e-3)
    assert pd["work"]["scale_factor"].units.is_compatible_with("dimensionless")

    # Unit normalization is performed on a copy used for fitting.
    assert pd["work"]["signal"].units.is_compatible_with("cm")
    np.testing.assert_allclose(pd["work"]["signal"].signal, work_signal_cm)
    np.testing.assert_allclose(pd["work"]["signal"].uncertainties["propagate_to_all"], work_sigma_cm)


def test_find_scale_factor_rejects_incompatible_dependent_units_before_mutation():
    x = np.linspace(1.0, 10.0, 20)
    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x, np.ones_like(x), y_units="s")
    pd["ref"] = _make_curve_bundle(x, np.ones_like(x), y_units="m")

    with pytest.raises(ValueError, match="Units are not compatible"):
        _run_step(pd, {"with_processing_keys": ["work", "ref"]})

    assert "scale_factor" not in pd["work"]


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


def test_find_scale_factor_lognormal_uses_selected_uncertainty_component():
    x_work = np.linspace(1.0, 10.0, 200)
    x_reference = np.linspace(1.5, 9.5, 170)
    work = 1.0 + np.exp(-x_work / 4.0)
    true_scale = 3.25
    reference = true_scale * (1.0 + np.exp(-x_reference / 4.0))
    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x_work, work, sigma_y=0.02)
    pd["reference"] = _make_curve_bundle(x_reference, reference, sigma_y=0.03)

    _run_step(
        pd,
        {
            "with_processing_keys": ["work", "reference"],
            "fit_model": "lognormal",
            "uncertainty_weight_key": "propagate_to_all",
            "fit_min_val": 2.0,
            "fit_max_val": 9.0,
            "scale_uncertainty_key": "scale_fit",
            "diagnostic_prefix": "gain_fit",
        },
    )

    scale = pd["work"]["scale_factor"]
    assert float(scale.signal.item()) == pytest.approx(true_scale, rel=1.0e-4)
    assert float(scale.uncertainties["scale_fit"].item()) > 0.0
    assert float(pd["work"]["gain_fit_point_count"].signal.item()) > 2
    assert float(pd["work"]["gain_fit_x_min"].signal.item()) >= 2.0
    assert float(pd["work"]["gain_fit_x_max"].signal.item()) <= 9.0
    assert float(pd["work"]["gain_fit_reduced_chi_square"].signal.item()) >= 0.0


def test_find_scale_factor_lognormal_requires_explicit_uncertainty_key():
    x = np.linspace(1.0, 2.0, 5)
    pd = ProcessingData()
    pd["work"] = _make_curve_bundle(x, np.ones(5))
    pd["reference"] = _make_curve_bundle(x, np.ones(5))

    with pytest.raises(ValueError, match="requires uncertainty_weight_key"):
        _run_step(
            pd,
            {"with_processing_keys": ["work", "reference"], "fit_model": "lognormal"},
        )


def test_find_scale_factor_dependency_contract_is_exact():
    step = FindScaleFactor1D(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["work", "reference"],
            "signal_key": "intensity",
            "independent_axis_key": "q",
            "scale_output_key": "gain",
            "fit_background": True,
            "background_output_key": "offset",
            "diagnostic_prefix": "fit",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"work.intensity", "work.q", "reference.intensity", "reference.q"},
        processing_writes={
            "work.gain",
            "work.offset",
            "work.fit_point_count",
            "work.fit_x_min",
            "work.fit_x_max",
            "work.fit_reduced_chi_square",
        },
    )
