# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["OpenAI ChatGPT"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "17/02/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.lake_desmearing import LakeDesmearing


def _make_processing_data(
    q_axis: np.ndarray,
    smeared: np.ndarray,
    sigma: float | np.ndarray | None = None,
    *,
    extra_base_data: dict[str, np.ndarray | float] | None = None,
) -> ProcessingData:
    units_q = ureg.Unit("1/nm")
    axis_bd = BaseData(signal=q_axis, units=units_q, rank_of_data=1)

    uncertainties = {}
    if sigma is not None:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.size == 1:
            sigma_arr = np.full_like(smeared, float(sigma_arr), dtype=float)
        uncertainties = {"sigma": sigma_arr}

    signal_bd = BaseData(
        signal=smeared,
        units=ureg.dimensionless,
        uncertainties=uncertainties,
        rank_of_data=1,
        axes=[axis_bd],
    )

    bundle = DataBundle(
        {
            "signal": signal_bd,
            "Q": axis_bd,
        }
    )

    if extra_base_data:
        for name, values in extra_base_data.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 1:
                arr = np.full_like(q_axis, float(arr), dtype=float)
            elif arr.shape != q_axis.shape:
                raise ValueError(f"Extra BaseData '{name}' must be scalar or match q_axis shape.")
            bundle[name] = BaseData(signal=arr, units=units_q, rank_of_data=1)

    processing_data = ProcessingData()
    processing_data["sample"] = bundle
    return processing_data


def _fwhm(axis: np.ndarray, signal: np.ndarray) -> float:
    half_max = float(np.max(signal)) / 2.0
    indices = np.where(signal >= half_max)[0]
    if indices.size < 2:
        raise ValueError("Signal does not have a well-defined FWHM.")
    return float(axis[indices[-1]] - axis[indices[0]])


def test_lake_desmearing_recovers_true_signal():
    step = LakeDesmearing(io_sources=IoSources())

    q = np.linspace(0.005, 0.3, 120)
    true_signal = np.exp(-(((q - 0.05) / 0.025) ** 2)) + 0.1 * np.cos(50 * q)
    primary_top = 0.0012
    primary_bottom = 0.002
    secondary_top = 0.0008
    secondary_bottom = 0.0014

    kernel, _, _ = LakeDesmearing._build_resolution_kernel(
        q,
        np.full_like(q, primary_top),
        np.full_like(q, primary_bottom),
        np.full_like(q, secondary_top),
        np.full_like(q, secondary_bottom),
    )
    smeared = LakeDesmearing.smear_signal(true_signal, kernel)

    processing_data = _make_processing_data(q, smeared)

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["sample"],
            "signal_key": "signal",
            "axis_key": "Q",
            "resolution_top_halfwidth_value": primary_top,
            "resolution_halfwidth_value": primary_bottom,
            "secondary_resolution_top_halfwidth_value": secondary_top,
            "secondary_resolution_halfwidth_value": secondary_bottom,
            "output_signal_key": "desmeared_signal",
            "clip_non_negative": False,
            "regularization_lambda": 1e-8,
        }
    )

    result = step.calculate()

    assert "sample" in result
    desmeared_bd = result["sample"]["desmeared_signal"]
    assert desmeared_bd.units == ureg.dimensionless

    assert_allclose(desmeared_bd.signal, true_signal, rtol=5e-3, atol=5e-4)


def test_lake_desmearing_uncertainty_propagation_matches_theory():
    step = LakeDesmearing(io_sources=IoSources())

    q = np.linspace(0.01, 0.25, 100)
    true_signal = np.sin(6 * q) ** 2 + 0.05
    primary_top = 0.0010
    primary_bottom = 0.0015
    secondary_top = 0.0006
    secondary_bottom = 0.0009

    kernel, _, _ = LakeDesmearing._build_resolution_kernel(
        q,
        np.full_like(q, primary_top),
        np.full_like(q, primary_bottom),
        np.full_like(q, secondary_top),
        np.full_like(q, secondary_bottom),
    )
    smeared = LakeDesmearing.smear_signal(true_signal, kernel)

    sigma_meas = 0.02
    processing_data = _make_processing_data(q, smeared, sigma=sigma_meas)

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["sample"],
            "resolution_top_halfwidth_value": primary_top,
            "resolution_halfwidth_value": primary_bottom,
            "secondary_resolution_top_halfwidth_value": secondary_top,
            "secondary_resolution_halfwidth_value": secondary_bottom,
            "output_signal_key": "desmeared_signal",
            "clip_non_negative": False,
            "regularization_lambda": 5e-8,
        }
    )

    result = step.calculate()
    desmeared_bd = result["sample"]["desmeared_signal"]

    desmear_operator = LakeDesmearing._compute_desmearing_operator(kernel, 5e-8)
    sigma_expected = np.sqrt((desmear_operator**2) @ np.full(q.shape, sigma_meas**2))
    assert "desmeared" in desmeared_bd.uncertainties
    assert_allclose(desmeared_bd.uncertainties["desmeared"], sigma_expected, rtol=1e-12, atol=1e-12)


def test_lake_desmearing_requires_halfwidth_configuration():
    step = LakeDesmearing(io_sources=IoSources())
    q = np.linspace(0.01, 0.05, 10)
    smeared = np.exp(-q)
    processing_data = _make_processing_data(q, smeared)

    step.processing_data = processing_data
    step.configuration.update({"with_processing_keys": ["sample"]})

    with pytest.raises(ValueError):
        _ = step.calculate()


def test_lake_desmearing_accepts_secondary_halfwidth_key():
    step = LakeDesmearing(io_sources=IoSources())

    q = np.linspace(0.02, 0.25, 80)
    true_signal = 0.8 * np.exp(-(((q - 0.08) / 0.02) ** 2)) + 0.2
    primary_top = 0.0008
    primary_bottom = 0.0018
    secondary_top = np.linspace(0.0004, 0.0012, q.size)
    secondary_bottom = np.linspace(0.0007, 0.0016, q.size)

    kernel, _, _ = LakeDesmearing._build_resolution_kernel(
        q,
        np.full_like(q, primary_top),
        np.full_like(q, primary_bottom),
        secondary_top,
        secondary_bottom,
    )
    smeared = LakeDesmearing.smear_signal(true_signal, kernel)

    processing_data = _make_processing_data(
        q,
        smeared,
        extra_base_data={
            "secondary_top_hw": secondary_top,
            "secondary_bottom_hw": secondary_bottom,
        },
    )

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["sample"],
            "resolution_top_halfwidth_value": primary_top,
            "resolution_halfwidth_value": primary_bottom,
            "secondary_resolution_top_halfwidth_key": "secondary_top_hw",
            "secondary_resolution_halfwidth_key": "secondary_bottom_hw",
            "output_signal_key": "desmeared_signal",
            "clip_non_negative": False,
            "regularization_lambda": 1e-8,
        }
    )

    result = step.calculate()
    desmeared_bd = result["sample"]["desmeared_signal"]

    assert_allclose(desmeared_bd.signal, true_signal, rtol=5e-3, atol=5e-4)


def test_smear_signal_broadens_gaussian():
    q = np.linspace(-0.1, 0.1, 801)
    true_signal = np.exp(-((q / 0.01) ** 2))

    primary_top = 0.0015
    primary_bottom = 0.0025
    secondary_top = 0.0009
    secondary_bottom = 0.0015

    kernel, _, _ = LakeDesmearing._build_resolution_kernel(
        q,
        np.full_like(q, primary_top),
        np.full_like(q, primary_bottom),
        np.full_like(q, secondary_top),
        np.full_like(q, secondary_bottom),
    )
    smeared = LakeDesmearing.smear_signal(true_signal, kernel)

    assert smeared.max() < true_signal.max()
    assert _fwhm(q, smeared) > _fwhm(q, true_signal)
    assert_allclose(np.trapezoid(smeared, q), np.trapezoid(true_signal, q), rtol=1e-6, atol=1e-8)
