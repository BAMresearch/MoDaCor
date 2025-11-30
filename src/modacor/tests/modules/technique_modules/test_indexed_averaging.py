# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Tests for the IndexedAverager processing step.

We test:
- Basic 1D averaging with a simple, hand-crafted pixel_index map.
- Correct handling of Mask and pixel_index == -1.
- Uncertainty propagation from per-pixel uncertainties to bin-mean.
- SEM ("SEM" key) behaviour for signal.
- Integration-style test using prepare_execution() + calculate().
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.indexed_averaging import IndexedAverager

# ---------------------------------------------------------------------------
# Small helpers to build simple test databundles
# ---------------------------------------------------------------------------


def make_1d_bundle_basic():
    """
    Simple 1D test bundle with 4 pixels and 2 bins.

    pixel_index:
      - pixels 0, 1 -> bin 0
      - pixels 2, 3 -> bin 1
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    Q = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    Psi = np.array([0.0, 0.0, np.pi, np.pi], dtype=float)

    signal_bd = BaseData(
        signal=signal,
        units=ureg.dimensionless,
        rank_of_data=1,
    )
    Q_bd = BaseData(
        signal=Q,
        units=ureg.dimensionless,
        rank_of_data=1,
    )
    Psi_bd = BaseData(
        signal=Psi,
        units=ureg.radian,
        rank_of_data=1,
    )
    pixel_index = np.array([0, 0, 1, 1], dtype=float)
    pix_bd = BaseData(
        signal=pixel_index,
        units=ureg.dimensionless,
        rank_of_data=1,
    )

    bundle = DataBundle(
        {
            "signal": signal_bd,
            "Q": Q_bd,
            "Psi": Psi_bd,
            "pixel_index": pix_bd,
        }
    )
    return bundle


def make_1d_bundle_with_mask_and_uncertainties():
    """
    1D test bundle with:
    - Mask on one pixel.
    - Per-pixel uncertainties on signal and Q.
    - 2 bins as before.

    pixel_index:
      - pixel 0 -> bin 0
      - pixel 1 -> bin 0 (masked)
      - pixel 2 -> bin 1
      - pixel 3 -> -1 (ignored)
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    Q = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    Psi = np.array([0.0, 0.0, np.pi, np.pi], dtype=float)

    # uncertainties: "sigma" for signal, "dq" for Q
    sigma_signal = np.array([0.1, 0.1, 0.2, 0.2], dtype=float)
    dq = np.array([0.5, 0.5, 1.0, 1.0], dtype=float)

    signal_bd = BaseData(
        signal=signal,
        units=ureg.dimensionless,
        uncertainties={"sigma": sigma_signal},
        rank_of_data=1,
    )
    Q_bd = BaseData(
        signal=Q,
        units=ureg.dimensionless,
        uncertainties={"dq": dq},
        rank_of_data=1,
    )
    Psi_bd = BaseData(
        signal=Psi,
        units=ureg.radian,
        rank_of_data=1,
    )

    pixel_index = np.array([0, 0, 1, -1], dtype=float)
    pix_bd = BaseData(
        signal=pixel_index,
        units=ureg.dimensionless,
        rank_of_data=1,
    )

    # Mask out pixel 1; booleans, True = masked
    mask_arr = np.array([False, True, False, False], dtype=bool)
    mask_bd = BaseData(
        signal=mask_arr,
        units=ureg.dimensionless,
        rank_of_data=1,
    )

    bundle = DataBundle(
        {
            "signal": signal_bd,
            "Q": Q_bd,
            "Psi": Psi_bd,
            "pixel_index": pix_bd,
            "Mask": mask_bd,
        }
    )
    return bundle


def make_2d_bundle_basic():
    """
    Simple 2D test bundle (2 x 3) with 3 bins.

    Layout (row-major):

        signal =
            [[1, 2, 3],
             [4, 5, 6]]

        Q =
            [[10, 20, 30],
             [40, 50, 60]]

        Psi = all zeros (for a trivial circular mean)

        pixel_index =
            [[0, 0, 1],
             [1, 2, 2]]

    So:
        bin 0: pixels (0,0), (0,1) → signal 1,2; Q 10,20
        bin 1: pixels (0,2), (1,0) → signal 3,4; Q 30,40
        bin 2: pixels (1,1), (1,2) → signal 5,6; Q 50,60
    """
    signal = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=float,
    )
    Q = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dtype=float,
    )
    # Psi all zeros in radians → mean Psi per bin is trivially 0
    Psi = np.zeros_like(signal, dtype=float)

    # Simple per-pixel uncertainties (same everywhere)
    sigma_signal = np.full_like(signal, 0.1, dtype=float)
    dq = np.full_like(signal, 0.5, dtype=float)

    signal_bd = BaseData(
        signal=signal,
        units=ureg.dimensionless,
        uncertainties={"sigma": sigma_signal},
        rank_of_data=2,
    )
    Q_bd = BaseData(
        signal=Q,
        units=ureg.dimensionless,
        uncertainties={"dq": dq},
        rank_of_data=2,
    )
    Psi_bd = BaseData(
        signal=Psi,
        units=ureg.radian,
        rank_of_data=2,
    )

    pixel_index = np.array(
        [[0, 0, 1], [1, 2, 2]],
        dtype=float,
    )
    pix_bd = BaseData(
        signal=pixel_index,
        units=ureg.dimensionless,
        rank_of_data=2,
    )

    bundle = DataBundle(
        {
            "signal": signal_bd,
            "Q": Q_bd,
            "Psi": Psi_bd,
            "pixel_index": pix_bd,
        }
    )
    return bundle


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_indexedaverager_1d_basic_unweighted_mean():
    """
    For a simple 1D bundle without mask and with equal weights:
    - Bin 0 averages pixels 0 and 1.
    - Bin 1 averages pixels 2 and 3.
    We check signal, Q, Psi means and axis wiring.
    """
    step = IndexedAverager(io_sources=IoSources())

    processing_data = ProcessingData()
    processing_data["bundle"] = make_1d_bundle_basic()

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["bundle"],
            "output_processing_key": None,
            "averaging_direction": "azimuthal",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": False,
            "uncertainty_weight_key": None,
        }
    )

    step.prepare_execution()
    out = step.calculate()

    assert "bundle" in out
    db_out = out["bundle"]

    assert "signal" in db_out
    assert "Q" in db_out
    assert "Psi" in db_out

    sig_1d = db_out["signal"]
    Q_1d = db_out["Q"]
    Psi_1d = db_out["Psi"]

    # 2 bins
    assert sig_1d.signal.shape == (2,)
    assert Q_1d.signal.shape == (2,)
    assert Psi_1d.signal.shape == (2,)

    # Simple averages:
    # bin 0: (1+2)/2 = 1.5, (10+20)/2 = 15, Psi = 0
    # bin 1: (3+4)/2 = 3.5, (30+40)/2 = 35, Psi = pi
    assert_allclose(sig_1d.signal, np.array([1.5, 3.5]), rtol=1e-12, atol=1e-12)
    assert_allclose(Q_1d.signal, np.array([15.0, 35.0]), rtol=1e-12, atol=1e-12)
    assert_allclose(Psi_1d.signal, np.array([0.0, np.pi]), rtol=1e-12, atol=1e-12)

    # Axis wiring: for averaging_direction="azimuthal", signal axes should reference Q
    assert len(sig_1d.axes) == 1
    assert sig_1d.axes[0] is Q_1d


def test_indexedaverager_mask_and_negative_index():
    """
    Check that:
    - Masked pixels are excluded.
    - Pixels with pixel_index == -1 are excluded.
    - Means are computed only from remaining pixels.
    """
    step = IndexedAverager(io_sources=IoSources())

    processing_data = ProcessingData()
    processing_data["bundle"] = make_1d_bundle_with_mask_and_uncertainties()

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["bundle"],
            "output_processing_key": None,
            "averaging_direction": "azimuthal",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": False,
            "uncertainty_weight_key": None,
        }
    )

    step.prepare_execution()
    out = step.calculate()

    db_out = out["bundle"]
    sig_1d = db_out["signal"]
    Q_1d = db_out["Q"]
    Psi_1d = db_out["Psi"]

    # Valid pixels:
    # - pixel 0: index 0, not masked -> bin 0
    # - pixel 1: index 0, masked      -> ignored
    # - pixel 2: index 1, not masked  -> bin 1
    # - pixel 3: index -1             -> ignored
    # So:
    # bin 0: signal=1, Q=10, Psi=0
    # bin 1: signal=3, Q=30, Psi=pi
    assert_allclose(sig_1d.signal, np.array([1.0, 3.0]), rtol=1e-12, atol=1e-12)
    assert_allclose(Q_1d.signal, np.array([10.0, 30.0]), rtol=1e-12, atol=1e-12)
    assert_allclose(Psi_1d.signal, np.array([0.0, np.pi]), rtol=1e-12, atol=1e-12)


def test_indexedaverager_uncertainty_propagation_and_sem():
    """
    For the basic 1D setup without mask, with a simple per-pixel uncertainty:
    - Check that per-bin propagated uncertainties on signal match the expected
      sqrt(sum(w^2 sigma^2)) / sum_w for equal weights.
    - Check that SEM ("SEM") is present and behaves as expected.
    """
    step = IndexedAverager(io_sources=IoSources())

    # Start from the basic bundle and add a sigma uncertainty
    bundle = make_1d_bundle_basic()
    sigma_signal = np.array([0.1, 0.1, 0.2, 0.2], dtype=float)
    bundle["signal"].uncertainties = {"sigma": sigma_signal}

    processing_data = ProcessingData()
    processing_data["bundle"] = bundle

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["bundle"],
            "output_processing_key": None,
            "averaging_direction": "azimuthal",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": False,
            "uncertainty_weight_key": None,
        }
    )

    step.prepare_execution()
    out = step.calculate()

    sig_1d = out["bundle"]["signal"]
    assert "sigma" in sig_1d.uncertainties
    assert "SEM" in sig_1d.uncertainties

    # Expected propagated sigma on bin means:
    # bin 0: pixels 0,1, sigma=0.1 each
    #   sigma_mean0 = sqrt(0.1^2 + 0.1^2) / 2 = 0.1 / sqrt(2)
    # bin 1: pixels 2,3, sigma=0.2 each
    #   sigma_mean1 = sqrt(0.2^2 + 0.2^2) / 2 = 0.2 / sqrt(2)
    expected_sigma = np.array([0.1 / np.sqrt(2.0), 0.2 / np.sqrt(2.0)], dtype=float)

    assert_allclose(sig_1d.uncertainties["sigma"], expected_sigma, rtol=1e-12, atol=1e-12)

    # SEM from scatter:
    # For basic bundle signal = [1,2,3,4] and means [1.5, 3.5]
    # bin 0: dev = [-0.5, +0.5], sum(dev^2) = 0.5, sum_w=2 -> var_spread=0.25
    # bin 1: same pattern
    # Effective N_eff = (sum_w^2 / sum_w2) = 4/2 = 2
    # sem = sqrt(var_spread / N_eff) = sqrt(0.25/2) = sqrt(0.125)
    expected_sem = np.full(2, np.sqrt(0.125), dtype=float)

    sem = sig_1d.uncertainties["SEM"]
    assert_allclose(sem, expected_sem, rtol=1e-12, atol=1e-12)


def test_indexedaverager_raises_on_missing_uncertainty_weight_key():
    """
    If use_signal_uncertainty_weights=True but uncertainty_weight_key is None,
    a ValueError should be raised.
    """
    step = IndexedAverager(io_sources=IoSources())

    processing_data = ProcessingData()
    processing_data["bundle"] = make_1d_bundle_basic()

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["bundle"],
            "output_processing_key": None,
            "averaging_direction": "azimuthal",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": True,
            "uncertainty_weight_key": None,
        }
    )

    step.prepare_execution()
    with pytest.raises(ValueError):
        _ = step.calculate()


def test_indexedaverager_prepare_and_calculate_integration_radial_axis():
    """
    Integration-style test:
    - Build a small 1D bundle.
    - Run prepare_execution() and calculate().
    - Check that for averaging_direction='radial', signal.axes[0] references Psi.
    """
    step = IndexedAverager(io_sources=IoSources())

    processing_data = ProcessingData()
    processing_data["bundle"] = make_1d_bundle_basic()

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["bundle"],
            "output_processing_key": None,
            "averaging_direction": "radial",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": False,
            "uncertainty_weight_key": None,
        }
    )

    step.prepare_execution()
    out = step.calculate()

    db_out = out["bundle"]
    sig_1d = db_out["signal"]
    Psi_1d = db_out["Psi"]

    assert len(sig_1d.axes) == 1
    # For radial averaging, we expect signal.axes[0] to be Psi
    assert sig_1d.axes[0] is Psi_1d


def test_indexedaverager_2d_basic_unweighted_mean():
    """
    2D main-use-case test:

    - 2x3 signal, Q, Psi, pixel_index (3 bins).
    - No mask, equal weights.
    - Check binned 1D means for signal, Q, Psi.
    - Check propagated signal uncertainty shape and values.
    """
    step = IndexedAverager(io_sources=IoSources())

    processing_data = ProcessingData()
    processing_data["img"] = make_2d_bundle_basic()

    step.processing_data = processing_data
    step.configuration.update(
        {
            "with_processing_keys": ["img"],
            "output_processing_key": None,
            "averaging_direction": "azimuthal",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": False,
            "uncertainty_weight_key": None,
        }
    )

    step.prepare_execution()
    out = step.calculate()

    assert "img" in out
    db_out = out["img"]

    sig_1d = db_out["signal"]
    Q_1d = db_out["Q"]
    Psi_1d = db_out["Psi"]

    # We expect 3 bins
    assert sig_1d.signal.shape == (3,)
    assert Q_1d.signal.shape == (3,)
    assert Psi_1d.signal.shape == (3,)

    # Means (from the docstring of make_2d_bundle_basic):
    # bin 0: (1, 2) -> 1.5;  (10, 20) -> 15
    # bin 1: (3, 4) -> 3.5;  (30, 40) -> 35
    # bin 2: (5, 6) -> 5.5;  (50, 60) -> 55
    expected_signal = np.array([1.5, 3.5, 5.5], dtype=float)
    expected_Q = np.array([15.0, 35.0, 55.0], dtype=float)
    expected_Psi = np.zeros(3, dtype=float)  # all Psi were zero

    assert_allclose(sig_1d.signal, expected_signal, rtol=1e-12, atol=1e-12)
    assert_allclose(Q_1d.signal, expected_Q, rtol=1e-12, atol=1e-12)
    assert_allclose(Psi_1d.signal, expected_Psi, rtol=1e-12, atol=1e-12)

    # For averaging_direction="azimuthal", axis should reference Q
    assert len(sig_1d.axes) == 1
    assert sig_1d.axes[0] is Q_1d

    # Uncertainty propagation sanity check:
    # signal sigma per pixel = 0.1 everywhere, equal weights.
    # Each bin has 2 pixels → sigma_mean = 0.1 / sqrt(2) for all bins.
    assert "sigma" in sig_1d.uncertainties
    sigma_binned = sig_1d.uncertainties["sigma"]
    expected_sigma = np.full(3, 0.1 / np.sqrt(2.0), dtype=float)

    assert_allclose(sigma_binned, expected_sigma, rtol=1e-12, atol=1e-12)
