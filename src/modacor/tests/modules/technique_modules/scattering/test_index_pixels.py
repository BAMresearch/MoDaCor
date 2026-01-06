# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "30/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports
__version__ = "20251130.1"

"""
Tests for the IndexPixels processing step.

We test:
- Basic azimuthal binning in Q (1D case).
- Basic radial binning in Psi (1D case).
- Unit conversion for q_limits_unit.
- Psi wrap-around masking for azimuthal direction.
- A small integration-style test using prepare_execution() + calculate().
"""

import numpy as np

# import pytest
import pint
from numpy.testing import assert_array_equal

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.index_pixels import IndexPixels

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def make_1d_signal_bundle(
    q_values: np.ndarray,
    psi_values: np.ndarray,
    q_unit: pint.Unit = ureg.Unit("1/nm"),
    psi_unit: pint.Unit = ureg.Unit("radian"),
) -> DataBundle:
    """
    Build a minimal 1D databundle with signal, Q, Psi.
    """
    n = q_values.size
    assert psi_values.size == n

    signal_bd = BaseData(
        signal=np.ones(n, dtype=float),
        units=ureg.dimensionless,
        rank_of_data=1,
    )
    q_bd = BaseData(
        signal=np.asarray(q_values, dtype=float),
        units=q_unit,
        rank_of_data=1,
    )
    psi_bd = BaseData(
        signal=np.asarray(psi_values, dtype=float),
        units=psi_unit,
        rank_of_data=1,
    )

    # simple axis metadata (optional, but useful to check propagation)
    axis_bd = BaseData(
        signal=np.arange(n, dtype=float),
        units=ureg.pixel,
        rank_of_data=0,
    )
    signal_bd.axes = [axis_bd]

    bundle = DataBundle(
        {
            "signal": signal_bd,
            "Q": q_bd,
            "Psi": psi_bd,
        }
    )
    return bundle


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_indexpixels_azimuthal_basic_binning_1d_linear():
    """
    Azimuthal direction:
    - Bins along Q (linear bins).
    - Psi ROI is full circle.
    - Check that obvious Q values fall into expected bins.
    """
    # Q values: two points clearly in each bin, others outside range
    q_vals = np.array([0.5, 1.25, 1.75, 2.25, 2.75, 3.5], dtype=float)
    # Psi irrelevant here beyond being finite and inside ROI
    psi_vals = np.zeros_like(q_vals, dtype=float)

    bundle = make_1d_signal_bundle(
        q_values=q_vals,
        psi_values=psi_vals,
        q_unit=ureg.dimensionless,
        psi_unit=ureg.degree,
    )

    processing_data = ProcessingData()
    processing_data["test"] = bundle

    step = IndexPixels(io_sources=IoSources())
    step.processing_data = processing_data
    step._prepared_data = {}

    step.configuration = {
        "with_processing_keys": ["test"],
        "output_processing_key": None,
        # azimuthal: binning along Q
        "averaging_direction": "azimuthal",
        "n_bins": 2,
        "bin_type": "linear",
        "q_min": 1.0,
        "q_max": 3.0,
        "q_limits_unit": None,  # same as Q.units
        # full-circle Psi in degrees
        "psi_min": 0.0,
        "psi_max": 360.0,
        "psi_limits_unit": "degree",
    }

    step.prepare_execution()
    out = step.calculate()

    assert "test" in out
    db = out["test"]
    assert "pixel_index" in db

    pix = db["pixel_index"]
    assert pix.signal.shape == q_vals.shape
    assert pix.units == ureg.dimensionless

    # Edges are [1, 2, 3]. With searchsorted(side="right") - 1:
    # Q:   0.5   1.25   1.75   2.25   2.75   3.5
    # idx:  -1     0      0      1      1    -1  (plus Q-range mask)
    expected = np.array([-1, 0, 0, 1, 1, -1], dtype=int)
    # pixel_index stored as float; cast to int for comparison
    result = pix.signal.astype(int)
    assert_array_equal(result, expected)


def test_indexpixels_radial_basic_binning_1d_linear_psi():
    """
    Radial direction:
    - Bins along Psi (linear bins from psi_min to psi_max).
    - Q ROI wide enough to include all pixels.
    - Check straightforward bin assignment for four angles.
    """
    # Four angles: 0°, 90°, 180°, 270°
    psi_deg = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
    psi_rad = np.deg2rad(psi_deg)
    q_vals = np.ones_like(psi_deg, dtype=float)  # simple constant Q

    bundle = make_1d_signal_bundle(
        q_values=q_vals,
        psi_values=psi_rad,
        q_unit=ureg.dimensionless,
        psi_unit=ureg.radian,
    )

    processing_data = ProcessingData()
    processing_data["test"] = bundle

    step = IndexPixels(io_sources=IoSources())
    step.processing_data = processing_data
    step._prepared_data = {}

    step.configuration = {
        "with_processing_keys": ["test"],
        "output_processing_key": None,
        "averaging_direction": "radial",
        "n_bins": 4,
        "bin_type": "linear",
        # Wide Q ROI, everything included
        "q_min": 0.0,
        "q_max": 2.0,
        "q_limits_unit": None,
        # Psi bin range 0..360° in radians
        "psi_min": 0.0,
        "psi_max": 360.0,
        "psi_limits_unit": "degree",
    }

    step.prepare_execution()
    out = step.calculate()

    db = out["test"]
    pix = db["pixel_index"]

    assert pix.signal.shape == psi_deg.shape
    assert pix.units == ureg.dimensionless

    # Edges (in radians) correspond to [0, 90, 180, 270, 360] deg.
    # Values: 0°, 90°, 180°, 270° → bins 0, 1, 2, 3
    expected = np.array([0, 1, 2, 3], dtype=int)
    result = pix.signal.astype(int)
    assert_array_equal(result, expected)


def test_indexpixels_azimuthal_q_limits_unit_conversion():
    """
    Check that q_min/q_max specified in q_limits_unit are converted
    correctly to the Q units.

    Example:
      - Q in 1/Å,
      - q_limits_unit = 1/nm,
      - q_min = 1.5 1/nm  → 0.15 1/Å
      - q_max = 2.5 1/nm  → 0.25 1/Å
    Only Q values in [0.15, 0.25] should be included.
    """
    Q_unit = ureg.Unit("1/angstrom")

    q_vals = np.array([0.10, 0.20, 0.30], dtype=float)  # in 1/Å
    psi_vals = np.zeros_like(q_vals, dtype=float)

    bundle = make_1d_signal_bundle(
        q_values=q_vals,
        psi_values=psi_vals,
        q_unit=Q_unit,
        psi_unit=ureg.radian,
    )

    processing_data = ProcessingData()
    processing_data["test"] = bundle

    step = IndexPixels(io_sources=IoSources())
    step.processing_data = processing_data
    step._prepared_data = {}

    step.configuration = {
        "with_processing_keys": ["test"],
        "output_processing_key": None,
        "averaging_direction": "azimuthal",
        "n_bins": 1,
        "bin_type": "linear",
        # In 1/nm; will be converted to 1/Å
        "q_min": 1.5,
        "q_max": 2.5,
        "q_limits_unit": "1/nm",
        # full circle in Psi
        "psi_min": 0.0,
        "psi_max": 2.0 * np.pi,
        "psi_limits_unit": "radian",
    }

    step.prepare_execution()
    out = step.calculate()

    db = out["test"]
    pix = db["pixel_index"]

    # Only the middle value (0.20 1/Å) lies between 0.15 and 0.25
    # and within the Q-range mask; with 1 bin, its index should be 0.
    expected = np.array([-1, 0, -1], dtype=int)
    result = pix.signal.astype(int)
    assert_array_equal(result, expected)


def test_indexpixels_azimuthal_psi_wraparound_mask():
    """
    Azimuthal direction with Psi wrap-around ROI:
      psi_min > psi_max, e.g. 300°..60°.

    Pixels with Psi in [300°, 360°) U [0°, 60°] should be included;
    others should be masked to -1.
    """
    psi_deg = np.array([10.0, 100.0, 200.0, 350.0], dtype=float)
    psi_rad = np.deg2rad(psi_deg)
    # Q values chosen to be all in-range for binning
    q_vals = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)

    bundle = make_1d_signal_bundle(
        q_values=q_vals,
        psi_values=psi_rad,
        q_unit=ureg.dimensionless,
        psi_unit=ureg.radian,
    )

    processing_data = ProcessingData()
    processing_data["test"] = bundle

    step = IndexPixels(io_sources=IoSources())
    step.processing_data = processing_data
    step._prepared_data = {}

    step.configuration = {
        "with_processing_keys": ["test"],
        "output_processing_key": None,
        "averaging_direction": "azimuthal",
        "n_bins": 4,
        "bin_type": "linear",
        "q_min": 0.5,
        "q_max": 3.0,
        "q_limits_unit": None,
        # wrap-around: 300° .. 60°
        "psi_min": 300.0,
        "psi_max": 60.0,
        "psi_limits_unit": "degree",
    }

    step.prepare_execution()
    out = step.calculate()

    db = out["test"]
    pix = db["pixel_index"]

    # The first (10°) and last (350°) pixels lie inside the wrap-around ROI.
    # The internal binning is along Q; with 4 linear bins between 0.5 and 3.0:
    # edges [0.5, 1.125, 1.75, 2.375, 3.0]
    # q = [1.0, 1.5, 2.0, 2.5] -> indices [0, 1, 2, 3] if unmasked.
    # Applying the Psi ROI:
    #   10°  -> index 0
    #   100° -> masked -> -1
    #   200° -> masked -> -1
    #   350° -> index 3
    expected = np.array([0, -1, -1, 3], dtype=int)
    result = pix.signal.astype(int)
    assert_array_equal(result, expected)


def test_indexpixels_prepare_and_calculate_integration_2d():
    """
    Integration-style test (2D case):
    - Build a 2D signal, Q, Psi.
    - Run prepare_execution() and calculate().
    - Check that the pixel_index BaseData is present with expected shape,
      axes, and rank_of_data.
    """
    n0, n1 = 4, 6
    spatial_shape = (n0, n1)

    # Simple Q: monotonically increasing across the flattened array
    q_vals = np.linspace(0.1, 2.0, num=n0 * n1, dtype=float).reshape(spatial_shape)
    # Psi: zeros (full-circle ROI)
    psi_vals = np.zeros(spatial_shape, dtype=float)

    signal_bd = BaseData(
        signal=np.ones(spatial_shape, dtype=float),
        units=ureg.dimensionless,
        rank_of_data=2,
    )
    q_bd = BaseData(
        signal=q_vals,
        units=ureg.Unit("1/nm"),
        rank_of_data=2,
    )
    psi_bd = BaseData(
        signal=psi_vals,
        units=ureg.radian,
        rank_of_data=2,
    )

    # Add two spatial axes for completeness
    axis_row = BaseData(
        signal=np.arange(n0, dtype=float),
        units=ureg.pixel,
        rank_of_data=0,
    )
    axis_col = BaseData(
        signal=np.arange(n1, dtype=float),
        units=ureg.pixel,
        rank_of_data=0,
    )
    signal_bd.axes = [axis_row, axis_col]

    bundle = DataBundle(
        {
            "signal": signal_bd,
            "Q": q_bd,
            "Psi": psi_bd,
        }
    )

    processing_data = ProcessingData()
    processing_data["image"] = bundle

    step = IndexPixels(io_sources=IoSources())
    step.processing_data = processing_data
    step._prepared_data = {}

    step.configuration = {
        "with_processing_keys": ["image"],
        "output_processing_key": None,
        "averaging_direction": "azimuthal",
        "n_bins": 8,
        "bin_type": "log",
        "q_min": None,  # auto from data
        "q_max": None,
        "q_limits_unit": None,
        "psi_min": 0.0,
        "psi_max": 2.0 * np.pi,
        "psi_limits_unit": "radian",
    }

    step.prepare_execution()
    out = step.calculate()

    assert "image" in out
    db = out["image"]
    assert "pixel_index" in db

    pix = db["pixel_index"]
    assert isinstance(pix, BaseData)
    assert pix.signal.shape == spatial_shape
    assert pix.rank_of_data == signal_bd.rank_of_data

    # axes should match the last rank_of_data axes of the original signal
    assert len(pix.axes) == len(signal_bd.axes[-signal_bd.rank_of_data :])
    # dimensionless units
    assert pix.units == ureg.dimensionless
