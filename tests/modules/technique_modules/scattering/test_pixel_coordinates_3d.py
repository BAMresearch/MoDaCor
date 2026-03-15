# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "06/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports
__version__ = "20260106.1"

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.geometry_helpers import prepare_static_scalar
from modacor.modules.technique_modules.scattering.pixel_coordinates_3d import CanonicalDetectorFrame, PixelCoordinates3D

# ----------------------------
# helpers
# ----------------------------


def _make_processing_data_2d(shape: tuple[int, int] = (11, 20), *, rod: int = 2) -> ProcessingData:
    pd = ProcessingData()
    b = DataBundle()
    b["signal"] = BaseData(signal=np.zeros(shape, dtype=float), units=ureg.dimensionless, rank_of_data=rod)
    pd["sample"] = b
    return pd


def _make_frame(
    *,
    det_x: BaseData,
    det_y: BaseData,
    det_z: BaseData,
    pitch_slow: BaseData,
    pitch_fast: BaseData,
    e_fast=(1.0, 0.0, 0.0),
    e_slow=(0.0, 1.0, 0.0),
    e_norm=(0.0, 0.0, 1.0),
) -> CanonicalDetectorFrame:
    """
    In the new structure, the PixelCoordinates3D implementation reduces "static config"
    arrays (NeXus-style) to scalars via prepare_static_scalar(...) during loading.

    Because this test double bypasses IO loading (_load_canonical_frame), we perform the same
    reduction here so the frame matches the module’s expectations (scalar det_coord_* / pitches).
    """
    det_z_s = prepare_static_scalar(det_z, require_units=ureg.m, uncertainty_key="detector_position_jitter")
    det_x_s = prepare_static_scalar(det_x, require_units=ureg.m, uncertainty_key="detector_position_jitter")
    det_y_s = prepare_static_scalar(det_y, require_units=ureg.m, uncertainty_key="detector_position_jitter")

    pitch_slow_s = prepare_static_scalar(
        pitch_slow,
        require_units=ureg.m / ureg.pixel,
        uncertainty_key="pixel_pitch_jitter",
    )
    pitch_fast_s = prepare_static_scalar(
        pitch_fast,
        require_units=ureg.m / ureg.pixel,
        uncertainty_key="pixel_pitch_jitter",
    )

    return CanonicalDetectorFrame(
        det_coord_z=det_z_s,
        det_coord_x=det_x_s,
        det_coord_y=det_y_s,
        e_fast=np.array(e_fast, dtype=float),
        e_slow=np.array(e_slow, dtype=float),
        e_normal=np.array(e_norm, dtype=float),
        pixel_pitch_slow=pitch_slow_s,
        pixel_pitch_fast=pitch_fast_s,
    )


class DummyPixelCoordinates3D(PixelCoordinates3D):
    """
    Test double that bypasses IO loading and returns a fixed CanonicalDetectorFrame.
    """

    def __init__(self, *, frame: CanonicalDetectorFrame, **kwargs):
        super().__init__(**kwargs)
        self._frame = frame

    def _load_canonical_frame(self, *, RoD, detector_shape, reference_signal):
        return self._frame


# ----------------------------
# tests: static-scalar preparation (moved from pixel module to helpers)
# ----------------------------


def test_prepare_static_scalar_passes_through_scalar():
    bd = BaseData(signal=np.array(2.5), units=ureg.m, rank_of_data=0)
    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="detector_position_jitter")
    assert np.size(out.signal) == 1
    assert out.rank_of_data == 0
    assert out.units.is_compatible_with(ureg.m)
    np.testing.assert_allclose(out.signal, 2.5)


def test_prepare_static_scalar_reduces_shape_5_1_1_1_to_scalar_mean_and_sem():
    # Mimics NeXus vector stored as [5,1,1,1]
    values = np.array([2.50, 2.52, 2.48, 2.51, 2.49], dtype=float).reshape(5, 1, 1, 1)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)

    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="detector_position_jitter")

    assert np.size(out.signal) == 1
    assert out.rank_of_data == 0
    assert out.units.is_compatible_with(ureg.m)

    exp_mean = float(np.mean(values))

    # For equal weights, this helper uses:
    #   var = mean((x-mean)^2)   (population-style)
    #   sem = sqrt(var) / sqrt(N)
    flat = values.ravel()
    exp_var = float(np.mean((flat - exp_mean) ** 2))
    exp_sem = float(np.sqrt(exp_var) / np.sqrt(flat.size))

    np.testing.assert_allclose(out.signal, exp_mean, rtol=0, atol=1e-15)
    assert "detector_position_jitter" in out.uncertainties
    np.testing.assert_allclose(out.uncertainties["detector_position_jitter"], exp_sem, rtol=0, atol=1e-15)


def test_prepare_static_scalar_rejects_wrong_units():
    bd = BaseData(signal=np.array([1.0, 2.0, 3.0]), units=ureg.pixel, rank_of_data=0)
    with pytest.raises(ValueError, match="Value must be in"):
        prepare_static_scalar(bd, require_units=ureg.m)


# ----------------------------
# tests: pixel coordinate math
# ----------------------------


def test_pixel_coordinates_2d_identity_basis_constant_z_and_expected_x_y():
    """
    2D detector: (slow, fast) = (11, 20)

    Convention under test:
    - det_coord_* is the lab-frame position of the *pixel-grid origin corner* (before +0.5 center shift)
    - pixel centers at (j+0.5, i+0.5)
    - identity basis: fast->x, slow->y, no z components => coord_z should be constant at det_coord_z
    """
    pd = _make_processing_data_2d((11, 20), rod=2)

    # det_z given as a NeXus-like array (5,1,1,1), reduced to scalar in _make_frame()
    det_z_vals = np.array([2.507, 2.508, 2.509, 2.507, 2.508], dtype=float).reshape(5, 1, 1, 1)
    det_z = BaseData(signal=det_z_vals, units=ureg.m, rank_of_data=0)

    det_x = BaseData(signal=np.array(0.0), units=ureg.m, rank_of_data=0)
    det_y = BaseData(signal=np.array(0.0), units=ureg.m, rank_of_data=0)

    pitch_fast = BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0)  # 1 mm/px
    pitch_slow = BaseData(signal=np.array(2e-3), units=ureg.m / ureg.pixel, rank_of_data=0)  # 2 mm/px

    frame = _make_frame(
        det_x=det_x,
        det_y=det_y,
        det_z=det_z,
        pitch_slow=pitch_slow,
        pitch_fast=pitch_fast,
        e_fast=(1.0, 0.0, 0.0),
        e_slow=(0.0, 1.0, 0.0),
        e_norm=(0.0, 0.0, 1.0),
    )

    step = DummyPixelCoordinates3D(io_sources=IoSources(), frame=frame)
    step.configuration["with_processing_keys"] = ["sample"]

    step.execute(pd)
    out = pd["sample"]

    cx = out["coord_x"].signal
    cy = out["coord_y"].signal
    cz = out["coord_z"].signal

    assert cx.shape == (11, 20)
    assert cy.shape == (11, 20)
    assert cz.shape == (11, 20)

    # expected x: (i + 0.5) * pitch_fast
    i = np.arange(20, dtype=float) + 0.5
    exp_x = np.broadcast_to(i[None, :] * 1e-3, (11, 20))

    # expected y: (j + 0.5) * pitch_slow
    j = np.arange(11, dtype=float) + 0.5
    exp_y = np.broadcast_to(j[:, None] * 2e-3, (11, 20))

    # expected z: scalar mean(det_z_vals) broadcast
    exp_z_scalar = float(np.mean(det_z_vals))
    exp_z = np.full((11, 20), exp_z_scalar, dtype=float)

    np.testing.assert_allclose(cx, exp_x)
    np.testing.assert_allclose(cy, exp_y)
    np.testing.assert_allclose(cz, exp_z)

    assert out["coord_x"].units.is_compatible_with(ureg.m)
    assert out["coord_x"].rank_of_data == 2


def test_pixel_coordinates_2d_offset_origin_shifts_coordinates():
    pd = _make_processing_data_2d((11, 20), rod=2)

    det_z = BaseData(signal=np.array(2.0), units=ureg.m, rank_of_data=0)
    det_x = BaseData(signal=np.array(0.10), units=ureg.m, rank_of_data=0)  # 10 cm offset
    det_y = BaseData(signal=np.array(-0.05), units=ureg.m, rank_of_data=0)  # -5 cm offset

    pitch_fast = BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0)
    pitch_slow = BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0)

    frame = _make_frame(
        det_x=det_x,
        det_y=det_y,
        det_z=det_z,
        pitch_slow=pitch_slow,
        pitch_fast=pitch_fast,
        e_fast=(1.0, 0.0, 0.0),
        e_slow=(0.0, 1.0, 0.0),
        e_norm=(0.0, 0.0, 1.0),
    )

    step = DummyPixelCoordinates3D(io_sources=IoSources(), frame=frame)
    step.configuration["with_processing_keys"] = ["sample"]
    step.execute(pd)
    out = pd["sample"]

    cx = out["coord_x"].signal
    cy = out["coord_y"].signal
    cz = out["coord_z"].signal

    # Spot check a few pixels
    # pixel (slow=j, fast=i) => center at (j+0.5, i+0.5)
    j0, i0 = 0, 0
    np.testing.assert_allclose(cx[j0, i0], 0.10 + 0.5e-3)
    np.testing.assert_allclose(cy[j0, i0], -0.05 + 0.5e-3)
    np.testing.assert_allclose(cz[j0, i0], 2.0)

    j1, i1 = 10, 19
    np.testing.assert_allclose(cx[j1, i1], 0.10 + 19.5e-3)
    np.testing.assert_allclose(cy[j1, i1], -0.05 + 10.5e-3)
    np.testing.assert_allclose(cz[j1, i1], 2.0)


def test_pixel_coordinates_rod0_returns_scalars():
    pd = ProcessingData()
    b = DataBundle()
    b["signal"] = BaseData(signal=np.array(1.0), units=ureg.dimensionless, rank_of_data=0)
    pd["sample"] = b

    frame = _make_frame(
        det_x=BaseData(signal=np.array(0.1), units=ureg.m, rank_of_data=0),
        det_y=BaseData(signal=np.array(0.2), units=ureg.m, rank_of_data=0),
        det_z=BaseData(signal=np.array(2.0), units=ureg.m, rank_of_data=0),
        pitch_slow=BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0),
        pitch_fast=BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0),
    )

    step = DummyPixelCoordinates3D(io_sources=IoSources(), frame=frame)
    step.configuration["with_processing_keys"] = ["sample"]

    step.execute(pd)
    out = pd["sample"]

    assert np.size(out["coord_x"].signal) == 1
    assert np.size(out["coord_y"].signal) == 1
    assert np.size(out["coord_z"].signal) == 1

    np.testing.assert_allclose(out["coord_x"].signal, 0.1)
    np.testing.assert_allclose(out["coord_y"].signal, 0.2)
    np.testing.assert_allclose(out["coord_z"].signal, 2.0)
