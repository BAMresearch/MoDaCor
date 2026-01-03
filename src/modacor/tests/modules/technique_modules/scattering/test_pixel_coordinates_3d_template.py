# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "03/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.pixel_coordinates_3d_template import (
    CanonicalDetectorFrame,
    PixelCoordinates3DTemplate,
)


class DummyPixelCoordinates(PixelCoordinates3DTemplate):
    """Template test double: returns a fixed canonical frame."""

    def __init__(self, *, frame: CanonicalDetectorFrame, **kwargs):
        super().__init__(**kwargs)
        self._frame = frame

    def _load_canonical_frame(self, *, RoD, detector_shape, reference_signal):
        return self._frame


def _make_processing_data(signal_shape: tuple[int, ...], *, rod: int) -> ProcessingData:
    pd = ProcessingData()
    b = DataBundle()
    b["signal"] = BaseData(signal=np.zeros(signal_shape, dtype=float), units=ureg.dimensionless, rank_of_data=rod)
    pd["sample"] = b
    return pd


def test_template_2d_coordinates_identity_basis():
    # detector: 2x3 (slow, fast)
    pd = _make_processing_data((2, 3), rod=2)

    origin = BaseData(signal=np.array([0.0, 0.0, 1.0]), units=ureg.m, rank_of_data=0)
    pitch_slow = BaseData(signal=np.array(2e-3), units=ureg.m / ureg.pixel, rank_of_data=0)  # 2 mm/px
    pitch_fast = BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0)  # 1 mm/px
    bc_slow = BaseData(signal=np.array(0.0), units=ureg.pixel, rank_of_data=0)
    bc_fast = BaseData(signal=np.array(0.0), units=ureg.pixel, rank_of_data=0)

    frame = CanonicalDetectorFrame(
        origin=origin,
        e_fast=np.array([1.0, 0.0, 0.0]),
        e_slow=np.array([0.0, 1.0, 0.0]),
        e_normal=np.array([0.0, 0.0, 1.0]),
        pixel_pitch_slow=pitch_slow,
        pixel_pitch_fast=pitch_fast,
        beam_center_slow_px=bc_slow,
        beam_center_fast_px=bc_fast,
    )

    step = DummyPixelCoordinates(io_sources=IoSources(), frame=frame)
    step.configuration["with_processing_keys"] = ["sample"]

    step.execute(pd)
    out = pd["sample"]

    n_slow, n_fast = out["coord_x"].signal.shape

    fast = (np.arange(n_fast) + 0.5) * 1e-3
    slow = (np.arange(n_slow) + 0.5) * 2e-3

    exp_x = np.broadcast_to(fast[None, :], (n_slow, n_fast))
    exp_y = np.broadcast_to(slow[:, None], (n_slow, n_fast))
    exp_z = np.ones((n_slow, n_fast)) * 1.0

    np.testing.assert_allclose(out["coord_x"].signal, exp_x)
    np.testing.assert_allclose(out["coord_y"].signal, exp_y)
    np.testing.assert_allclose(out["coord_z"].signal, exp_z)

    assert out["coord_x"].rank_of_data == 2
    assert out["coord_x"].units.is_compatible_with(ureg.m)

    assert np.all(out["coord_x"].signal[0, :] == out["coord_x"].signal[1, :])
    assert np.all(out["coord_y"].signal[:, 0] == out["coord_y"].signal[:, 1])


def test_template_1d_coordinates_identity_basis():
    pd = _make_processing_data((4,), rod=1)

    origin = BaseData(signal=np.array([0.0, 0.0, 2.0]), units=ureg.m, rank_of_data=0)
    pitch_fast = BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0)
    pitch_slow = BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0)  # unused in RoD=1
    bc_fast = BaseData(signal=np.array(0.0), units=ureg.pixel, rank_of_data=0)
    bc_slow = BaseData(signal=np.array(0.0), units=ureg.pixel, rank_of_data=0)

    frame = CanonicalDetectorFrame(
        origin=origin,
        e_fast=np.array([1.0, 0.0, 0.0]),
        e_slow=np.array([0.0, 1.0, 0.0]),
        e_normal=np.array([0.0, 0.0, 1.0]),
        pixel_pitch_slow=pitch_slow,
        pixel_pitch_fast=pitch_fast,
        beam_center_slow_px=bc_slow,
        beam_center_fast_px=bc_fast,
    )

    step = DummyPixelCoordinates(io_sources=IoSources(), frame=frame)
    step.configuration["with_processing_keys"] = ["sample"]

    step.execute(pd)
    out = pd["sample"]

    exp_x = np.array([0.5, 1.5, 2.5, 3.5]) * 1e-3
    exp_y = np.zeros(4)
    exp_z = np.ones(4) * 2.0

    np.testing.assert_allclose(out["coord_x"].signal, exp_x)
    np.testing.assert_allclose(out["coord_y"].signal, exp_y)
    np.testing.assert_allclose(out["coord_z"].signal, exp_z)


def test_template_rejects_wrong_units_fast_beam_center():
    pd = _make_processing_data((2, 2), rod=2)

    frame = CanonicalDetectorFrame(
        origin=BaseData(signal=np.array([0.0, 0.0, 1.0]), units=ureg.m, rank_of_data=0),
        e_fast=np.array([1.0, 0.0, 0.0]),
        e_slow=np.array([0.0, 1.0, 0.0]),
        e_normal=np.array([0.0, 0.0, 1.0]),
        pixel_pitch_slow=BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0),
        pixel_pitch_fast=BaseData(signal=np.array(1e-3), units=ureg.m / ureg.pixel, rank_of_data=0),
        beam_center_slow_px=BaseData(signal=np.array(0.0), units=ureg.pixel, rank_of_data=0),
        beam_center_fast_px=BaseData(signal=np.array(0.0), units=ureg.m, rank_of_data=0),  # WRONG
    )

    step = DummyPixelCoordinates(io_sources=IoSources(), frame=frame)
    step.configuration["with_processing_keys"] = ["sample"]

    with pytest.raises(ValueError, match="beam_center_fast_px must be in pixels"):
        step.execute(pd)
