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
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.xs_geometry_from_pixel_coordinates import (
    XSGeometryFromPixelCoordinates,
)

# ----------------------------
# helpers
# ----------------------------


def _make_processing_data_with_coords(shape=(11, 20), *, rod=2) -> ProcessingData:
    """
    Create ProcessingData with a single databundle 'sample' containing coord_x/coord_y/coord_z.
    """
    n_slow, n_fast = shape

    # Choose pitches that make expected arrays easy to compute
    pitch_fast = 1e-3  # m
    pitch_slow = 2e-3  # m

    x = (np.arange(n_fast, dtype=float) + 0.5)[None, :] * pitch_fast
    y = (np.arange(n_slow, dtype=float) + 0.5)[:, None] * pitch_slow

    coord_x = np.broadcast_to(x, shape)
    coord_y = np.broadcast_to(y, shape)

    det_z = 2.5  # m
    coord_z = np.full(shape, det_z, dtype=float)

    pd = ProcessingData()
    b = DataBundle()
    b["coord_x"] = BaseData(signal=coord_x, units=ureg.m, rank_of_data=rod)
    b["coord_y"] = BaseData(signal=coord_y, units=ureg.m, rank_of_data=rod)
    b["coord_z"] = BaseData(signal=coord_z, units=ureg.m, rank_of_data=rod)
    pd["sample"] = b
    return pd


class DummyXSGeometryFromPixelCoordinates(XSGeometryFromPixelCoordinates):
    """
    Test double that bypasses IO loading and returns fixed BaseData objects for config sources.
    """

    def __init__(self, *, sources: dict[str, BaseData], **kwargs):
        super().__init__(**kwargs)
        self._sources = sources

    def _load_from_sources(self, key: str) -> BaseData:
        return self._sources[key]


def _expected_geometry_arrays(
    *,
    coord_x: np.ndarray,
    coord_y: np.ndarray,
    coord_z: np.ndarray,
    sample_z: float,
    wavelength: float,
    pitch_fast: float,
    pitch_slow: float,
    detector_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """
    Numpy-only expected values matching the implementation.
    """
    dx = coord_x
    dy = coord_y
    dz = coord_z - sample_z

    r_perp = np.sqrt(dx * dx + dy * dy)
    R = np.sqrt(dx * dx + dy * dy + dz * dz)

    two_theta = np.arctan(r_perp / dz)
    psi = np.arctan2(dy, dx)

    k = (2.0 * np.pi) / wavelength  # 1/m

    rhat_x = dx / R
    rhat_y = dy / R
    rhat_z = dz / R

    Q0 = k * rhat_x
    Q1 = k * rhat_y
    Q2 = k * (rhat_z - 1.0)
    Q = np.sqrt(Q0 * Q0 + Q1 * Q1 + Q2 * Q2)

    n = np.asarray(detector_normal, dtype=float)
    n = n / np.linalg.norm(n)
    cos_alpha = rhat_x * n[0] + rhat_y * n[1] + rhat_z * n[2]

    area = pitch_fast * pitch_slow  # m^2
    omega = (area * cos_alpha) / (R * R)

    return two_theta, psi, Q0, Q1, Q2, Q, cos_alpha, omega


# ----------------------------
# tests
# ----------------------------


def test_geometry_from_pixel_coordinates_2d_identity_normal_matches_expected_arrays():
    pd = _make_processing_data_with_coords((11, 20), rod=2)
    b = pd["sample"]

    # sample_z as NeXus-like (5,1,1,1) array -> should be reduced to scalar mean
    sample_z_vals = np.array([0.10, 0.12, 0.08, 0.11, 0.09], dtype=float).reshape(5, 1, 1, 1)
    sample_z_bd = BaseData(signal=sample_z_vals, units=ureg.m, rank_of_data=0)

    # wavelength scalar
    wavelength_bd = BaseData(signal=np.array(1.0e-10, dtype=float), units=ureg.m, rank_of_data=0)

    # pitches are scalar detector element sizes in length units
    pitch_fast_bd = BaseData(signal=np.array(1e-3, dtype=float), units=ureg.m, rank_of_data=0)
    pitch_slow_bd = BaseData(signal=np.array(2e-3, dtype=float), units=ureg.m, rank_of_data=0)

    sources = {
        "sample_z": sample_z_bd,
        "wavelength": wavelength_bd,
        "pixel_pitch_fast": pitch_fast_bd,
        "pixel_pitch_slow": pitch_slow_bd,
    }

    step = DummyXSGeometryFromPixelCoordinates(io_sources=IoSources(), sources=sources)
    step.configuration["with_processing_keys"] = ["sample"]
    step.configuration["detector_normal"] = (0.0, 0.0, 1.0)

    step.execute(pd)

    # compute expected
    exp_sample_z = float(np.mean(sample_z_vals))
    exp_two_theta, exp_psi, exp_Q0, exp_Q1, exp_Q2, exp_Q, exp_cos_alpha, exp_omega = _expected_geometry_arrays(
        coord_x=b["coord_x"].signal,
        coord_y=b["coord_y"].signal,
        coord_z=b["coord_z"].signal,
        sample_z=exp_sample_z,
        wavelength=float(wavelength_bd.signal),
        pitch_fast=float(pitch_fast_bd.signal),
        pitch_slow=float(pitch_slow_bd.signal),
        detector_normal=(0.0, 0.0, 1.0),
    )

    out = pd["sample"]

    np.testing.assert_allclose(out["TwoTheta"].signal, exp_two_theta)
    np.testing.assert_allclose(out["Psi"].signal, exp_psi)

    np.testing.assert_allclose(out["Q0"].signal, exp_Q0)
    np.testing.assert_allclose(out["Q1"].signal, exp_Q1)
    np.testing.assert_allclose(out["Q2"].signal, exp_Q2)
    np.testing.assert_allclose(out["Q"].signal, exp_Q)

    np.testing.assert_allclose(out["CosAlpha"].signal, exp_cos_alpha)
    np.testing.assert_allclose(out["Omega"].signal, exp_omega)

    # basic metadata checks
    assert out["Q"].rank_of_data == 2
    assert out["TwoTheta"].units.is_compatible_with(ureg.radian)
    assert out["Psi"].units.is_compatible_with(ureg.radian)
    assert out["Q"].units.is_compatible_with(ureg.m**-1)
    assert out["CosAlpha"].units.is_compatible_with(ureg.dimensionless)
    assert out["Omega"].units == ureg.steradian


def test_geometry_from_pixel_coordinates_detector_normal_is_normalized():
    """
    detector_normal=(0,0,2) should behave identically to (0,0,1).
    """
    pd = _make_processing_data_with_coords((11, 20), rod=2)
    b = pd["sample"]

    sample_z_bd = BaseData(
        signal=np.array([0.10, 0.12, 0.08, 0.11, 0.09], dtype=float).reshape(5, 1, 1, 1), units=ureg.m, rank_of_data=0
    )
    wavelength_bd = BaseData(signal=np.array(1.0e-10, dtype=float), units=ureg.m, rank_of_data=0)
    pitch_fast_bd = BaseData(signal=np.array(1e-3, dtype=float), units=ureg.m, rank_of_data=0)
    pitch_slow_bd = BaseData(signal=np.array(2e-3, dtype=float), units=ureg.m, rank_of_data=0)

    sources = {
        "sample_z": sample_z_bd,
        "wavelength": wavelength_bd,
        "pixel_pitch_fast": pitch_fast_bd,
        "pixel_pitch_slow": pitch_slow_bd,
    }

    # run with non-unit normal
    step = DummyXSGeometryFromPixelCoordinates(io_sources=IoSources(), sources=sources)
    step.configuration["with_processing_keys"] = ["sample"]
    step.configuration["detector_normal"] = (0.0, 0.0, 2.0)
    step.execute(pd)
    omega_nonunit = pd["sample"]["Omega"].signal.copy()

    # expected omega with unit normal
    exp_sample_z = float(np.mean(sample_z_bd.signal))
    *_rest, exp_omega_unit = _expected_geometry_arrays(
        coord_x=b["coord_x"].signal,
        coord_y=b["coord_y"].signal,
        coord_z=b["coord_z"].signal,
        sample_z=exp_sample_z,
        wavelength=float(wavelength_bd.signal),
        pitch_fast=float(pitch_fast_bd.signal),
        pitch_slow=float(pitch_slow_bd.signal),
        detector_normal=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(omega_nonunit, exp_omega_unit)


def test_geometry_from_pixel_coordinates_uses_nexus_detector_frame(tmp_path):
    h5py = pytest.importorskip("h5py")
    calibration_path = tmp_path / "calibration.nxs"

    def write_transform(dataset, *, transformation_type, units, vector, depends_on="."):
        dataset.attrs["transformation_type"] = transformation_type
        dataset.attrs["units"] = units
        dataset.attrs["vector"] = np.asarray(vector, dtype=float)
        dataset.attrs["depends_on"] = depends_on

    with h5py.File(calibration_path, "w") as h5:
        module = h5.require_group("/entry1/instrument/detector/detector_module")
        offset = module.create_dataset("module_offset", data=2.5)
        write_transform(offset, transformation_type="translation", units="m", vector=[0.0, 0.0, 1.0])

        fast = module.create_dataset("fast_pixel_direction", data=3.0)
        write_transform(
            fast,
            transformation_type="translation",
            units="mm",
            vector=[1.0, 0.0, 0.0],
            depends_on="./module_offset",
        )

        slow = module.create_dataset("slow_pixel_direction", data=4.0)
        write_transform(
            slow,
            transformation_type="translation",
            units="mm",
            vector=[0.0, 1.0, 0.0],
            depends_on="./module_offset",
        )

    pd = _make_processing_data_with_coords((3, 5), rod=2)
    b = pd["sample"]
    sources = {
        "wavelength": BaseData(signal=np.asarray(1.0e-10), units=ureg.m, rank_of_data=0),
    }
    io_sources = IoSources()
    io_sources.register_source(HDFSource(source_reference="calibration", resource_location=calibration_path))

    step = DummyXSGeometryFromPixelCoordinates(io_sources=io_sources, sources=sources)
    step.configuration["with_processing_keys"] = ["sample"]
    step.configuration["detector_frame"] = {
        "type": "nexus",
        "source": "calibration",
        "detector_path": "/entry1/instrument/detector",
    }
    step.configuration["sample_z_override"] = {"value": 0.5, "units": "m"}

    step.execute(pd)

    *_, exp_omega = _expected_geometry_arrays(
        coord_x=b["coord_x"].signal,
        coord_y=b["coord_y"].signal,
        coord_z=b["coord_z"].signal,
        sample_z=0.5,
        wavelength=1.0e-10,
        pitch_fast=3.0e-3,
        pitch_slow=4.0e-3,
        detector_normal=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(pd["sample"]["Omega"].signal, exp_omega)


def test_geometry_from_pixel_coordinates_uses_nexus_sample_z_override(tmp_path):
    h5py = pytest.importorskip("h5py")
    measurement_path = tmp_path / "measurement.nxs"

    def write_transform(dataset, *, transformation_type, units, vector, depends_on="."):
        dataset.attrs["transformation_type"] = transformation_type
        dataset.attrs["units"] = units
        dataset.attrs["vector"] = np.asarray(vector, dtype=float)
        dataset.attrs["depends_on"] = depends_on

    with h5py.File(measurement_path, "w") as h5:
        transformations = h5.require_group("/entry1/sample/transformations")
        base_z = transformations.create_dataset("base_z", data=1.0)
        write_transform(base_z, transformation_type="translation", units="m", vector=[0.0, 0.0, 1.0])

        sample_z = transformations.create_dataset("sample_z", data=50.0)
        write_transform(
            sample_z,
            transformation_type="translation",
            units="mm",
            vector=[0.0, 0.0, 1.0],
            depends_on="./base_z",
        )

    pd = _make_processing_data_with_coords((3, 5), rod=2)
    b = pd["sample"]
    sources = {
        "wavelength": BaseData(signal=np.asarray(1.0e-10), units=ureg.m, rank_of_data=0),
        "pixel_pitch_fast": BaseData(signal=np.asarray(1.0e-3), units=ureg.m, rank_of_data=0),
        "pixel_pitch_slow": BaseData(signal=np.asarray(2.0e-3), units=ureg.m, rank_of_data=0),
    }
    io_sources = IoSources()
    io_sources.register_source(HDFSource(source_reference="sample", resource_location=measurement_path))

    step = DummyXSGeometryFromPixelCoordinates(io_sources=io_sources, sources=sources)
    step.configuration["with_processing_keys"] = ["sample"]
    step.configuration["sample_z_override"] = {
        "type": "nexus",
        "source": "sample",
        "transform_path": "/entry1/sample/transformations/sample_z",
    }

    step.execute(pd)

    exp_two_theta, *_ = _expected_geometry_arrays(
        coord_x=b["coord_x"].signal,
        coord_y=b["coord_y"].signal,
        coord_z=b["coord_z"].signal,
        sample_z=1.05,
        wavelength=1.0e-10,
        pitch_fast=1.0e-3,
        pitch_slow=2.0e-3,
        detector_normal=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(pd["sample"]["TwoTheta"].signal, exp_two_theta)
