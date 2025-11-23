# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from modacor.dataclasses.databundle import DataBundle

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

"""
Tests for the XSGeometry processing step.

We test:
- Low-level geometry helpers (_compute_coordinates, _compute_angles, _compute_Q, ...)
- Basic symmetry properties for a simple 2D detector
- Simple checks for 1D and 0D cases
- A thin integration test for prepare_execution() + calculate()
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.xs_geometry import XSGeometry

# ---------------------------------------------------------------------------
# Small helpers for building geometry BaseData
# ---------------------------------------------------------------------------


def _bd_scalar(value: float, unit) -> BaseData:
    return BaseData(signal=np.array(float(value), dtype=float), units=unit)


def _bd_vector(values, unit) -> BaseData:
    return BaseData(signal=np.asarray(values, dtype=float), units=unit)


def make_geom_2d(n0: int, n1: int):
    """
    Convenience helper: simple 2D geometry

    - Detector: n0 x n1
    - Detector distance: 1.0 m with 1 mm uncertainty
    - Pixel size: 1e-3 m in both directions (with small uncertainty)
    - Beam center: exact center pixel (with 0.25 pixel uncertainty)
    - Wavelength: 1 Å (1e-10 m) with 2% uncertainty
    """
    # --- detector distance: 1.0 m ± 1 mm ---
    D_value = 1.0
    D_unc = 1e-3  # 1 mm
    D_bd = BaseData(
        signal=np.array(D_value, dtype=float),
        units=ureg.meter,
        # propagate_to_all so this uncertainty participates in all keys
        uncertainties={"propagate_to_all": np.array(D_unc, dtype=float)},
    )

    # --- pixel size: 1e-3 m ± 1e-6 m (just something small) ---
    pixel_signal = np.asarray([1e-3, 1e-3], dtype=float)
    pixel_unc = np.full_like(pixel_signal, 1e-6, dtype=float)
    pixel_size_bd = BaseData(
        signal=pixel_signal,
        units=ureg.meter,
        uncertainties={"propagate_to_all": pixel_unc},
    )

    # --- beam centre: at the centre pixel, ±0.25 pixel index ---
    center_row = (n0 - 1) / 2.0
    center_col = (n1 - 1) / 2.0
    beam_signal = np.asarray([center_col, center_row], dtype=float)
    beam_unc = np.full_like(beam_signal, 0.25, dtype=float)  # ±0.25 pixel
    beam_center_bd = BaseData(
        signal=beam_signal,
        units=ureg.dimensionless,
        uncertainties={"pixel_index": beam_unc},
    )

    # --- wavelength: 1 Å ± 2% ---
    lambda_value = 1.0e-10  # 1 Å in meters
    lambda_unc = 0.02 * lambda_value  # 2%
    wavelength_bd = BaseData(
        signal=np.array(lambda_value, dtype=float),
        units=ureg.meter,
        uncertainties={"propagate_to_all": np.array(lambda_unc, dtype=float)},
    )

    return D_bd, pixel_size_bd, beam_center_bd, wavelength_bd


def make_geom_1d(n: int):
    """
    Simple 1D geometry: n pixels in a line.

    - Detector distance: 1.0 m with 1 mm uncertainty
    - Pixel size: 1e-3 m (first component used)
    - Beam center: central pixel with ±0.25 pixel uncertainty
    - Wavelength: 1 Å with 2% uncertainty
    """
    # --- detector distance: 1.0 m ± 1 mm ---
    D_value = 1.0
    D_unc = 1e-3  # 1 mm
    D_bd = BaseData(
        signal=np.array(D_value, dtype=float),
        units=ureg.meter,
        uncertainties={"propagate_to_all": np.array(D_unc, dtype=float)},
    )

    # --- pixel size: 1e-3 m ± 1e-6 m ---
    pixel_signal = np.asarray([1e-3, 1e-3], dtype=float)
    pixel_unc = np.full_like(pixel_signal, 1e-6, dtype=float)
    pixel_size_bd = BaseData(
        signal=pixel_signal,
        units=ureg.meter,
        uncertainties={"propagate_to_all": pixel_unc},
    )

    # --- beam centre: central pixel ±0.25 pixel index ---
    center = (n - 1) / 2.0
    beam_signal = np.asarray([center], dtype=float)
    beam_unc = np.full_like(beam_signal, 0.25, dtype=float)
    beam_center_bd = BaseData(
        signal=beam_signal,
        units=ureg.dimensionless,
        uncertainties={"pixel_index": beam_unc},
    )

    # --- wavelength: 1 Å ± 2% ---
    lambda_value = 1.0e-10  # 1 Å in meters
    lambda_unc = 0.02 * lambda_value
    wavelength_bd = BaseData(
        signal=np.array(lambda_value, dtype=float),
        units=ureg.meter,
        uncertainties={"propagate_to_all": np.array(lambda_unc, dtype=float)},
    )

    return D_bd, pixel_size_bd, beam_center_bd, wavelength_bd


# ---------------------------------------------------------------------------
# Tests for helper methods (math / symmetry)
# ---------------------------------------------------------------------------


def test_xsgeometry_2d_center_q_zero_and_symmetry():
    """
    For a symmetric 2D detector with the beam at the center:
    - Q at the center pixel should be ≈ 0.
    - Q0 should be antisymmetric left-right, Q1 symmetric left-right.
    - Q1 should be antisymmetric up-down, Q0 symmetric up-down.
    - Psi should behave like atan2(y, x):
        right of center  ~ 0
        left of center   ~ ±π
        above center     ~ +π/2
        below center     ~ -π/2
    - Omega (solid angle) should be largest at the beam center and smaller at corners.
    """
    step = XSGeometry(io_sources=IoSources())

    n0, n1 = 5, 5
    spatial_shape = (n0, n1)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_2d(n0, n1)
    px0_bd, px1_bd = step._extract_pixel_pitches(pixel_size_bd)

    # coordinates
    x0_bd, x1_bd, r_perp_bd, R_bd = step._compute_coordinates(
        RoD=2,
        spatial_shape=spatial_shape,
        beam_center_bd=beam_center_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    # angles, Q magnitude, components, Psi, Omega
    _, theta_bd, sin_theta_bd = step._compute_angles(
        r_perp_bd=r_perp_bd,
        detector_distance_bd=D_bd,
    )
    Q_bd = step._compute_Q(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
    )
    Q0_bd, Q1_bd, Q2_bd = step._compute_Q_components(
        Q_bd=Q_bd,
        x0_bd=x0_bd,
        x1_bd=x1_bd,
        r_perp_bd=r_perp_bd,
    )
    Psi_bd = step._compute_psi(x0_bd=x0_bd, x1_bd=x1_bd)
    Omega_bd = step._compute_solid_angle(
        R_bd=R_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    center = (n0 // 2, n1 // 2)
    row_c, col_c = center

    # Q at center ≈ 0
    assert_allclose(Q_bd.signal[center], 0.0, atol=1e-12)
    assert_allclose(Q0_bd.signal[center], 0.0, atol=1e-12)
    assert_allclose(Q1_bd.signal[center], 0.0, atol=1e-12)
    assert_allclose(Q2_bd.signal[center], 0.0, atol=1e-12)

    # left-right symmetry
    col_left = col_c - 1
    col_right = col_c + 1
    q0_left = Q0_bd.signal[row_c, col_left]
    q0_right = Q0_bd.signal[row_c, col_right]
    q1_left = Q1_bd.signal[row_c, col_left]
    q1_right = Q1_bd.signal[row_c, col_right]

    # |Q0_left| == |Q0_right|, opposite sign
    assert_allclose(abs(q0_left), abs(q0_right), rtol=1e-6, atol=1e-12)
    assert q0_left == pytest.approx(-q0_right, rel=1e-6, abs=1e-12)
    # Q1 symmetric
    assert_allclose(q1_left, q1_right, rtol=1e-6, atol=1e-12)

    # up-down symmetry
    row_up = row_c - 1
    row_down = row_c + 1
    q0_up = Q0_bd.signal[row_up, col_c]
    q0_down = Q0_bd.signal[row_down, col_c]
    q1_up = Q1_bd.signal[row_up, col_c]
    q1_down = Q1_bd.signal[row_down, col_c]

    # |Q1_up| == |Q1_down|, opposite sign
    assert_allclose(abs(q1_up), abs(q1_down), rtol=1e-6, atol=1e-12)
    assert q1_up == pytest.approx(-q1_down, rel=1e-6, abs=1e-12)
    # Q0 symmetric
    assert_allclose(q0_up, q0_down, rtol=1e-6, atol=1e-12)

    # Psi behaviour (image coords: rows increase downward)
    psi_right = Psi_bd.signal[row_c, col_right]
    psi_left = Psi_bd.signal[row_c, col_left]
    psi_up = Psi_bd.signal[row_up, col_c]
    psi_down = Psi_bd.signal[row_down, col_c]

    assert psi_right == pytest.approx(0.0, abs=1e-4)
    # left should be close to ±π
    assert abs(abs(psi_left) - np.pi) < 1e-4
    # above centre → x0 ~ 0, x1 < 0 → atan2(neg, 0) ≈ -π/2
    assert psi_up == pytest.approx(-np.pi / 2.0, abs=1e-4)
    # below centre → x1 > 0 → atan2(pos, 0) ≈ +π/2
    assert psi_down == pytest.approx(+np.pi / 2.0, abs=1e-4)

    # Omega largest at center, smaller at corner
    omega_center = Omega_bd.signal[center]
    omega_corner = Omega_bd.signal[0, 0]
    assert omega_corner < omega_center
    # and Omega positive everywhere
    assert np.all(Omega_bd.signal > 0.0)

    # sanity: theta increases with radius
    theta_row = theta_bd.signal[row_c, :]
    assert theta_row[col_c] == pytest.approx(0.0, abs=1e-12)
    assert theta_row[col_left] > theta_row[col_c]
    assert theta_row[col_right] > theta_row[col_c]


def test_xsgeometry_1d_center_q_zero_and_monotonic():
    """
    For a symmetric 1D detector:
    - Q at the center pixel should be ≈ 0.
    - |Q| should increase as we move away from the center.
    - Q1 and Q2 should be zero.
    """
    step = XSGeometry(io_sources=IoSources())

    n = 7
    spatial_shape = (n,)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_1d(n)
    px0_bd, px1_bd = step._extract_pixel_pitches(pixel_size_bd)

    x0_bd, x1_bd, r_perp_bd, R_bd = step._compute_coordinates(
        RoD=1,
        spatial_shape=spatial_shape,
        beam_center_bd=beam_center_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    _, theta_bd, sin_theta_bd = step._compute_angles(
        r_perp_bd=r_perp_bd,
        detector_distance_bd=D_bd,
    )
    Q_bd = step._compute_Q(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
    )
    Q0_bd, Q1_bd, Q2_bd = step._compute_Q_components(
        Q_bd=Q_bd,
        x0_bd=x0_bd,
        x1_bd=x1_bd,
        r_perp_bd=r_perp_bd,
    )

    center = n // 2

    # center ≈ 0
    assert_allclose(Q_bd.signal[center], 0.0, atol=1e-12)
    assert_allclose(Q0_bd.signal[center], 0.0, atol=1e-12)
    # Q1, Q2 should be identically zero
    assert_allclose(Q1_bd.signal, 0.0, atol=1e-12)
    assert_allclose(Q2_bd.signal, 0.0, atol=1e-12)

    # |Q| grows away from center
    abs_Q = np.abs(Q_bd.signal)
    assert abs_Q[center + 1] > abs_Q[center]
    assert abs_Q[center + 2] > abs_Q[center + 1]


def test_xsgeometry_0d_shapes_and_units():
    """
    For a 0D detector (scalar signal):
    - All geometry outputs should be scalars.
    - Q and its components should be zero.
    - Omega should be positive scalar.
    """
    step = XSGeometry(io_sources=IoSources())

    # 0D: no spatial shape
    spatial_shape: tuple[int, ...] = ()
    D_bd = _bd_scalar(1.0, ureg.meter)
    # pixel size / beam center technically irrelevant for RoD=0, but we supply valid shapes
    pixel_size_bd = _bd_vector([1e-3, 1e-3], ureg.meter)
    beam_center_bd = _bd_vector([0.0], ureg.dimensionless)
    wavelength_bd = _bd_scalar(1.0, ureg.meter)

    px0_bd, px1_bd = step._extract_pixel_pitches(pixel_size_bd)
    x0_bd, x1_bd, r_perp_bd, R_bd = step._compute_coordinates(
        RoD=0,
        spatial_shape=spatial_shape,
        beam_center_bd=beam_center_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    _, theta_bd, sin_theta_bd = step._compute_angles(
        r_perp_bd=r_perp_bd,
        detector_distance_bd=D_bd,
    )
    Q_bd = step._compute_Q(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
    )
    Q0_bd, Q1_bd, Q2_bd = step._compute_Q_components(
        Q_bd=Q_bd,
        x0_bd=x0_bd,
        x1_bd=x1_bd,
        r_perp_bd=r_perp_bd,
    )
    Omega_bd = step._compute_solid_angle(
        R_bd=R_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    # all scalars
    assert Q_bd.signal.shape == ()
    assert Q0_bd.signal.shape == ()
    assert Q1_bd.signal.shape == ()
    assert Q2_bd.signal.shape == ()
    assert theta_bd.signal.shape == ()
    assert Omega_bd.signal.shape == ()

    # Q and components should be zero
    assert_allclose(Q_bd.signal, 0.0, atol=1e-12)
    assert_allclose(Q0_bd.signal, 0.0, atol=1e-12)
    assert_allclose(Q1_bd.signal, 0.0, atol=1e-12)
    assert_allclose(Q2_bd.signal, 0.0, atol=1e-12)

    # solid angle > 0
    assert Omega_bd.signal > 0.0


@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
def test_xsgeometry_pixel_index_uncertainty_propagates_to_coordinates():
    """
    Check that the 'pixel_index' uncertainty defined on beam_center and the index grid
    shows up on the detector-plane coordinates x0 and r_perp.
    """
    step = XSGeometry(io_sources=IoSources())

    n0, n1 = 5, 5
    spatial_shape = (n0, n1)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_2d(n0, n1)
    px0_bd, px1_bd = step._extract_pixel_pitches(pixel_size_bd)

    x0_bd, x1_bd, r_perp_bd, R_bd = step._compute_coordinates(
        RoD=2,
        spatial_shape=spatial_shape,
        beam_center_bd=beam_center_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    # We expect 'pixel_index' uncertainties to be present on x0 and r_perp
    assert "pixel_index" in x0_bd.uncertainties
    assert "pixel_index" in r_perp_bd.uncertainties

    unc_x0 = x0_bd.uncertainties["pixel_index"]
    unc_r = r_perp_bd.uncertainties["pixel_index"]

    # Off-centre pixels should have finite, non-zero uncertainties
    row_c, col_c = n0 // 2, n1 // 2
    col_right = col_c + 1

    assert np.isfinite(unc_x0[row_c, col_right])
    assert unc_x0[row_c, col_right] > 0.0

    assert np.isfinite(unc_r[row_c, col_right])
    assert unc_r[row_c, col_right] > 0.0


@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
def test_xsgeometry_Q_has_nonzero_uncertainty_off_center():
    """
    Check that the 'pixel_index' uncertainties propagate all the way to Q
    and are non-zero away from the beam centre.
    """
    step = XSGeometry(io_sources=IoSources())

    n0, n1 = 5, 5
    spatial_shape = (n0, n1)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_2d(n0, n1)
    px0_bd, px1_bd = step._extract_pixel_pitches(pixel_size_bd)

    x0_bd, x1_bd, r_perp_bd, R_bd = step._compute_coordinates(
        RoD=2,
        spatial_shape=spatial_shape,
        beam_center_bd=beam_center_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    _, theta_bd, sin_theta_bd = step._compute_angles(
        r_perp_bd=r_perp_bd,
        detector_distance_bd=D_bd,
    )
    Q_bd = step._compute_Q(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
    )

    # We expect the 'pixel_index' uncertainty key to exist on Q as well
    assert "pixel_index" in Q_bd.uncertainties

    unc_Q = Q_bd.uncertainties["pixel_index"]

    row_c, col_c = n0 // 2, n1 // 2
    col_right = col_c + 1

    # At the exact beam pixel, Q ≈ 0 and derivative is singular; uncertainty may be inf/NaN.
    # Off-centre, we expect finite, non-zero uncertainty.
    assert np.isfinite(unc_Q[row_c, col_right])
    assert unc_Q[row_c, col_right] > 0.0


# ---------------------------------------------------------------------------
# Thin integration test using prepare_execution + calculate
# ---------------------------------------------------------------------------


def test_xsgeometry_prepare_and_calculate_integration():
    """
    Integration-style test:
    - Build a minimal processing_data and configuration.
    - Override _load_geometry to return synthetic BaseData.
    - Run prepare_execution() and calculate().
    - Check that the expected geometry keys are present in the output databundle.
    """
    step = XSGeometry(io_sources=IoSources())

    # Build a simple 2D signal databundle
    n0, n1 = 5, 5
    signal_bd = BaseData(
        signal=np.ones((n0, n1), dtype=float),
        units=ureg.dimensionless,
        rank_of_data=2,
    )

    processing_data = ProcessingData()
    # match what XSGeometry expects: processing_data["signal"]["signal"] is a BaseData
    processing_data["signal"] = DataBundle({"signal": signal_bd})

    # Fake geometry via helper; we inject this directly into _load_geometry.
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_2d(n0, n1)
    fake_geom = {
        "detector_distance": D_bd,
        "pixel_size": pixel_size_bd,
        "beam_center": beam_center_bd,
        "wavelength": wavelength_bd,
    }

    # Minimal configuration
    step.configuration = {
        "with_processing_keys": ["signal"],
        "output_processing_key": None,
    }

    # Attach processing_data and bypass I/O in _load_geometry
    step.processing_data = processing_data
    step._prepared_data = {}
    step._load_geometry = lambda: fake_geom  # type: ignore[assignment]

    # Execute prepare + calculate directly (no ProcessStep.execute merge)
    step.prepare_execution()
    out = step.calculate()

    assert "signal" in out
    databundle = out["signal"]

    for key in ["Q", "Q0", "Q1", "Q2", "Psi", "Theta", "Omega"]:
        assert key in databundle, f"Missing geometry key '{key}' in databundle."
        assert isinstance(databundle[key], BaseData), f"{key} is not a BaseData."

    # simple sanity check on Q field
    Q_bd = databundle["Q"]
    center = (n0 // 2, n1 // 2)
    assert_allclose(Q_bd.signal[center], 0.0, atol=1e-12)
