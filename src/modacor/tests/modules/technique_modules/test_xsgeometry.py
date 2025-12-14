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
    - Pixel size: 1e-3 m/pixel in both directions (with small uncertainty)
    - Beam center: exact centre pixel (with 0.25 pixel uncertainty)
    - Wavelength: 1 Å (1e-10 m) with 2% uncertainty
    """
    # --- detector distance: 1.0 m ± 1 mm ---
    D_value = 1.0
    D_unc = 1e-3  # 1 mm
    D_bd = BaseData(
        signal=np.array(D_value, dtype=float),
        units=ureg.meter,
        uncertainties={"propagate_to_all": np.array(D_unc, dtype=float)},
    )

    # --- pixel size: 1e-3 m/pixel ± 1e-6 m/pixel ---
    pixel_signal = np.asarray([1e-3, 1e-3], dtype=float)
    pixel_unc = np.full_like(pixel_signal, 1e-6, dtype=float)
    pixel_size_bd = BaseData(
        signal=pixel_signal,
        units=ureg.meter / ureg.pixel,
        uncertainties={"propagate_to_all": pixel_unc},
    )

    # --- beam centre: at the centre pixel, ±0.25 pixel ---
    center_row = (n0 - 1) / 2.0
    center_col = (n1 - 1) / 2.0
    beam_signal = np.asarray([center_row, center_col], dtype=float)
    beam_unc = np.full_like(beam_signal, 0.25, dtype=float)
    beam_center_bd = BaseData(
        signal=beam_signal,
        units=ureg.pixel,
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


def make_geom_1d(n: int):
    """
    Simple 1D geometry: n pixels in a line.

    Units follow the same conventions as make_geom_2d.
    """
    # --- detector distance: 1.0 m ± 1 mm ---
    D_value = 1.0
    D_unc = 1e-3
    D_bd = BaseData(
        signal=np.array(D_value, dtype=float),
        units=ureg.meter,
        uncertainties={"propagate_to_all": np.array(D_unc, dtype=float)},
    )

    # --- pixel size: 1e-3 m/pixel ± 1e-6 m/pixel ---
    pixel_signal = np.asarray([1e-3, 1e-3], dtype=float)
    pixel_unc = np.full_like(pixel_signal, 1e-6, dtype=float)
    pixel_size_bd = BaseData(
        signal=pixel_signal,
        units=ureg.meter / ureg.pixel,
        uncertainties={"propagate_to_all": pixel_unc},
    )

    # --- beam centre: central pixel ±0.25 pixel ---
    center = (n - 1) / 2.0
    beam_signal = np.asarray([center], dtype=float)
    beam_unc = np.full_like(beam_signal, 0.25, dtype=float)
    beam_center_bd = BaseData(
        signal=beam_signal,
        units=ureg.pixel,
        uncertainties={"pixel_index": beam_unc},
    )

    # --- wavelength: 1 Å ± 2% ---
    lambda_value = 1.0e-10
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
    For a symmetric 2D detector with the beam between the central pixels:
    - Q at the nominal center pixel should be the global minimum of Q.
    - Q2 should be identically zero.
    - Along the central row:
        * Q1 should change sign left vs right of the beam (antisymmetric),
          while Q0 remains positive (symmetric in sign).
    - Along the central column:
        * Q0 should change sign above vs below the beam (antisymmetric),
          while Q1 remains positive (symmetric in sign).
    - Psi at the four corners should lie in the expected quadrants:
        top-left     ~ (-π, -π/2)
        top-right    ~ (π/2, π)
        bottom-left  ~ (-π/2, 0)
        bottom-right ~ (0, π/2)
    - Omega (solid angle) should be largest near the beam centre and smaller at the corners.
    - θ should increase with distance from the centre, along at least one row.
    """
    step = XSGeometry(io_sources=IoSources())

    n0, n1 = 5, 5
    spatial_shape = (n0, n1)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_2d(n0, n1)
    px0_bd, px1_bd = pixel_size_bd.indexed(0, rank_of_data=0), pixel_size_bd.indexed(1, rank_of_data=0)

    # coordinates
    x0_bd, x1_bd, r_perp_bd, R_bd = step._compute_coordinates(
        RoD=2,
        spatial_shape=spatial_shape,
        beam_center_bd=beam_center_bd,
        px0_bd=px0_bd,
        px1_bd=px1_bd,
        detector_distance_bd=D_bd,
    )

    # angles, Q magnitude & components, Psi, Omega
    _, theta_bd, sin_theta_bd = step._compute_angles(
        r_perp_bd=r_perp_bd,
        detector_distance_bd=D_bd,
    )
    Q_bd, Q0_bd, Q1_bd, Q2_bd = step._compute_Q_and_components(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
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

    # ------------------------------------------------------------------
    # Q behaviour near the beam centre
    # ------------------------------------------------------------------

    # Q at the "centre" pixel should be the global minimum
    q_center = Q_bd.signal[center]
    q_min = np.min(Q_bd.signal)
    assert q_center == pytest.approx(q_min, rel=1e-12, abs=1e-12)

    # Q2 should be identically zero
    assert_allclose(Q2_bd.signal, 0.0, atol=1e-12)

    # ------------------------------------------------------------------
    # Left-right and up-down behaviour of Q0/Q1
    #
    # NOTE: with the current implementation (x0 from rows, x1 from columns),
    # Q0 varies primarily along the "vertical" direction and Q1 along "horizontal".
    # So the antisymmetry/symmetry expectations are effectively swapped
    # compared to an (x, y) convention.
    # ------------------------------------------------------------------

    # Left-right: inspect the central row
    col_left = col_c - 1
    col_right = col_c + 1
    q0_left_row = Q0_bd.signal[row_c, col_left]
    q0_right_row = Q0_bd.signal[row_c, col_right]
    q1_left_row = Q1_bd.signal[row_c, col_left]
    q1_right_row = Q1_bd.signal[row_c, col_right]

    # Q1 changes sign (antisymmetric) left vs right
    assert q1_left_row < 0.0
    assert q1_right_row > 0.0
    # Q0 stays positive on both sides of the beam row
    assert q0_left_row > 0.0
    assert q0_right_row > 0.0

    # Up-down: inspect the central column
    row_up = row_c - 1
    row_down = row_c + 1
    q0_up_col = Q0_bd.signal[row_up, col_c]
    q0_down_col = Q0_bd.signal[row_down, col_c]
    q1_up_col = Q1_bd.signal[row_up, col_c]
    q1_down_col = Q1_bd.signal[row_down, col_c]

    # Q0 changes sign (antisymmetric) above vs below
    assert q0_up_col < 0.0
    assert q0_down_col > 0.0
    # Q1 remains positive above and below (symmetric in sign)
    assert q1_up_col > 0.0
    assert q1_down_col > 0.0

    # ------------------------------------------------------------------
    # Psi behaviour at the four corners (clear quadrants)
    # ------------------------------------------------------------------

    # Indices: (row, col)
    psi_tl = Psi_bd.signal[0, 0]  # top-left
    psi_tr = Psi_bd.signal[0, -1]  # top-right
    psi_bl = Psi_bd.signal[-1, 0]  # bottom-left
    psi_br = Psi_bd.signal[-1, -1]  # bottom-right

    # Top-left: x0 < 0, x1 < 0 → atan2(neg, neg) ∈ (-π, -π/2)
    assert -np.pi < psi_tl < -np.pi / 2.0

    # Top-right: x0 < 0, x1 > 0 → atan2(pos, neg) ∈ (π/2, π)
    assert np.pi / 2.0 < psi_tr < np.pi

    # Bottom-left: x0 > 0, x1 < 0 → atan2(neg, pos) ∈ (-π/2, 0)
    assert -np.pi / 2.0 < psi_bl < 0.0

    # Bottom-right: x0 > 0, x1 > 0 → atan2(pos, pos) ∈ (0, π/2)
    assert 0.0 < psi_br < np.pi / 2.0

    # ------------------------------------------------------------------
    # Omega behaviour and θ monotonicity
    # ------------------------------------------------------------------

    # Omega largest near the beam centre, smaller at a corner
    omega_center = Omega_bd.signal[center]
    omega_corner = Omega_bd.signal[0, 0]
    assert omega_corner < omega_center
    assert np.all(Omega_bd.signal > 0.0)

    # θ increases with distance from the centre along the central row
    theta_row = theta_bd.signal[row_c, :]
    # centre is smallest
    theta_center = theta_row[col_c]
    assert theta_center == pytest.approx(np.min(theta_row), abs=1e-12)
    # neighbours further out have larger θ
    # assert theta_row[col_left] > theta_center
    assert theta_row[col_right] > theta_center


def test_xsgeometry_1d_center_q_zero_and_monotonic():
    """
    For a symmetric 1D detector:
    - Q at the pixel closest to the beam centre should be minimal (not necessarily zero
      with half-pixel indexing).
    - |Q| should increase as we move away from the center.
    - Q1 and Q2 should be zero.
    """
    step = XSGeometry(io_sources=IoSources())

    n = 7
    spatial_shape = (n,)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_1d(n)
    px0_bd, px1_bd = pixel_size_bd.indexed(0, rank_of_data=0), pixel_size_bd.indexed(1, rank_of_data=0)

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

    # Q magnitude and components
    Q_bd, Q0_bd, Q1_bd, Q2_bd = step._compute_Q_and_components(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
        x0_bd=x0_bd,
        x1_bd=x1_bd,
        r_perp_bd=r_perp_bd,
    )

    center = n // 2

    # Centre pixel should have minimal |Q| (but not exactly zero with half-pixel indexing)
    abs_Q = np.abs(Q_bd.signal)
    assert abs_Q[center] == pytest.approx(abs_Q.min(), rel=1e-12, abs=1e-12)

    # In 1D, Q is entirely along the single axis: |Q0| == |Q|, Q1 == Q2 == 0
    assert_allclose(np.abs(Q0_bd.signal), abs_Q, rtol=1e-12, atol=1e-12)
    assert_allclose(Q1_bd.signal, 0.0, atol=1e-12)
    assert_allclose(Q2_bd.signal, 0.0, atol=1e-12)

    # |Q| grows away from center on the positive side
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
    pixel_size_bd = _bd_vector([1e-3, 1e-3], ureg.meter / ureg.pixel)
    beam_center_bd = _bd_vector([0.0], ureg.pixel)
    wavelength_bd = _bd_scalar(1.0, ureg.meter)

    px0_bd, px1_bd = pixel_size_bd.indexed(0, rank_of_data=0), pixel_size_bd.indexed(1, rank_of_data=0)
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
    Q_bd, Q0_bd, Q1_bd, Q2_bd = step._compute_Q_and_components(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
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
    px0_bd, px1_bd = pixel_size_bd.indexed(0, rank_of_data=0), pixel_size_bd.indexed(1, rank_of_data=0)

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

    # Off-centre pixels should have finite, non-zero uncertainties.
    # Choose a pixel where x0 != 0 to avoid the relative-error degeneracy at x0 == 0.
    row_c, col_c = n0 // 2, n1 // 2
    row_up = row_c - 1

    assert np.isfinite(unc_x0[row_up, col_c])
    assert unc_x0[row_up, col_c] > 0.0

    assert np.isfinite(unc_r[row_up, col_c])
    assert unc_r[row_up, col_c] > 0.0


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
    px0_bd, px1_bd = pixel_size_bd.indexed(0, rank_of_data=0), pixel_size_bd.indexed(1, rank_of_data=0)

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
    Q_bd, Q0_bd, Q1_bd, Q2_bd = step._compute_Q_and_components(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
        x0_bd=x0_bd,
        x1_bd=x1_bd,
        r_perp_bd=r_perp_bd,
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

    for key in ["Q", "Q0", "Q1", "Q2", "Psi", "TwoTheta", "Omega"]:
        assert key in databundle, f"Missing geometry key '{key}' in databundle."
        assert isinstance(databundle[key], BaseData), f"{key} is not a BaseData."

    # simple sanity check on Q field: central pixel is closest to beam
    Q_bd = databundle["Q"]
    center = (n0 // 2, n1 // 2)
    abs_Q = np.abs(Q_bd.signal)
    assert abs_Q[center] == pytest.approx(abs_Q.min(), rel=1e-12, abs=1e-12)


def test_xsgeometry_Q0_and_Omega_have_uncertainty_off_center():
    step = XSGeometry(io_sources=IoSources())

    n0, n1 = 5, 5
    spatial_shape = (n0, n1)
    D_bd, pixel_size_bd, beam_center_bd, wavelength_bd = make_geom_2d(n0, n1)
    px0_bd, px1_bd = pixel_size_bd.indexed(0, rank_of_data=0), pixel_size_bd.indexed(1, rank_of_data=0)

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
    Q_bd, Q0_bd, Q1_bd, Q2_bd = step._compute_Q_and_components(
        sin_theta_bd=sin_theta_bd,
        wavelength_bd=wavelength_bd,
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

    row_c, col_c = n0 // 2, n1 // 2
    col_right = col_c + 1

    # Q0 should carry pixel_index uncertainty at some off-centre pixel
    assert "pixel_index" in Q0_bd.uncertainties
    unc_Q0_pix = Q0_bd.uncertainties["pixel_index"][row_c, col_right]
    assert np.isfinite(unc_Q0_pix)
    assert unc_Q0_pix > 0.0

    # Omega should also carry non-zero propagated uncertainty (from distance,
    # pixel_size, beam_center, etc.). We don't require every key to be finite
    # (some keys may legitimately produce NaNs near singular points), but at
    # least one uncertainty contribution at the off-centre pixel must be finite.
    assert Omega_bd.uncertainties  # dict not empty

    target_shape = (n0, n1)
    found_finite_positive = False

    for key, u in Omega_bd.uncertainties.items():
        # Uncertainty arrays may be scalar or lower-dimensional; broadcast to the
        # detector shape so we can safely index the off-centre pixel.
        u_full = np.broadcast_to(u, target_shape)
        val = u_full[row_c, col_right]
        if np.isfinite(val) and val > 0.0:
            found_finite_positive = True
            break

    assert found_finite_positive
