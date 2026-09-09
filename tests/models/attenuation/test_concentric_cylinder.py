# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pytest

from modacor.geometry import ConcentricCylinderGeometry
from modacor.models.attenuation import (
    adaptive_attenuation_factors_on_grid,
    attenuation_factors_at_detectors,
    attenuation_factors_for_phase_at_detectors,
    beam_chord_quadrature,
    direct_beam_transmission,
    uniform_cross_section_quadrature,
)


def _reference_inputs():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]))
    points, weights = uniform_cross_section_quadrature(radius=1.0, radial_order=8, azimuthal_order=32)
    detectors = np.array([[0.0, 0.0, 100.0], [0.0, 100.0, 0.0], [0.0, 0.0, -100.0]])
    return geometry, points, weights, detectors


def _detector_grid(*, slow_size=33, fast_size=35, extent=500.0, distance=1000.0):
    slow = np.linspace(-extent, extent, slow_size)
    fast = np.linspace(-extent, extent, fast_size)
    fast_grid, slow_grid = np.meshgrid(fast, slow)
    return np.stack(
        (fast_grid, slow_grid, np.full_like(fast_grid, distance)),
        axis=-1,
    )


def test_cross_section_weights_sum_to_disk_area_and_points_lie_in_plane():
    points, weights = uniform_cross_section_quadrature(
        radius=2.0,
        axis=[1.0, 1.0, 0.0],
        centre=[2.0, -2.0, 3.0],
        radial_order=6,
        azimuthal_order=20,
    )
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

    assert points.shape == (120, 3)
    assert weights.shape == (120,)
    assert weights.sum() == pytest.approx(4.0 * np.pi)
    np.testing.assert_allclose((points - [2.0, -2.0, 3.0]) @ axis, 0.0, atol=1e-14)


def test_beam_chord_quadrature_weights_equal_illuminated_phase_volume():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))
    beam_points = np.array([[0.0, 0.0, 0.0]])

    sample_points, sample_weights = beam_chord_quadrature(
        geometry=geometry,
        beam_points=beam_points,
        beam_weights=[2.0],
        phase_index=0,
        chord_order=5,
    )
    wall_points, wall_weights = beam_chord_quadrature(
        geometry=geometry,
        beam_points=beam_points,
        beam_weights=[2.0],
        phase_index=1,
        chord_order=5,
    )

    assert sample_points.shape == (5, 3)
    assert wall_points.shape == (10, 3)
    assert sample_weights.sum() == pytest.approx(2.0 * 2.0)
    assert wall_weights.sum() == pytest.approx(2.0 * 2.0 * 0.25)
    assert np.all(np.linalg.norm(sample_points[:, 1:], axis=1) < 1.0)
    assert np.all(np.linalg.norm(wall_points[:, 1:], axis=1) > 1.0)


def test_annular_quadrature_keeps_full_chord_when_beam_misses_inner_phase():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 2.0]))

    points, weights = beam_chord_quadrature(
        geometry=geometry,
        beam_points=[[0.0, 1.5, 0.0]],
        beam_weights=[1.0],
        phase_index=1,
        chord_order=6,
    )

    assert points.shape == (6, 3)
    assert weights.sum() == pytest.approx(2.0 * np.sqrt(2.0**2 - 1.5**2))


def test_direct_transmission_is_beam_area_weighted_and_includes_missed_rays():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))
    sample_mu = 0.4
    wall_mu = 1.2

    transmission = direct_beam_transmission(
        geometry=geometry,
        attenuation_coefficients=[sample_mu, wall_mu],
        beam_points=[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        beam_weights=[3.0, 1.0],
    )

    central_transmission = np.exp(-(2.0 * sample_mu + 2.0 * 0.25 * wall_mu))
    assert transmission == pytest.approx((3.0 * central_transmission + 1.0) / 4.0)


def test_equal_sample_and_wall_coefficients_match_one_outer_phase():
    layered = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))
    homogeneous = ConcentricCylinderGeometry(radii=np.array([1.25]))
    beam_points = np.array([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    beam_weights = np.array([0.5, 1.0, 0.5])
    detectors = np.array([[0.0, 100.0, 100.0], [0.0, -50.0, 100.0]])

    scattering_points, volume_weights = beam_chord_quadrature(
        geometry=layered,
        beam_points=beam_points,
        beam_weights=beam_weights,
        phase_index=0,
        chord_order=12,
    )
    layered_factor = attenuation_factors_at_detectors(
        geometry=layered,
        attenuation_coefficients=[0.8, 0.8],
        scattering_points=scattering_points,
        volume_weights=volume_weights,
        detector_positions=detectors,
    )
    homogeneous_factor = attenuation_factors_at_detectors(
        geometry=homogeneous,
        attenuation_coefficients=[0.8],
        scattering_points=scattering_points,
        volume_weights=volume_weights,
        detector_positions=detectors,
    )

    np.testing.assert_allclose(layered_factor, homogeneous_factor, rtol=1e-14, atol=1e-14)


def test_zero_wall_attenuation_matches_sample_only_geometry():
    layered = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))
    sample_only = ConcentricCylinderGeometry(radii=np.array([1.0]))
    beam_points = np.array([[0.0, -0.4, 0.0], [0.0, 0.4, 0.0]])
    detectors = np.array([[0.0, 100.0, 100.0], [0.0, -50.0, 100.0]])
    common = {
        "beam_points": beam_points,
        "beam_weights": [1.0, 1.0],
        "origin_phase_index": 0,
        "detector_positions": detectors,
        "chord_order": 12,
    }

    layered_factor = attenuation_factors_for_phase_at_detectors(
        geometry=layered,
        attenuation_coefficients=[0.8, 0.0],
        **common,
    )
    sample_only_factor = attenuation_factors_for_phase_at_detectors(
        geometry=sample_only,
        attenuation_coefficients=[0.8],
        **common,
    )

    np.testing.assert_allclose(layered_factor, sample_only_factor, rtol=1e-14, atol=1e-14)


def test_sample_absorption_reduces_wall_origin_factor_in_filled_capillary():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))
    detectors = np.array([[0.0, 100.0, 100.0], [0.0, -100.0, 100.0]])
    common = {
        "geometry": geometry,
        "beam_points": [[0.0, 0.0, 0.0]],
        "beam_weights": [1.0],
        "origin_phase_index": 1,
        "detector_positions": detectors,
        "chord_order": 20,
    }

    empty_factor = attenuation_factors_for_phase_at_detectors(
        attenuation_coefficients=[0.0, 1.2],
        **common,
    )
    filled_factor = attenuation_factors_for_phase_at_detectors(
        attenuation_coefficients=[0.8, 1.2],
        **common,
    )

    assert np.all(filled_factor < empty_factor)


def test_beam_chord_and_cross_section_quadratures_converge_to_same_factor():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]))
    transverse_nodes, transverse_weights = np.polynomial.legendre.leggauss(32)
    beam_points = np.column_stack((np.zeros_like(transverse_nodes), transverse_nodes, np.zeros_like(transverse_nodes)))
    chord_points, chord_weights = beam_chord_quadrature(
        geometry=geometry,
        beam_points=beam_points,
        beam_weights=transverse_weights,
        phase_index=0,
        chord_order=32,
    )
    disk_points, disk_weights = uniform_cross_section_quadrature(
        radius=1.0,
        radial_order=24,
        azimuthal_order=192,
    )
    detector = np.array([[0.0, 1000.0, 1000.0]])

    chord_factor = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[1.5],
        scattering_points=chord_points,
        volume_weights=chord_weights,
        detector_positions=detector,
    )
    disk_factor = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[1.5],
        scattering_points=disk_points,
        volume_weights=disk_weights,
        detector_positions=detector,
    )

    np.testing.assert_allclose(chord_factor, disk_factor, rtol=1e-4)


@pytest.mark.parametrize(
    "mu_radius, expected",
    [
        (0.5, [0.4378234194, 0.4591165885, 0.4836187785]),
        (1.5, [0.1009213475, 0.1482592384, 0.1968237080]),
        (3.0, [0.0212966799, 0.0583137080, 0.0973230705]),
    ],
)
def test_homogeneous_factors_match_diffpy_labpdfproc_reference(mu_radius, expected):
    """Cross-check Chen et al.'s published reference implementation (muD=2 muR)."""
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]))
    points, weights = uniform_cross_section_quadrature(
        radius=1.0,
        radial_order=24,
        azimuthal_order=192,
    )
    angles = np.deg2rad([30.0, 90.0, 150.0])
    detectors = 1.0e8 * np.column_stack((np.zeros_like(angles), np.sin(angles), np.cos(angles)))

    actual = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[mu_radius],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detectors,
    )

    # diffpy.labpdfproc's interpolation itself differs from converged direct
    # integration by up to roughly 0.25% at muR=3, so this is a reference-code
    # agreement tolerance rather than our quadrature convergence tolerance.
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=1e-7)


def test_zero_attenuation_returns_one_for_every_requested_detector():
    geometry, points, weights, detectors = _reference_inputs()

    factors = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[0.0],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detectors,
    )

    np.testing.assert_array_equal(factors, np.ones(3))


def test_chunking_does_not_change_point_ray_result():
    geometry, points, weights, detectors = _reference_inputs()
    common = {
        "geometry": geometry,
        "attenuation_coefficients": [0.7],
        "scattering_points": points,
        "volume_weights": weights,
        "detector_positions": detectors,
    }

    one_chunk = attenuation_factors_at_detectors(**common, detector_chunk_size=100)
    single_detector_chunks = attenuation_factors_at_detectors(**common, detector_chunk_size=1)

    np.testing.assert_allclose(single_detector_chunks, one_chunk, rtol=1e-14, atol=1e-14)


def test_coefficient_sets_reuse_geometry_without_changing_point_ray_results():
    geometry, points, weights, detectors = _reference_inputs()
    coefficient_sets = np.array([[0.2], [0.7]])

    combined = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=coefficient_sets,
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detectors,
    )
    separate = np.stack(
        [
            attenuation_factors_at_detectors(
                geometry=geometry,
                attenuation_coefficients=coefficients,
                scattering_points=points,
                volume_weights=weights,
                detector_positions=detectors,
            )
            for coefficients in coefficient_sets
        ]
    )

    assert combined.shape == (2, detectors.shape[0])
    np.testing.assert_allclose(combined, separate, rtol=1e-14, atol=1e-14)


def test_adaptive_grid_agrees_with_withheld_exact_pixel_rays():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]), axis=[1.0, 0.3, 0.1])
    points, weights = uniform_cross_section_quadrature(
        radius=1.0,
        axis=geometry.axis,
        radial_order=8,
        azimuthal_order=32,
    )
    detector_grid = _detector_grid(slow_size=65, fast_size=67)

    adaptive = adaptive_attenuation_factors_on_grid(
        geometry=geometry,
        attenuation_coefficients=[1.5],
        scattering_points=points,
        volume_weights=weights,
        detector_position_grid=detector_grid,
        relative_tolerance=5e-4,
    )
    withheld = ~adaptive.evaluated_mask
    exact = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[1.5],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detector_grid[withheld],
    )

    assert np.any(withheld)
    assert np.count_nonzero(adaptive.evaluated_mask) < detector_grid.shape[0] * detector_grid.shape[1]
    np.testing.assert_allclose(adaptive.factors[withheld], exact, rtol=1.5e-3, atol=1e-12)


def test_adaptive_grid_uses_one_mesh_for_multiple_coefficient_sets():
    geometry, points, weights, _detectors = _reference_inputs()
    detector_grid = _detector_grid(slow_size=17, fast_size=19)
    coefficient_sets = np.array([[0.2], [1.5]])

    combined = adaptive_attenuation_factors_on_grid(
        geometry=geometry,
        attenuation_coefficients=coefficient_sets,
        scattering_points=points,
        volume_weights=weights,
        detector_position_grid=detector_grid,
        relative_tolerance=5e-4,
    )
    exact = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=coefficient_sets,
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detector_grid.reshape(-1, 3),
    ).reshape(2, *detector_grid.shape[:2])

    assert combined.factors.shape == (2, *detector_grid.shape[:2])
    np.testing.assert_allclose(combined.factors, exact, rtol=1.5e-3, atol=1e-12)


def test_adaptive_grid_falls_back_to_exact_pixels_at_maximum_depth():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]), axis=[1.0, 0.3, 0.1])
    points, weights = uniform_cross_section_quadrature(
        radius=1.0,
        axis=geometry.axis,
        radial_order=6,
        azimuthal_order=24,
    )
    detector_grid = _detector_grid(slow_size=9, fast_size=9)

    adaptive = adaptive_attenuation_factors_on_grid(
        geometry=geometry,
        attenuation_coefficients=[3.0],
        scattering_points=points,
        volume_weights=weights,
        detector_position_grid=detector_grid,
        relative_tolerance=1e-12,
        max_depth=0,
    )
    exact = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[3.0],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detector_grid.reshape(-1, 3),
    ).reshape(detector_grid.shape[:2])

    assert adaptive.exact_fallback_pixel_count == detector_grid.shape[0] * detector_grid.shape[1]
    assert np.all(adaptive.evaluated_mask)
    np.testing.assert_array_equal(adaptive.factors, exact)


def test_adaptive_grid_never_evaluates_masked_detector_pixels():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]), axis=[1.0, 0.3, 0.1])
    points, weights = uniform_cross_section_quadrature(
        radius=1.0,
        axis=geometry.axis,
        radial_order=6,
        azimuthal_order=24,
    )
    detector_grid = _detector_grid(slow_size=17, fast_size=19)
    active = np.ones(detector_grid.shape[:2], dtype=bool)
    active[8, :] = False
    active[:, 9] = False

    adaptive = adaptive_attenuation_factors_on_grid(
        geometry=geometry,
        attenuation_coefficients=[1.5],
        scattering_points=points,
        volume_weights=weights,
        detector_position_grid=detector_grid,
        active_mask=active,
        relative_tolerance=5e-4,
    )
    exact = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[1.5],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detector_grid[active],
    )

    assert not np.any(adaptive.evaluated_mask[~active])
    np.testing.assert_array_equal(adaptive.factors[~active], 1.0)
    np.testing.assert_allclose(adaptive.factors[active], exact, rtol=1.5e-3, atol=1e-12)


def test_adaptive_grid_with_no_active_pixels_returns_identity_without_evaluation():
    geometry, points, weights, _detectors = _reference_inputs()
    detector_grid = _detector_grid(slow_size=5, fast_size=7)

    adaptive = adaptive_attenuation_factors_on_grid(
        geometry=geometry,
        attenuation_coefficients=[0.5],
        scattering_points=points,
        volume_weights=weights,
        detector_position_grid=detector_grid,
        active_mask=np.zeros(detector_grid.shape[:2], dtype=bool),
    )

    np.testing.assert_array_equal(adaptive.factors, 1.0)
    assert not np.any(adaptive.evaluated_mask)
    assert adaptive.exact_fallback_pixel_count == 0


def test_adaptive_grid_allows_nonfinite_coordinates_only_at_masked_pixels():
    geometry, points, weights, _detectors = _reference_inputs()
    detector_grid = _detector_grid(slow_size=5, fast_size=7)
    detector_grid[2, 3] = np.nan
    active = np.ones(detector_grid.shape[:2], dtype=bool)
    active[2, 3] = False

    adaptive = adaptive_attenuation_factors_on_grid(
        geometry=geometry,
        attenuation_coefficients=[0.5],
        scattering_points=points,
        volume_weights=weights,
        detector_position_grid=detector_grid,
        active_mask=active,
    )

    assert adaptive.factors[2, 3] == 1.0
    with pytest.raises(ValueError, match="Active detector positions"):
        adaptive_attenuation_factors_on_grid(
            geometry=geometry,
            attenuation_coefficients=[0.5],
            scattering_points=points,
            volume_weights=weights,
            detector_position_grid=detector_grid,
        )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"detector_position_grid": np.zeros((3, 3))}, "detector_position_grid"),
        ({"relative_tolerance": 0.0}, "relative_tolerance"),
        ({"absolute_tolerance": -1.0}, "absolute_tolerance"),
        ({"max_depth": -1}, "max_depth"),
        ({"active_mask": np.zeros((3, 3))}, "active_mask"),
    ],
)
def test_invalid_adaptive_grid_inputs_are_rejected(override, message):
    geometry, points, weights, _detectors = _reference_inputs()
    arguments = {
        "geometry": geometry,
        "attenuation_coefficients": [0.5],
        "scattering_points": points,
        "volume_weights": weights,
        "detector_position_grid": _detector_grid(slow_size=5, fast_size=5),
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        adaptive_attenuation_factors_on_grid(**arguments)


def test_point_ray_integral_matches_explicit_detector_loop():
    geometry, points, weights, detectors = _reference_inputs()
    coefficient = 0.4

    actual = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[coefficient],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detectors,
    )
    incoming = geometry.forward_path_lengths(points, [0.0, 0.0, -1.0])[:, 0]
    expected = []
    for detector in detectors:
        outgoing = geometry.forward_path_lengths(points, detector - points)[:, 0]
        expected.append(np.average(np.exp(-coefficient * (incoming + outgoing)), weights=weights))

    np.testing.assert_allclose(actual, expected)


def test_rigid_rotation_preserves_detector_factors():
    geometry, points, weights, detectors = _reference_inputs()
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    reference = attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=[0.6],
        scattering_points=points,
        volume_weights=weights,
        detector_positions=detectors,
    )
    rotated_geometry = ConcentricCylinderGeometry(radii=np.array([1.0]), axis=rotation @ geometry.axis)

    rotated = attenuation_factors_at_detectors(
        geometry=rotated_geometry,
        attenuation_coefficients=[0.6],
        scattering_points=points @ rotation.T,
        volume_weights=weights,
        detector_positions=detectors @ rotation.T,
        incident_direction=rotation @ np.array([0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(rotated, reference, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"attenuation_coefficients": [-1.0]}, "attenuation_coefficients"),
        ({"volume_weights": np.array([])}, "volume_weights"),
        ({"detector_chunk_size": 0}, "detector_chunk_size"),
    ],
)
def test_invalid_integrator_inputs_are_rejected(override, message):
    geometry, points, weights, detectors = _reference_inputs()
    arguments = {
        "geometry": geometry,
        "attenuation_coefficients": [0.5],
        "scattering_points": points,
        "volume_weights": weights,
        "detector_positions": detectors,
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        attenuation_factors_at_detectors(**arguments)
