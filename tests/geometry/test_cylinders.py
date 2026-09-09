# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pytest

from modacor.geometry import ConcentricCylinderGeometry


def test_central_ray_returns_radius_and_wall_thickness():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))

    lengths = geometry.forward_path_lengths([0.0, 0.0, 0.0], [0.0, 0.0, 4.0])

    np.testing.assert_allclose(lengths, [1.0, 0.25])


def test_ray_entering_from_outside_crosses_both_sides():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))

    lengths = geometry.forward_path_lengths([0.0, 0.0, -2.0], [0.0, 0.0, 1.0])

    np.testing.assert_allclose(lengths, [2.0, 0.5])


def test_ray_pointing_away_from_cylinder_has_no_path():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.25]))

    lengths = geometry.forward_path_lengths([0.0, 0.0, 2.0], [0.0, 0.0, 1.0])

    np.testing.assert_array_equal(lengths, [0.0, 0.0])


def test_off_centre_sample_origin_has_analytic_path_lengths():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 2.0]))

    lengths = geometry.forward_path_lengths([0.0, 0.5, 0.0], [0.0, 0.0, 1.0])

    inner = np.sqrt(1.0 - 0.5**2)
    outer = np.sqrt(2.0**2 - 0.5**2)
    np.testing.assert_allclose(lengths, [inner, outer - inner])


def test_wall_origin_ray_can_cross_sample_and_wall_twice():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 2.0]))

    lengths = geometry.forward_path_lengths([0.0, 0.0, 1.5], [0.0, 0.0, -1.0])

    np.testing.assert_allclose(lengths, [2.0, 1.5])


def test_full_line_intervals_and_phase_lengths_are_consistent():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 2.0]))

    intervals = geometry.line_boundary_intervals([0.0, 0.5, 0.0], [0.0, 0.0, 2.0])
    lengths = geometry.line_phase_path_lengths([0.0, 0.5, 0.0], [0.0, 0.0, 2.0])

    inner_half_chord = np.sqrt(1.0 - 0.5**2)
    outer_half_chord = np.sqrt(2.0**2 - 0.5**2)
    np.testing.assert_allclose(
        intervals, [[-inner_half_chord, inner_half_chord], [-outer_half_chord, outer_half_chord]]
    )
    np.testing.assert_allclose(lengths, [2.0 * inner_half_chord, 2.0 * (outer_half_chord - inner_half_chord)])


def test_full_line_that_misses_inner_phase_has_only_wall_length():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 2.0]))

    intervals = geometry.line_boundary_intervals([0.0, 1.5, 0.0], [0.0, 0.0, 1.0])
    lengths = geometry.line_phase_path_lengths([0.0, 1.5, 0.0], [0.0, 0.0, 1.0])

    np.testing.assert_array_equal(np.isnan(intervals[0]), [True, True])
    np.testing.assert_allclose(lengths, [0.0, 2.0 * np.sqrt(2.0**2 - 1.5**2)])


def test_origins_and_directions_broadcast():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 2.0]))
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])

    lengths = geometry.forward_path_lengths(origins, [0.0, 0.0, 1.0])

    assert lengths.shape == (2, 2)
    np.testing.assert_allclose(lengths[0], [1.0, 1.0])
    np.testing.assert_allclose(lengths[1].sum(), np.sqrt(2.0**2 - 0.5**2))


def test_axis_sign_and_rigid_rotation_do_not_change_lengths():
    origin = np.array([0.2, -0.3, 0.1])
    direction = np.array([0.1, 0.4, 0.8])
    axis = np.array([1.0, 1.0, 0.25])
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0, 1.4]), axis=axis)

    reference = geometry.forward_path_lengths(origin, direction)
    reversed_axis = ConcentricCylinderGeometry(radii=np.array([1.0, 1.4]), axis=-axis)
    rotated = ConcentricCylinderGeometry(radii=np.array([1.0, 1.4]), axis=rotation @ axis)

    np.testing.assert_allclose(reversed_axis.forward_path_lengths(origin, direction), reference)
    np.testing.assert_allclose(
        rotated.forward_path_lengths(rotation @ origin, rotation @ direction),
        reference,
    )


def test_tangent_ray_has_zero_length():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]))

    lengths = geometry.forward_path_lengths([0.0, 1.0, -2.0], [0.0, 0.0, 1.0])

    np.testing.assert_allclose(lengths, [0.0])


def test_parallel_ray_is_rejected_for_infinite_cylinder():
    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]))

    with pytest.raises(ValueError, match="parallel"):
        geometry.forward_path_lengths([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "radii",
    [[], [0.0], [-1.0], [1.0, 1.0], [2.0, 1.0], [1.0, np.nan]],
)
def test_invalid_radii_are_rejected(radii):
    with pytest.raises(ValueError, match="radii"):
        ConcentricCylinderGeometry(radii=np.asarray(radii))


def test_invalid_axis_and_zero_direction_are_rejected():
    with pytest.raises(ValueError, match="axis"):
        ConcentricCylinderGeometry(radii=np.array([1.0]), axis=np.zeros(3))

    geometry = ConcentricCylinderGeometry(radii=np.array([1.0]))
    with pytest.raises(ValueError, match="directions"):
        geometry.forward_path_lengths([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
