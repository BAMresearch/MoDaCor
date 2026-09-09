# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pytest

from modacor.models.attenuation import (
    gaussian_beam_profile,
    image_beam_profile,
    trapezoid_beam_profile,
)


def test_image_profile_uses_calibrated_pixel_centres_and_normalized_weights():
    profile = image_beam_profile(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        pixel_pitch=(2.0, 1.0),
    )

    np.testing.assert_allclose(profile.points, [[0.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
    np.testing.assert_allclose(profile.weights, [0.25, 0.75])
    assert profile.retained_weight_fraction == 1.0


def test_image_downsampling_preserves_weight_and_first_spatial_moment():
    image = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    full = image_beam_profile(image, pixel_pitch=(2.0, 1.0))
    reduced = image_beam_profile(image, pixel_pitch=(2.0, 1.0), downsample=(2, 3))

    assert reduced.points.shape == (1, 3)
    np.testing.assert_allclose(
        np.average(reduced.points, weights=reduced.weights, axis=0),
        np.average(full.points, weights=full.weights, axis=0),
    )
    assert reduced.weights.sum() == pytest.approx(1.0)


def test_image_cutoff_records_discarded_incident_weight():
    profile = image_beam_profile(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        pixel_pitch=(2.0, 1.0),
        relative_weight_cutoff=0.5,
    )

    np.testing.assert_allclose(profile.points, [[1.0, 1.0, 0.0]])
    np.testing.assert_array_equal(profile.weights, [1.0])
    assert profile.retained_weight_fraction == pytest.approx(0.75)


def test_image_centre_calibrates_fractional_reference_pixel():
    profile = image_beam_profile(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        pixel_pitch=(2.0, 1.0),
        image_centre=(0.0, 0.0),
        centre=(10.0, 20.0, 30.0),
    )

    np.testing.assert_allclose(profile.points, [[11.0, 20.0, 30.0], [12.0, 22.0, 30.0]])


def test_beam_plane_rotation_and_direction_keep_points_in_normal_plane():
    direction = np.array([0.2, -0.1, 1.0])
    direction /= np.linalg.norm(direction)
    centre = np.array([3.0, 4.0, 5.0])
    profile = image_beam_profile(
        [[1.0, 1.0], [1.0, 1.0]],
        pixel_pitch=(2.0, 1.0),
        centre=centre,
        incident_direction=direction,
        rotation=0.7,
    )

    np.testing.assert_allclose((profile.points - centre) @ direction, 0.0, atol=1e-14)


def test_gaussian_profile_reproduces_principal_standard_deviations():
    profile = gaussian_beam_profile(
        standard_deviations=(0.4, 0.8),
        quadrature_order=24,
        truncation_sigma=(4.0, 4.0),
    )
    mean = np.average(profile.points[:, :2], weights=profile.weights, axis=0)
    variance = np.average((profile.points[:, :2] - mean) ** 2, weights=profile.weights, axis=0)

    np.testing.assert_allclose(mean, 0.0, atol=1e-14)
    # Four-sigma truncation lowers the variance by about 0.1%.
    np.testing.assert_allclose(variance, [0.8**2, 0.4**2], rtol=2e-3)
    assert profile.retained_weight_fraction == pytest.approx(0.999873319, rel=1e-8)


def test_symmetric_trapezoid_profile_has_expected_extent_and_centroid():
    profile = trapezoid_beam_profile(
        plateau_width=(1.0, 2.0),
        ramp_widths=((0.5, 0.5), (1.0, 1.0)),
        quadrature_order_per_region=4,
    )
    mean = np.average(profile.points, weights=profile.weights, axis=0)

    assert profile.points.shape == (144, 3)
    np.testing.assert_allclose(mean, 0.0, atol=1e-14)
    assert np.max(np.abs(profile.points[:, 0])) < 2.0
    assert np.max(np.abs(profile.points[:, 1])) < 1.0
    assert profile.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "function, arguments, message",
    [
        (image_beam_profile, {"image": [[-1.0]], "pixel_pitch": (1.0, 1.0)}, "image"),
        (
            gaussian_beam_profile,
            {"standard_deviations": (0.0, 1.0)},
            "standard_deviations",
        ),
        (
            trapezoid_beam_profile,
            {"plateau_width": (0.0, 0.0), "ramp_widths": ((0.0, 0.0), (1.0, 1.0))},
            "trapezoid axis",
        ),
    ],
)
def test_invalid_profile_inputs_are_rejected(function, arguments, message):
    with pytest.raises(ValueError, match=message):
        function(**arguments)
