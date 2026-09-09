# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from modacor.models.attenuation import (
    flat_plate_relative_attenuation,
    flat_plate_relative_attenuation_derivative,
)


def test_flat_plate_relative_attenuation_matches_depth_integral():
    transmission = 0.5
    cos_alpha = np.array([1.0, 0.8, 0.5])
    x = (1.0 / cos_alpha - 1.0) * np.log(transmission)
    expected = np.ones_like(x)
    expected[1:] = np.expm1(x[1:]) / x[1:]
    np.testing.assert_allclose(
        flat_plate_relative_attenuation(transmission, cos_alpha),
        expected,
    )


def test_flat_plate_derivative_matches_central_difference():
    transmission = 0.5
    cos_alpha = np.array([1.0, 0.999999, 0.8, 0.5])
    delta = 1e-6
    numerical = (
        flat_plate_relative_attenuation(transmission + delta, cos_alpha)
        - flat_plate_relative_attenuation(transmission - delta, cos_alpha)
    ) / (2.0 * delta)
    np.testing.assert_allclose(
        flat_plate_relative_attenuation_derivative(transmission, cos_alpha),
        numerical,
        rtol=2e-5,
        atol=2e-10,
    )
