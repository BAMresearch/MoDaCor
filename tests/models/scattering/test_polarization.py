from __future__ import annotations

import numpy as np

from modacor.models.scattering import linear_polarization_factor


def test_unpolarized_factor_is_azimuth_independent() -> None:
    two_theta = np.full(4, np.pi / 3.0)
    psi = np.array([0.0, np.pi / 4.0, np.pi / 2.0, np.pi])

    result = linear_polarization_factor(two_theta, psi, fraction=0.5)

    np.testing.assert_allclose(result, (1.0 + np.cos(two_theta) ** 2) / 2.0)


def test_offset_rotates_linear_polarization_axis() -> None:
    two_theta = np.array([np.pi / 2.0])

    horizontal = linear_polarization_factor(two_theta, np.array([0.0]), fraction=1.0)
    rotated = linear_polarization_factor(
        two_theta,
        np.array([np.pi / 2.0]),
        fraction=1.0,
        offset_radian=np.pi / 2.0,
    )

    np.testing.assert_allclose(rotated, horizontal)
