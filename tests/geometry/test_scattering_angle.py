from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor.geometry.scattering_angle import q_angle_factor, signed_q_from_angle


def test_signed_q_from_angle_retains_wing_sign():
    angle = np.array([-1.0e-5, 0.0, 1.0e-5])
    result = signed_q_from_angle(angle, np.array(0.1))

    assert_allclose(result, 4.0 * np.pi / 0.1 * np.sin(angle / 2.0))
    assert result[0] < 0.0 < result[2]


def test_signed_q_rejects_nonpositive_wavelength():
    with pytest.raises(ValueError, match="positive"):
        signed_q_from_angle(np.array([0.0]), np.array([0.0]))


def test_angle_conventions_use_scattering_or_bragg_angle():
    angle = np.array([0.2])
    wavelength = np.array(0.1)

    assert_allclose(
        signed_q_from_angle(angle, wavelength, convention="two_theta"),
        4.0 * np.pi / wavelength * np.sin(angle / 2.0),
    )
    assert_allclose(
        signed_q_from_angle(angle, wavelength, convention="theta"),
        4.0 * np.pi / wavelength * np.sin(angle),
    )
    assert q_angle_factor("SCATTERING_ANGLE") == 0.5
    with pytest.raises(ValueError, match="Unsupported angle convention"):
        q_angle_factor("yaw")
