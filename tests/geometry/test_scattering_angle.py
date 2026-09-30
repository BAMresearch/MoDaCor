from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor.geometry.scattering_angle import signed_q_from_scattering_angle


def test_signed_q_from_scattering_angle_retains_wing_sign():
    angle = np.array([-1.0e-5, 0.0, 1.0e-5])
    result = signed_q_from_scattering_angle(angle, np.array(0.1))

    assert_allclose(result, 4.0 * np.pi / 0.1 * np.sin(angle / 2.0))
    assert result[0] < 0.0 < result[2]


def test_signed_q_rejects_nonpositive_wavelength():
    with pytest.raises(ValueError, match="positive"):
        signed_q_from_scattering_angle(np.array([0.0]), np.array([0.0]))
