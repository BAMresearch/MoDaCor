from __future__ import annotations

import numpy as np

from modacor.models.attenuation.planar import planar_absorption_efficiency, planar_transmission


def test_planar_transmission_uses_angle_dependent_path_length() -> None:
    cos_alpha = np.array([1.0, 0.5])

    result = planar_transmission(2.0, 0.25, cos_alpha)

    np.testing.assert_allclose(result, np.exp(np.array([-0.5, -1.0])))


def test_planar_absorption_is_complement_of_transmission() -> None:
    cos_alpha = np.array([1.0, 0.75, 0.5])
    transmission = planar_transmission(2.0, 0.25, cos_alpha)

    np.testing.assert_allclose(planar_absorption_efficiency(2.0, 0.25, cos_alpha), 1.0 - transmission)
