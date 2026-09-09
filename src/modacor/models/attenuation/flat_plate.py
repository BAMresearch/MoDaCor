# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["flat_plate_relative_attenuation", "flat_plate_relative_attenuation_derivative"]

import numpy as np


def flat_plate_relative_attenuation(transmission: float, cos_alpha) -> np.ndarray:
    """Return the depth-averaged relative attenuation for a uniform flat plate."""
    cos_alpha = np.asarray(cos_alpha, dtype=float)
    x = ((1.0 / cos_alpha) - 1.0) * np.log(float(transmission))
    result = np.ones_like(x, dtype=float)
    regular = np.abs(x) >= 1e-6
    result[regular] = np.expm1(x[regular]) / x[regular]
    small = ~regular
    result[small] = 1.0 + x[small] / 2.0 + x[small] ** 2 / 6.0
    return result


def flat_plate_relative_attenuation_derivative(transmission: float, cos_alpha) -> np.ndarray:
    """Return the derivative of relative attenuation with respect to transmission."""
    transmission = float(transmission)
    cos_alpha = np.asarray(cos_alpha, dtype=float)
    coefficient = (1.0 / cos_alpha) - 1.0
    x = coefficient * np.log(transmission)
    derivative_x = np.empty_like(x, dtype=float)
    regular = np.abs(x) >= 1e-6
    derivative_x[regular] = (np.exp(x[regular]) * x[regular] - np.expm1(x[regular])) / x[regular] ** 2
    small = ~regular
    derivative_x[small] = 0.5 + x[small] / 3.0 + x[small] ** 2 / 8.0
    return derivative_x * coefficient / transmission
