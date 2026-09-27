# SPDX-License-Identifier: BSD-3-Clause
"""Polarization-factor kernels for scattering data."""

from __future__ import annotations

__all__ = ["linear_polarization_factor"]

import numpy as np


def linear_polarization_factor(
    two_theta,
    psi,
    fraction: float,
    offset_radian: float = 0.0,
) -> np.ndarray:
    """Return the linear-polarization intensity factor for radian angles."""

    two_theta = np.asarray(two_theta, dtype=float)
    psi = np.asarray(psi, dtype=float)
    phi = psi - float(offset_radian)
    sin2 = np.sin(two_theta) ** 2
    cos_phi2 = np.cos(phi) ** 2
    sin_phi2 = np.sin(phi) ** 2
    return float(fraction) * (1.0 - sin2 * cos_phi2) + (1.0 - float(fraction)) * (1.0 - sin2 * sin_phi2)
