# SPDX-License-Identifier: BSD-3-Clause
"""Numerical scattering-angle geometry kernels."""

from __future__ import annotations

__all__ = ["signed_q_from_scattering_angle"]

import numpy as np


def signed_q_from_scattering_angle(angle_radians: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """Return ``4*pi/wavelength * sin(angle/2)`` without folding the sign."""

    angle = np.asarray(angle_radians, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("signed_q_from_scattering_angle wavelength must be finite and positive.")
    return 4.0 * np.pi * np.sin(angle / 2.0) / wavelength
