# SPDX-License-Identifier: BSD-3-Clause
"""Numerical scattering-angle geometry kernels."""

from __future__ import annotations

__all__ = ["q_angle_factor", "signed_q_from_angle"]

import numpy as np

_ANGLE_FACTORS = {
    "scattering_angle": 0.5,
    "two_theta": 0.5,
    "bragg_angle": 1.0,
    "theta": 1.0,
}


def q_angle_factor(convention: str) -> float:
    """Return the sine-argument factor for a supported diffraction angle convention."""

    normalized = str(convention).strip().lower()
    try:
        return _ANGLE_FACTORS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ANGLE_FACTORS))
        raise ValueError(f"Unsupported angle convention {convention!r}; expected one of: {supported}.") from exc


def signed_q_from_angle(
    angle_radians: np.ndarray,
    wavelength: np.ndarray,
    *,
    convention: str = "scattering_angle",
) -> np.ndarray:
    """Return signed Q for an angle in radians and wavelength in any length unit."""

    angle = np.asarray(angle_radians, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("signed_q_from_angle wavelength must be finite and positive.")
    return 4.0 * np.pi * np.sin(angle * q_angle_factor(convention)) / wavelength
