# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["identity_matrix4", "rotation_matrix4", "translation_matrix4"]

import numpy as np

from modacor.geometry.vectors import unit_vector3


def identity_matrix4() -> np.ndarray:
    """Return a 4 by 4 homogeneous identity transform."""
    return np.eye(4, dtype=float)


def translation_matrix4(vector) -> np.ndarray:
    """Return a homogeneous translation transform for a finite three-vector."""
    translation = np.asarray(vector, dtype=float)
    if translation.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {translation.shape}.")
    if not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain only finite values.")
    matrix = identity_matrix4()
    matrix[:3, 3] = translation
    return matrix


def rotation_matrix4(axis, angle_radians: float) -> np.ndarray:
    """Return a homogeneous right-handed rotation about an axis through the origin."""
    x, y, z = unit_vector3(axis, name="rotation axis")
    angle = float(angle_radians)
    if not np.isfinite(angle):
        raise ValueError("rotation angle must be finite.")
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus_cosine = 1.0 - cosine
    matrix = identity_matrix4()
    matrix[:3, :3] = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )
    return matrix
