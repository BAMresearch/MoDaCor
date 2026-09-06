# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["unit_vector3"]

import numpy as np


def unit_vector3(value, *, name: str = "vector") -> np.ndarray:
    """Return a finite three-vector normalized to unit length."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm
