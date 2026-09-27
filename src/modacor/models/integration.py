# SPDX-License-Identifier: BSD-3-Clause
"""Format-independent numerical integration kernels."""

from __future__ import annotations

__all__ = ["quadrature_weights_1d"]

import numpy as np
from scipy.integrate import simpson


def quadrature_weights_1d(axis: np.ndarray, method: str) -> np.ndarray:
    """Return coefficients whose dot product with samples is their integral."""

    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("quadrature_weights_1d requires a one-dimensional axis with at least two points.")
    if method == "trapezoid":
        weights = np.empty(axis.size, dtype=float)
        weights[0] = 0.5 * (axis[1] - axis[0])
        weights[-1] = 0.5 * (axis[-1] - axis[-2])
        if axis.size > 2:
            weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
        return weights
    if method == "simpson":
        return np.asarray(simpson(np.eye(axis.size), x=axis, axis=1), dtype=float)
    raise ValueError("method must be 'trapezoid' or 'simpson'.")
