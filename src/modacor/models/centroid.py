# SPDX-License-Identifier: BSD-3-Clause
"""Format-independent intensity-centroid kernels."""

from __future__ import annotations

__all__ = ["CentroidResult", "intensity_centroid_1d"]

import numpy as np
from attrs import define


@define(frozen=True, slots=True)
class CentroidResult:
    """Numerical centroid result and first-order propagation sensitivities."""

    center: float
    intensity_sum: float
    contributor_count: int
    contributors: np.ndarray
    signal_sensitivity: np.ndarray
    axis_sensitivity: np.ndarray


def intensity_centroid_1d(
    axis: np.ndarray,
    intensity: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    baseline: float = 0.0,
) -> CentroidResult:
    """Compute an intensity centroid after baseline subtraction and clipping."""

    axis = np.asarray(axis, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if axis.ndim != 1 or intensity.ndim != 1 or axis.shape != intensity.shape:
        raise ValueError("intensity_centroid_1d requires matching one-dimensional arrays.")
    if axis.size == 0:
        raise ValueError("intensity_centroid_1d requires at least one point.")
    if valid is None:
        valid_array = np.ones(axis.shape, dtype=bool)
    else:
        valid_array = np.asarray(valid, dtype=bool)
        if valid_array.shape != axis.shape:
            raise ValueError("intensity_centroid_1d valid mask must match the input shape.")
    valid_array &= np.isfinite(axis) & np.isfinite(intensity)

    corrected = np.clip(intensity - float(baseline), 0.0, None)
    contributors = valid_array & (corrected > 0.0)
    intensity_sum = float(np.sum(corrected[contributors]))
    if not np.isfinite(intensity_sum) or intensity_sum <= 0.0:
        raise ValueError("intensity_centroid_1d has no positive intensity weight.")

    center = float(np.sum(axis[contributors] * corrected[contributors]) / intensity_sum)
    signal_sensitivity = np.zeros(axis.shape, dtype=float)
    signal_sensitivity[contributors] = (axis[contributors] - center) / intensity_sum
    axis_sensitivity = np.zeros(axis.shape, dtype=float)
    axis_sensitivity[contributors] = corrected[contributors] / intensity_sum
    return CentroidResult(
        center=center,
        intensity_sum=intensity_sum,
        contributor_count=int(np.count_nonzero(contributors)),
        contributors=contributors,
        signal_sensitivity=signal_sensitivity,
        axis_sensitivity=axis_sensitivity,
    )
