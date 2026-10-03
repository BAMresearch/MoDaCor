# SPDX-License-Identifier: BSD-3-Clause
"""Format-independent one-dimensional remapping kernels."""

from __future__ import annotations

__all__ = ["remap_coefficients_1d"]

import numpy as np


def remap_coefficients_1d(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    *,
    mode: str,
    max_gap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, coefficients, and validity for 1D remapping.

    ``source_axis`` must be finite and strictly increasing. Targets outside its
    closed domain are invalid, so this kernel never extrapolates. In ``nearest``
    mode ties select the lower coordinate. In ``linear`` mode the two bracketing
    samples are returned. ``max_gap`` optionally limits bracketing interval
    width in linear mode and is not accepted in setting-free nearest mode.
    """

    source = np.asarray(source_axis, dtype=float)
    target = np.asarray(target_axis, dtype=float)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("remap_coefficients_1d requires one-dimensional axes.")
    if source.size == 0:
        raise ValueError("remap_coefficients_1d requires at least one source point.")
    if not np.all(np.isfinite(source)):
        raise ValueError("remap_coefficients_1d source_axis must be finite.")
    if source.size > 1 and not np.all(np.diff(source) > 0.0):
        raise ValueError("remap_coefficients_1d source_axis must be strictly increasing.")
    if max_gap is not None and (not np.isfinite(max_gap) or max_gap < 0.0):
        raise ValueError("remap_coefficients_1d max_gap must be finite and non-negative.")

    mode = str(mode).strip().lower()
    finite_target = np.isfinite(target)
    in_domain = finite_target & (target >= source[0]) & (target <= source[-1])

    if mode == "nearest":
        if max_gap is not None:
            raise ValueError("max_gap is only supported for linear remapping.")
        insertion = np.searchsorted(source, target, side="left")
        right = np.clip(insertion, 0, source.size - 1)
        left = np.clip(insertion - 1, 0, source.size - 1)
        choose_left = np.abs(target - source[left]) <= np.abs(source[right] - target)
        index = np.where(choose_left, left, right)
        valid = in_domain
        return index, index, np.ones(target.size), np.zeros(target.size), valid

    if mode == "linear":
        if source.size < 2:
            raise ValueError("linear remapping requires at least two source points.")
        insertion = np.searchsorted(source, target, side="right")
        right = np.clip(insertion, 1, source.size - 1)
        left = right - 1
        span = source[right] - source[left]
        fraction = (target - source[left]) / span
        valid = in_domain
        if max_gap is not None:
            exact_source_point = np.isclose(fraction, 0.0) | np.isclose(fraction, 1.0)
            valid &= (span <= max_gap) | exact_source_point
        return left, right, 1.0 - fraction, fraction, valid

    raise ValueError("remap_coefficients_1d mode must be 'nearest' or 'linear'.")
