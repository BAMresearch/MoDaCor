# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WeightedScatterEstimates", "finalize_weighted_scatter"]


@dataclass(frozen=True)
class WeightedScatterEstimates:
    """Finalized scatter estimates derived from weighted reduction moments."""

    effective_sample_size: np.ndarray
    variance: np.ndarray
    standard_deviation: np.ndarray
    standard_error_mean: np.ndarray
    standard_error_sum: np.ndarray


def finalize_weighted_scatter(
    *,
    sum_w: np.ndarray,
    sum_w2: np.ndarray,
    sum_w_squared_deviations: np.ndarray,
    ddof: int = 1,
) -> WeightedScatterEstimates:
    """Finalize weighted scatter statistics from caller-provided moments.

    The caller owns observation selection and accumulation. This lets regular
    axis reducers and indexed/bin reducers share the statistical definitions
    without coupling their different grouping implementations.
    """

    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}.")

    sum_w_arr, sum_w2_arr, squared_deviations_arr = np.broadcast_arrays(
        np.asarray(sum_w, dtype=float),
        np.asarray(sum_w2, dtype=float),
        np.asarray(sum_w_squared_deviations, dtype=float),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = (sum_w_arr**2) / sum_w2_arr
        population_variance = squared_deviations_arr / sum_w_arr
        variance = population_variance * n_eff / (n_eff - ddof)

    valid = (
        np.isfinite(sum_w_arr)
        & np.isfinite(sum_w2_arr)
        & np.isfinite(squared_deviations_arr)
        & np.isfinite(n_eff)
        & (sum_w_arr > 0.0)
        & (sum_w2_arr > 0.0)
        & (n_eff > ddof)
    )
    variance = np.where(valid, np.maximum(variance, 0.0), np.nan)
    standard_deviation = np.sqrt(variance)

    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error_mean = np.sqrt(variance / n_eff)
    standard_error_sum = np.abs(sum_w_arr) * standard_error_mean

    return WeightedScatterEstimates(
        effective_sample_size=np.asarray(n_eff),
        variance=np.asarray(variance),
        standard_deviation=np.asarray(standard_deviation),
        standard_error_mean=np.asarray(standard_error_mean),
        standard_error_sum=np.asarray(standard_error_sum),
    )
