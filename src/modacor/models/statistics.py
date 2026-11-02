# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WeightedScatterEstimates", "finalize_weighted_scatter", "static_weighted_mean_sem"]


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


def static_weighted_mean_sem(signal: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    """Return weighted mean and population-scatter SEM of static observations.

    Preserves the static-metadata convention: effective count is sum(w)^2 /
    sum(w^2), with no ddof correction or propagation of measured input sigmas.
    """
    x = signal.ravel()

    # --- robust weights handling ---
    if weights is None:
        w = np.ones_like(x)
    else:
        w_raw = np.asarray(weights, dtype=float)

        # allow scalar/length-1 weights
        if w_raw.size == 1:
            w = np.full_like(x, float(w_raw.reshape(-1)[0]))
        else:
            # allow broadcastable weights (e.g. (5,1,1,1) vs (5,))
            try:
                w = np.broadcast_to(w_raw, np.shape(signal)).ravel()
            except ValueError as e:
                raise ValueError(f"weights shape {w_raw.shape} does not match signal shape {np.shape(signal)}") from e

        if w.size != x.size:
            raise ValueError(f"weights size {w.size} does not match signal size {x.size}")

    wsum = float(np.sum(w))
    if wsum <= 0:
        raise ValueError("weights must sum to > 0")

    mean = float(np.sum(w * x) / wsum)

    # effective N for SEM (works for equal weights too)
    n_eff = float((wsum**2) / np.sum(w**2))

    # weighted population variance about the weighted mean
    var = float(np.sum(w * (x - mean) ** 2) / wsum)
    sem = float(np.sqrt(var) / np.sqrt(n_eff))

    return mean, sem
