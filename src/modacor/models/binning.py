"""Numerical helpers for one-dimensional bin assignment."""

from __future__ import annotations

import numpy as np

__all__ = ["assign_bin_indices", "generate_bin_edges", "validate_bin_edges"]


def validate_bin_edges(edges: np.ndarray) -> np.ndarray:
    """Return validated, strictly increasing one-dimensional bin edges."""
    result = np.asarray(edges, dtype=float)
    if result.ndim != 1:
        raise ValueError("Bin edges must be one-dimensional.")
    if result.size < 2:
        raise ValueError("At least two bin edges are required.")
    if not np.all(np.isfinite(result)):
        raise ValueError("Bin edges must all be finite.")
    if not np.all(np.diff(result) > 0.0):
        raise ValueError("Bin edges must be strictly increasing.")
    return result


def generate_bin_edges(
    values: np.ndarray,
    *,
    lower: float | None,
    upper: float | None,
    n_bins: int,
    spacing: str,
) -> np.ndarray:
    """Generate linear or logarithmic edges from finite numerical values."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot generate bin edges without finite coordinate values.")

    spacing = str(spacing).lower()
    if spacing not in {"linear", "log"}:
        raise ValueError("spacing must be 'linear' or 'log'.")

    if lower is None:
        if spacing == "log":
            positive = finite_values[finite_values > 0.0]
            if positive.size == 0:
                raise ValueError("Logarithmic binning requires at least one positive coordinate value.")
            lower = float(np.min(positive))
        else:
            lower = float(np.min(finite_values))
    if upper is None:
        upper = float(np.max(finite_values))

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"Invalid bin range: lower={lower!r}, upper={upper!r}.")
    if spacing == "log" and lower <= 0.0:
        raise ValueError("Logarithmic binning requires a positive lower edge.")

    if spacing == "log":
        return np.geomspace(lower, upper, num=n_bins + 1, dtype=float)
    return np.linspace(lower, upper, num=n_bins + 1, dtype=float)


def assign_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values to left-inclusive bins, including the final right edge.

    Non-finite and out-of-range values receive index ``-1``.
    """
    validated_edges = validate_bin_edges(edges)
    coordinates = np.asarray(values, dtype=float)
    indices = np.full(coordinates.shape, -1, dtype=np.int64)

    finite = np.isfinite(coordinates)
    final_edge = np.isclose(coordinates, validated_edges[-1], rtol=1e-12, atol=0.0)
    in_range = finite & (coordinates >= validated_edges[0]) & ((coordinates <= validated_edges[-1]) | final_edge)
    if not np.any(in_range):
        return indices

    assigned = np.searchsorted(validated_edges, coordinates[in_range], side="right") - 1
    assigned[final_edge[in_range]] = validated_edges.size - 2
    indices[in_range] = assigned
    return indices
