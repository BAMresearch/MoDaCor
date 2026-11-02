# SPDX-License-Identifier: BSD-3-Clause
"""Format-independent numerical integration kernels."""

from __future__ import annotations

__all__ = [
    "quadrature_weights_1d",
    "IntegrationDomain",
    "prepare_integration_domain",
    "collapse_duplicate_values",
    "integrated_uncertainty",
    "integrate_samples_1d",
]

import numpy as np
from attrs import define
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


@define(frozen=True, slots=True)
class IntegrationDomain:
    common: np.ndarray
    sort_order: np.ndarray
    duplicate_groups: np.ndarray | None
    duplicate_counts: np.ndarray | None
    quadrature_weights: np.ndarray


def prepare_integration_domain(
    axis: np.ndarray,
    common: np.ndarray,
    *,
    method: str,
    sort_axis: bool,
    duplicate_axis: str,
) -> IntegrationDomain:
    """Group a caller-selected domain, retaining sorting and duplicate policies."""
    valid_axis = axis[common]
    if valid_axis.size < 2:
        raise ValueError("Integrate1D common domain contains fewer than two valid points.")

    sort_order = np.arange(valid_axis.size)
    if sort_axis:
        sort_order = np.argsort(valid_axis, kind="stable")
        valid_axis = valid_axis[sort_order]
    differences = np.diff(valid_axis)
    if not (np.all(differences >= 0.0) or np.all(differences <= 0.0)):
        raise ValueError("Integrate1D axis must be strictly monotonic.")

    if duplicate_axis not in {"error", "mean"}:
        raise ValueError("Integrate1D duplicate_axis must be 'error' or 'mean'.")
    duplicate_groups: np.ndarray | None = None
    duplicate_counts: np.ndarray | None = None
    if np.any(differences == 0.0):
        if duplicate_axis == "error":
            raise ValueError("Integrate1D axis must be strictly monotonic.")
        group_starts = np.concatenate(([True], differences != 0.0))
        duplicate_groups = np.cumsum(group_starts) - 1
        duplicate_counts = np.bincount(duplicate_groups)
        valid_axis = valid_axis[group_starts]
    if valid_axis.size < 2:
        raise ValueError("Integrate1D common domain contains fewer than two distinct coordinates.")

    return IntegrationDomain(
        common=common,
        sort_order=sort_order,
        duplicate_groups=duplicate_groups,
        duplicate_counts=duplicate_counts,
        quadrature_weights=quadrature_weights_1d(valid_axis, method),
    )


def collapse_duplicate_values(values: np.ndarray, domain: IntegrationDomain) -> np.ndarray:
    """Average samples sharing each coordinate, or return ungrouped values."""
    if domain.duplicate_groups is None:
        return values
    if domain.duplicate_counts is None:  # pragma: no cover - construction invariant
        raise RuntimeError("Duplicate coordinate groups are missing their sample counts.")
    return np.bincount(domain.duplicate_groups, weights=values) / domain.duplicate_counts


def integrated_uncertainty(
    component: np.ndarray,
    signal_shape: tuple[int, ...],
    domain: IntegrationDomain,
) -> np.ndarray:
    """Collapse independent duplicate sigmas and contract quadrature weights."""
    values = np.broadcast_to(component, signal_shape).squeeze()[domain.common][domain.sort_order]
    if domain.duplicate_groups is not None:
        if domain.duplicate_counts is None:  # pragma: no cover - construction invariant
            raise RuntimeError("Duplicate coordinate groups are missing their sample counts.")
        values = np.sqrt(np.bincount(domain.duplicate_groups, weights=values**2)) / domain.duplicate_counts
    return np.asarray(np.sqrt(np.sum((values * domain.quadrature_weights) ** 2)))


def integrate_samples_1d(
    signal: np.ndarray, components: dict[str, np.ndarray], domain: IntegrationDomain
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Integrate selected samples and independent sigmas after duplicate collapse."""
    values = np.asarray(signal, dtype=float).squeeze()[domain.common][domain.sort_order]
    values = collapse_duplicate_values(values, domain)
    uncertainties = {
        name: integrated_uncertainty(component, signal.shape, domain) for name, component in components.items()
    }
    return np.asarray(np.sum(values * domain.quadrature_weights)), uncertainties
