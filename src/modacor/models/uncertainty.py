# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = [
    "combine_uncertainty_component",
    "nominal_and_finite_difference",
    "uncertainties_from_derivative",
]

import numpy as np


def combine_uncertainty_component(target: dict[str, np.ndarray], name: str, values) -> None:
    """Combine an independent uncertainty component into ``target`` in quadrature."""
    component = np.asarray(values, dtype=float)
    if name in target:
        component = np.hypot(target[name], component)
    target[name] = component


def nominal_and_finite_difference(
    values: np.ndarray, delta: float | None, central: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split batched nominal/perturbed results and calculate their finite difference."""
    if delta is None:
        return values, None
    nominal = values[0]
    if central:
        return nominal, (values[1] - values[2]) / (2.0 * delta)
    return nominal, (values[1] - nominal) / delta


def uncertainties_from_derivative(
    derivative: np.ndarray | float | None, parameter_uncertainties: dict[str, float]
) -> dict[str, np.ndarray]:
    """Propagate named independent standard uncertainties through one derivative."""
    if derivative is None:
        return {}
    return {name: np.abs(derivative) * uncertainty for name, uncertainty in parameter_uncertainties.items()}
