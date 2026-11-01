# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = [
    "combine_uncertainty_component",
    "nominal_and_finite_difference",
    "uncertainties_from_derivative",
    "validate_uncertainty_keys",
    "with_uncertainty_estimates",
]

import logging

import numpy as np

logger = logging.getLogger(__name__)


def validate_uncertainty_keys(uncertainties) -> None:
    """Reject mixed reserved-key mappings, including mutable dictionary inputs."""
    if "propagate_to_all" in uncertainties and len(uncertainties) != 1:
        raise ValueError("propagate_to_all must be the only uncertainty key; keep it alone or rename it.")


def with_uncertainty_estimates(existing, estimates) -> dict[str, np.ndarray]:
    """Prepare an atomic estimate update, folding a global term into every new name.

    A batch uses the same original global sigma for every estimate. Subsequent
    calls on an already-named mapping cannot recover that removed global term.
    The caller installs the returned mapping with one assignment.
    """
    validate_uncertainty_keys(existing)
    validate_uncertainty_keys(estimates)
    if estimates and "propagate_to_all" in existing and "propagate_to_all" not in estimates:
        result = {key: np.asarray(np.hypot(value, existing["propagate_to_all"])) for key, value in estimates.items()}
        logger.info("Stage 0 legacy fold into uncertainty estimates %s", sorted(result))
        return result
    result = {**existing, **{key: np.asarray(value) for key, value in estimates.items()}}
    validate_uncertainty_keys(result)
    return result


def combine_uncertainty_component(target: dict[str, np.ndarray], name: str, values) -> None:
    """Combine an independent uncertainty component into ``target`` in quadrature."""
    validate_uncertainty_keys(target)
    validate_uncertainty_keys({**target, name: values})
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
    validate_uncertainty_keys(parameter_uncertainties)
    if derivative is None:
        return {}
    return {name: np.abs(derivative) * uncertainty for name, uncertainty in parameter_uncertainties.items()}
