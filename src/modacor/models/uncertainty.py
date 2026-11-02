# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = [
    "combine_uncertainty_component",
    "combine_uncertainty_components",
    "uncertainty_key_plan",
    "nominal_and_finite_difference",
    "uncertainties_from_derivative",
    "validate_uncertainty_keys",
    "with_uncertainty_estimates",
]

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def validate_uncertainty_keys(uncertainties) -> None:
    """Reject mixed reserved-key mappings, including mutable dictionary inputs."""
    if "propagate_to_all" in uncertainties and len(uncertainties) != 1:
        raise ValueError("propagate_to_all must be the only uncertainty key; keep it alone or rename it.")


def uncertainty_key_plan(
    left_unc: Dict[str, np.ndarray],
    right_unc: Dict[str, np.ndarray],
) -> tuple[set[str], Any | None, Any | None]:
    validate_uncertainty_keys(left_unc)
    validate_uncertainty_keys(right_unc)
    left_global = left_unc.get("propagate_to_all")
    right_global = right_unc.get("propagate_to_all")
    left_keys, right_keys = set(left_unc), set(right_unc)
    if left_global is not None and right_keys - {"propagate_to_all"}:
        out_keys = right_keys
    elif right_global is not None and left_keys - {"propagate_to_all"}:
        out_keys = left_keys
    else:
        out_keys = left_keys | right_keys
    if left_keys and right_keys and left_global is None and right_global is None and left_keys != right_keys:
        logger.debug("Uncertainty key sets differ; propagating their union with absent terms treated as zero.")
    return out_keys, left_global, right_global


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


def combine_uncertainty_components(existing, components) -> dict[str, np.ndarray]:
    """Combine independent sigma mappings using the Stage 0 §4.3 key table.

    Empty terms contribute zero; a global-only term contributes to every named
    estimate of the other mapping, in either order. All arrays are prepared
    before installation; invalid mixed mappings never change either input.
    """
    keys, global_existing, global_components = uncertainty_key_plan(existing, components)
    result = {}
    for key in keys:
        left = existing.get(key, global_existing)
        right = components.get(key, global_components)
        if left is None:
            result[key] = np.array(right, copy=True)
        elif right is None:
            result[key] = np.array(left, copy=True)
        else:
            result[key] = np.asarray(np.hypot(left, right))
    if keys and "propagate_to_all" not in keys and (global_existing is not None or global_components is not None):
        logger.info("Stage 0 legacy fold into uncertainty estimates %s", sorted(keys))
    return result


def combine_uncertainty_component(target: dict[str, np.ndarray], name: str, values) -> None:
    """Atomically attach one independent component using the shared key table."""
    combined = combine_uncertainty_components(target, {name: values})
    target.clear()
    target.update(combined)


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
