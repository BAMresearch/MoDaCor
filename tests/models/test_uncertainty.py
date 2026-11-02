# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from modacor.models.uncertainty import (
    combine_uncertainty_component,
    nominal_and_finite_difference,
    uncertainties_from_derivative,
)


def test_nominal_and_finite_difference_supports_forward_and_central_batches():
    forward = np.array([[2.0, 4.0], [2.3, 4.6]])
    nominal, derivative = nominal_and_finite_difference(forward, 0.1, central=False)
    np.testing.assert_array_equal(nominal, [2.0, 4.0])
    np.testing.assert_allclose(derivative, [3.0, 6.0])

    central = np.array([[2.0, 4.0], [2.3, 4.6], [1.7, 3.4]])
    nominal, derivative = nominal_and_finite_difference(central, 0.1, central=True)
    np.testing.assert_array_equal(nominal, [2.0, 4.0])
    np.testing.assert_allclose(derivative, [3.0, 6.0])


def test_named_uncertainty_helpers_scale_and_combine_independent_components():
    propagated = uncertainties_from_derivative(np.array([-2.0, 3.0]), {"sample": 0.1})
    np.testing.assert_allclose(propagated["sample"], [0.2, 0.3])
    combine_uncertainty_component(propagated, "sample", np.array([0.15, 0.4]))
    np.testing.assert_allclose(propagated["sample"], np.hypot([0.2, 0.3], [0.15, 0.4]))


def test_stage0_numerical_helpers_reject_mixed_keys_before_writes():
    import pytest

    target = {"propagate_to_all": np.array(3.0), "named": np.array(4.0)}
    before = dict(target)
    with pytest.raises(ValueError, match="propagate_to_all"):
        combine_uncertainty_component(target, "other", np.array(5.0))
    assert target.keys() == before.keys()
    assert all(target[key] is before[key] for key in target)
    with pytest.raises(ValueError, match="propagate_to_all"):
        uncertainties_from_derivative(2.0, target)


def test_sensitivity_contraction_matches_release_expression_and_centroid_boundary():
    from types import SimpleNamespace

    from modacor.dataclasses.basedata import BaseData
    from modacor.models.uncertainty import sensitivity_contraction
    from modacor.modules import FindCenterOfMass1D

    signal_sensitivity = np.array([0.1, -0.2, 0.4])
    axis_sensitivity = np.array([0.3, 0.5, 0.2])
    signal = BaseData(np.ones(3), "count", {"measured": np.array([0.2, 0.3, 0.5])})
    axis = BaseData(np.arange(3.0), "m", {"calibration": 0.1})
    for sensitivity, sigma in [
        (signal_sensitivity, signal.uncertainties["measured"]),
        (axis_sensitivity, axis.uncertainties["calibration"]),
    ]:
        actual = sensitivity_contraction(sensitivity, sigma)
        np.testing.assert_allclose(actual, np.sqrt(np.sum((sensitivity * sigma) ** 2)), rtol=1e-12, atol=0)
    result = SimpleNamespace(signal_sensitivity=signal_sensitivity, axis_sensitivity=axis_sensitivity)
    components = FindCenterOfMass1D._propagated_uncertainties(result, signal, axis)
    np.testing.assert_allclose(
        components["signal:measured"],
        sensitivity_contraction(signal_sensitivity, signal.uncertainties["measured"]),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        components["axis:calibration"],
        sensitivity_contraction(axis_sensitivity, axis.uncertainties["calibration"]),
        rtol=1e-12,
        atol=0,
    )
    assert np.isnan(sensitivity_contraction(np.array([0.0, 1.0]), np.array([np.nan, 1.0])))
