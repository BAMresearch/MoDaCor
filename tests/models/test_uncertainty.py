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
