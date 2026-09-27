from __future__ import annotations

import numpy as np
import pytest

from modacor.models.integration import quadrature_weights_1d


@pytest.mark.parametrize("method", ["trapezoid", "simpson"])
def test_quadrature_weights_reproduce_direct_integral(method: str) -> None:
    axis = np.array([0.0, 0.2, 0.7, 1.0])
    values = axis**2 + 2.0 * axis + 3.0

    weighted = np.dot(quadrature_weights_1d(axis, method), values)

    if method == "trapezoid":
        expected = np.trapezoid(values, axis)
    else:
        from scipy.integrate import simpson

        expected = simpson(values, x=axis)
    assert weighted == pytest.approx(expected)


def test_quadrature_weights_validate_axis_and_method() -> None:
    with pytest.raises(ValueError, match="one-dimensional axis"):
        quadrature_weights_1d(np.array([1.0]), "trapezoid")
    with pytest.raises(ValueError, match="method"):
        quadrature_weights_1d(np.array([0.0, 1.0]), "rectangle")
