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


@pytest.mark.parametrize("method", ["trapezoid", "simpson"])
@pytest.mark.parametrize("descending", [False, True])
def test_duplicate_grouping_and_sums_match_release_formula_and_module(method, descending):
    from modacor.dataclasses.basedata import BaseData
    from modacor.models.integration import (
        collapse_duplicate_values,
        integrate_samples_1d,
        integrated_uncertainty,
        prepare_integration_domain,
    )
    from modacor.modules import Integrate1D

    axis = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    if descending:
        axis = axis[::-1]
    signal = 2 + axis**2
    components = {"measured": np.full(axis.shape, 0.1), "broadcast": np.asarray(0.2)}
    domain = prepare_integration_domain(
        axis, np.ones(axis.shape, bool), method=method, sort_axis=False, duplicate_axis="mean"
    )
    collapsed = collapse_duplicate_values(signal, domain)
    expected = np.array([2.0, 2.25, 3.0])
    if descending:
        expected = expected[::-1]
    np.testing.assert_allclose(collapsed, expected, rtol=1e-12, atol=0)
    nominal, uncertainties = integrate_samples_1d(signal, components, domain)
    for key, sigma in [("measured", 0.1), ("broadcast", 0.2)]:
        expected_sigma = np.sqrt(np.sum((sigma / np.sqrt(2) * domain.quadrature_weights) ** 2))
        np.testing.assert_allclose(uncertainties[key], expected_sigma, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            integrated_uncertainty(components[key], signal.shape, domain), expected_sigma, rtol=1e-12, atol=0
        )
    actual = Integrate1D._integrate_signal(BaseData(signal, "count", components), BaseData(axis, "1/m"), domain)
    np.testing.assert_allclose(actual.signal, nominal, rtol=1e-12, atol=0)
    for key in uncertainties:
        np.testing.assert_allclose(actual.uncertainties[key], uncertainties[key], rtol=1e-12, atol=0)


def test_integration_domain_keeps_selection_sort_and_duplicate_policy():
    from modacor.models.integration import prepare_integration_domain

    axis = np.array([1.0, 0.0, np.nan, 0.5, 0.5])
    domain = prepare_integration_domain(
        axis, np.isfinite(axis), method="trapezoid", sort_axis=True, duplicate_axis="mean"
    )
    np.testing.assert_allclose(domain.duplicate_counts, [1, 2, 1], rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="strictly monotonic"):
        prepare_integration_domain(axis, np.isfinite(axis), method="trapezoid", sort_axis=True, duplicate_axis="error")
