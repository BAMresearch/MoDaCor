# SPDX-License-Identifier: BSD-3-Clause
"""Central conversion and scalar-ingestion contracts."""

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData


@pytest.mark.parametrize("target", ["m", "mm"])
def test_conversion_equivalence_ownership_integer_promotion_and_weights(target):
    axis = BaseData([0, 1], "s")
    original = BaseData(
        np.array([1000, 2000]),
        "mm",
        uncertainties={"SEM": np.array([10, 20]), "STD": 0.5},
        weights=np.array([2.0, 3.0]),
        axes=[axis],
        rank_of_data=1,
    )
    converted = original.in_units(target)
    inplace = original.copy()
    weights = inplace.weights
    inplace.to_units(target)
    assert inplace.weights is weights
    assert converted.signal.dtype.kind == inplace.signal.dtype.kind == "f"
    assert converted.axes[0] is axis
    for actual in [inplace, converted]:
        np.testing.assert_array_equal(actual.signal, original.signal * ureg.Unit(target).m_from(ureg.mm))
        np.testing.assert_array_equal(actual.weights, original.weights)
        for key in original.uncertainties:
            np.testing.assert_array_equal(
                actual.uncertainties[key], original.uncertainties[key] * ureg.Unit(target).m_from(ureg.mm)
            )
            assert not np.shares_memory(converted.uncertainties[key], original.uncertainties[key])
    assert not np.shares_memory(converted.signal, original.signal)
    np.testing.assert_array_equal(original.signal, [1000, 2000])


@pytest.mark.parametrize("shape", [(), (1,), (2, 3)])
def test_percent_accessors_preserve_global_mapping_and_input(shape):
    data = BaseData(np.full(shape, 50.0), "%", uncertainties={"propagate_to_all": np.full(shape, 1.0)})
    signal, uncertainties = data.magnitudes("dimensionless")
    np.testing.assert_array_equal(signal, np.full(shape, 0.5))
    np.testing.assert_array_equal(uncertainties["propagate_to_all"], np.full(shape, 0.01))
    assert data.scalar_components("dimensionless") == (0.5, {"propagate_to_all": 0.01})
    np.testing.assert_array_equal(data.signal, np.full(shape, 50.0))


@pytest.mark.parametrize(
    "signal,uncertainties", [([1, 2], {}), ([1, 1], {"SEM": [1, 2]}), ([], {}), (1, {"SEM": np.inf})]
)
def test_scalar_boundary_rejects_nonconstant_or_invalid_components(signal, uncertainties):
    with pytest.raises(ValueError):
        BaseData(np.asarray(signal), "m", uncertainties=uncertainties).scalar_components("m")


@pytest.mark.parametrize(
    "units,target,factor",
    [
        ("count/%", "count", 100),
        ("count/sr", "count/sr", 1),
        ("m/mm", "", 1000),
        ("rad", "rad", 1),
        ("deg", "deg", 1),
        ("px", "px", 1),
        ("%", "", 0.01),
        ("permille", "", 0.001),
        ("ppm", "", 1e-6),
        ("count/deg", "count/deg", 1),
        ("sr/%", "sr", 100),
        ("mm", "mm", 1),
        ("mrad", "mrad", 1),
        ("rad/deg", "rad/deg", 1),
    ],
)
def test_scalar_reduction(units, target, factor):
    data = BaseData.from_scalar(2, units, {"SEM": 0.1})
    assert data.units == ureg.Unit(target)
    assert float(data.signal) == pytest.approx(2 * factor, rel=1e-12)
    assert float(data.uncertainties["SEM"]) == pytest.approx(0.1 * factor, rel=1e-12)
    # General constructors and arithmetic do not apply ingestion reduction.
    assert BaseData(2, units).units == ureg.Unit(units)
    assert (BaseData(2, units) * 1).units == ureg.Unit(units)


def test_scalar_constructor_does_not_bypass_validation():
    with pytest.raises(ValueError):
        BaseData.from_scalar(50, "%", {"propagate_to_all": 1, "SEM": 2})


def test_conversion_adopts_fresh_arrays(monkeypatch):
    data = BaseData(np.ones((3, 4)), "mm", uncertainties={"SEM": 0.1, "STD": 0.2})
    signal, uncertainties = data.magnitudes("m")
    original = BaseData.magnitudes
    monkeypatch.setattr(
        BaseData, "magnitudes", lambda self, units: (signal, uncertainties) if self is data else original(self, units)
    )
    result = data.in_units("m")
    assert result.signal is signal
    assert all(result.uncertainties[key] is value for key, value in uncertainties.items())


@pytest.mark.parametrize("method", ["to_units", "magnitudes", "in_units", "scalar_components"])
def test_invalid_conversion_units_raise_valueerror(method):
    data = BaseData(1.0, "m")
    with pytest.raises(ValueError, match="Invalid unit provided: not_a_real_unit"):
        getattr(data, method)("not_a_real_unit")


@pytest.mark.parametrize("target", [None, "m", ureg.m])
def test_native_magnitudes_skip_pint_compatibility_and_factor(monkeypatch, target):
    data = BaseData(1.0, "m", uncertainties={"SEM": 0.1})

    def unexpected(*args, **kwargs):
        pytest.fail("Native-unit access must skip Pint conversion work")

    monkeypatch.setattr(type(data.units), "is_compatible_with", unexpected)
    monkeypatch.setattr(type(data.units), "m_from", unexpected)
    signal, uncertainties = data.magnitudes(target)
    assert signal is data.signal
    assert uncertainties is data.uncertainties


@pytest.mark.parametrize(
    "signal,units,uncertainties,target,detail",
    [
        (1.0, "s", {}, "m", "Cannot convert"),
        ([1.0, 2.0], "m", {}, "m", "scalar or constant-valued"),
        (1.0, "m", {"SEM": np.inf}, "m", "finite and non-negative"),
        (1.0, "m", {}, "not_a_real_unit", "Invalid unit provided"),
    ],
)
def test_scalar_components_errors_include_parameter_name(signal, units, uncertainties, target, detail):
    with pytest.raises(ValueError, match=f"sample_mu: .*{detail}"):
        BaseData(signal, units, uncertainties=uncertainties).scalar_components(target, name="sample_mu")
