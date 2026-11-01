"""Stage 0 acceptance tests; requirement ids link to the normative specification."""

import operator
import tracemalloc

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from modacor import ureg
from modacor.dataclasses.basedata import BaseData, dict_signal_converter
from modacor.dataclasses.uncertainty_tools import combine_uncertainty_keys, quadrature_aggregator


def test_t0_06_constructor_and_converter_ownership():
    source = np.full(4, 3.0)
    source.flags.writeable = False
    p = BaseData(source, ureg.m, {"Poisson": source}, weights=source)
    q = BaseData(source, ureg.m, {"Poisson": source})
    converted = dict_signal_converter({"u": source})
    for value in (p.signal, p.weights, p.uncertainties["Poisson"], converted["u"]):
        assert not np.shares_memory(value, source)
        assert value.flags.writeable
    p /= 3
    assert_array_equal(q.signal, source)
    assert_array_equal(q.uncertainties["Poisson"], source)


def test_s0_02_copy_and_conversion_detach_external_arrays():
    p = BaseData(np.ones(3), ureg.m, {"u": np.ones(3)})
    q = p.copy()
    assert not np.shares_memory(q.uncertainties["u"], p.uncertainties["u"])
    external = np.ones(3)
    p.uncertainties["u"] = external
    p.to_units(ureg.cm)
    assert_array_equal(external, np.ones(3))
    assert_allclose(p.uncertainties["u"], 100)
    assert_allclose(q.uncertainties["u"], 1)


# E, G and several overlapping/disjoint named mappings exercise every key-table row.

MAPPINGS = ({}, {"propagate_to_all": 0.3}, {"u": 0.2}, {"u": 0.4, "v": 0.5}, {"w": 0.6})
OPS = (operator.add, operator.sub, operator.mul, operator.truediv)


def assert_data_equal(actual, expected):
    assert actual.units == expected.units
    assert_allclose(actual.signal, expected.signal, equal_nan=True)
    assert set(actual.uncertainties) == set(expected.uncertainties)
    for key in expected.uncertainties:
        assert_allclose(actual.uncertainties[key], expected.uncertainties[key], equal_nan=True)


@pytest.mark.parametrize("left_unc", MAPPINGS)
@pytest.mark.parametrize("right_unc", MAPPINGS)
@pytest.mark.parametrize("op", OPS)
def test_t0_01_key_table_out_of_place(left_unc, right_unc, op):
    a = BaseData(np.array([4.0, 9.0]), ureg.m, left_unc)
    b = BaseData(2.0, ureg.m, right_unc)
    result = op(a, b)
    named_a = set(left_unc) - {"propagate_to_all"}
    named_b = set(right_unc) - {"propagate_to_all"}
    keys = named_a | named_b
    if not keys and (left_unc or right_unc):
        keys = {"propagate_to_all"}
    assert set(result.uncertainties) == keys
    for key in keys:
        sigma_a = left_unc.get(key, left_unc.get("propagate_to_all", 0.0))
        sigma_b = right_unc.get(key, right_unc.get("propagate_to_all", 0.0))
        if op in (operator.add, operator.sub):
            expected = np.hypot(sigma_a, sigma_b)
        elif op is operator.mul:
            expected = np.hypot(b.signal * sigma_a, a.signal * sigma_b)
        else:
            expected = np.hypot(sigma_a / b.signal, a.signal * sigma_b / b.signal**2)
        assert_allclose(result.uncertainties[key], np.broadcast_to(expected, result.shape))


def test_t0_02_global_divisor_and_t0_05_named_union():
    a = BaseData(100.0, ureg.count)
    t = BaseData(2.0, ureg.s, {"propagate_to_all": 0.1})
    assert_allclose((a / t).uncertainties["propagate_to_all"], 2.5)
    result = BaseData(100.0, ureg.count, {"Poisson": 10.0, "SEM": 15.0}) - BaseData(50.0, ureg.count, {"Poisson": 7.0})
    assert_allclose(result.uncertainties["Poisson"], np.hypot(10, 7))
    assert_allclose(result.uncertainties["SEM"], 15)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x.copy(),
        lambda x: x.indexed(0),
        lambda x: x.to_units(ureg.m),
        lambda x: x.to_dimensionless(),
        lambda x: -x,
        lambda x: x.squeeze(),
        lambda x: x.log(),
        lambda x: x + 1,
        lambda x: 1 / x,
        lambda x: x.__iadd__(1),
        lambda x: x.__isub__(1),
        lambda x: x.__itruediv__(2),
    ],
)
def test_t0_03_mutable_mixed_mapping_entrypoints(operation):
    x = BaseData(np.ones(2), ureg.dimensionless, {"propagate_to_all": 0.2})
    x.uncertainties["u"] = np.ones(2)
    before_signal = x.signal.copy()
    before_unc = {k: v.copy() for k, v in x.uncertainties.items()}
    with pytest.raises(ValueError, match="propagate_to_all"):
        operation(x)
    assert_array_equal(x.signal, before_signal)
    assert x.units == ureg.dimensionless
    for key in before_unc:
        assert_array_equal(x.uncertainties[key], before_unc[key])


def test_t0_03_constructor_and_combinations_reject_reserved_key():
    with pytest.raises(ValueError, match="propagate_to_all"):
        BaseData(1.0, ureg.m, {"propagate_to_all": 0.1, "u": 0.2})
    x = BaseData(1.0, ureg.m, {"propagate_to_all": 0.1})
    with pytest.raises(ValueError, match="source key"):
        combine_uncertainty_keys(
            basedata=x,
            combinations={"u": ("propagate_to_all",)},
            aggregator=quadrature_aggregator,
            drop_sources=True,
            ignore_missing=False,
        )
    assert set(x.uncertainties) == {"propagate_to_all"}
    x = BaseData(1.0, ureg.m, {"u": 0.1})
    with pytest.raises(ValueError, match="propagate_to_all"):
        combine_uncertainty_keys(
            basedata=x,
            combinations={"propagate_to_all": ("u",)},
            aggregator=quadrature_aggregator,
            drop_sources=False,
            ignore_missing=False,
        )
    assert set(x.uncertainties) == {"u"}


def test_t0_09_scalar_peak_allocation():
    frame = BaseData(np.ones((2048, 2048)), ureg.count, {str(k): np.ones((2048, 2048)) for k in range(3)})
    tracemalloc.start()
    result = frame * 2.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert_allclose(result.signal[0, 0], 2)
    # Python bookkeeping is not frame storage; permit 64 KiB overhead.
    assert peak <= 4 * frame.signal.nbytes + 65536
