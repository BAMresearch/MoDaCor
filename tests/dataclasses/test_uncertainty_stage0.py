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


IOPS = (operator.iadd, operator.isub, operator.imul, operator.itruediv)


@pytest.mark.parametrize("left_unc", MAPPINGS)
@pytest.mark.parametrize("right_unc", MAPPINGS)
@pytest.mark.parametrize("op,iop", zip(OPS, IOPS))
@pytest.mark.parametrize("reverse", [False, True])
def test_t0_01_key_table_in_place(left_unc, right_unc, op, iop, reverse):
    a = BaseData(np.array([4.0, 9.0]), ureg.m, left_unc)
    b = BaseData(2.0, ureg.m, right_unc)
    if reverse:
        a, b = b, a
    expected = op(a, b)
    reference = a
    result = iop(a, b)
    assert result is reference
    assert_data_equal(result, expected)


@pytest.mark.parametrize("op,iop", zip(OPS, IOPS))
@pytest.mark.parametrize("alias", ["self", "signal", "uncertainty", "cross_array", "multiple_keys"])
def test_t0_07_t0_11_t0_12_snapshot_alias_safety(op, iop, alias):
    a = BaseData(np.array([4.0, 9.0]), ureg.m, {"u": np.array([2.0, 3.0])})
    b = BaseData(np.array([2.0, 3.0]), ureg.m, {"u": np.array([0.1, 0.2])})
    if alias == "self":
        b = a
    elif alias == "signal":
        b.signal = a.signal[::-1]
    elif alias == "uncertainty":
        b.uncertainties["u"] = a.uncertainties["u"][::-1]
    elif alias == "cross_array":
        b.signal = a.uncertainties["u"]
        b.uncertainties["u"] = a.signal
    else:
        a.uncertainties["v"] = a.uncertainties["u"]
    expected = op(a.copy(), b.copy())
    original_b = b.copy()
    result = iop(a, b)
    assert result is a
    assert_data_equal(result, expected)
    if b is not a:
        assert_data_equal(b, original_b)
    if alias == "self" and op is operator.truediv:
        assert_allclose(result.uncertainties["u"], [np.sqrt(0.5), np.sqrt(2) / 3])
    if alias == "self" and op is operator.sub:
        assert_allclose(result.uncertainties["u"], np.sqrt(2) * np.array([2.0, 3.0]))


@pytest.mark.parametrize("op,iop", zip(OPS, IOPS))
@pytest.mark.parametrize("failure", ["broadcast", "units", "mixed", "metadata", "signal_shape"])
def test_t0_08_t0_12_failure_is_atomic(op, iop, failure):
    a = BaseData(np.full(3, 10.0), ureg.m, {"u": np.ones(3)}, rank_of_data=1)
    b = BaseData(np.ones(3), ureg.m, {"u": np.ones(3)}, rank_of_data=1)
    if failure == "broadcast":
        b.uncertainties["u"] = np.ones(5)
    elif failure == "units":
        b.units = ureg.degC if op in (operator.mul, operator.truediv) else ureg.s
    elif failure == "mixed":
        b.uncertainties["propagate_to_all"] = np.ones(3)
    elif failure == "metadata":
        b.rank_of_data = 0
        b.axes = [None]
    else:
        b.signal = np.ones(5)
    before = a.copy()
    signal_ref = a.signal
    unc_ref = a.uncertainties["u"]
    with pytest.raises((ValueError, TypeError)) as regular_error:
        op(a, b)
    with pytest.raises(type(regular_error.value)):
        iop(a, b)
    assert_data_equal(a, before)
    assert a.signal is signal_ref
    assert a.uncertainties["u"] is unc_ref
    assert a.axes == before.axes
    assert a.rank_of_data == before.rank_of_data
    assert_array_equal(a.weights, before.weights)


@pytest.mark.parametrize(
    "operand", [2.0, 2.0 * ureg.s, BaseData(np.array([2.0, 3.0]), ureg.s, {"v": np.array([0.1, 0.2])})]
)
def test_t0_12_multiply_keeps_container_identity(operand):
    a = BaseData(np.array([4.0, 9.0]), ureg.m, {"u": np.array([2.0, 3.0])})
    expected = a * operand
    reference = a
    a *= operand
    assert a is reference
    assert_data_equal(reference, expected)


@pytest.mark.parametrize(
    "function,units,target",
    [
        ("log", ureg.percent, ureg.dimensionless),
        ("exp", ureg.percent, ureg.dimensionless),
        ("arcsin", ureg.percent, ureg.dimensionless),
        ("arccos", ureg.percent, ureg.dimensionless),
        ("arctan", ureg.percent, ureg.dimensionless),
        ("sin", ureg.degree, ureg.radian),
        ("cos", ureg.degree, ureg.radian),
        ("tan", ureg.degree, ureg.radian),
    ],
)
def test_t0_10_unary_inline_conversion_preserves_input(function, units, target):
    x = BaseData(np.array([20.0, 40.0]), units, {"u": np.array([1.0, 2.0]), "v": 0.5})
    before = x.copy()
    signal_ref = x.signal
    unc_refs = dict(x.uncertainties)
    converted = x.copy()
    converted.to_units(target)
    expected = getattr(converted, function)()
    result = getattr(x, function)()
    assert_data_equal(result, expected)
    assert_data_equal(x, before)
    assert x.signal is signal_ref
    for key in unc_refs:
        assert x.uncertainties[key] is unc_refs[key]
    if function == "sin":
        assert_allclose(result.signal, np.sin(np.deg2rad([20.0, 40.0])))
        assert_allclose(result.uncertainties["u"], np.abs(np.cos(np.deg2rad([20.0, 40.0]))) * np.deg2rad([1.0, 2.0]))
    if function == "exp":
        assert_allclose(result.uncertainties["u"], np.exp([0.2, 0.4]) * [0.01, 0.02])


def test_t0_04_actual_poisson_legacy_fold(caplog):
    from modacor.dataclasses.databundle import DataBundle
    from modacor.dataclasses.processing_data import ProcessingData
    from modacor.io.io_sources import IoSources
    from modacor.modules import PoissonUncertainties

    x = BaseData(np.array([100.0]), ureg.count, {"propagate_to_all": 5.0})
    original_mapping = x.uncertainties
    processing = ProcessingData()
    processing["sample"] = DataBundle(signal=x)
    step = PoissonUncertainties(io_sources=IoSources(), configuration={"with_processing_keys": ["sample"]})
    with caplog.at_level("INFO"):
        step.execute(processing)
    assert set(x.uncertainties) == {"Poisson"}
    assert set(original_mapping) == {"propagate_to_all"}
    assert_allclose(x.uncertainties["Poisson"], np.hypot(10, 5))
    assert_allclose((x / 2).uncertainties["Poisson"], np.sqrt(125) / 2)
    records = [record for record in caplog.records if "legacy fold" in record.message]
    assert len(records) == 1
    assert "Poisson" in records[0].message


def test_t0_04b_batch_fold_and_later_producer_limitation(caplog):
    from modacor.models.uncertainty import with_uncertainty_estimates

    x = BaseData(100.0, ureg.count, {"propagate_to_all": 5.0})
    original_mapping = x.uncertainties
    with caplog.at_level("INFO"):
        x.uncertainties = with_uncertainty_estimates(x.uncertainties, {"SEM": np.array(10.0), "STD": np.array(20.0)})
        x.uncertainties = with_uncertainty_estimates(x.uncertainties, {"later": np.array(2.0)})
    assert set(x.uncertainties) == {"SEM", "STD", "later"}
    assert all(isinstance(value, np.ndarray) for value in x.uncertainties.values())
    assert_allclose(x.uncertainties["SEM"], 11.18, atol=5e-3)
    assert_allclose(x.uncertainties["STD"], 20.62, atol=5e-3)
    assert_allclose(x.uncertainties["SEM"], np.hypot(10, 5))
    assert_allclose(x.uncertainties["STD"], np.hypot(20, 5))
    assert_allclose(x.uncertainties["later"], 2)
    assert set(original_mapping) == {"propagate_to_all"}
    records = [record for record in caplog.records if "legacy fold" in record.message]
    assert len(records) == 1
    assert "SEM" in records[0].message and "STD" in records[0].message


def test_variance_setters_validate_and_fold_atomically():
    x = BaseData(np.ones(2), ureg.m, {"propagate_to_all": 3.0})
    old = x.uncertainties
    x.variances["u"] = np.array([16.0, 16.0])
    assert set(old) == {"propagate_to_all"}
    assert set(x.uncertainties) == {"u"}
    assert_allclose(x.uncertainties["u"], 5)
    snapshot = x.copy()
    for invalid in ({"propagate_to_all": 1.0, "u": 2.0}, {"u": np.ones(3)}, {"u": -1.0}):
        with pytest.raises(ValueError):
            x.variances = invalid
        assert_data_equal(x, snapshot)
    with pytest.raises(ValueError):
        x.variances["propagate_to_all"] = 1.0
    assert_data_equal(x, snapshot)
    with pytest.raises(ValueError):
        x.variances["u"] = -1.0
    assert_data_equal(x, snapshot)


@pytest.mark.parametrize("op,iop", zip(OPS, IOPS))
@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("reverse", [False, True])
def test_t0_01_mixed_row_in_both_orders(op, iop, mapping, reverse):
    a = BaseData(np.array([4.0, 9.0]), ureg.m, {"propagate_to_all": 0.2})
    a.uncertainties["u"] = np.ones(2)
    b = BaseData(2.0, ureg.m, mapping)
    if reverse:
        a, b = b, a
    before_signal = a.signal.copy()
    before_units = a.units
    before_unc = {k: np.copy(v) for k, v in a.uncertainties.items()}
    for operation in (op, iop):
        with pytest.raises(ValueError, match="propagate_to_all"):
            operation(a, b)
        assert_array_equal(a.signal, before_signal)
        assert a.units == before_units
        for key in before_unc:
            assert_array_equal(a.uncertainties[key], before_unc[key])


def test_s0_02_unit_conversion_preserves_masked_arrays():
    x = BaseData(np.ma.array([1.0, 2.0], mask=[False, True]), ureg.m)
    source = x.signal
    x.to_units(ureg.cm)
    assert isinstance(x.signal, np.ma.MaskedArray)
    assert_array_equal(x.signal.mask, [False, True])
    assert_allclose(x.signal.compressed(), [100.0])
    assert_allclose(source.data, [1.0, 2.0])


@pytest.mark.parametrize("op,iop", zip(OPS, IOPS))
def test_i0_01_nan_inf_policy_matches_independent_quadrature(op, iop):
    a = BaseData(np.array([4.0]), ureg.m, {"u": np.array([np.nan])})
    b = BaseData(np.array([2.0]), ureg.m, {"u": np.array([np.inf])})
    assert np.isnan(op(a, b).uncertainties["u"][0])
    assert np.isnan(iop(a, b).uncertainties["u"][0])


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("right_unit", [ureg.m, ureg.cm])
def test_s0_07_binary_allocation_and_pint_conversion(op, right_unit):
    shape = (1000, 1000)
    a = BaseData(np.full(shape, 4.0), ureg.m, {"u": np.full(shape, 0.2), "v": np.full(shape, 0.3)})
    b = BaseData(np.full(shape, 2.0), right_unit, {"u": np.full(shape, 0.1), "v": np.full(shape, 0.4)})
    tracemalloc.start()
    result = op(a, b)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= 3 * a.signal.nbytes + 65536
    nominal = op(4 * ureg.m, 2 * right_unit)
    assert result.units == nominal.units
    assert_allclose(result.signal[0, 0], nominal.magnitude)
    if op in (operator.add, operator.sub):
        assert_allclose(result.uncertainties["u"][0, 0], np.hypot(0.2, (0.1 * right_unit).to(ureg.m).magnitude))
