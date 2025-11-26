# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "18/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest

from modacor import ureg

# import tiled.client  # not sure what the class of tiled.client is...
from modacor.dataclasses.basedata import (  # adjust the import path as needed
    BaseData,
    signal_converter,
    validate_broadcast,
    validate_rank_of_data,
)


@pytest.fixture
def simple_basedata():
    sig = np.arange(6, dtype=float).reshape((2, 3))
    uncs = {
        "poisson": np.full((2, 3), 0.5),
        "sem": 0.2,  # scalar uncertainty
    }
    return BaseData(signal=sig, uncertainties=uncs, units=ureg.dimensionless)


# ---------------------------------------------------------------------------
# Basic helpers & broadcast tests
# ---------------------------------------------------------------------------


def test_signal_converter_converts_scalars_and_preserves_arrays():
    # Scalar int → array
    arr1 = signal_converter(5)
    assert isinstance(arr1, np.ndarray)
    assert arr1.shape == ()
    assert arr1.item() == 5.0

    # Scalar float → array
    arr2 = signal_converter(3.14)
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == ()
    assert arr2.item() == pytest.approx(3.14)

    # ndarray → unchanged
    original = np.array([[1, 2], [3, 4]], dtype=float)
    arr3 = signal_converter(original)
    assert arr3 is original


def test_validate_broadcast_accepts_scalars_and_matching_shapes():
    signal = np.zeros((4, 5, 6))

    # Scalar arrays (size 1) always okay
    scalar = np.array(2.0)
    validate_broadcast(signal, scalar, "scalar")

    # Exact shape
    arr_full = np.ones((4, 5, 6))
    validate_broadcast(signal, arr_full, "full")

    # Broadcastable suffix shape
    arr_suffix = np.ones((5, 6))
    validate_broadcast(signal, arr_suffix, "suffix")

    # Leading ones OK
    arr_leading = np.ones((1, 5, 6))
    validate_broadcast(signal, arr_leading, "leading")

    # Single dimension match
    arr_last = np.ones((6,))
    validate_broadcast(signal, arr_last, "last_dim")


def test_validate_broadcast_raises_on_incompatible_shapes():
    signal = np.zeros((3, 4, 5))

    # Totally incompatible
    with pytest.raises(ValueError):
        validate_broadcast(signal, np.ones((2, 2)), "bad1")

    # Broadcasts to wrong shape (e.g., (3,4,4) → (3,4,5))
    with pytest.raises(ValueError):
        validate_broadcast(signal, np.ones((3, 4, 4)), "bad2")


# ---------------------------------------------------------------------------
# Variances / uncertainties
# ---------------------------------------------------------------------------


def test_initial_variances_and_uncertainties(simple_basedata):
    bd = simple_basedata
    # variances property squares each uncertainty
    vars_dict = bd.variances
    assert np.allclose(vars_dict["poisson"], 0.5**2)
    assert np.allclose(vars_dict["sem"], 0.2**2)

    # ensure uncertainties remain unchanged
    assert "poisson" in bd.uncertainties and "sem" in bd.uncertainties
    assert bd.uncertainties["sem"] == pytest.approx(0.2)


def test_variances_setter_updates_uncertainties_and_validates_shape(simple_basedata):
    bd = simple_basedata
    # valid new variances (scalar and full array)
    new_vars = {
        "poisson": np.full((2, 3), 0.25),
        "sem": 0.04,
    }
    bd.variances = new_vars
    # uncertainties become sqrt(var)
    assert np.allclose(bd.uncertainties["poisson"], 0.25**0.5)
    assert bd.uncertainties["sem"] == pytest.approx(0.04**0.5)

    # invalid shape (wrong shape)
    with pytest.raises(ValueError):
        bd.variances = {"poisson": np.ones((3, 2))}


def test_variances_setitem_updates_underlying_uncertainties(simple_basedata):
    bd = simple_basedata

    new_var = np.array([[4.0, 9.0, 16.0], [25.0, 36.0, 49.0]], dtype=float)

    bd.variances["poisson"] = new_var

    # Underlying uncertainties should now be sqrt(new_var)
    np.testing.assert_allclose(bd.uncertainties["poisson"], np.sqrt(new_var))

    # Reading variances again returns the original variance values
    np.testing.assert_allclose(bd.variances["poisson"], new_var)


def test_variances_setitem_rejects_incompatible_shape(simple_basedata):
    """
    Assigning a variance array that cannot broadcast to signal.shape should raise.
    """
    bd = simple_basedata

    bad_var = np.ones((2,), dtype=float)  # signal has shape (2,3)

    with pytest.raises(ValueError):
        bd.variances["bad"] = bad_var


def test_variances_setter_rejects_non_dict(simple_basedata):
    bd = simple_basedata

    with pytest.raises(TypeError):
        bd.variances = ["not", "a", "dict"]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rank-of-data validation
# ---------------------------------------------------------------------------


def test_validate_rank_of_data_bounds_and_ndim():
    class Dummy:
        def __init__(self, signal, rank):
            self.signal = signal
            self.rank_of_data = rank

    sig1 = np.zeros((2, 3))
    # Valid ranks: 0, 1, 2
    for r in (0, 1, 2):
        dummy = Dummy(sig1, r)
        validate_rank_of_data(dummy, type("A", (), {"name": "rank_of_data"}), r)

    # Negative or >3 invalid
    for r in (-1, 4):
        dummy = Dummy(sig1, r)
        with pytest.raises(ValueError):
            validate_rank_of_data(dummy, type("A", (), {"name": "rank_of_data"}), r)

    # Rank > ndim invalid
    dummy2 = Dummy(np.zeros((5,)), 2)
    with pytest.raises(ValueError):
        validate_rank_of_data(dummy2, type("A", (), {"name": "rank_of_data"}), 2)


def test_rank_of_data_validation_errors(simple_basedata):
    bd = simple_basedata
    # valid rank
    bd.rank_of_data = 1  # <= ndim
    assert bd.rank_of_data == 1

    # invalid rank, 3 > ndim
    with pytest.raises(ValueError):
        bd.rank_of_data = 3

    # invalid rank > 3
    with pytest.raises(ValueError):
        bd.rank_of_data = 5


# ---------------------------------------------------------------------------
# Weights broadcast validation and axes tests
# ---------------------------------------------------------------------------


def test_weighting_broadcast_validation(simple_basedata):
    bd = simple_basedata
    # valid weighting (broadcastable to (2,3))
    bd.weights = np.array([1.0, 2.0, 3.0])
    bd.__attrs_post_init__()  # should not raise

    # invalid weighting shape
    with pytest.raises(ValueError):
        bd.weights = np.ones((3, 2))
        bd.__attrs_post_init__()


def test_axes_sanity_check_logs_when_length_mismatched(simple_basedata, caplog):
    bd = simple_basedata
    # signal.ndim == 2, but axes length is 1 → mismatch
    bd.axes = [None]

    with caplog.at_level(logging.DEBUG):
        bd.__attrs_post_init__()

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("BaseData.axes length" in msg for msg in messages)


def test_axes_sanity_check_no_log_when_length_matches(simple_basedata, caplog):
    bd = simple_basedata
    # signal.ndim == 2, axes length == 2 → OK
    bd.axes = [None, None]

    with caplog.at_level(logging.DEBUG):
        bd.__attrs_post_init__()

    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("BaseData.axes length" in msg for msg in messages)


# ---------------------------------------------------------------------------
# Unit conversion behaviour
# ---------------------------------------------------------------------------


def test_to_units_converts_properly():
    sig = np.array([[1.0, 2.0], [3.0, 4.0]])
    bd = BaseData(signal=sig.copy(), units=ureg.meter)

    bd.to_units(ureg.centimeter)
    expected = sig * 100  # m to cm
    assert bd.units == ureg.centimeter
    np.testing.assert_allclose(bd.signal, expected)


def test_to_units_multiplicative_conversion_scales_signal_and_uncertainties():
    sig = np.array([1.0, 2.0, 3.0], dtype=float)
    uncs = {"stat": np.array([0.1, 0.2, 0.3], dtype=float)}
    bd = BaseData(signal=sig.copy(), units=ureg.meter, uncertainties=uncs)

    signal_before = bd.signal.copy()
    uncs_before = {k: v.copy() for k, v in bd.uncertainties.items()}

    # Use same pint logic as BaseData.to_units
    cfact = ureg.millimeter.m_from(ureg.meter)

    bd.to_units(ureg.millimeter, multiplicative_conversion=True)

    # Units updated
    assert bd.units == ureg.millimeter

    # Signal scaled
    np.testing.assert_allclose(bd.signal, signal_before * cfact)

    # Each uncertainty scaled by the same factor
    for key, unc_after in bd.uncertainties.items():
        np.testing.assert_allclose(unc_after, uncs_before[key] * cfact)


def test_to_units_same_units_is_noop(simple_basedata):
    bd = simple_basedata

    signal_before = bd.signal.copy()
    uncs_before = {k: v.copy() for k, v in bd.uncertainties.items()}

    bd.to_units(ureg.dimensionless, multiplicative_conversion=True)

    # Nothing should have changed
    np.testing.assert_allclose(bd.signal, signal_before)
    for key, unc_after in bd.uncertainties.items():
        np.testing.assert_allclose(unc_after, uncs_before[key])
    assert bd.units == ureg.dimensionless


def test_to_units_incompatible_units_raises(simple_basedata):
    bd = simple_basedata
    # dimensionless vs. time is not compatible
    with pytest.raises(ValueError):
        bd.to_units(ureg.second, multiplicative_conversion=True)


def test_to_units_non_multiplicative_path_not_implemented(simple_basedata):
    """
    Once the non-multiplicative branch in BaseData.to_units is guarded
    with NotImplementedError, this ensures we don't silently do the wrong thing.
    """
    bd = simple_basedata
    bd.units = ureg.kelvin

    with pytest.raises(NotImplementedError):
        bd.to_units(ureg.rankine, multiplicative_conversion=False)


# ---------------------------------------------------------------------------
# Metadata preservation in ops
# ---------------------------------------------------------------------------


def test_binary_ops_preserve_rank_axes_and_weights(simple_basedata):
    bd = simple_basedata

    # Set some non-default metadata
    bd.rank_of_data = 2
    bd.axes = [None, None]  # two-dimensional signal
    bd.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    other = BaseData(signal=np.ones_like(bd.signal), units=bd.units)

    # BaseData / BaseData
    res = bd / other
    assert res.rank_of_data == bd.rank_of_data
    assert res.axes == bd.axes
    assert res.axes is not bd.axes  # new list, no aliasing
    np.testing.assert_allclose(res.weights, bd.weights)

    # BaseData / scalar
    res2 = bd / 2.0
    assert res2.rank_of_data == bd.rank_of_data
    assert res2.axes == bd.axes
    assert res2.axes is not bd.axes
    np.testing.assert_allclose(res2.weights, bd.weights)


def test_unary_ops_preserve_rank_axes_and_weights(simple_basedata):
    bd = simple_basedata

    bd.rank_of_data = 1
    bd.axes = [None, None]  # arbitrary axes metadata
    bd.weights = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    neg = -bd
    sqrt_bd = bd.sqrt()
    log_bd = bd.log()  # valid for all positive elements; 0 will yield NaN, which is fine

    for out in (neg, sqrt_bd, log_bd):
        assert out.rank_of_data == bd.rank_of_data
        assert out.axes == bd.axes
        assert out.axes is not bd.axes
        np.testing.assert_allclose(out.weights, bd.weights)


# ---------------------------------------------------------------------------
# Copy tests:
# ---------------------------------------------------------------------------


def test_copy_creates_independent_arrays_and_axes_list(simple_basedata):
    bd = simple_basedata

    # Set some metadata so we can check it is carried over
    bd.rank_of_data = 2
    bd.axes = [None, None]
    bd.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    bd_copy = bd.copy()

    # Different object
    assert bd_copy is not bd
    assert isinstance(bd_copy, BaseData)

    # Signal copied, not aliased
    np.testing.assert_allclose(bd_copy.signal, bd.signal)
    assert bd_copy.signal is not bd.signal

    # Weights copied, not aliased
    np.testing.assert_allclose(bd_copy.weights, bd.weights)
    assert bd_copy.weights is not bd.weights

    # Uncertainties copied, not aliased
    assert set(bd_copy.uncertainties.keys()) == set(bd.uncertainties.keys())
    for key in bd.uncertainties:
        np.testing.assert_allclose(bd_copy.uncertainties[key], bd.uncertainties[key])
        assert bd_copy.uncertainties[key] is not bd.uncertainties[key]

    # Axes list shallow-copied: new list, same elements
    assert bd_copy.axes == bd.axes
    assert bd_copy.axes is not bd.axes
    # Elements themselves are the same objects (shallow copy)
    for a_orig, a_copy in zip(bd.axes, bd_copy.axes):
        assert a_orig is a_copy

    # rank_of_data and units preserved
    assert bd_copy.rank_of_data == bd.rank_of_data
    assert bd_copy.units == bd.units


def test_copy_without_axes_uses_empty_axes(simple_basedata):
    bd = simple_basedata
    bd.axes = [None, None]

    bd_copy = bd.copy(with_axes=False)

    # Axes dropped
    assert bd_copy.axes == []
    # Other content still copied
    np.testing.assert_allclose(bd_copy.signal, bd.signal)
    for key in bd.uncertainties:
        np.testing.assert_allclose(bd_copy.uncertainties[key], bd.uncertainties[key])
