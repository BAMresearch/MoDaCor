# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

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
from ..dataclasses.basedata import (  # adjust the import path as needed
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


def test_initial_variances_and_uncertainties(simple_basedata):
    bd = simple_basedata
    # variances property squares each uncertainty
    vars_dict = bd.variances
    assert np.allclose(vars_dict["poisson"], 0.5**2)
    assert np.allclose(vars_dict["sem"], 0.2**2)

    # ensure uncertainties remain unchanged
    assert "poisson" in bd.uncertainties and "sem" in bd.uncertainties
    assert bd.uncertainties["sem"] == pytest.approx(0.2)


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


def test_scalar_variance_property_and_setter(simple_basedata):
    bd = simple_basedata
    # default scalar_uncertainty = 0 → variance = 0
    assert bd.scaling_variance == 0.0

    # set scalar_variance via scalar
    bd.scaling_variance = 9.0
    assert bd.scaling_uncertainty == pytest.approx(3.0)
    assert bd.scaling_variance == pytest.approx(9.0)

    # setting with array of size 1 is allowed
    bd.scaling_variance = np.array([16.0])
    assert bd.scaling_uncertainty == pytest.approx(4.0)

    # array of size >1 should error
    with pytest.raises(ValueError):
        bd.scaling_variance = np.array([1.0, 2.0])


def test_weighting_broadcast_validation(simple_basedata):
    bd = simple_basedata
    # valid weighting (broadcastable to (2,3))
    bd.weights = np.array([1.0, 2.0, 3.0])
    bd.__attrs_post_init__()  # should not raise

    # invalid weighting shape
    with pytest.raises(ValueError):
        bd.weights = np.ones((3, 2))
        bd.__attrs_post_init__()


def test_apply_scalar_affects_signal_and_uncertainty(simple_basedata):
    bd = simple_basedata
    # Set a non-default scalar and uncertainty
    bd.scaling = 2.0
    bd.scaling_uncertainty = 0.5  # so scalar_variance = 0.25
    original_signal = bd.signal.copy()
    bd.apply_scaling()

    # signal doubled
    assert np.allclose(bd.signal, original_signal * 2.0)
    # scalar reset
    assert bd.scaling == 1.0
    # scalar_uncertainty /= scalar_before (0.5/2 = 0.25)
    assert bd.scaling_uncertainty == pytest.approx(0.25)


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


def test_apply_offset(simple_basedata):
    bd = simple_basedata
    original = bd.signal.copy()
    bd.offset = 3.5
    bd.offset_uncertainty = 0.5
    bd.apply_offset()
    expected = original + 3.5
    np.testing.assert_allclose(bd.signal, expected)


def test_to_units_converts_properly():
    sig = np.array([[1.0, 2.0], [3.0, 4.0]])
    bd = BaseData(signal=sig.copy(), units=ureg.meter)

    bd.to_units(ureg.centimeter)
    bd.apply_scaling()  # unit conversion is applied to scalar
    bd.apply_offset()
    expected = sig * 100  # m to cm
    assert bd.units == ureg.centimeter
    np.testing.assert_allclose(bd.signal, expected)
