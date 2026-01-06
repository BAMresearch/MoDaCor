# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "06/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports
__version__ = "20260106.1"

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.modules.technique_modules.scattering.geometry_helpers import (
    prepare_static_scalar,
    require_scalar,
    unit_vec3,
)

# ----------------------------
# unit_vec3
# ----------------------------


def test_unit_vec3_normalizes():
    v = np.array([3.0, 0.0, 4.0])
    u = unit_vec3(v)
    np.testing.assert_allclose(np.linalg.norm(u), 1.0)
    np.testing.assert_allclose(u, np.array([0.6, 0.0, 0.8]))


def test_unit_vec3_rejects_zero_vector():
    with pytest.raises(ValueError, match="must be non-zero"):
        unit_vec3((0.0, 0.0, 0.0), name="basis_fast")


# ----------------------------
# require_scalar
# ----------------------------


def test_require_scalar_passes_scalar_and_sets_rod0():
    # scalar signal => ndim=0 => rank_of_data MUST be 0
    bd = BaseData(signal=np.array(2.5), units=ureg.m, rank_of_data=0)
    out = require_scalar("z", bd)
    assert np.size(out.signal) == 1
    assert out.rank_of_data == 0
    assert out.units.is_compatible_with(ureg.m)
    np.testing.assert_allclose(out.signal, 2.5)


def test_require_scalar_squeezes_singleton_array():
    # singleton array is valid; RoD must not exceed ndim
    bd = BaseData(signal=np.array([[[[2.5]]]]), units=ureg.m, rank_of_data=0)
    out = require_scalar("z", bd)
    assert np.size(out.signal) == 1
    assert out.rank_of_data == 0
    np.testing.assert_allclose(out.signal, 2.5)


def test_require_scalar_rejects_non_scalar():
    bd = BaseData(signal=np.array([1.0, 2.0]), units=ureg.m, rank_of_data=0)
    with pytest.raises(ValueError, match="must be scalar"):
        require_scalar("det_z", bd)


# ----------------------------
# prepare_static_scalar
# ----------------------------


def test_prepare_static_scalar_passes_through_scalar():
    # scalar signal => ndim=0 => RoD must be 0
    bd = BaseData(signal=np.array(2.5), units=ureg.m, rank_of_data=0)
    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="jitter")
    assert np.size(out.signal) == 1
    assert out.rank_of_data == 0
    assert out.units.is_compatible_with(ureg.m)
    np.testing.assert_allclose(out.signal, 2.5)
    # passthrough: don't assert uncertainties content


def test_prepare_static_scalar_reduces_shape_5_1_1_1_uniform_weights_mean_and_sem():
    values = np.array([2.50, 2.52, 2.48, 2.51, 2.49], dtype=float).reshape(5, 1, 1, 1)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)

    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")

    exp_mean = float(np.mean(values))
    flat = values.ravel()
    exp_var = float(np.mean((flat - exp_mean) ** 2))  # population var
    exp_sem = float(np.sqrt(exp_var) / np.sqrt(flat.size))

    np.testing.assert_allclose(out.signal, exp_mean, rtol=0, atol=1e-15)
    assert out.rank_of_data == 0
    assert "sem" in out.uncertainties
    np.testing.assert_allclose(out.uncertainties["sem"], exp_sem, rtol=0, atol=1e-15)


def test_prepare_static_scalar_reduces_1d_shape_5_to_scalar_mean_and_sem():
    """
    New: common case where NeXus/HDF5 read yields a squeezed vector shape (5,)
    (e.g. after user preprocessing or reader behavior).
    """
    values = np.array([2.50, 2.52, 2.48, 2.51, 2.49], dtype=float)  # shape (5,)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)

    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")

    exp_mean = float(np.mean(values))
    exp_var = float(np.mean((values - exp_mean) ** 2))  # population var
    exp_sem = float(np.sqrt(exp_var) / np.sqrt(values.size))

    assert np.size(out.signal) == 1
    assert out.rank_of_data == 0
    np.testing.assert_allclose(out.signal, exp_mean, rtol=0, atol=1e-15)
    assert "sem" in out.uncertainties
    np.testing.assert_allclose(out.uncertainties["sem"], exp_sem, rtol=0, atol=1e-15)


def test_prepare_static_scalar_accepts_scalar_weights_broadcasts():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=float).reshape(4, 1, 1, 1)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)
    bd.weights = np.array([1.0])  # scalar/size-1 weights

    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")

    exp_mean = float(np.mean(values))
    flat = values.ravel()
    exp_var = float(np.mean((flat - exp_mean) ** 2))
    exp_sem = float(np.sqrt(exp_var) / np.sqrt(flat.size))

    np.testing.assert_allclose(out.signal, exp_mean, rtol=0, atol=1e-15)
    np.testing.assert_allclose(out.uncertainties["sem"], exp_sem, rtol=0, atol=1e-15)


def test_prepare_static_scalar_accepts_broadcastable_weights():
    values = np.array([10.0, 20.0, 30.0, 40.0], dtype=float).reshape(4, 1, 1, 1)
    weights = np.array([1.0, 1.0, 2.0, 2.0], dtype=float).reshape(4, 1, 1, 1)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)
    bd.weights = weights

    out = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")

    x = values.ravel()
    w = weights.ravel()
    wsum = float(np.sum(w))
    exp_mean = float(np.sum(w * x) / wsum)
    n_eff = float((wsum**2) / np.sum(w**2))
    exp_var = float(np.sum(w * (x - exp_mean) ** 2) / wsum)
    exp_sem = float(np.sqrt(exp_var) / np.sqrt(n_eff))

    np.testing.assert_allclose(out.signal, exp_mean, rtol=0, atol=1e-15)
    np.testing.assert_allclose(out.uncertainties["sem"], exp_sem, rtol=0, atol=1e-15)


def test_prepare_static_scalar_rejects_non_broadcastable_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=float).reshape(4, 1, 1, 1)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)
    bd.weights = np.array([1.0, 2.0, 3.0])  # not broadcastable

    with pytest.raises(ValueError, match="weights shape .* does not match signal shape"):
        prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")


def test_prepare_static_scalar_rejects_wrong_units():
    bd = BaseData(signal=np.array([1.0, 2.0, 3.0]), units=ureg.pixel, rank_of_data=0)
    with pytest.raises(ValueError, match="Value must be in"):
        prepare_static_scalar(bd, require_units=ureg.m)


def test_prepare_static_scalar_rejects_nonpositive_weight_sum():
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    bd = BaseData(signal=values, units=ureg.m, rank_of_data=0)
    bd.weights = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="weights must sum to > 0"):
        prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")


def test_prepare_static_scalar_signal_1d_weights_5_1_1_1_raises():
    # signal is already squeezed to (5,)
    x = np.array([2.50, 2.52, 2.48, 2.51, 2.49], dtype=float)  # shape (5,)

    # weights come from a NeXus/HDF5-like dataset: shape (5,1,1,1)
    w = np.array([1.0, 1.0, 2.0, 2.0, 4.0], dtype=float).reshape(5, 1, 1, 1)

    bd = BaseData(signal=x, units=ureg.m, rank_of_data=0)
    bd.weights = w

    with pytest.raises(ValueError, match="weights shape .* does not match signal shape"):
        _ = prepare_static_scalar(bd, require_units=ureg.m, uncertainty_key="sem")
