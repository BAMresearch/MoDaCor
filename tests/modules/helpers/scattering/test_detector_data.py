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

import logging

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.models.statistics import static_weighted_mean_sem
from modacor.modules.helpers.scattering.detector_data import (
    prepare_static_scalar,
    require_scalar,
)

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
    bd = BaseData(signal=np.array([1.0, 2.0, 3.0]), units=ureg.second, rank_of_data=0)
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


@pytest.mark.parametrize("sigma", [0.4, [0.4], [0.4, 0.4, 0.4, 0.4]])
def test_static_scalar_keeps_uniform_measurement_component(sigma):
    data = BaseData(np.array([1.0, 2.0, 3.0, 4.0]), ureg.m, uncertainties={"measurement": sigma})
    mean, sem = static_weighted_mean_sem(data.signal, data.weights)
    output = prepare_static_scalar(data, uncertainty_key="jitter")
    np.testing.assert_allclose(output.uncertainties["measurement"], 0.4, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.signal, mean, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["jitter"], sem, rtol=1e-12, atol=0)
    assert output.units == data.units


def test_static_scalar_propagates_weighted_components_and_preserves_mean_sem():
    data = BaseData(
        np.array([[2.0], [4.0], [6.0]]),
        ureg.m,
        uncertainties={"measurement": np.array([[0.2], [0.3], [0.4]]), "calibration": 0.6},
        weights=np.array([[1.0], [2.0], [3.0]]),
    )
    mean, sem = static_weighted_mean_sem(data.signal, data.weights)
    output = prepare_static_scalar(data, uncertainty_key="jitter")
    np.testing.assert_allclose(output.signal, mean, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["jitter"], sem, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["measurement"], 1.0 / 3.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["calibration"], 0.6, rtol=1e-12, atol=0)
    np.testing.assert_allclose(data.uncertainties["measurement"], [[0.2], [0.3], [0.4]], rtol=1e-12, atol=0)
    assert set(data.uncertainties) == {"measurement", "calibration"}


@pytest.mark.parametrize("weights", [None, 1.0, [1.0, 2.0, 3.0]])
def test_static_scalar_without_components_matches_previous_result(weights):
    data = BaseData(np.array([2.0, 4.0, 6.0]), ureg.m)
    if weights is not None:
        data.weights = np.asarray(weights)
    mean, sem = static_weighted_mean_sem(data.signal, data.weights)
    output = prepare_static_scalar(data, uncertainty_key="jitter")
    np.testing.assert_allclose(output.signal, mean, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["jitter"], sem, rtol=1e-12, atol=0)
    assert set(output.uncertainties) == {"jitter"}


def test_static_scalar_rejects_component_collision_with_jitter_key():
    data = BaseData(np.array([1.0, 2.0, 3.0]), ureg.m, uncertainties={"calibration_jitter": 0.1})
    with pytest.raises(ValueError, match="calibration_jitter.*conflict"):
        prepare_static_scalar(data, uncertainty_key="calibration_jitter")


def test_static_scalar_passthrough_keeps_components_even_with_jitter_name():
    data = BaseData.from_scalar(2.0, ureg.m, uncertainties={"jitter": 0.3})
    output = prepare_static_scalar(data, uncertainty_key="jitter")
    np.testing.assert_allclose(output.signal, data.signal, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["jitter"], data.uncertainties["jitter"], rtol=1e-12, atol=0)


def test_static_scalar_preserves_nan_propagation_with_measurement_components():
    data = BaseData(np.array([1.0, np.nan, 3.0]), ureg.m, uncertainties={"measurement": 0.3})
    output = prepare_static_scalar(data, uncertainty_key="jitter")
    assert np.isnan(output.signal) and np.isnan(output.uncertainties["jitter"])
    np.testing.assert_allclose(output.uncertainties["measurement"], 0.3, rtol=1e-12, atol=0)


@pytest.mark.parametrize("weights", [1.0, [1.0, 2.0, 3.0, 4.0]])
def test_static_scalar_folds_unreduced_global_component_into_jitter(weights, caplog):
    data = BaseData(
        np.array([1.0, 2.0, 3.0, 4.0]),
        ureg.m,
        uncertainties={"propagate_to_all": 0.4},
        weights=np.asarray(weights),
    )
    mean, sem = static_weighted_mean_sem(data.signal, data.weights)
    global_mean = 0.4
    with caplog.at_level(logging.INFO, logger="modacor.models.uncertainty"):
        output = prepare_static_scalar(data, uncertainty_key="jitter")
    assert set(output.uncertainties) == {"jitter"}
    np.testing.assert_allclose(output.signal, mean, rtol=1e-12, atol=0)
    np.testing.assert_allclose(output.uncertainties["jitter"], np.hypot(sem, global_mean), rtol=1e-12, atol=0)
    assert set(data.uncertainties) == {"propagate_to_all"}
    records = [record for record in caplog.records if "Stage 0 legacy fold" in record.message]
    assert len(records) == 1 and "jitter" in records[0].message


def test_static_scalar_component_reduction_accepts_singleton_weight_shape():
    data = BaseData(np.arange(4.0), ureg.m, uncertainties={"measurement": 0.4})
    data.weights = np.ones((1, 1, 1))
    output = prepare_static_scalar(data, uncertainty_key="jitter")
    np.testing.assert_allclose(output.uncertainties["measurement"], 0.4, rtol=1e-12, atol=0)


def test_static_scalar_component_does_not_shrink_with_frame_count():
    components = []
    for count in (5, 16):
        data = BaseData(np.full(count, 0.15406), ureg.nm, uncertainties={"wavelength_error": 0.005})
        output = prepare_static_scalar(data, require_units=ureg.nm, uncertainty_key="wavelength_jitter")
        components.append(output.uncertainties["wavelength_error"])
    np.testing.assert_allclose(components, [0.005, 0.005], rtol=1e-12, atol=0)
