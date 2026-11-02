# SPDX-License-Identifier: BSD-3-Clause
"""Array kernels retain the release modules' reduction policies."""

import numpy as np
import pytest

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.models.reduction import (
    axis_count,
    axis_scatter_moments,
    effective_values_and_weights,
    group_observations,
    grouped_mean,
    grouped_scatter_statistics,
    grouped_uncertainties,
    grouped_weights,
    scalar_weight_reduction,
    weighted_axis_reduction,
)
from modacor.modules import IndexedAverager, ReduceDimensionality


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("nan_policy", ["omit", "propagate"])
@pytest.mark.parametrize("scalar_weight", [0.0, 2.0, np.nan])
def test_scalar_kernel_matches_module_boundary(reduction, nan_policy, scalar_weight):
    data = BaseData(
        np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 8.0]]), "count", uncertainties={"measured": 0.1}, weights=scalar_weight
    )
    nominal, components, weights = scalar_weight_reduction(
        data.signal, data.uncertainties, axis=0, nan_policy=nan_policy, reduction=reduction, scalar_weight=scalar_weight
    )
    expected = ReduceDimensionality._scalar_weight_reduction(
        bd=data, axis=0, nan_policy=nan_policy, reduction=reduction, scalar_weight=scalar_weight
    )
    np.testing.assert_allclose(nominal, expected.signal, rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_allclose(
        components["measured"], expected.uncertainties["measured"], rtol=1e-12, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(weights, expected.weights, rtol=1e-12, atol=0, equal_nan=True)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("nan_policy", ["omit", "propagate"])
@pytest.mark.parametrize("mask", [None, np.array([[True, False, False], [False, False, False]])])
def test_weighted_kernel_matches_module_and_release_formula(reduction, nan_policy, mask):
    x = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 8.0]])
    w = np.array([[1.0], [2.0]])
    data = BaseData(x, "count", uncertainties={"measured": 0.1}, weights=w)
    nominal, components, _ = weighted_axis_reduction(
        x,
        data.uncertainties,
        w,
        axis=0,
        use_weights=True,
        nan_policy=nan_policy,
        reduction=reduction,
        exclusion_mask=mask,
    )
    expected = ReduceDimensionality._weighted_mean_with_uncertainty(
        data, 0, True, nan_policy, reduction=reduction, exclusion_mask=mask
    )
    np.testing.assert_allclose(nominal, expected.signal, rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_allclose(
        components["measured"], expected.uncertainties["measured"], rtol=1e-12, atol=0, equal_nan=True
    )
    selected = np.zeros(x.shape, bool) if mask is None else mask.copy()
    if nan_policy == "omit":
        selected |= np.isnan(x)
    weff = np.where(selected, 0.0, np.broadcast_to(w, x.shape))
    xeff = np.where(selected, 0.0, x)
    denominator = np.where(np.sum(weff, axis=0) == 0.0, np.nan, np.sum(weff, axis=0)) if reduction == "mean" else 1.0
    np.testing.assert_allclose(nominal, np.sum(weff * xeff, axis=0) / denominator, rtol=1e-12, atol=0, equal_nan=True)


def test_effective_values_and_scatter_moments_keep_zero_weight_policy():
    x, w, mask = effective_values_and_weights(
        signal=np.array([[1.0, np.nan], [3.0, 8.0]]),
        weights=np.array([[1.0], [2.0]]),
        use_weights=True,
        nan_policy="omit",
        exclusion_mask=None,
    )
    np.testing.assert_allclose(x, [[1.0, 0.0], [3.0, 8.0]], rtol=1e-12, atol=0)
    np.testing.assert_allclose(w, [[1.0, 0.0], [2.0, 2.0]], rtol=1e-12, atol=0)
    assert mask[0, 1]
    sum_w, sum_w2, scatter = axis_scatter_moments(x, w, axis=0)
    np.testing.assert_allclose(sum_w, [3.0, 2.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(sum_w2, [5.0, 4.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(scatter, [8 / 3, 0.0], rtol=1e-12, atol=0)
    assert axis_count(x.shape, None) == 4
    assert axis_count(x.shape, (0, -1)) == 4
    assert axis_count(x.shape, -1) == 2


def test_grouped_kernels_match_module_outputs_and_release_moments():
    x = np.array([1.0, 3.0, 2.0, 6.0])
    indices = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 2.0, 0.0, 1.0])
    grouped = group_observations(indices, x, None, weights, np.ones(4, bool))
    means = grouped_mean(grouped.values, grouped.dense_indices, grouped.weights, grouped.sum_weights)
    components = grouped_uncertainties(
        {"measured": np.full(4, 0.1)},
        dense_indices=grouped.dense_indices,
        weights=grouped.weights,
        sum_weights=grouped.sum_weights,
    )
    sem, std = grouped_scatter_statistics(
        grouped.values, means, grouped.dense_indices, grouped.weights, grouped.sum_weights, grouped.sum_squared_weights
    )
    np.testing.assert_allclose(means, [7 / 3, 6.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(components["measured"], [0.1 * np.sqrt(5) / 3, 0.1], rtol=1e-12, atol=0)
    np.testing.assert_allclose(std, [np.sqrt(8 / 9), np.nan], rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_allclose(sem, [np.sqrt(40 / 81), np.nan], rtol=1e-12, atol=0, equal_nan=True)
    bundle = DataBundle(
        signal=BaseData(x, "count", {"measured": 0.1}, weights=weights), bin_index=BaseData(indices, "")
    )
    result = IndexedAverager(configuration={"axis_key": None})._reduce_bundle(bundle)["signal"]
    np.testing.assert_allclose(means, result.signal, rtol=1e-12, atol=0)
    for key, expected in [("measured", components["measured"]), ("SEM", sem), ("STD", std)]:
        np.testing.assert_allclose(expected, result.uncertainties[key], rtol=1e-12, atol=0, equal_nan=True)


def test_inverse_variance_weight_kernel():
    weights, valid = grouped_weights(
        (4,), np.ones(4, bool), value_weights=np.array([1.0, 2.0, 0.0, 1.0]), sigma=np.array([1.0, 2.0, 0.0, np.nan])
    )
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.0, 1.0], rtol=1e-12, atol=0)
    assert valid.tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="negative"):
        grouped_weights((1,), np.ones(1, bool), value_weights=np.array([-1.0]))
