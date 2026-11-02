import numpy as np
import pytest

from modacor.models.statistics import finalize_weighted_scatter, static_weighted_mean_sem


def test_finalize_weighted_scatter_matches_indexed_averager_population_formula():
    estimates = finalize_weighted_scatter(
        sum_w=np.array([2.0]),
        sum_w2=np.array([2.0]),
        sum_w_squared_deviations=np.array([0.5]),
        ddof=0,
    )

    np.testing.assert_allclose(estimates.effective_sample_size, [2.0])
    np.testing.assert_allclose(estimates.standard_deviation, [0.5])
    np.testing.assert_allclose(estimates.standard_error_mean, [np.sqrt(0.125)])


def test_finalize_weighted_scatter_applies_effective_ddof_correction():
    estimates = finalize_weighted_scatter(
        sum_w=np.array(3.0),
        sum_w2=np.array(5.0),
        sum_w_squared_deviations=np.array(6.0),
        ddof=1,
    )

    assert estimates.effective_sample_size == pytest.approx(1.8)
    assert estimates.variance == pytest.approx(4.5)
    assert estimates.standard_deviation == pytest.approx(np.sqrt(4.5))
    assert estimates.standard_error_mean == pytest.approx(np.sqrt(2.5))
    assert estimates.standard_error_sum == pytest.approx(np.sqrt(22.5))


def test_finalize_weighted_scatter_returns_nan_when_ddof_exhausts_effective_count():
    estimates = finalize_weighted_scatter(
        sum_w=np.array(1.0),
        sum_w2=np.array(1.0),
        sum_w_squared_deviations=np.array(0.0),
        ddof=1,
    )

    assert np.isnan(estimates.variance)
    assert np.isnan(estimates.standard_deviation)
    assert np.isnan(estimates.standard_error_mean)
    assert np.isnan(estimates.standard_error_sum)


@pytest.mark.parametrize("weights", [None, np.array(1.0), np.array([[1.0, 2.0, 3.0]])])
def test_static_statistics_match_module_boundary(weights):
    from modacor.dataclasses.basedata import BaseData
    from modacor.modules.helpers.scattering.detector_data import prepare_static_scalar

    values = np.array([[1.0, 3.0, 8.0], [2.0, 5.0, 9.0]])
    mean, sem = static_weighted_mean_sem(values, weights)
    data = BaseData(values, "m", weights=1.0 if weights is None else weights)
    actual = prepare_static_scalar(data)
    np.testing.assert_allclose(actual.signal, mean, rtol=1e-12, atol=0)
    np.testing.assert_allclose(actual.uncertainties["static_config_jitter"], sem, rtol=1e-12, atol=0)
    w = np.ones_like(values) if weights is None else np.broadcast_to(weights, values.shape)
    expected_mean = np.sum(w * values) / np.sum(w)
    expected_var = np.sum(w * (values - expected_mean) ** 2) / np.sum(w)
    expected_sem = np.sqrt(expected_var) / np.sqrt(np.sum(w) ** 2 / np.sum(w**2))
    np.testing.assert_allclose([mean, sem], [expected_mean, expected_sem], rtol=1e-12, atol=0)


def test_static_statistics_keep_zero_weight_and_nan_policies():
    with pytest.raises(ValueError, match="sum to > 0"):
        static_weighted_mean_sem(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    mean, sem = static_weighted_mean_sem(np.array([1.0, np.nan]), None)
    assert np.isnan(mean) and np.isnan(sem)
