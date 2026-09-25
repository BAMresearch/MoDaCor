import numpy as np
import pytest

from modacor.modules.helpers.statistics import finalize_weighted_scatter


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
