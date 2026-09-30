from __future__ import annotations

import numpy as np
import pytest

from modacor.models.scaling import (
    DependentData1D,
    FitData1D,
    fit_lognormal_scale_factor_1d,
    fit_scale_factor_1d,
    prepare_scale_fit_data,
)


def _dependent(values: np.ndarray) -> DependentData1D:
    return DependentData1D(
        y=np.asarray(values, dtype=float),
        sigma=np.full(np.shape(values), 0.1),
        weights=np.ones(np.shape(values)),
    )


def test_prepare_and_fit_scale_with_background() -> None:
    x_ref = np.linspace(0.0, 4.0, 5)
    work = _dependent(x_ref + 1.0)
    reference = _dependent(2.5 * work.y + 3.0)
    fit_data = prepare_scale_fit_data(
        x_work=x_ref,
        dep_work=work,
        x_ref=x_ref,
        dep_ref=reference,
        require_overlap=True,
        interpolation_kind="linear",
        fit_min=0.0,
        fit_max=4.0,
        use_weights=True,
    )

    result = fit_scale_factor_1d(
        fit_data,
        fit_background=True,
        robust_loss="linear",
        robust_fscale=1.0,
    )

    assert result.scale == pytest.approx(2.5)
    assert result.background == pytest.approx(3.0)


def test_prepare_scale_fit_data_rejects_disjoint_axes() -> None:
    with pytest.raises(ValueError, match="No overlap"):
        prepare_scale_fit_data(
            x_work=np.array([0.0, 1.0]),
            dep_work=_dependent(np.array([1.0, 2.0])),
            x_ref=np.array([2.0, 3.0]),
            dep_ref=_dependent(np.array([3.0, 4.0])),
            require_overlap=True,
            interpolation_kind="linear",
            fit_min=0.0,
            fit_max=3.0,
            use_weights=True,
        )


def test_prepare_scale_fit_data_uncertainty_averages_duplicate_coordinates() -> None:
    work = DependentData1D(
        y=np.asarray([1.0, 2.0, 4.0, 5.0]),
        sigma=np.asarray([0.1, 0.2, 0.4, 0.5]),
        weights=np.ones(4),
    )
    fit_data = prepare_scale_fit_data(
        x_work=np.asarray([0.0, 1.0, 1.0, 2.0]),
        dep_work=work,
        x_ref=np.asarray([0.0, 1.0, 2.0]),
        dep_ref=_dependent(np.asarray([1.0, 2.4, 5.0])),
        require_overlap=True,
        interpolation_kind="linear",
        fit_min=0.0,
        fit_max=2.0,
        use_weights=True,
    )

    np.testing.assert_allclose(fit_data.y_work, [1.0, 2.4, 5.0])
    np.testing.assert_allclose(fit_data.sigma_work, [0.1, 1.0 / np.sqrt(31.25), 0.5])


def test_lognormal_scale_is_uncertainty_weighted_geometric_ratio():
    fit_data = FitData1D(
        x=np.arange(3.0),
        y_ref=np.array([2.0, 8.0, 18.0]),
        y_work=np.array([1.0, 2.0, 3.0]),
        sigma_ref=np.array([0.2, 0.8, 1.8]),
        sigma_work=np.array([0.1, 0.2, 0.3]),
        weights=np.ones(3),
    )

    result = fit_lognormal_scale_factor_1d(fit_data)

    relative_variance = 0.1**2 + 0.1**2
    expected_log_scale = np.mean(np.log([2.0, 4.0, 6.0]))
    assert result.scale == pytest.approx(np.exp(expected_log_scale))
    assert result.scale_sigma == pytest.approx(result.scale * np.sqrt(relative_variance / 3.0))
