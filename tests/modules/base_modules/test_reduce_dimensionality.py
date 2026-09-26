# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import logging
import unittest

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources

# adjust this import path to where you put the step:
from modacor.modules.base_modules.reduce_dimensionality import ReduceDimensionality  # noqa: E402

TEST_IO_SOURCES = IoSources()


class TestReduceDimensionality(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/reduce_dim_weighted_average.py"""

    def setUp(self):
        # Simple 2x3 example so we can verify by hand:
        #
        # x = [[1, 2, 3],
        #      [4, 5, 6]]
        #
        self.signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

        # absolute 1σ uncertainties = 0.1 everywhere
        self.unc = 0.1 * np.ones_like(self.signal)

        # weights: second row has weight 2, first row weight 1
        # (broadcastable to signal.shape)
        self.weights = np.array([[1.0], [2.0]], dtype=float)

        self.test_processing_data = ProcessingData()
        self.test_basedata = BaseData(
            signal=self.signal,
            units=ureg.Unit("count"),
            uncertainties={"u": self.unc},
            weights=self.weights,
        )
        self.test_data_bundle = DataBundle(signal=self.test_basedata)
        self.test_processing_data["bundle"] = self.test_data_bundle

    def tearDown(self):
        pass

    # ------------------------------------------------------------------
    # Basic unweighted mean (use_weights=False, nan_policy='propagate')
    # ------------------------------------------------------------------

    def test_unweighted_mean_axis0(self):
        """
        Unweighted mean over axis=0 should match np.mean(signal, axis=0)
        and propagate uncertainties as:
            σ_mean = sqrt(σ1^2 + σ2^2) / N.
        """
        avg_step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
        avg_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            axes=0,
            use_weights=False,
            nan_policy="propagate",
        )
        avg_step.processing_data = self.test_processing_data

        avg_step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Expected mean
        expected_mean = np.mean(self.signal, axis=0)
        np.testing.assert_allclose(result_bd.signal, expected_mean)

        # Expected uncertainty:
        # Two points each with σ=0.1 -> σ_mean = sqrt(0.1^2 + 0.1^2) / 2
        expected_sigma = np.sqrt(0.1**2 + 0.1**2) / 2.0
        expected_u = np.full_like(expected_mean, expected_sigma)
        np.testing.assert_allclose(result_bd.uncertainties["u"], expected_u)

        # Units should be preserved
        self.assertEqual(result_bd.units, ureg.Unit("count"))

    # ------------------------------------------------------------------
    # Weighted mean (use_weights=True)
    # ------------------------------------------------------------------

    def test_weighted_mean_axis0(self):
        """
        Weighted mean over axis=0 using BaseData.weights.

        For each column:
            μ = (1*x1 + 2*x2) / (1+2)
            σ^2 = (1^2 σ1^2 + 2^2 σ2^2) / (1+2)^2
        with σ1 = σ2 = 0.1.
        """
        avg_step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
        avg_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            axes=0,
            use_weights=True,
            nan_policy="propagate",
        )
        avg_step.processing_data = self.test_processing_data

        avg_step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Expected weighted mean along axis 0
        w1, w2 = 1.0, 2.0
        w_sum = w1 + w2
        expected_mean = (w1 * self.signal[0, :] + w2 * self.signal[1, :]) / w_sum
        np.testing.assert_allclose(result_bd.signal, expected_mean)

        # Uncertainty:
        # σ_μ^2 = (w1^2 σ1^2 + w2^2 σ2^2) / (w_sum^2)
        sigma1 = sigma2 = 0.1
        var_num = w1**2 * sigma1**2 + w2**2 * sigma2**2  # = (1 + 4)*0.01 = 0.05
        expected_sigma = np.sqrt(var_num) / w_sum
        expected_u = np.full_like(expected_mean, expected_sigma)
        np.testing.assert_allclose(result_bd.uncertainties["u"], expected_u)

        self.assertEqual(result_bd.units, ureg.Unit("count"))

    def test_weighted_sum_axis0(self):
        """
        Weighted sum over axis=0 using BaseData.weights.

        For each column:
            S = Σ w_i x_i = 1*x1 + 2*x2
            σ_S^2 = Σ w_i^2 σ_i^2 = (1^2 + 2^2) * σ^2
        with σ = 0.1 everywhere.
        """
        avg_step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
        avg_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            axes=0,
            use_weights=True,
            nan_policy="propagate",
            reduction="sum",  # NEW
        )
        avg_step.processing_data = self.test_processing_data

        avg_step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Expected weighted sum along axis 0
        w1, w2 = 1.0, 2.0
        expected_sum = w1 * self.signal[0, :] + w2 * self.signal[1, :]
        np.testing.assert_allclose(result_bd.signal, expected_sum)

        # Uncertainty:
        # σ_S^2 = (w1^2 + w2^2) * σ^2
        sigma = 0.1
        var_factor = w1**2 + w2**2  # 1 + 4 = 5
        expected_sigma = np.sqrt(var_factor * sigma**2)
        expected_u = np.full_like(expected_sum, expected_sigma)
        np.testing.assert_allclose(result_bd.uncertainties["u"], expected_u)

        # Units preserved
        self.assertEqual(result_bd.units, ureg.Unit("count"))

    # ------------------------------------------------------------------
    # nan_policy='omit'
    # ------------------------------------------------------------------

    def test_nanmean_omit(self):
        """
        When nan_policy='omit', NaNs in the signal are ignored.
        For columns with only one finite value, the mean is that value
        and the uncertainty is its σ.
        """
        # Introduce NaNs: one partial column, one fully NaN column
        signal_nan = self.signal.copy()
        signal_nan[0, 1] = np.nan  # second column: [NaN, 5]
        signal_nan[:, 2] = np.nan  # third column: [NaN, NaN]

        # Update processing data with this modified signal
        bd_nan = BaseData(
            signal=signal_nan,
            units=ureg.Unit("count"),
            uncertainties={"u": self.unc},  # still 0.1 everywhere
            weights=self.weights,
        )
        self.test_processing_data["bundle"]["signal"] = bd_nan

        avg_step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
        avg_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            axes=0,
            use_weights=False,
            nan_policy="omit",
        )
        avg_step.processing_data = self.test_processing_data

        avg_step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Column 0: mean of [1, 4]
        expected_col0_mean = (1.0 + 4.0) / 2.0
        # Column 1: mean of [5] (since NaN is omitted)
        expected_col1_mean = 5.0
        # Column 2: all NaN -> NaN
        expected_mean = np.array([expected_col0_mean, expected_col1_mean, np.nan])
        np.testing.assert_allclose(result_bd.signal, expected_mean, equal_nan=True)

        # Uncertainties:
        # Col0: two points with σ=0.1 -> σ_mean = sqrt(0.1^2+0.1^2)/2
        col0_sigma = np.sqrt(0.1**2 + 0.1**2) / 2.0
        # Col1: single finite point with σ=0.1 -> σ_mean = 0.1
        col1_sigma = 0.1
        # Col2: no finite points -> NaN
        expected_u = np.array([col0_sigma, col1_sigma, np.nan])
        np.testing.assert_allclose(result_bd.uncertainties["u"], expected_u, equal_nan=True)

        self.assertEqual(result_bd.units, ureg.Unit("count"))

    # ------------------------------------------------------------------
    # Execution via __call__ shortcut
    # ------------------------------------------------------------------

    def test_weighted_average_execution_via_call(self):
        """
        Ensure the ProcessStep __call__ interface works, like for PoissonUncertainties.
        """
        avg_step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
        avg_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            axes=0,
            use_weights=True,
            nan_policy="omit",
        )

        # Execute via __call__
        avg_step(self.test_processing_data)

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Basic sanity checks: shape reduced along axis 0 → (3,)
        self.assertEqual(result_bd.signal.shape, (3,))
        # Units preserved
        self.assertEqual(result_bd.units, ureg.Unit("count"))
        # Uncertainty key still present
        self.assertIn("u", result_bd.uncertainties)
        # No unexpected NaNs for this simple case
        self.assertFalse(np.isnan(result_bd.signal).any())
        self.assertFalse(np.isnan(result_bd.uncertainties["u"]).any())


def test_reduce_dimensionality_rank_and_axes_reduce_as_expected():
    # 2D signal with matching axes metadata
    sig = np.arange(12.0).reshape(3, 4)
    bd = BaseData(signal=sig, units=ureg.dimensionless)

    # original rank and axes
    bd.rank_of_data = 2
    axis0 = BaseData(signal=np.arange(3.0), units=ureg.dimensionless)
    axis1 = BaseData(signal=np.arange(4.0), units=ureg.dimensionless)
    bd.axes = [axis0, axis1]

    # --- reduce over axis=1 -> shape (3,) ---
    out_axis1 = ReduceDimensionality._weighted_mean_with_uncertainty(
        bd=bd,
        axis=1,
        use_weights=False,
        nan_policy="omit",
        reduction="mean",
    )

    # shape reduced correctly
    assert out_axis1.signal.shape == (3,)
    # rank reduced: min(old_rank=2, new_ndim=1) -> 1
    assert out_axis1.rank_of_data == 1
    # axes: axis 1 removed, axis 0 preserved
    assert len(out_axis1.axes) == 1
    assert out_axis1.axes[0] is axis0

    # --- reduce over all axes -> scalar ---
    out_all = ReduceDimensionality._weighted_mean_with_uncertainty(
        bd=bd,
        axis=None,
        use_weights=False,
        nan_policy="omit",
        reduction="mean",
    )

    # scalar output
    assert out_all.signal.shape == ()
    # rank cannot exceed new_ndim (0)
    assert out_all.rank_of_data == 0
    # axes should be empty for scalar
    assert out_all.axes == []


def test_reduce_dimensionality_nan_uncertainty_omit_policy_is_explicit():
    signal = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    unc = np.array([[0.1, np.nan], [0.2, 0.3]], dtype=float)

    bd = BaseData(
        signal=signal,
        units=ureg.Unit("count"),
        uncertainties={"u": unc},
        weights=np.ones_like(signal),
    )

    out = ReduceDimensionality._weighted_mean_with_uncertainty(
        bd=bd,
        axis=0,
        use_weights=False,
        nan_policy="omit",
        reduction="mean",
    )

    np.testing.assert_allclose(out.signal, np.array([2.0, 3.0]))
    assert out.uncertainties["u"][0] == pytest.approx(np.sqrt(0.1**2 + 0.2**2) / 2.0)
    assert np.isnan(out.uncertainties["u"][1])


def test_scalar_weight_fast_path_matches_explicit_unit_weights():
    signal = np.arange(24.0).reshape(2, 3, 4)
    unc = {"u": np.sqrt(signal + 1.0)}

    scalar_weight_bd = BaseData(
        signal=signal,
        units=ureg.count,
        uncertainties=unc,
        weights=np.array(1.0),
    )
    explicit_weight_bd = BaseData(
        signal=signal,
        units=ureg.count,
        uncertainties=unc,
        weights=np.ones_like(signal),
    )

    fast = ReduceDimensionality._weighted_mean_with_uncertainty(
        bd=scalar_weight_bd,
        axis=(0, 1),
        use_weights=True,
        nan_policy="propagate",
        reduction="mean",
    )
    explicit = ReduceDimensionality._weighted_mean_with_uncertainty(
        bd=explicit_weight_bd,
        axis=(0, 1),
        use_weights=True,
        nan_policy="propagate",
        reduction="mean",
    )

    np.testing.assert_allclose(fast.signal, explicit.signal)
    np.testing.assert_allclose(fast.uncertainties["u"], explicit.uncertainties["u"])


def test_reduce_dimensionality_emits_info_and_debug_logs(caplog):
    """
    Ensure that ReduceDimensionality.calculate() emits at least one INFO and
    one DEBUG log record via the MessageHandler-backed logger.
    """
    signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    unc = 0.1 * np.ones_like(signal)
    weights = np.array([[1.0], [2.0]], dtype=float)

    processing_data = ProcessingData()
    bd = BaseData(
        signal=signal,
        units=ureg.Unit("count"),
        uncertainties={"u": unc},
        weights=weights,
    )
    bundle = DataBundle(signal=bd)
    processing_data["bundle"] = bundle

    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["bundle"],
        axes=0,
        use_weights=True,
        nan_policy="omit",
        reduction="mean",
    )
    step.processing_data = processing_data

    logger_name = "modacor.modules.base_modules.reduce_dimensionality"

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        step.calculate()

    # Sanity: output exists
    assert "bundle" in processing_data
    out_bd: BaseData = processing_data["bundle"]["signal"]
    assert isinstance(out_bd, BaseData)

    # Collect log records from expected logger
    records = [rec for rec in caplog.records if rec.name == logger_name]
    assert records, "Expected at least one log record from ReduceDimensionality logger."

    levels = {rec.levelno for rec in records}
    assert logging.INFO in levels, "Expected at least one INFO log from ReduceDimensionality."
    assert logging.DEBUG in levels, "Expected at least one DEBUG log from ReduceDimensionality."


def test_non_data_axes_reduce_all_leading_dimensions_and_preserve_data_axes():
    signal = np.arange(2 * 3 * 4 * 5.0).reshape(2, 3, 4, 5)
    axes = [BaseData(signal=np.arange(size), units=ureg.dimensionless) for size in signal.shape]
    processing_data = ProcessingData(
        sample=DataBundle(
            signal=BaseData(
                signal=signal,
                units=ureg.count,
                uncertainties={"u": np.ones_like(signal)},
                axes=axes,
                rank_of_data=2,
            )
        )
    )

    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes="non_data",
        use_weights=False,
        nan_policy="propagate",
    )
    step.execute(processing_data)

    result = processing_data["sample"]["signal"]
    np.testing.assert_allclose(result.signal, np.mean(signal, axis=(0, 1)))
    assert result.signal.shape == (4, 5)
    assert result.rank_of_data == 2
    assert result.axes == axes[-2:]


def test_non_data_axes_are_a_true_noop_at_data_rank():
    signal = BaseData(
        signal=np.arange(12.0).reshape(3, 4),
        units=ureg.count,
        uncertainties={"u": np.ones((3, 4))},
        weights=np.full((3, 4), 2.0),
        rank_of_data=2,
    )
    processing_data = ProcessingData(sample=DataBundle(signal=signal))

    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample"], axes="non_data")
    step.execute(processing_data)

    assert processing_data["sample"]["signal"] is signal


def test_non_data_axes_resolve_per_processing_key():
    first = np.arange(6.0).reshape(2, 3)
    second = np.arange(24.0).reshape(2, 3, 4)
    processing_data = ProcessingData(
        first=DataBundle(signal=BaseData(signal=first, units=ureg.count, rank_of_data=1)),
        second=DataBundle(signal=BaseData(signal=second, units=ureg.count, rank_of_data=2)),
    )

    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["first", "second"],
        axes="non_data",
        use_weights=False,
        nan_policy="propagate",
    )
    step.execute(processing_data)

    np.testing.assert_allclose(processing_data["first"]["signal"].signal, np.mean(first, axis=0))
    np.testing.assert_allclose(processing_data["second"]["signal"].signal, np.mean(second, axis=0))


def test_non_data_axes_with_rank_zero_reduce_to_scalar():
    processing_data = ProcessingData(
        sample=DataBundle(signal=BaseData(signal=np.arange(6.0).reshape(2, 3), units=ureg.count, rank_of_data=0))
    )
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes="non_data",
        use_weights=False,
        nan_policy="propagate",
    )
    step.execute(processing_data)

    result = processing_data["sample"]["signal"]
    assert result.signal.shape == ()
    assert result.rank_of_data == 0


def test_non_data_axes_reject_unknown_symbolic_mode():
    processing_data = ProcessingData(
        sample=DataBundle(signal=BaseData(signal=np.ones((2, 3)), units=ureg.count, rank_of_data=1))
    )
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample"], axes="automatic")

    with pytest.raises(ValueError, match="non_data"):
        step.execute(processing_data)


def _run_reduction_with_estimators(
    *,
    signal,
    estimators,
    reduction="mean",
    weights=1.0,
    use_weights=False,
    nan_policy="propagate",
    uncertainties=None,
    collision_policy="error",
    axes=0,
):
    original = BaseData(
        signal=np.asarray(signal, dtype=float),
        units=ureg.count,
        weights=np.asarray(weights, dtype=float),
        uncertainties={} if uncertainties is None else uncertainties,
    )
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=axes,
        reduction=reduction,
        use_weights=use_weights,
        nan_policy=nan_policy,
        uncertainty_estimation={
            "collision_policy": collision_policy,
            "estimators": estimators,
        },
    )
    step.execute(processing_data)
    return processing_data["sample"]["signal"]


def test_uncertainty_estimators_add_unweighted_std_and_sem_under_selected_keys():
    signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    propagated = np.full_like(signal, 0.1)

    result = _run_reduction_with_estimators(
        signal=signal,
        uncertainties={"detector": propagated},
        estimators={
            "frame_STD": {"method": "standard_deviation", "ddof": 1},
            "frame_SEM": {"method": "standard_error_mean", "ddof": 1},
        },
    )

    expected_std = np.std(signal, axis=0, ddof=1)
    np.testing.assert_allclose(result.uncertainties["frame_STD"], expected_std)
    np.testing.assert_allclose(result.uncertainties["frame_SEM"], expected_std / np.sqrt(2.0))
    np.testing.assert_allclose(
        result.uncertainties["detector"],
        np.sqrt(np.sum(propagated**2, axis=0)) / 2.0,
    )


def test_uncertainty_estimators_use_weighted_effective_sample_size():
    signal = np.array([[1.0, 2.0], [4.0, 5.0]])
    weights = np.array([[1.0], [2.0]])

    result = _run_reduction_with_estimators(
        signal=signal,
        weights=weights,
        use_weights=True,
        estimators={
            "weighted_STD": {"method": "standard_deviation", "ddof": 1},
            "weighted_SEM": {"method": "standard_error_mean", "ddof": 1},
        },
    )

    np.testing.assert_allclose(result.uncertainties["weighted_STD"], np.sqrt(4.5))
    np.testing.assert_allclose(result.uncertainties["weighted_SEM"], np.sqrt(2.5))


@pytest.mark.parametrize(
    ("use_weights", "weights", "expected"),
    [
        (False, np.array(1.0), 3.0),
        (True, np.array([[1.0], [2.0]]), np.sqrt(22.5)),
    ],
)
def test_standard_error_sum_estimates_uncertainty_of_sum(use_weights, weights, expected):
    signal = np.array([[1.0, 2.0], [4.0, 5.0]])

    result = _run_reduction_with_estimators(
        signal=signal,
        reduction="sum",
        weights=weights,
        use_weights=use_weights,
        estimators={"sum_repeatability": {"method": "standard_error_sum", "ddof": 1}},
    )

    np.testing.assert_allclose(result.uncertainties["sum_repeatability"], expected)


def test_standard_deviation_is_available_as_input_scatter_for_sum():
    signal = np.array([[1.0, 2.0], [4.0, 5.0]])
    result = _run_reduction_with_estimators(
        signal=signal,
        reduction="sum",
        estimators={"input_scatter": {"method": "standard_deviation", "ddof": 1}},
    )

    np.testing.assert_allclose(result.uncertainties["input_scatter"], np.std(signal, axis=0, ddof=1))


def test_uncertainty_estimators_follow_nan_omit_and_insufficient_count_rules():
    signal = np.array([[1.0, np.nan, np.nan], [4.0, 5.0, np.nan]])
    result = _run_reduction_with_estimators(
        signal=signal,
        nan_policy="omit",
        estimators={
            "STD": {"method": "standard_deviation", "ddof": 1},
            "SEM": {"method": "standard_error_mean", "ddof": 1},
        },
    )

    np.testing.assert_allclose(
        result.uncertainties["STD"],
        np.array([np.std([1.0, 4.0], ddof=1), np.nan, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.uncertainties["SEM"],
        np.array([np.std([1.0, 4.0], ddof=1) / np.sqrt(2.0), np.nan, np.nan]),
        equal_nan=True,
    )


def test_uncertainty_estimators_follow_nan_propagate_policy():
    signal = np.array([[1.0, np.nan], [4.0, 5.0]])
    result = _run_reduction_with_estimators(
        signal=signal,
        nan_policy="propagate",
        estimators={"STD": {"method": "standard_deviation", "ddof": 1}},
    )

    np.testing.assert_allclose(
        result.uncertainties["STD"],
        np.array([np.std([1.0, 4.0], ddof=1), np.nan]),
        equal_nan=True,
    )


def test_uncertainty_estimators_reduce_multiple_axes():
    signal = np.arange(24.0).reshape(2, 3, 4)
    result = _run_reduction_with_estimators(
        signal=signal,
        axes=(0, 1),
        estimators={"STD": {"method": "standard_deviation", "ddof": 1}},
    )

    np.testing.assert_allclose(result.uncertainties["STD"], np.std(signal, axis=(0, 1), ddof=1))


@pytest.mark.parametrize("disabled_configuration", [None, {}, {"estimators": {}}])
def test_empty_uncertainty_estimation_configuration_preserves_existing_behavior(disabled_configuration):
    signal = np.arange(6.0).reshape(2, 3)
    uncertainty = np.full_like(signal, 0.2)
    processing_data = ProcessingData(
        sample=DataBundle(signal=BaseData(signal=signal, units=ureg.count, uncertainties={"u": uncertainty}))
    )
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=0,
        use_weights=False,
        uncertainty_estimation=disabled_configuration,
    )
    step.execute(processing_data)

    assert set(processing_data["sample"]["signal"].uncertainties) == {"u"}


def test_estimator_collision_error_leaves_existing_basedata_unmodified():
    original = BaseData(
        signal=np.array([[1.0, 2.0], [4.0, 5.0]]),
        units=ureg.count,
        uncertainties={"repeatability": np.full((2, 2), 0.1)},
    )
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=0,
        use_weights=False,
        uncertainty_estimation={
            "collision_policy": "error",
            "estimators": {"repeatability": {"method": "standard_deviation"}},
        },
    )

    with pytest.raises(ValueError, match="already exists after propagation"):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


@pytest.mark.parametrize("collision_policy", ["overwrite_existing", "keep_existing", "propagate"])
def test_estimator_collision_policies(collision_policy):
    signal = np.array([[1.0, 2.0], [4.0, 5.0]])
    input_uncertainty = np.full_like(signal, 0.2)
    result = _run_reduction_with_estimators(
        signal=signal,
        uncertainties={"repeatability": input_uncertainty},
        collision_policy=collision_policy,
        estimators={"repeatability": {"method": "standard_deviation", "ddof": 1}},
    )

    propagated = np.sqrt(2.0 * 0.2**2) / 2.0
    estimated = np.std(signal, axis=0, ddof=1)
    if collision_policy == "overwrite_existing":
        expected = estimated
    elif collision_policy == "keep_existing":
        expected = propagated
    else:
        expected = np.hypot(propagated, estimated)
    np.testing.assert_allclose(result.uncertainties["repeatability"], expected)


def test_estimator_collision_policy_can_be_overridden_per_estimator():
    signal = np.array([[1.0, 2.0], [4.0, 5.0]])
    result = _run_reduction_with_estimators(
        signal=signal,
        uncertainties={"repeatability": np.full_like(signal, 0.2)},
        collision_policy="error",
        estimators={
            "repeatability": {
                "method": "standard_deviation",
                "collision_policy": "overwrite_existing",
            }
        },
    )

    np.testing.assert_allclose(result.uncertainties["repeatability"], np.std(signal, axis=0, ddof=1))


@pytest.mark.parametrize(
    ("reduction", "method"),
    [("sum", "standard_error_mean"), ("mean", "standard_error_sum")],
)
def test_estimator_rejects_method_for_wrong_reduction_before_mutation(reduction, method):
    original = BaseData(signal=np.arange(6.0).reshape(2, 3), units=ureg.count)
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=0,
        reduction=reduction,
        uncertainty_estimation={"estimators": {"estimate": {"method": method}}},
    )

    with pytest.raises(ValueError, match="is not valid"):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


def test_estimator_rejects_negative_weights_without_replacing_basedata():
    original = BaseData(
        signal=np.array([[1.0, 2.0], [4.0, 5.0]]),
        units=ureg.count,
        weights=np.array([[1.0], [-1.0]]),
    )
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=0,
        use_weights=True,
        uncertainty_estimation={"estimators": {"STD": {"method": "standard_deviation"}}},
    )

    with pytest.raises(ValueError, match="non-negative effective weights"):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


def test_estimator_returns_nan_for_zero_effective_weight():
    result = _run_reduction_with_estimators(
        signal=np.array([[1.0, 2.0], [4.0, 5.0]]),
        weights=np.zeros((2, 1)),
        use_weights=True,
        estimators={"STD": {"method": "standard_deviation", "ddof": 1}},
    )

    assert np.isnan(result.uncertainties["STD"]).all()


@pytest.mark.parametrize(
    ("uncertainty_estimation", "message"),
    [
        ({"collision_policy": "replace", "estimators": {}}, "collision_policy"),
        ({"estimators": {"u": {"method": "variance"}}}, "Unknown uncertainty estimator"),
        ({"estimators": {"u": {"method": "standard_deviation", "ddof": -1}}}, "ddof"),
        ({"estimators": {"u": {"method": "standard_deviation", "extra": True}}}, "unknown key"),
    ],
)
def test_invalid_estimator_configuration_fails_before_mutation(uncertainty_estimation, message):
    original = BaseData(signal=np.arange(6.0).reshape(2, 3), units=ureg.count)
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=0,
        uncertainty_estimation=uncertainty_estimation,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


def test_non_data_noop_does_not_add_configured_estimators():
    original = BaseData(signal=np.arange(6.0).reshape(2, 3), units=ureg.count, rank_of_data=2)
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes="non_data",
        uncertainty_estimation={"estimators": {"STD": {"method": "standard_deviation"}}},
    )
    step.execute(processing_data)

    assert processing_data["sample"]["signal"] is original
    assert "STD" not in original.uncertainties


def _run_direct_mask_reduction(
    *,
    signal,
    mask,
    axes=0,
    reduction="mean",
    use_weights=False,
    weights=1.0,
    nan_policy="propagate",
    mask_bits=None,
    uncertainties=None,
    uncertainty_estimation=None,
):
    signal_bd = BaseData(
        signal=np.asarray(signal),
        units=ureg.count,
        weights=np.asarray(weights, dtype=float),
        uncertainties={} if uncertainties is None else uncertainties,
    )
    bundle = DataBundle(
        signal=signal_bd,
        mask=BaseData(signal=np.asarray(mask), units=ureg.dimensionless),
    )
    processing_data = ProcessingData(sample=bundle)
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=axes,
        reduction=reduction,
        use_weights=use_weights,
        nan_policy=nan_policy,
        mask_key="mask",
        mask_bits=mask_bits,
        uncertainty_estimation=uncertainty_estimation,
    )
    step.execute(processing_data)
    return signal_bd, processing_data["sample"]["signal"]


def test_direct_mask_excludes_values_without_mutating_input_signal():
    signal = np.array([[1.0, 10.0], [3.0, 20.0]])
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint32)
    uncertainty = np.full_like(signal, 0.1)

    original, result = _run_direct_mask_reduction(
        signal=signal,
        mask=mask,
        uncertainties={"detector": uncertainty},
    )

    np.testing.assert_array_equal(original.signal, signal)
    np.testing.assert_allclose(result.signal, np.array([2.0, 20.0]))
    np.testing.assert_allclose(
        result.uncertainties["detector"],
        np.array([np.sqrt(2.0) * 0.1 / 2.0, 0.1]),
    )


def test_direct_mask_is_omitted_even_when_nan_policy_is_propagate():
    signal = np.array([[1.0, np.nan], [3.0, 20.0]])
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint32)

    _, result = _run_direct_mask_reduction(signal=signal, mask=mask, nan_policy="propagate")

    np.testing.assert_allclose(result.signal, np.array([2.0, 20.0]))


def test_direct_mask_is_shared_by_scatter_estimators():
    signal = np.array([[1.0, 10.0], [3.0, 20.0]])
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint32)

    _, result = _run_direct_mask_reduction(
        signal=signal,
        mask=mask,
        uncertainty_estimation={
            "estimators": {
                "frame_STD": {"method": "standard_deviation", "ddof": 1},
                "frame_SEM": {"method": "standard_error_mean", "ddof": 1},
            }
        },
    )

    np.testing.assert_allclose(
        result.uncertainties["frame_STD"],
        np.array([np.std([1.0, 3.0], ddof=1), np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.uncertainties["frame_SEM"],
        np.array([np.std([1.0, 3.0], ddof=1) / np.sqrt(2.0), np.nan]),
        equal_nan=True,
    )


def test_direct_mask_excludes_values_from_weighted_sum_and_sum_uncertainty():
    signal = np.array([[1.0, 10.0], [3.0, 20.0]])
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint32)
    weights = np.array([[1.0], [2.0]])
    uncertainty = np.full_like(signal, 0.1)

    _, result = _run_direct_mask_reduction(
        signal=signal,
        mask=mask,
        reduction="sum",
        use_weights=True,
        weights=weights,
        uncertainties={"detector": uncertainty},
        uncertainty_estimation={"estimators": {"sum_repeatability": {"method": "standard_error_sum", "ddof": 1}}},
    )

    np.testing.assert_allclose(result.signal, np.array([7.0, 40.0]))
    np.testing.assert_allclose(result.uncertainties["detector"], np.array([np.sqrt(5.0) * 0.1, 0.2]))
    assert result.uncertainties["sum_repeatability"][0] == pytest.approx(np.sqrt(10.0))
    assert np.isnan(result.uncertainties["sum_repeatability"][1])


def test_direct_mask_can_select_reason_bits():
    signal = np.array([[1.0, 10.0], [3.0, 20.0]])
    mask = np.array([[1, 2], [0, 0]], dtype=np.uint32)

    _, selected = _run_direct_mask_reduction(signal=signal, mask=mask, mask_bits=1)
    _, all_bits = _run_direct_mask_reduction(signal=signal, mask=mask)

    np.testing.assert_allclose(selected.signal, np.array([3.0, 15.0]))
    np.testing.assert_allclose(all_bits.signal, np.array([3.0, 20.0]))


def test_direct_mask_broadcasts_over_leading_signal_dimensions():
    signal = np.arange(8.0).reshape(2, 2, 2)
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint32)

    _, result = _run_direct_mask_reduction(signal=signal, mask=mask, axes=0)

    expected = np.mean(signal, axis=0)
    expected[0, 1] = np.nan
    np.testing.assert_allclose(result.signal, expected, equal_nan=True)


@pytest.mark.parametrize(("reduction", "expected_signal"), [("mean", np.nan), ("sum", 0.0)])
def test_direct_mask_all_excluded_follows_empty_reduction_semantics(reduction, expected_signal):
    signal = np.array([[1.0], [3.0]])
    mask = np.ones_like(signal, dtype=np.uint32)

    _, result = _run_direct_mask_reduction(signal=signal, mask=mask, reduction=reduction)

    if reduction == "mean":
        assert np.isnan(result.signal[0])
    else:
        assert result.signal[0] == expected_signal
        assert result.weights[0] == 0.0


@pytest.mark.parametrize(
    ("mask", "mask_bits", "message"),
    [
        (np.zeros((2, 2), dtype=float), None, "integer dtype"),
        (np.zeros((3, 2), dtype=np.uint32), None, "cannot broadcast"),
        (np.zeros((2, 2), dtype=np.uint32), 0, "between 1"),
        (np.zeros((2, 2), dtype=np.uint32), [], "must not be empty"),
    ],
)
def test_direct_mask_rejects_invalid_mask_inputs(mask, mask_bits, message):
    original = BaseData(signal=np.arange(4.0).reshape(2, 2), units=ureg.count)
    processing_data = ProcessingData(
        sample=DataBundle(signal=original, mask=BaseData(signal=mask, units=ureg.dimensionless))
    )
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        axes=0,
        mask_key="mask",
        mask_bits=mask_bits,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


def test_mask_bits_requires_mask_key_during_preparation():
    original = BaseData(signal=np.arange(4.0).reshape(2, 2), units=ureg.count)
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample"], axes=0, mask_bits=1)

    with pytest.raises(ValueError, match="requires mask_key"):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


def test_direct_mask_requires_configured_key_in_each_processed_bundle():
    original = BaseData(signal=np.arange(4.0).reshape(2, 2), units=ureg.count)
    processing_data = ProcessingData(sample=DataBundle(signal=original))
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample"], axes=0, mask_key="missing_mask")

    with pytest.raises(KeyError, match="missing_mask"):
        step.execute(processing_data)
    assert processing_data["sample"]["signal"] is original


def test_direct_mask_dependency_contract_reads_mask_and_signal_but_only_writes_signal():
    step = ReduceDimensionality(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample"], mask_key="quality_mask")

    contract = step.dependency_contract()

    assert contract.source_refs == frozenset()
    assert contract.processing_reads == frozenset({"sample.signal", "sample.quality_mask"})
    assert contract.processing_writes == frozenset({"sample.signal"})
