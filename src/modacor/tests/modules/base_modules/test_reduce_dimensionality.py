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

import unittest

import numpy as np

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
