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

import modacor.modules.base_modules.multiply as multiply_module
from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.multiply import Multiply

TEST_IO_SOURCES = IoSources()


class TestMultiplyProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/multiply.py"""

    def setUp(self):
        # Simple 2x3 BaseData
        signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
        data_unc = 0.1 * np.ones_like(signal)

        self.test_processing_data = ProcessingData()
        self.base = BaseData(
            signal=signal,
            units=ureg.Unit("count"),
            uncertainties={"u": data_unc},
        )
        self.test_data_bundle = DataBundle(signal=self.base)
        self.test_processing_data["bundle"] = self.test_data_bundle

        # Factor: scalar BaseData with its own unit + uncertainty
        self.factor = BaseData(
            signal=2.0,
            units=ureg.Unit("second"),
            uncertainties={"propagate_to_all": np.array(0.2, dtype=float)},
        )

        # Ground truth using BaseData.__mul__
        self.expected_result = self.base * self.factor

        # Monkeypatch basedata_from_sources
        self._orig_basedata_from_sources = multiply_module.basedata_from_sources
        multiply_module.basedata_from_sources = self._fake_basedata_from_sources

    def tearDown(self):
        # Restore original helper
        multiply_module.basedata_from_sources = self._orig_basedata_from_sources

    def _fake_basedata_from_sources(self, io_sources, signal_source, units_source=None, uncertainty_sources=None):
        """Fake basedata_from_sources that always returns self.factor."""
        return self.factor

    def test_multiply_calculation(self):
        """
        Multiply.calculate() should multiply the DataBundle's BaseData by the factor
        returned from basedata_from_sources, using BaseData.__mul__ semantics.
        """
        step = Multiply(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            multiplier_source="dummy",  # ignored by fake basedata_from_sources
        )
        step.processing_data = self.test_processing_data

        step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Compare with ground truth
        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)
        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )
        self.assertEqual(result_bd.units, self.expected_result.units)

    def test_multiply_execution_via_call(self):
        """
        Multiply.__call__ should run the step and update ProcessingData in-place.
        """
        step = Multiply(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            multiplier_source="dummy",
        )

        step(self.test_processing_data)

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)
        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )
        self.assertEqual(result_bd.units, self.expected_result.units)
