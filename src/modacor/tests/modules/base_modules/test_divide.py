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

import modacor.modules.base_modules.divide as divide_module
from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources

# The processing step under test
from modacor.modules.base_modules.divide import Divide

TEST_IO_SOURCES = IoSources()


class TestDivideProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/divide.py"""

    def setUp(self):
        # Small test signal: shape (2, 3) to avoid trivial broadcasting bugs
        signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

        # absolute 1σ uncertainties on the data
        data_unc = 0.1 * np.ones_like(signal)

        self.test_processing_data = ProcessingData()
        self.test_basedata = BaseData(
            signal=signal,
            units=ureg.Unit("count"),
            uncertainties={"u": data_unc},
        )
        self.test_data_bundle = DataBundle(signal=self.test_basedata)
        self.test_processing_data["bundle"] = self.test_data_bundle

        # Divisor: scalar BaseData with uncertainty
        # (propagate_to_all for reuse with any key)
        self.divisor = BaseData(
            signal=2.0,
            units=ureg.Unit("second"),
            uncertainties={"propagate_to_all": np.array(0.2, dtype=float)},
        )

        # Ground truth result using the already-tested BaseData.__truediv__
        self.expected_result = self.test_basedata / self.divisor

        # Monkeypatch basedata_from_sources to return our known divisor
        self._orig_basedata_from_sources = divide_module.basedata_from_sources
        divide_module.basedata_from_sources = self._fake_basedata_from_sources

    def tearDown(self):
        # Restore original helper to avoid leaking the patch to other tests
        divide_module.basedata_from_sources = self._orig_basedata_from_sources

    # ------------------------------------------------------------------ #
    # Helper used for monkeypatch
    # ------------------------------------------------------------------ #

    def _fake_basedata_from_sources(self, io_sources, signal_source, units_source=None, uncertainty_sources=None):
        """
        Fake basedata_from_sources that ignores its inputs and returns
        the pre-constructed self.divisor.
        """
        return self.divisor

    # ------------------------------------------------------------------ #
    # Actual tests
    # ------------------------------------------------------------------ #

    def test_divide_calculation(self):
        """
        Divide.calculate() should divide the DataBundle's BaseData by the divisor
        returned from basedata_from_sources, using BaseData.__truediv__ semantics.
        """
        divide_step = Divide(io_sources=TEST_IO_SOURCES)
        divide_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            divisor_source="dummy",  # ignored by our fake basedata_from_sources
        )
        divide_step.processing_data = self.test_processing_data

        divide_step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Signal should match the pre-computed expected_result
        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)

        # Uncertainties per key should also match
        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )

        # Units should be derived via pint from count / second
        self.assertEqual(result_bd.units, self.expected_result.units)
        # double-check as I'm having issues..
        self.assertEqual(result_bd.units, ureg.Unit("count / second"))

    def test_divide_execution_via_call(self):
        """
        Divide.__call__ should behave like in other ProcessSteps:
        calling the object with ProcessingData runs the step and updates in-place.
        """
        divide_step = Divide(io_sources=TEST_IO_SOURCES)
        divide_step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            divisor_source="dummy",
        )

        # Execute via __call__
        divide_step(self.test_processing_data)

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        # Same checks as in test_divide_calculation
        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)

        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )

        self.assertEqual(result_bd.units, self.expected_result.units)
