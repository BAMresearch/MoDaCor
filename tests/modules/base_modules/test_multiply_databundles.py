# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import unittest

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.multiply_databundles import MultiplyDatabundles

TEST_IO_SOURCES = IoSources()


class TestMultiplyDatabundles(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/multiply_databundles.py"""

    def setUp(self):
        # Two simple 2x3 BaseData objects to multiply
        signal1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float)
        signal2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

        # Absolute 1σ uncertainties
        unc1 = 0.5 * np.ones_like(signal1)
        unc2 = 0.2 * np.ones_like(signal2)

        self.bd1 = BaseData(
            signal=signal1,
            units=ureg.Unit("count"),
            uncertainties={"u": unc1},
        )
        self.bd2 = BaseData(
            signal=signal2,
            units=ureg.Unit("count"),
            uncertainties={"u": unc2},
        )

        self.bundle1 = DataBundle(signal=self.bd1)
        self.bundle2 = DataBundle(signal=self.bd2)

        self.processing_data = ProcessingData()
        self.processing_data["bundle1"] = self.bundle1
        self.processing_data["bundle2"] = self.bundle2

        # Ground truth using BaseData.__mul__
        self.expected_result = self.bd1 * self.bd2

    def tearDown(self):
        pass

    # ------------------------------------------------------------------ #
    # Tests
    # ------------------------------------------------------------------ #

    def test_multiply_databundles_calculation(self):
        """
        MultiplyDatabundles.calculate() should multiply the second DataBundle's
        signal with the first, using BaseData.__mul__ semantics.
        """
        step = MultiplyDatabundles(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle1", "bundle2"],
        )
        step.processing_data = self.processing_data

        output = step.calculate()

        # Only the multiplicand key should be in output
        self.assertEqual(list(output.keys()), ["bundle1"])

        result_bd: BaseData = self.processing_data["bundle1"]["signal"]

        # Signal and uncertainties should match the precomputed result
        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)
        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )

        # Units should be preserved (count * count → count)
        self.assertEqual(result_bd.units, self.expected_result.units)

    def test_multiply_databundles_execution_via_call(self):
        """
        MultiplyDatabundles.__call__ should run the step and update ProcessingData in-place.
        """
        # Re-initialize processing_data to original state
        processing_data = ProcessingData()
        processing_data["bundle1"] = DataBundle(signal=self.bd1)
        processing_data["bundle2"] = DataBundle(signal=self.bd2)

        step = MultiplyDatabundles(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle1", "bundle2"],
        )

        step(processing_data)

        result_bd: BaseData = processing_data["bundle1"]["signal"]

        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)
        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )
        self.assertEqual(result_bd.units, self.expected_result.units)

    def test_requires_exactly_two_keys(self):
        """
        MultiplyDatabundles should assert if 'with_processing_keys' does not
        contain exactly two keys.
        """
        step = MultiplyDatabundles(io_sources=TEST_IO_SOURCES)
        # Only one key → should trigger the assertion in calculate()
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle1"],
        )
        step.processing_data = self.processing_data

        with self.assertRaises(AssertionError):
            step.calculate()
