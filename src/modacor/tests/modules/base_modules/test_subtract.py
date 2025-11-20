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

import modacor.modules.base_modules.subtract as subtract_module
from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.subtract import Subtract

TEST_IO_SOURCES = IoSources()


class TestSubtractProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/subtract.py"""

    def setUp(self):
        # 2x3 BaseData
        signal = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float)
        data_unc = 0.5 * np.ones_like(signal)

        self.test_processing_data = ProcessingData()
        self.base = BaseData(
            signal=signal,
            units=ureg.Unit("count"),
            uncertainties={"u": data_unc},
        )
        self.test_data_bundle = DataBundle(signal=self.base)
        self.test_processing_data["bundle"] = self.test_data_bundle

        # Subtrahend: scalar BaseData with same units so subtraction is valid
        self.subtrahend = BaseData(
            signal=5.0,
            units=ureg.Unit("count"),
            uncertainties={"propagate_to_all": np.array(0.2, dtype=float)},
        )

        # Ground truth using BaseData.__sub__
        self.expected_result = self.base - self.subtrahend

        # Monkeypatch basedata_from_sources
        self._orig_basedata_from_sources = subtract_module.basedata_from_sources
        subtract_module.basedata_from_sources = self._fake_basedata_from_sources

    def tearDown(self):
        subtract_module.basedata_from_sources = self._orig_basedata_from_sources

    def _fake_basedata_from_sources(self, io_sources, signal_source, units_source=None, uncertainty_sources=None):
        """Fake basedata_from_sources that always returns self.subtrahend."""
        return self.subtrahend

    def test_subtract_calculation(self):
        """
        Subtract.calculate() should subtract the subtrahend from the DataBundle's BaseData,
        using BaseData.__sub__ semantics.
        """
        step = Subtract(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            subtrahend_source="dummy",  # ignored
        )
        step.processing_data = self.test_processing_data

        step.calculate()

        result_bd: BaseData = self.test_processing_data["bundle"]["signal"]

        np.testing.assert_allclose(result_bd.signal, self.expected_result.signal)
        for key in self.expected_result.uncertainties:
            np.testing.assert_allclose(
                result_bd.uncertainties[key],
                self.expected_result.uncertainties[key],
            )
        self.assertEqual(result_bd.units, self.expected_result.units)

    def test_subtract_execution_via_call(self):
        """
        Subtract.__call__ should run the step and update ProcessingData in-place.
        """
        step = Subtract(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_kwargs(
            with_processing_keys=["bundle"],
            subtrahend_source="dummy",
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
