# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Anja F. Hörmann"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "04/08/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import unittest

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources

# module under test
from modacor.modules.base_modules.threshold_mask import ThresholdMask

TEST_IO_SOURCES = IoSources()


class TestApplyMaskProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/apply_mask.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()

        signal = np.array([[0, 1, 4e9], [2, 3, 7e5]], dtype=np.uint32)

        db = DataBundle(
            signal=BaseData(signal=signal, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

    def _make_step(self, *, mask="threshold_mask", sources=None) -> BitwiseOrMasks:
        step = ThresholdMask(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": mask,
            "threshold": 1e9,
        }
        step.processing_data = self.test_processing_data
        return step

    # ------------------------------------------------------------------ #
    # Actual tests
    # ------------------------------------------------------------------ #

    def test_threshold_mask_expected_mask(self):
        "Test that the right pixels are masked on a small array"

        self.setUp()

        step = self._make_step()

        step.processing_data = self.test_processing_data
        step.calculate()
        out = step.processing_data["sample"]["threshold_mask"].signal

        expected = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_threshold_mask_ndim_signal(self):
        """
        If the signal isn't of rank 2 the step should
        average the frames before determining the mask.
        """
        self.test_processing_data = ProcessingData()

        signal = np.array([[[0, 1, 4e9], [2, 3, 7e5]],
                           [[0, 1, 4e9], [2, 3, 7e5]]], dtype=np.uint32)

        db = DataBundle(
            signal=BaseData(signal=signal, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = self._make_step()
        step.processing_data = self.test_processing_data

        step.calculate()

        out = self.test_processing_data["sample"]["threshold_mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        self.assertEqual(out.ndim, 2)
