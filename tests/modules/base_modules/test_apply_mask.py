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
from modacor.modules.base_modules.apply_mask import ApplyMask

TEST_IO_SOURCES = IoSources()

class TestApplyMaskProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/apply_mask.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()

        tgt = np.ones((2, 3), dtype=np.uint32)
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)

        db = DataBundle(
            mask=BaseData(signal=mask, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

    def _make_step(self, *, mask="mask", sources=None) -> BitwiseOrMasks:
        step = ApplyMask(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "mask_key": mask,
            "basedata_to_mask": ["signal"],
        }
        step.processing_data = self.test_processing_data
        return step

    # ------------------------------------------------------------------ #
    # Actual tests
    # ------------------------------------------------------------------ #

    def test_apply_mask_expected_signal(self):
        "Test that the right pixels are masked on a small array"

        self.setUp()

        step = self._make_step()

        step.processing_data = self.test_processing_data
        step.calculate()
        out = step.processing_data["sample"]["signal"].signal

        expected = np.array([[0, 1, 1], [1, 0, 1]])
        np.testing.assert_array_equal(out, expected)

    def test_target_non_uint32_is_upcast_to_uint32_once(self):
        """
        If the mask isn't uint32 (e.g. int64), the step should convert it to uint32
        (one-time allocation) and then OR into that.
        """
        self.test_processing_data = ProcessingData()

        mask_i64 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
        tgt = np.ones((2, 3), dtype=np.uint32)

        
        db = DataBundle(
            mask=BaseData(signal=mask_i64, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = self._make_step()
        step.processing_data = self.test_processing_data

        before_id = id(self.test_processing_data["sample"]["mask"].signal)
        step.calculate()

        out = self.test_processing_data["sample"]["mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        self.assertNotEqual(id(out), before_id)  # replacement happened due to upcast

        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)
