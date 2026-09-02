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
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources

# module under test
from modacor.modules.base_modules.apply_mask import ApplyMask

TEST_IO_SOURCES = IoSources()


class TestApplyMaskProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/apply_mask.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()

        tgt = np.ones((2, 3), dtype=float)
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)

        db = DataBundle(
            mask=BaseData(signal=mask, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

    def _make_step(self, *, mask="mask", sources=None, masked_value=0) -> ApplyMask:
        step = ApplyMask(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_dict(
            {
                "with_processing_keys": ["sample"],
                "mask_key": mask,
                "basedata_to_mask": list(sources or ["signal"]),
                "masked_value": masked_value,
            }
        )
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

    def test_apply_mask_default_masked_value_is_nan(self):
        self.setUp()

        step = ApplyMask(io_sources=TEST_IO_SOURCES)
        step.modify_config_by_dict(
            {
                "with_processing_keys": ["sample"],
                "mask_key": "mask",
                "basedata_to_mask": ["signal"],
            }
        )
        step.processing_data = self.test_processing_data

        step.calculate()

        out = step.processing_data["sample"]["signal"].signal
        expected = np.array([[np.nan, 1.0, 1.0], [1.0, np.nan, 1.0]])
        np.testing.assert_allclose(out, expected)

    def test_apply_mask_treats_any_nonzero_bit_as_masked(self):
        self.test_processing_data = ProcessingData()

        tgt = np.ones((2, 3), dtype=np.uint32)
        mask = np.array([[4, 0, 0], [0, 32, 0]], dtype=np.uint32)

        db = DataBundle(
            mask=BaseData(signal=mask, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = self._make_step()
        step.calculate()

        expected = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint32)
        out = self.test_processing_data["sample"]["signal"].signal
        self.assertEqual(out.dtype, np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_apply_mask_promotes_integer_signal_for_nan(self):
        self.test_processing_data = ProcessingData()

        tgt = np.ones((2, 3), dtype=np.uint32)
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)

        self.test_processing_data["sample"] = DataBundle(
            mask=BaseData(signal=mask, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
        )

        step = self._make_step(masked_value="nan")
        step.calculate()

        out = self.test_processing_data["sample"]["signal"].signal
        assert np.issubdtype(out.dtype, np.floating)
        expected = np.array([[np.nan, 1.0, 1.0], [1.0, np.nan, 1.0]])
        np.testing.assert_allclose(out, expected)

    def test_apply_mask_accepts_negative_masked_value(self):
        self.test_processing_data = ProcessingData()

        tgt = np.ones((2, 3), dtype=np.uint32)
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)

        self.test_processing_data["sample"] = DataBundle(
            mask=BaseData(signal=mask, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
        )

        step = self._make_step(masked_value=-1)
        step.calculate()

        out = self.test_processing_data["sample"]["signal"].signal
        expected = np.array([[-1, 1, 1], [1, -1, 1]])
        np.testing.assert_array_equal(out, expected)

    def test_apply_mask_broadcasts_detector_mask_over_frame_axis(self):
        self.test_processing_data = ProcessingData()

        tgt = np.ones((2, 2, 3), dtype=np.uint32)
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)

        db = DataBundle(
            mask=BaseData(signal=mask, units=ureg.dimensionless, uncertainties={}),
            signal=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}, rank_of_data=2),
        )
        self.test_processing_data["sample"] = db

        step = self._make_step()
        step.calculate()

        expected = np.array(
            [
                [[0, 1, 1], [1, 0, 1]],
                [[0, 1, 1], [1, 0, 1]],
            ],
            dtype=np.uint32,
        )
        np.testing.assert_array_equal(self.test_processing_data["sample"]["signal"].signal, expected)

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

    def test_apply_mask_dependency_contract_is_exact(self):
        self.test_processing_data["sample"]["variance"] = BaseData(
            signal=np.ones((2, 3), dtype=np.uint32),
            units=ureg.dimensionless,
            uncertainties={},
        )
        step = self._make_step(sources=["signal", "variance"])

        contract = step.dependency_contract()

        self.assertIsInstance(contract, ProcessStepDependencies)
        self.assertEqual(contract.source_refs, frozenset())
        self.assertEqual(
            contract.processing_reads,
            frozenset({"sample.mask", "sample.signal", "sample.variance"}),
        )
        self.assertEqual(contract.processing_writes, frozenset({"sample.signal", "sample.variance"}))
