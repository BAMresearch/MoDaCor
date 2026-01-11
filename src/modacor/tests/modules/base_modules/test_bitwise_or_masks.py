# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
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
from modacor.modules.base_modules.bitwise_or_masks import BitwiseOrMasks

TEST_IO_SOURCES = IoSources()


class TestBitwiseOrMasksProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/bitwise_or_masks.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()

        tgt = np.zeros((2, 3), dtype=np.uint32)
        bs = np.array([[4, 0, 0], [0, 32, 0]], dtype=np.uint32)

        db = DataBundle(
            mask=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=bs, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

    def _make_step(self, *, target="mask", sources=None) -> BitwiseOrMasks:
        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": target,
            "source_mask_keys": list(sources or []),
        }
        step.processing_data = self.test_processing_data
        return step

    # ------------------------------------------------------------------ #
    # Actual tests
    # ------------------------------------------------------------------ #

    def test_bitwise_or_masks_calculate_inplace_when_target_is_uint32(self):
        """
        If the target is already uint32, the OR should be performed truly in-place
        (target array object remains the same).
        """
        step = self._make_step(sources=["bs_mask"])

        before_id = id(self.test_processing_data["sample"]["mask"].signal)
        step.calculate()

        out = self.test_processing_data["sample"]["mask"].signal
        self.assertEqual(id(out), before_id)  # in-place, no replacement

        expected = np.array([[4, 0, 0], [0, 32, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_bitwise_or_masks_execution_via_call(self):
        """
        __call__ should run the step and update processing_data in-place.
        """
        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }

        step(self.test_processing_data)

        out = self.test_processing_data["sample"]["mask"].signal
        expected = np.array([[4, 0, 0], [0, 32, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_uint8_source_is_converted_and_or_applied(self):
        """
        A uint8 mask source should be accepted and widened to uint32 only if needed.
        """
        self.test_processing_data = ProcessingData()

        tgt = np.zeros((2, 3), dtype=np.uint32)
        # uint8 source containing bit 2 and bit 5
        src_u8 = np.array([[4, 0, 0], [0, 32, 0]], dtype=np.uint8)

        db = DataBundle(
            mask=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=src_u8, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }
        step.processing_data = self.test_processing_data
        step.calculate()

        out = self.test_processing_data["sample"]["mask"].signal
        self.assertEqual(out.dtype, np.uint32)  # canonical dtype
        expected = np.array([[4, 0, 0], [0, 32, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_target_non_uint32_is_upcast_to_uint32_once(self):
        """
        If the target mask isn't uint32 (e.g. int64), the step should convert it to uint32
        (one-time allocation) and then OR into that.
        """
        self.test_processing_data = ProcessingData()

        tgt_i64 = np.zeros((2, 3), dtype=np.int64)
        src = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.uint32)

        db = DataBundle(
            mask=BaseData(signal=tgt_i64, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=src, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }
        step.processing_data = self.test_processing_data

        before_id = id(self.test_processing_data["sample"]["mask"].signal)
        step.calculate()

        out = self.test_processing_data["sample"]["mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        self.assertNotEqual(id(out), before_id)  # replacement happened due to upcast

        expected = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_mixed_signed_int_preserves_high_bit_after_uint32_canonicalization(self):
        """
        OR in bit 31 (0x80000000), stored as negative in int32.
        After canonicalization, target is uint32 and should contain 0x80000000.
        """
        self.test_processing_data = ProcessingData()

        tgt = np.zeros((1,), dtype=np.uint32)
        src = np.array([np.int32(-2147483648)], dtype=np.int32)  # 0x80000000

        db = DataBundle(
            mask=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=src, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }
        step.processing_data = self.test_processing_data
        step.calculate()

        out = self.test_processing_data["sample"]["mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        self.assertEqual(int(out[0]), 0x80000000)

        # source unchanged
        self.assertEqual(src.dtype, np.int32)
        self.assertEqual(int(src[0]), -2147483648)

    def test_rejects_non_integer_masks(self):
        """
        Float masks should be rejected (masks must be integer dtype).
        """
        self.test_processing_data = ProcessingData()

        tgt = np.zeros((2, 3), dtype=np.uint32)
        bad = np.zeros((2, 3), dtype=float)

        db = DataBundle(
            mask=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=bad, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }
        step.processing_data = self.test_processing_data

        with self.assertRaises(AssertionError):
            step.calculate()

    def test_broadcasting_and_shape_errors(self):
        """
        Broadcasting is handled by NumPy; incompatible shapes should raise ValueError.
        """
        # broadcastable (2,3) |= (2,1)
        self.test_processing_data = ProcessingData()

        tgt = np.zeros((2, 3), dtype=np.uint32)
        src = np.array([[1], [2]], dtype=np.uint8)  # now explicitly uint8

        db = DataBundle(
            mask=BaseData(signal=tgt, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=src, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

        step = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }
        step.processing_data = self.test_processing_data
        step.calculate()

        expected = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint32)
        np.testing.assert_array_equal(self.test_processing_data["sample"]["mask"].signal, expected)

        # incompatible -> ValueError from NumPy ufunc broadcasting rules (especially with out=)
        self.test_processing_data = ProcessingData()

        tgt2 = np.zeros((2, 3), dtype=np.uint32)
        bad = np.zeros((4,), dtype=np.uint8)

        db2 = DataBundle(
            mask=BaseData(signal=tgt2, units=ureg.dimensionless, uncertainties={}),
            bs_mask=BaseData(signal=bad, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db2

        step2 = BitwiseOrMasks(io_sources=TEST_IO_SOURCES)
        step2.configuration = {
            "with_processing_keys": ["sample"],
            "target_mask_key": "mask",
            "source_mask_keys": ["bs_mask"],
        }
        step2.processing_data = self.test_processing_data

        with self.assertRaises(ValueError):
            step2.calculate()
