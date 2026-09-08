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
from modacor.modules.base_modules.threshold_mask import ThresholdMask

TEST_IO_SOURCES = IoSources()


class TestThresholdMaskProcessingStep(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/threshold_mask.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()

        signal = np.array([[0, 1, 4e9], [2, 3, 7e5]], dtype=np.uint32)
        flatfield = np.array([[0.75, 0.9, 1.0], [1.1, 1.25, 1.2]], dtype=float)

        db = DataBundle(
            signal=BaseData(signal=signal, units=ureg.dimensionless, uncertainties={}),
            flatfield=BaseData(signal=flatfield, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db

    def _make_step(self, **configuration) -> ThresholdMask:
        step = ThresholdMask(io_sources=TEST_IO_SOURCES)
        base_configuration = {"with_processing_keys": ["sample"], "threshold": 1e9}
        base_configuration.update(configuration)
        step.modify_config_by_dict(base_configuration)
        step.processing_data = self.test_processing_data
        return step

    # ------------------------------------------------------------------ #
    # Actual tests
    # ------------------------------------------------------------------ #

    def test_threshold_mask_expected_mask(self):
        "Test legacy threshold behavior masks values above the threshold."

        self.setUp()

        step = self._make_step()

        step.processing_data = self.test_processing_data
        step.calculate()
        out = step.processing_data["sample"]["threshold_mask"].signal

        expected = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_threshold_mask_can_use_non_signal_basedata_outside_bounds(self):
        step = self._make_step(
            source_basedata_key="flatfield",
            target_mask_key="flatfield_mask",
            lower_bound=0.8,
            upper_bound=1.2,
            threshold=None,
            mask_mode="outside",
        )

        step.calculate()

        out = step.processing_data["sample"]["flatfield_mask"].signal
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_threshold_mask_can_mask_values_inside_bounds(self):
        step = self._make_step(
            source_basedata_key="flatfield",
            lower_bound=0.9,
            upper_bound=1.1,
            threshold=None,
            mask_mode="inside",
        )

        step.calculate()

        out = step.processing_data["sample"]["threshold_mask"].signal
        expected = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_threshold_mask_preserves_source_dimensionality(self):
        """
        Source dimensionality is part of the mask semantics; leading image or
        frame axes must not be averaged before thresholding.
        """
        self.test_processing_data = ProcessingData()

        signal = np.array([[[0, 1, 4e9], [2, 3, 7e5]], [[0, 1, 4e9], [2, 3, 7e5]]], dtype=np.uint32)

        db = DataBundle(
            signal=BaseData(signal=signal, units=ureg.dimensionless, uncertainties={}, rank_of_data=2),
        )
        self.test_processing_data["sample"] = db

        step = self._make_step()
        step.processing_data = self.test_processing_data

        step.calculate()

        out = self.test_processing_data["sample"]["threshold_mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        self.assertEqual(out.shape, signal.shape)
        self.assertEqual(self.test_processing_data["sample"]["threshold_mask"].rank_of_data, 2)

        expected = np.array([[[0, 0, 1], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]], dtype=np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_threshold_mask_preserves_intermittent_hot_pixels(self):
        self.test_processing_data = ProcessingData()
        signal = np.zeros((2, 2, 2, 3), dtype=float)
        signal[1, 0, 0, 2] = 4e9

        db = DataBundle(
            signal=BaseData(signal=signal, units=ureg.dimensionless, uncertainties={}, rank_of_data=2),
        )
        self.test_processing_data["sample"] = db

        step = self._make_step()
        step.calculate()

        out = self.test_processing_data["sample"]["threshold_mask"].signal
        expected = np.zeros(signal.shape, dtype=np.uint32)
        expected[1, 0, 0, 2] = 1
        np.testing.assert_array_equal(out, expected)

    def test_threshold_mask_accepts_1d_source_without_invalid_rank(self):
        self.test_processing_data = ProcessingData()
        line = np.array([0.75, 0.9, 1.1, 1.25], dtype=float)
        self.test_processing_data["sample"] = DataBundle(
            flatfield=BaseData(signal=line, units=ureg.dimensionless, uncertainties={}, rank_of_data=1),
        )

        step = self._make_step(
            source_basedata_key="flatfield",
            lower_bound=0.8,
            upper_bound=1.2,
            threshold=None,
        )
        step.calculate()

        out = self.test_processing_data["sample"]["threshold_mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        self.assertEqual(out.ndim, 1)
        np.testing.assert_array_equal(out, np.array([1, 0, 0, 1], dtype=np.uint32))

    def test_threshold_mask_dependency_contract_is_exact(self):
        step = self._make_step(
            source_basedata_key="flatfield",
            target_mask_key="flatfield_mask",
            lower_bound=0.8,
            upper_bound=1.2,
            threshold=None,
        )

        contract = step.dependency_contract()

        self.assertIsInstance(contract, ProcessStepDependencies)
        self.assertEqual(contract.source_refs, frozenset())
        self.assertEqual(contract.processing_reads, frozenset({"sample.flatfield"}))
        self.assertEqual(contract.processing_writes, frozenset({"sample.flatfield_mask"}))

    def test_threshold_mask_rejects_invalid_range_or_mode(self):
        step = self._make_step(lower_bound=1.2, upper_bound=0.8, threshold=None)

        with self.assertRaises(ValueError):
            step.calculate()

        step = self._make_step(lower_bound=0.8, upper_bound=1.2, threshold=None, mask_mode="middle")

        with self.assertRaises(ValueError):
            step.calculate()
