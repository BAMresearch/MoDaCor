# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.dilate_mask import DilateMask


class TestDilateMaskProcessingStep(unittest.TestCase):
    def setUp(self):
        self.processing_data = ProcessingData()
        self.processing_data["static"] = DataBundle(
            mask=BaseData(
                signal=np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.uint32,
                ),
                units=ureg.dimensionless,
                rank_of_data=2,
            )
        )

    def _make_step(self, **configuration) -> DilateMask:
        step = DilateMask(io_sources=IoSources())
        base_configuration = {"with_processing_keys": ["static"], "backend": "scipy"}
        base_configuration.update(configuration)
        step.modify_config_by_dict(base_configuration)
        step.processing_data = self.processing_data
        return step

    def test_dilates_2d_mask_with_square_footprint(self):
        step = self._make_step(radius=1, footprint_shape="square")

        step.calculate()

        expected = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint32,
        )
        out = self.processing_data["static"]["mask"].signal
        self.assertEqual(out.dtype, np.uint32)
        np.testing.assert_array_equal(out, expected)

    def test_preserves_bitfield_reasons_independently(self):
        source = np.zeros((5, 5), dtype=np.uint32)
        source[2, 2] = 1
        source[1, 3] = 4
        self.processing_data["static"]["mask"] = BaseData(
            signal=source,
            units=ureg.dimensionless,
            rank_of_data=2,
        )
        step = self._make_step(radius=1, footprint_shape="cross")

        step.calculate()

        out = self.processing_data["static"]["mask"].signal
        self.assertEqual(out[2, 3], 5)
        self.assertEqual(out[2, 2] & 1, 1)
        self.assertEqual(out[1, 3] & 4, 4)

    def test_dilates_plane_by_plane_over_selected_axes(self):
        source = np.zeros((2, 4, 4), dtype=np.uint32)
        source[0, 1, 1] = 1
        source[1, 2, 2] = 2
        self.processing_data["static"]["mask"] = BaseData(
            signal=source,
            units=ureg.dimensionless,
            rank_of_data=2,
        )
        step = self._make_step(radius=1, footprint_shape="cross")

        step.calculate()

        out = self.processing_data["static"]["mask"].signal
        self.assertEqual(out.shape, source.shape)
        self.assertEqual(out[0, 1, 2], 1)
        self.assertEqual(out[1, 2, 1], 2)
        self.assertEqual(out[0, 2, 2], 0)

    def test_can_write_to_different_target_key(self):
        step = self._make_step(source_mask_key="mask", target_mask_key="dilated_mask")

        step.calculate()

        self.assertIn("dilated_mask", self.processing_data["static"])
        self.assertIn("mask", self.processing_data["static"])

    def test_dependency_contract_is_exact(self):
        step = self._make_step(source_mask_key="mask", target_mask_key="dilated_mask")

        contract = step.dependency_contract()

        self.assertIsInstance(contract, ProcessStepDependencies)
        self.assertEqual(contract.source_refs, frozenset())
        self.assertEqual(contract.processing_reads, frozenset({"static.mask"}))
        self.assertEqual(contract.processing_writes, frozenset({"static.dilated_mask"}))

    def test_rejects_invalid_inputs(self):
        step = self._make_step(radius=-1)
        with self.assertRaises(ValueError):
            step.calculate()

        self.processing_data["static"]["mask"] = BaseData(
            signal=np.array([[0.0, 1.0]]),
            units=ureg.dimensionless,
        )
        step = self._make_step()
        with self.assertRaises(TypeError):
            step.calculate()
