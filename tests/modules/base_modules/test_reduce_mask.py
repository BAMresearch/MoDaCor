# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "02/09/2026"
__status__ = "Development"

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.reduce_mask import ReduceMask


def _run_reduce(mask, *, axes=(0, 1), reduction="any", rank_of_data=2):
    processing_data = ProcessingData()
    processing_data["sample"] = DataBundle(
        raw_count_mask=BaseData(signal=np.asarray(mask), units=ureg.dimensionless, rank_of_data=rank_of_data)
    )
    step = ReduceMask(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "source_mask_key": "raw_count_mask",
            "target_mask_key": "mask",
            "axes": axes,
            "reduction": reduction,
        }
    )
    step.execute(processing_data)
    return processing_data["sample"]["mask"]


def test_reduce_mask_any_preserves_reason_bits_over_leading_axes():
    mask = np.zeros((2, 3, 2, 2), dtype=np.uint32)
    mask[0, 0, 0, 1] = 1
    mask[1, 2, 1, 0] = 4

    out = _run_reduce(mask, axes=[0, 1], reduction="any", rank_of_data=2)

    expected = np.array([[0, 1], [4, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(out.signal, expected)
    assert out.signal.dtype == np.uint32
    assert out.rank_of_data == 2


def test_reduce_mask_all_preserves_only_bits_present_everywhere():
    mask = np.full((2, 2, 1), 3, dtype=np.uint32)
    mask[0, 0, 0] = 1

    out = _run_reduce(mask, axes=[0, 1], reduction="all", rank_of_data=1)

    np.testing.assert_array_equal(out.signal, np.array([1], dtype=np.uint32))
    assert out.rank_of_data == 1


def test_reduce_mask_rejects_float_masks():
    with pytest.raises(TypeError):
        _run_reduce(np.zeros((2, 2), dtype=float), axes=0)
