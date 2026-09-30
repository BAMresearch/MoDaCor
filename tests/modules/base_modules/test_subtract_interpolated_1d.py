from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules import SubtractInterpolated1D


def _curve(q, signal, *, signal_uncertainties=None, mask=None) -> DataBundle:
    q_bd = BaseData(np.asarray(q, dtype=float), ureg.Unit("1/nm"), rank_of_data=1)
    signal_bd = BaseData(
        np.asarray(signal, dtype=float),
        ureg.count,
        uncertainties=signal_uncertainties or {},
        axes=[q_bd],
        rank_of_data=1,
    )
    bundle = DataBundle(signal=signal_bd, Q=q_bd)
    if mask is not None:
        bundle["Mask"] = BaseData(np.asarray(mask), ureg.dimensionless, rank_of_data=1)
    return bundle


def test_nearest_subtraction_supports_descending_sample_and_marks_outside_domain():
    processing_data = ProcessingData()
    processing_data["sample"] = _curve([5, 3, 2, 1, 0, -1], [100] * 6)
    processing_data["background"] = _curve([0, 2, 4], [0, 20, 40])
    step = SubtractInterpolated1D(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample", "background"],
            "mode": "nearest",
            "remapped_background_key": "background_on_sample",
        }
    )

    output = step.calculate()["sample"]

    assert_allclose(output["signal"].signal, [np.nan, 80, 80, 100, 100, np.nan], equal_nan=True)
    assert_allclose(
        output["background_on_sample"].signal,
        [np.nan, 20, 20, 0, 0, np.nan],
        equal_nan=True,
    )
    assert_array_equal(output["remap_mask"].signal, [True, False, False, False, False, True])
    assert_allclose(output["signal"].weights, [0, 1, 1, 1, 1, 0])
    assert output["signal"].axes == [output["Q"]]


def test_linear_subtraction_sorts_background_and_propagates_interpolation_uncertainty():
    processing_data = ProcessingData()
    processing_data["sample"] = _curve(
        [0.5, 1.5],
        [20, 30],
        signal_uncertainties={"sample": np.array([2.0, 2.0])},
    )
    processing_data["background"] = _curve(
        [2, 0, 1],
        [20, 0, 10],
        signal_uncertainties={"background": np.array([2.0, 1.0, 1.0])},
    )
    step = SubtractInterpolated1D(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict({"with_processing_keys": ["sample", "background"], "mode": "linear"})

    result = step.calculate()["sample"]["signal"]

    assert_allclose(result.signal, [15, 15])
    assert_allclose(result.uncertainties["sample"], [2, 2])
    assert_allclose(result.uncertainties["background"], [np.sqrt(0.5), np.sqrt(1.25)])


def test_linear_maximum_bracket_width_invalidates_remapping():
    processing_data = ProcessingData()
    processing_data["sample"] = _curve([0.5, 2.0], [10, 10])
    processing_data["background"] = _curve([0, 1, 3], [0, 1, 3])
    step = SubtractInterpolated1D(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample", "background"],
            "mode": "linear",
            "max_gap": 1.5,
        }
    )

    output = step.calculate()["sample"]

    assert_allclose(output["signal"].signal, [9.5, np.nan], equal_nan=True)
    assert_array_equal(output["remap_mask"].signal, [False, True])


def test_subtract_interpolated_dependency_contract_is_exact():
    step = SubtractInterpolated1D(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample", "background"],
            "mask_key": "Mask",
            "output_signal_key": "corrected",
            "remapped_background_key": "remapped",
            "output_mask_key": "invalid",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={
            "sample.signal",
            "sample.Q",
            "sample.Mask",
            "background.signal",
            "background.Q",
            "background.Mask",
        },
        processing_writes={"sample.corrected", "sample.remapped", "sample.invalid"},
    )
