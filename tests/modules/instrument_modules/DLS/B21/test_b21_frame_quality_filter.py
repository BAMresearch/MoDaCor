from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.instrument_modules.DLS.B21.b21_frame_quality_filter import (
    B21FrameQualityFilter,
)


def _processing_data() -> ProcessingData:
    processing = ProcessingData()
    processing["prefilter"] = DataBundle(
        signal=BaseData(
            signal=np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [10.0, 10.0, 18.0, 18.0],
                    [11.0, 11.0, 20.0, 20.0],
                ]
            ),
            units=ureg.count,
            rank_of_data=1,
        ),
        Q=BaseData(
            signal=np.array([0.01, 0.02, 0.10, 0.20]),
            units=ureg.angstrom**-1,
            rank_of_data=1,
        ),
    )
    return processing


def test_sets_independent_high_and_low_q_reason_bits() -> None:
    processing = _processing_data()
    step = B21FrameQualityFilter(
        io_sources=IoSources(),
        configuration={
            "with_processing_keys": ["prefilter"],
            "low_q_max": 0.03,
            "high_q_min": 0.10,
            "q_limits_unit": "1/angstrom",
            "high_q_min_fraction": 0.95,
            "low_q_max_factor": 1.05,
        },
    )
    step.execute(processing)

    result = processing["prefilter"]
    np.testing.assert_array_equal(result["frame_quality_flags"].signal, [0, 1, 2])
    assert result["frame_quality_flags"].signal.dtype == np.uint32
    np.testing.assert_allclose(result["high_q_total"].signal, [40.0, 36.0, 40.0])
    np.testing.assert_allclose(result["low_q_total"].signal, [20.0, 20.0, 22.0])
    assert result["high_q_reference"].signal == pytest.approx(40.0)
    assert result["low_q_reference"].signal == pytest.approx(20.0)


def test_converts_configured_q_limit_units() -> None:
    processing = _processing_data()
    step = B21FrameQualityFilter(
        io_sources=IoSources(),
        configuration={
            "with_processing_keys": ["prefilter"],
            "low_q_max": 0.3,
            "high_q_min": 1.0,
            "q_limits_unit": "1/nm",
        },
    )
    step.execute(processing)
    np.testing.assert_array_equal(processing["prefilter"]["frame_quality_flags"].signal, [0, 1, 2])


def test_missing_regional_data_sets_both_reason_bits() -> None:
    processing = _processing_data()
    signal = processing["prefilter"]["signal"]
    signal.signal = np.concatenate([signal.signal, np.full((1, 4), np.nan)], axis=0)
    step = B21FrameQualityFilter(
        io_sources=IoSources(),
        configuration={
            "with_processing_keys": ["prefilter"],
            "low_q_max": 0.03,
            "high_q_min": 0.10,
        },
    )
    step.execute(processing)
    np.testing.assert_array_equal(processing["prefilter"]["frame_quality_flags"].signal, [0, 1, 2, 3])


def test_rejects_overlapping_q_regions() -> None:
    processing = _processing_data()
    step = B21FrameQualityFilter(
        io_sources=IoSources(),
        configuration={
            "with_processing_keys": ["prefilter"],
            "low_q_max": 0.2,
            "high_q_min": 0.1,
        },
    )
    with pytest.raises(ValueError, match="low_q_max < high_q_min"):
        step.execute(processing)
