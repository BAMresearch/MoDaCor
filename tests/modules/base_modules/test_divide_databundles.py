# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.divide_databundles import DivideDatabundles


def _processing_data() -> tuple[ProcessingData, BaseData]:
    dividend = BaseData(
        signal=np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        units=ureg.count,
        uncertainties={"readout": np.full((2, 3), 0.5)},
        rank_of_data=1,
    )
    divisor = BaseData(
        signal=np.asarray([[2.0], [5.0]]),
        units=ureg.second,
        uncertainties={"timing": np.asarray([[0.1], [0.2]])},
        rank_of_data=1,
    )
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(raw=dividend)
    processing_data["count_time"] = DataBundle(duration=divisor)
    return processing_data, dividend / divisor


def test_divide_databundles_uses_configured_keys_and_broadcasts() -> None:
    processing_data, expected = _processing_data()
    step = DivideDatabundles(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["scan", "count_time"],
        dividend_data_key="raw",
        divisor_data_key="duration",
    )

    step.processing_data = processing_data
    output = step.calculate()

    assert list(output) == ["scan"]
    result = processing_data["scan"]["raw"]
    np.testing.assert_allclose(result.signal, expected.signal)
    assert result.units == expected.units
    assert result.uncertainties.keys() == expected.uncertainties.keys()
    for name, uncertainty in expected.uncertainties.items():
        np.testing.assert_allclose(result.uncertainties[name], uncertainty)


def test_divide_databundles_requires_two_processing_keys() -> None:
    processing_data, _expected = _processing_data()
    step = DivideDatabundles(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["scan"])

    with pytest.raises(AssertionError, match="exactly two processing keys"):
        step(processing_data)
