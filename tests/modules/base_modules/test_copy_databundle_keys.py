# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.copy_databundle_keys import CopyDataBundleKeys

TEST_IO_SOURCES = IoSources()


def _basedata(values: list[float]) -> BaseData:
    return BaseData(signal=np.asarray(values, dtype=float), units=ureg.dimensionless)


def _processing_data() -> ProcessingData:
    processing_data = ProcessingData()
    processing_data["sample"] = DataBundle(signal=_basedata([1.0, 2.0, 3.0]))
    processing_data["static"] = DataBundle(
        {
            "Q": _basedata([0.1, 0.2, 0.3]),
            "Psi": _basedata([10.0, 20.0, 30.0]),
        }
    )
    return processing_data


def test_copy_databundle_keys_copies_selected_keys() -> None:
    processing_data = _processing_data()
    step = CopyDataBundleKeys(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample", "static"], data_keys=["Q", "Psi"])

    step(processing_data)

    assert set(processing_data["sample"]) == {"signal", "Q", "Psi"}
    np.testing.assert_allclose(processing_data["sample"]["Q"].signal, [0.1, 0.2, 0.3])
    assert processing_data["sample"]["Q"] is not processing_data["static"]["Q"]
    processing_data["sample"]["Q"].signal[0] = 99.0
    assert processing_data["static"]["Q"].signal[0] == 0.1


def test_copy_databundle_keys_can_rename_keys() -> None:
    processing_data = _processing_data()
    step = CopyDataBundleKeys(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample", "static"], key_map={"Q": "Q_static"})
    step.processing_data = processing_data

    output = step.calculate()

    assert list(output) == ["sample"]
    np.testing.assert_allclose(processing_data["sample"]["Q_static"].signal, [0.1, 0.2, 0.3])
    assert "Q" not in processing_data["sample"]


def test_copy_databundle_keys_can_attach_by_reference() -> None:
    processing_data = _processing_data()
    step = CopyDataBundleKeys(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample", "static"], data_keys=["Q"], copy=False)

    step(processing_data)

    assert processing_data["sample"]["Q"] is processing_data["static"]["Q"]


def test_copy_databundle_keys_requires_two_processing_keys() -> None:
    processing_data = _processing_data()
    step = CopyDataBundleKeys(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample"], data_keys=["Q"])
    step.processing_data = processing_data

    with pytest.raises(AssertionError):
        step.calculate()


def test_copy_databundle_keys_requires_data_keys_or_key_map() -> None:
    processing_data = _processing_data()
    step = CopyDataBundleKeys(io_sources=TEST_IO_SOURCES)
    step.modify_config_by_kwargs(with_processing_keys=["sample", "static"])
    step.processing_data = processing_data

    with pytest.raises(ValueError, match="requires data_keys or key_map"):
        step.calculate()
