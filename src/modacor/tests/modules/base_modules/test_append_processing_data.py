# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "03/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from typing import Any

import numpy as np
import pytest
from attrs import define, field

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_source import IoSource
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.append_processing_data import AppendProcessingData


@define(kw_only=True)
class MemoryIoSource(IoSource):
    """
    Minimal in-memory IoSource for tests.

    - `data_key` is the part AFTER '<source_ref>::' (IoSources does that split).
    - get_data supports slicing via numpy indexing.
    - get_static_metadata returns whatever was stored.
    """

    data: dict[str, np.ndarray] = field(factory=dict)
    metadata: dict[str, Any] = field(factory=dict)

    def get_data(self, data_key: str, load_slice=...) -> np.ndarray:
        arr = self.data[data_key]
        return arr[load_slice]

    def get_data_shape(self, data_key: str) -> tuple[int, ...]:
        return tuple(self.data[data_key].shape) if data_key in self.data else ()

    def get_data_dtype(self, data_key: str):
        return self.data[data_key].dtype if data_key in self.data else None

    def get_data_attributes(self, data_key: str) -> dict[str, Any]:
        # Not needed in these tests
        return {}

    def get_static_metadata(self, data_key: str) -> Any:
        return self.metadata[data_key]


@pytest.fixture
def signal_array():
    return np.arange(6, dtype=float).reshape(2, 3)


@pytest.fixture
def io_sources(signal_array) -> IoSources:
    """
    Build a real IoSources with one registered MemoryIoSource under source_ref='sample'.
    """
    sources = IoSources()

    src = MemoryIoSource(
        source_reference="sample",
        data={
            "entry/instrument/detector/data": signal_array,
            "entry/instrument/detector/sigma": np.ones_like(signal_array),
        },
        metadata={
            "config/rank": 1,
            "entry/instrument/detector/data@units": "dimensionless",
        },
    )

    sources.register_source(src)
    return sources


def _make_step(io_sources: IoSources, processing_data: ProcessingData | None = None) -> AppendProcessingData:
    if processing_data is None:
        processing_data = ProcessingData()
    step = AppendProcessingData(io_sources=io_sources, processing_data=processing_data)
    return step


# --------------------------------------------------------------------------- #
# 1. Basic creation: new DataBundle with default output key "signal"
# --------------------------------------------------------------------------- #
def test_append_processing_data_creates_new_bundle(io_sources, signal_array):
    step = _make_step(io_sources)

    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample::entry/instrument/detector/data",
            "rank_of_data": 2,
            "units_location": None,
            "units_override": None,
            "uncertainties_sources": {},
        }
    )

    output = step.calculate()

    assert list(output.keys()) == ["sample"]
    bundle = output["sample"]
    assert isinstance(bundle, DataBundle)

    assert "sample" in step.processing_data
    assert step.processing_data["sample"] is bundle

    assert "signal" in bundle
    bd = bundle["signal"]
    assert isinstance(bd, BaseData)
    np.testing.assert_array_equal(bd.signal, signal_array)

    assert bd.units == ureg.dimensionless
    assert bd.rank_of_data == 2
    assert bundle.default_plot == "signal"


# --------------------------------------------------------------------------- #
# 2. Updating an existing DataBundle (reusing processing_key)
# --------------------------------------------------------------------------- #
def test_append_processing_data_updates_existing_bundle(io_sources, signal_array):
    processing_data = ProcessingData()

    existing_bundle = DataBundle()
    existing_bd = BaseData(signal=np.ones_like(signal_array), units=ureg.dimensionless)
    existing_bundle["existing"] = existing_bd
    existing_bundle.default_plot = "existing"
    processing_data["sample"] = existing_bundle

    step = _make_step(io_sources, processing_data=processing_data)

    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample::entry/instrument/detector/data",
            "rank_of_data": 1,
            "databundle_output_key": "I",
            "units_location": None,
            "units_override": None,
            "uncertainties_sources": {},
        }
    )

    output = step.calculate()

    assert list(output.keys()) == ["sample"]
    bundle = output["sample"]
    assert bundle is existing_bundle

    assert "existing" in bundle
    assert bundle["existing"] is existing_bd

    assert "I" in bundle
    bd_I = bundle["I"]
    np.testing.assert_array_equal(bd_I.signal, signal_array)
    assert bd_I.rank_of_data == 1

    assert bundle.default_plot == "existing"


# --------------------------------------------------------------------------- #
# 3. rank_of_data resolved from IoSources metadata (string reference)
# --------------------------------------------------------------------------- #
def test_append_processing_data_rank_from_metadata(io_sources):
    step = _make_step(io_sources)

    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample::entry/instrument/detector/data",
            "rank_of_data": "sample::config/rank",
            "units_location": None,
            "units_override": None,
            "uncertainties_sources": {},
        }
    )

    output = step.calculate()
    bd = output["sample"]["signal"]
    assert bd.rank_of_data == 1


# --------------------------------------------------------------------------- #
# 4. uncertainties_sources wiring
# --------------------------------------------------------------------------- #
def test_append_processing_data_adds_uncertainties(io_sources, signal_array):
    step = _make_step(io_sources)

    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample::entry/instrument/detector/data",
            "rank_of_data": 2,
            "units_location": None,
            "units_override": None,
            "uncertainties_sources": {
                "sigma": "sample::entry/instrument/detector/sigma",
            },
        }
    )

    output = step.calculate()
    bd = output["sample"]["signal"]

    assert "sigma" in bd.uncertainties
    np.testing.assert_array_equal(bd.uncertainties["sigma"], np.ones_like(signal_array))
