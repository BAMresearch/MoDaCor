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

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.modules.base_modules.append_processing_data import AppendProcessingData


class FakeIoSources:
    """
    Minimal IoSources stand-in for testing AppendProcessingData.

    It only implements the methods actually used by the step:
    - get_data
    - get_static_metadata
    """

    def __init__(self, data=None, metadata=None):
        self.data = data or {}
        self.metadata = metadata or {}

    def get_data(self, data_reference: str, load_slice=...):
        return self.data[data_reference]

    def get_static_metadata(self, data_reference: str):
        return self.metadata[data_reference]


@pytest.fixture
def signal_array():
    return np.arange(6, dtype=float).reshape(2, 3)


@pytest.fixture
def fake_io(signal_array):
    # Only the signal is needed for the simplest tests
    data = {
        "sample::entry/instrument/detector/data": signal_array,
        "sample::entry/instrument/detector/sigma": np.ones_like(signal_array),
    }
    metadata = {
        # for rank-of-data-from-metadata test
        "sample::config/rank": 1,
        # for units_location test (if you decide to use it later)
        "sample::entry/instrument/detector/data@units": "dimensionless",
    }
    return FakeIoSources(data=data, metadata=metadata)


def _make_step(fake_io, processing_data=None) -> AppendProcessingData:
    """
    Helper to create a minimally configured AppendProcessingData instance.
    """
    if processing_data is None:
        processing_data = ProcessingData()

    step = AppendProcessingData(io_sources=fake_io)
    step.processing_data = processing_data
    return step


# --------------------------------------------------------------------------- #
# 1. Basic creation: new DataBundle with default output key "signal"
# --------------------------------------------------------------------------- #
def test_append_processing_data_creates_new_bundle(fake_io, signal_array):
    step = _make_step(fake_io)

    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample::entry/instrument/detector/data",
            "rank_of_data": 2,
            # rely on default databundle_output_key = "signal"
            "units_location": None,
            "units_override": None,
            "uncertainties_sources": {},
        }
    )

    output = step.calculate()

    # Only one bundle, keyed by processing_key
    assert list(output.keys()) == ["sample"]
    bundle = output["sample"]
    assert isinstance(bundle, DataBundle)

    # New bundle is also stored in processing_data
    assert "sample" in step.processing_data
    assert step.processing_data["sample"] is bundle

    # A BaseData is stored under "signal" with the correct data
    assert "signal" in bundle
    bd = bundle["signal"]
    assert isinstance(bd, BaseData)
    np.testing.assert_array_equal(bd.signal, signal_array)

    # Default units (no units_location / override) → dimensionless
    assert bd.units == ureg.dimensionless

    # rank_of_data comes from configuration
    assert bd.rank_of_data == 2

    # default_plot should be set to the output key for a new bundle
    assert bundle.default_plot == "signal"


# --------------------------------------------------------------------------- #
# 2. Updating an existing DataBundle (reusing processing_key)
# --------------------------------------------------------------------------- #
def test_append_processing_data_updates_existing_bundle(fake_io, signal_array):
    processing_data = ProcessingData()

    # Pre-existing bundle with some content
    existing_bundle = DataBundle()
    existing_bd = BaseData(signal=np.ones_like(signal_array), units=ureg.dimensionless)
    existing_bundle["existing"] = existing_bd
    existing_bundle.default_plot = "existing"
    processing_data["sample"] = existing_bundle

    step = _make_step(fake_io, processing_data=processing_data)

    # Configure the step to write into "I" for the same processing_key
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

    # Still just one bundle, still the same object
    assert list(output.keys()) == ["sample"]
    bundle = output["sample"]
    assert bundle is existing_bundle

    # Existing entry retained
    assert "existing" in bundle
    assert bundle["existing"] is existing_bd

    # New entry added/overwritten at key "I"
    assert "I" in bundle
    bd_I = bundle["I"]
    np.testing.assert_array_equal(bd_I.signal, signal_array)
    assert bd_I.rank_of_data == 1

    # default_plot should NOT be changed if it was already set
    assert bundle.default_plot == "existing"


# --------------------------------------------------------------------------- #
# 3. rank_of_data resolved from IoSources metadata (string reference)
# --------------------------------------------------------------------------- #
def test_append_processing_data_rank_from_metadata(fake_io, signal_array):
    step = _make_step(fake_io)

    # rank_of_data is a metadata reference this time
    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample::entry/instrument/detector/data",
            "rank_of_data": "sample::config/rank",  # resolves to 1 in fake_io.metadata
            "units_location": None,
            "units_override": None,
            "uncertainties_sources": {},
        }
    )

    output = step.calculate()
    bd = output["sample"]["signal"]

    # Should have picked up rank 1 from metadata
    assert bd.rank_of_data == 1


# --------------------------------------------------------------------------- #
# 4. uncertainties_sources wiring
# --------------------------------------------------------------------------- #
def test_append_processing_data_adds_uncertainties(fake_io, signal_array):
    step = _make_step(fake_io)

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
    np.testing.assert_array_equal(
        bd.uncertainties["sigma"],
        np.ones_like(signal_array),
    )
