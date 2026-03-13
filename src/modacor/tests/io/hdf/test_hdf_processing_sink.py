# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "30/11/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink


@pytest.fixture
def processing_data_with_uncertainties() -> ProcessingData:
    pd = ProcessingData()
    bundle = DataBundle()

    signal = np.arange(6, dtype=float).reshape(2, 3)
    poisson = np.full_like(signal, 0.1, dtype=float)

    bundle["signal"] = BaseData(
        signal=signal,
        units=ureg.Unit("count"),
        uncertainties={"poisson": poisson},
    )
    pd["sample"] = bundle
    return pd


def _read_json_dataset(group: h5py.Group, name: str) -> dict | list:
    data = group[name][()]
    if isinstance(data, bytes):
        payload = data.decode("utf-8")
    else:
        payload = "".join(chr(c) for c in data.tolist())
    return json.loads(payload)


def _read_text_dataset(group: h5py.Group, name: str) -> str:
    data = group[name][()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


def _read_str_value(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def test_hdf_processing_sink_writes_result_and_metadata(
    tmp_path: Path, processing_data_with_uncertainties: ProcessingData
):
    out_file = tmp_path / "out.h5"
    sink = HDFProcessingSink(resource_location=out_file, iosink_method_kwargs={"compression": "gzip"})

    pipeline_spec = {"name": "demo", "version": "1.0"}
    pipeline_yaml = "name: demo\nsteps: {}\n"
    trace_events = [
        {
            "step_id": "S1",
            "module": "Example",
            "duration_s": 0.1,
            "config": {"a": 1},
            "datasets": {
                "sample.signal": {
                    "diff": ["units"],
                    "prev": {"units": "count"},
                    "now": {"units": "1/s"},
                }
            },
        }
    ]

    sink.write(
        "run1",
        processing_data_with_uncertainties,
        data_paths=["/sample/signal/signal"],
        pipeline_spec=pipeline_spec,
        pipeline_yaml=pipeline_yaml,
        trace_events=trace_events,
    )

    assert out_file.exists()

    with h5py.File(out_file, "r") as h5:
        signal_group = h5["processing/result/run1/sample/signal"]
        np.testing.assert_allclose(
            signal_group["signal"], processing_data_with_uncertainties["sample"]["signal"].signal
        )
        assert signal_group["signal"].attrs["units"] == "count"

        np.testing.assert_allclose(
            signal_group["uncertainties/poisson"],
            processing_data_with_uncertainties["sample"]["signal"].uncertainties["poisson"],
        )

        pipeline_group = h5["processing/pipeline/run1"]
        assert _read_json_dataset(pipeline_group, "spec") == pipeline_spec
        assert _read_text_dataset(pipeline_group, "yaml") == pipeline_yaml

        tracer_group = h5["processing/tracer/run1"]
        assert _read_json_dataset(tracer_group, "events") == trace_events
        assert tracer_group.attrs["schema_version"] == "1.1"
        assert _read_str_value(tracer_group["index/step_ids"][0]) == "S1"
        assert _read_str_value(tracer_group["index/modules"][0]) == "Example"
        assert bool(tracer_group["index/any_change"][0]) is True
        dataset_group = tracer_group["steps/0001_S1/datasets/sample.signal"]
        assert dataset_group.attrs["path"] == "sample.signal"
        assert _read_str_value(dataset_group["changed_kinds"][0]) == "units"
        assert json.loads(_read_text_dataset(dataset_group, "prev_json")) == {"units": "count"}
        assert json.loads(_read_text_dataset(dataset_group, "now_json")) == {"units": "1/s"}


def test_hdf_processing_sink_defaults_to_run_default(
    tmp_path: Path, processing_data_with_uncertainties: ProcessingData
):
    out_file = tmp_path / "out_default.h5"
    sink = HDFProcessingSink(resource_location=out_file)

    sink.write(
        "",
        processing_data_with_uncertainties,
        data_paths=["/sample/signal/signal"],
    )

    with h5py.File(out_file, "r") as h5:
        assert "processing/result/default/sample/signal/signal" in h5
        assert bool(h5["processing/pipeline/default"].attrs["empty"]) is True
        assert bool(h5["processing/tracer/default"].attrs["empty"]) is True


def test_hdf_processing_sink_accepts_string_data_path(
    tmp_path: Path, processing_data_with_uncertainties: ProcessingData
):
    out_file = tmp_path / "out_single_path.h5"
    sink = HDFProcessingSink(resource_location=out_file)

    sink.write(
        "run2",
        processing_data_with_uncertainties,
        data_paths="/sample/signal/signal",
    )

    with h5py.File(out_file, "r") as h5:
        assert "processing/result/run2/sample/signal/signal" in h5
