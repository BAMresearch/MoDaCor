# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from attrs import define, field, validators

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.csv.csv_sink import CSVSink
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.io.io_sink import IoSink
from modacor.io.pdh.pdh_sink import PDHSink
from modacor.io.runtime_support import build_sinks_from_specs, write_processing_data_hdf


@define(kw_only=True)
class CustomSink(IoSink):
    resource_location: Path = field(converter=Path, validator=validators.instance_of(Path))

    def write(self, subpath: str, *args, **kwargs):  # noqa: ARG002
        return self.resource_location


def test_build_sinks_from_specs_builds_csv_sink(tmp_path: Path):
    out_file = tmp_path / "out.csv"

    sinks = build_sinks_from_specs(
        [
            {
                "ref": "export_csv",
                "type": "csv",
                "location": out_file,
                "kwargs": {"delimiter": ","},
            }
        ]
    )

    sink = sinks.get_sink("export_csv")
    assert isinstance(sink, CSVSink)
    assert sink.resource_location == out_file
    assert sink.iosink_method_kwargs == {"delimiter": ","}


def test_build_sinks_from_specs_builds_hdf_processing_sink(tmp_path: Path):
    out_file = tmp_path / "out.h5"

    sinks = build_sinks_from_specs(
        [
            {
                "ref": "export_hdf",
                "type": "hdf_processing",
                "location": out_file,
                "kwargs": {"iosink_method_kwargs": {"compression": "gzip"}},
            }
        ]
    )

    sink = sinks.get_sink("export_hdf")
    assert isinstance(sink, HDFProcessingSink)
    assert sink.resource_location == out_file
    assert sink.iosink_method_kwargs == {"compression": "gzip"}


def test_build_sinks_from_specs_builds_pdh_sink(tmp_path: Path):
    out_file = tmp_path / "out.pdh"

    sinks = build_sinks_from_specs(
        [
            {
                "ref": "export_pdh",
                "type": "pdh",
                "location": out_file,
                "kwargs": {"xml_footer": ""},
            }
        ]
    )

    sink = sinks.get_sink("export_pdh")
    assert isinstance(sink, PDHSink)
    assert sink.resource_location == out_file
    assert sink.iosink_method_kwargs == {"xml_footer": ""}


def test_build_sinks_from_specs_supports_custom_sink(tmp_path: Path):
    out_file = tmp_path / "custom.out"

    sinks = build_sinks_from_specs(
        [
            {
                "ref": "custom",
                "type": "custom",
                "location": out_file,
                "kwargs": {
                    "class_path": "tests.io.test_runtime_support.CustomSink",
                    "iosink_method_kwargs": {"mode": "test"},
                },
            }
        ]
    )

    sink = sinks.get_sink("custom")
    assert isinstance(sink, CustomSink)
    assert sink.resource_location == out_file
    assert sink.iosink_method_kwargs == {"mode": "test"}


def test_build_sinks_from_specs_rejects_unsupported_type(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported sink type"):
        build_sinks_from_specs(
            [
                {
                    "ref": "bad",
                    "type": "unknown",
                    "location": tmp_path / "bad.out",
                }
            ]
        )


def test_write_processing_data_hdf_persists_tracer_snapshots(tmp_path: Path):
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(signal=np.array([1.0, 2.0]), units=ureg.Unit("count"))
    processing_data["sample"] = bundle

    out_file = tmp_path / "snapshot_export.h5"
    result = SimpleNamespace(
        processing_data=processing_data,
        pipeline=SimpleNamespace(to_spec=lambda: {"name": "demo"}),
        tracer=SimpleNamespace(
            processing_data_snapshots=[
                {
                    "step_id": "S1",
                    "module": "Example",
                    "processing_data": processing_data,
                }
            ]
        ),
    )

    write_processing_data_hdf(
        {
            "path": str(out_file),
            "data_paths": ["/sample/signal/signal"],
        },
        run_name="run1",
        result=result,
        pipeline_yaml="name: demo\nsteps: {}\n",
    )

    with h5py.File(out_file, "r") as h5:
        assert "processing/tracer/run1/steps/0001_S1/processing_data/sample/signal/signal" in h5


def test_build_sinks_from_specs_requires_custom_class_path(tmp_path: Path):
    with pytest.raises(ValueError, match="requires kwargs.class_path"):
        build_sinks_from_specs(
            [
                {
                    "ref": "custom",
                    "type": "custom",
                    "location": tmp_path / "custom.out",
                }
            ]
        )
