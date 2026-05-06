# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest
from attrs import define, field, validators

from modacor.io.csv.csv_sink import CSVSink
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.io.io_sink import IoSink
from modacor.io.runtime_support import build_sinks_from_specs


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
