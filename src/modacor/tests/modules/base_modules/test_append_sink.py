# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path

import pytest

from modacor.io.csv.csv_sink import CSVSink
from modacor.io.io_sinks import IoSinks
from modacor.modules.base_modules.append_sink import AppendSink


@pytest.fixture
def io_sinks():
    return IoSinks()


def _make_step(io_sinks: IoSinks) -> AppendSink:
    step = AppendSink()
    step.io_sinks = io_sinks
    return step


def test_append_sink_registers_csvsink(tmp_path: Path, io_sinks: IoSinks):
    out_file = tmp_path / "export.csv"

    step = _make_step(io_sinks)
    step.modify_config_by_dict(
        {
            "sink_identifier": ["export_csv"],
            "sink_location": [str(out_file)],
            "iosink_module": "modacor.io.csv.csv_sink.CSVSink",
            "iosink_method_kwargs": {"delimiter": ","},
        }
    )

    output = step.calculate()
    assert output == {}

    assert "export_csv" in io_sinks.defined_sinks
    sink = io_sinks.defined_sinks["export_csv"]
    assert isinstance(sink, CSVSink)
    assert sink.resource_location == out_file


def test_append_sink_does_not_overwrite_existing_sink(tmp_path: Path, io_sinks: IoSinks):
    out_file = tmp_path / "export.csv"

    step = _make_step(io_sinks)
    step.modify_config_by_dict(
        {
            "sink_identifier": ["export_csv"],
            "sink_location": [str(out_file)],
            "iosink_module": "modacor.io.csv.csv_sink.CSVSink",
            "iosink_method_kwargs": {"delimiter": ","},
        }
    )

    step.calculate()
    first = io_sinks.defined_sinks["export_csv"]

    # Run again: AppendSink should skip registering if id already exists
    step.calculate()
    second = io_sinks.defined_sinks["export_csv"]

    assert first is second
