# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "11/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sinks import IoSinks
from modacor.modules.base_modules.append_sink import AppendSink
from modacor.modules.base_modules.sink_processing_data import SinkProcessingData


@pytest.fixture
def processing_data_1d() -> ProcessingData:
    """
    Minimal ProcessingData with a single DataBundle containing two 1D BaseData entries.
    """
    pd = ProcessingData()
    b = DataBundle()

    q = BaseData(signal=np.linspace(0.1, 1.0, 5), units=ureg.Unit("1/nm"))
    i = BaseData(signal=np.array([10, 11, 12, 13, 14], dtype=float), units=ureg.dimensionless)

    b["Q"] = q
    b["signal"] = i
    pd["sample"] = b
    return pd


def _register_csv_sink(io_sinks: IoSinks, out_file: Path) -> None:
    """
    Register a CSVSink via AppendSink.
    """
    step = AppendSink(io_sources=None, io_sinks=io_sinks)
    step.modify_config_by_dict(
        {
            "sink_identifier": ["export_csv"],
            "sink_location": [str(out_file)],
            "iosink_module": "modacor.io.csv.csv_sink.CSVSink",
            # simplified: delimiter (and any np.savetxt kwargs) live here
            "iosink_method_kwargs": {"delimiter": ","},
        }
    )
    step.calculate()


def _run_sink_step(io_sinks: IoSinks, processing_data: ProcessingData, *, target: str, data_paths: list[str]):
    step = SinkProcessingData(io_sources=None, io_sinks=io_sinks, processing_data=processing_data)
    step.modify_config_by_dict({"target": target, "data_paths": data_paths})
    return step.calculate()


def test_sink_processing_data_writes_csv_numpy(tmp_path: Path, processing_data_1d: ProcessingData):
    out_file = tmp_path / "export.csv"
    io_sinks = IoSinks()
    _register_csv_sink(io_sinks, out_file)

    data_paths = ["/sample/Q/signal", "/sample/signal/signal"]
    output = _run_sink_step(io_sinks, processing_data_1d, target="export_csv::", data_paths=data_paths)

    assert output == {}
    assert out_file.is_file()

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 + 5  # 2 headers + 5 rows

    # header row: names derived from data_paths
    assert lines[0] == "sample/Q/signal,sample/signal/signal"

    # units row: inferred from BaseData units
    q_units = str(processing_data_1d["sample"]["Q"].units)
    i_units = str(processing_data_1d["sample"]["signal"].units)
    assert lines[1] == f"{q_units},{i_units}"  # noqa: E231

    # first numeric row should match first entries
    first_row = [float(x) for x in lines[2].split(",")]
    assert first_row == [
        float(processing_data_1d["sample"]["Q"].signal[0]),
        float(processing_data_1d["sample"]["signal"].signal[0]),
    ]


def test_sink_processing_data_rejects_csv_subpath(tmp_path: Path, processing_data_1d: ProcessingData):
    out_file = tmp_path / "export.csv"
    io_sinks = IoSinks()
    _register_csv_sink(io_sinks, out_file)

    with pytest.raises(ValueError):
        _run_sink_step(
            io_sinks,
            processing_data_1d,
            target="export_csv::not_supported",
            data_paths=["/sample/Q/signal"],
        )


def test_sink_processing_data_requires_explicit_leaf_path(tmp_path: Path, processing_data_1d: ProcessingData):
    out_file = tmp_path / "export.csv"
    io_sinks = IoSinks()
    _register_csv_sink(io_sinks, out_file)

    # Missing leaf (BaseData object root) -> CSVSink should refuse
    with pytest.raises(ValueError):
        _run_sink_step(io_sinks, processing_data_1d, target="export_csv::", data_paths=["/sample/Q"])
