# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "11/05/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.pdh.pdh_sink import PDHSink


@pytest.fixture
def processing_data_with_pdh_columns() -> ProcessingData:
    pd = ProcessingData()
    bundle = DataBundle()

    q = BaseData(signal=np.array([0.1, 0.2, 0.3], dtype=float), units=ureg.Unit("1/nm"))
    signal = BaseData(
        signal=np.array([10.0, 11.0, 12.0], dtype=float),
        units=ureg.dimensionless,
        uncertainties={"poisson": np.array([1.0, 1.1, 1.2], dtype=float)},
        axes=[q],
        rank_of_data=1,
    )

    bundle["Q"] = q
    bundle["signal"] = signal
    pd["sample"] = bundle
    return pd


def _float_line(values: list[float]) -> str:
    return " ".join(f"{float(value):14.6E}" for value in values) + " "


def test_pdh_sink_writes_fixed_header_data_and_minimal_xml_footer(
    tmp_path: Path,
    processing_data_with_pdh_columns: ProcessingData,
):
    out_file = tmp_path / "out.pdh"
    sink = PDHSink(resource_location=out_file)

    result = sink.write(
        "",
        processing_data_with_pdh_columns,
        data_paths=[
            "/sample/Q/signal",
            "/sample/signal/signal",
            "/sample/signal/uncertainties/poisson",
        ],
    )

    assert result == out_file
    content = out_file.read_bytes()
    assert b"\r\n" in content

    lines = content.decode("utf-8").splitlines()
    assert lines[0] == "{SAXSquantDirectMeasurement}"
    assert lines[1] == "SAXS BOX".ljust(80)
    assert lines[2] == "        3         0         0         0         0         0         0         0 "
    assert lines[3] == _float_line([0.0, 0.0, 0.0, 0.0, 0.0])
    assert lines[4] == _float_line([0.0, 0.0, 0.0, 0.0, 0.0])
    assert lines[5] == _float_line([0.1, 10.0, 1.0])
    assert lines[6] == _float_line([0.2, 11.0, 1.1])
    assert lines[7] == _float_line([0.3, 12.0, 1.2])
    assert lines[8] == '<?xml version="1.0" encoding="utf-8"?>'
    assert lines[9] == '<fileinfo version="3.80.110606"/>'


def test_pdh_sink_can_omit_xml_footer(tmp_path: Path, processing_data_with_pdh_columns: ProcessingData):
    out_file = tmp_path / "out_without_xml.pdh"
    sink = PDHSink(resource_location=out_file, iosink_method_kwargs={"xml_footer": ""})

    sink.write(
        "",
        processing_data_with_pdh_columns,
        data_paths=[
            "/sample/Q/signal",
            "/sample/signal/signal",
            "/sample/signal/uncertainties/poisson",
        ],
    )

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    assert not any(line.startswith("<?xml") for line in lines)


def test_pdh_sink_rejects_subpaths(tmp_path: Path, processing_data_with_pdh_columns: ProcessingData):
    sink = PDHSink(resource_location=tmp_path / "out.pdh")

    with pytest.raises(ValueError, match="does not support subpaths"):
        sink.write(
            "not_supported",
            processing_data_with_pdh_columns,
            data_paths=[
                "/sample/Q/signal",
                "/sample/signal/signal",
                "/sample/signal/uncertainties/poisson",
            ],
        )


def test_pdh_sink_requires_three_paths(tmp_path: Path, processing_data_with_pdh_columns: ProcessingData):
    sink = PDHSink(resource_location=tmp_path / "out.pdh")

    with pytest.raises(ValueError, match="exactly three paths"):
        sink.write("", processing_data_with_pdh_columns, data_paths=["/sample/Q/signal", "/sample/signal/signal"])


def test_pdh_sink_requires_explicit_leaf_paths(tmp_path: Path, processing_data_with_pdh_columns: ProcessingData):
    sink = PDHSink(resource_location=tmp_path / "out.pdh")

    with pytest.raises(ValueError, match="explicit leaf paths"):
        sink.write(
            "",
            processing_data_with_pdh_columns,
            data_paths=[
                "/sample/Q",
                "/sample/signal/signal",
                "/sample/signal/uncertainties/poisson",
            ],
        )


def test_pdh_sink_rejects_non_1d_columns(tmp_path: Path, processing_data_with_pdh_columns: ProcessingData):
    processing_data_with_pdh_columns["sample"]["bad"] = BaseData(
        signal=np.ones((2, 2), dtype=float),
        units=ureg.dimensionless,
    )
    sink = PDHSink(resource_location=tmp_path / "out.pdh")

    with pytest.raises(ValueError, match="expects 1D arrays only"):
        sink.write(
            "",
            processing_data_with_pdh_columns,
            data_paths=[
                "/sample/bad/signal",
                "/sample/signal/signal",
                "/sample/signal/uncertainties/poisson",
            ],
        )


def test_pdh_sink_rejects_mismatched_lengths(tmp_path: Path, processing_data_with_pdh_columns: ProcessingData):
    processing_data_with_pdh_columns["sample"]["short"] = BaseData(
        signal=np.array([1.0, 2.0], dtype=float),
        units=ureg.dimensionless,
    )
    sink = PDHSink(resource_location=tmp_path / "out.pdh")

    with pytest.raises(ValueError, match="identical length"):
        sink.write(
            "",
            processing_data_with_pdh_columns,
            data_paths=[
                "/sample/Q/signal",
                "/sample/short/signal",
                "/sample/signal/uncertainties/poisson",
            ],
        )
