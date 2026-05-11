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

__all__ = ["PDHSink"]

from pathlib import Path
from typing import Sequence

import numpy as np
from attrs import define, field, validators

from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sink import IoSink
from modacor.io.processing_path import parse_processing_path, resolve_processing_path

_PDH_EOL = "\r\n"
_PDH_FLOAT_WIDTH = 14
_PDH_INT_WIDTH = 9
_PDH_FLOAT_PRECISION = 6
_DEFAULT_XML_FOOTER = '<?xml version="1.0" encoding="utf-8"?>\n<fileinfo version="3.80.110606"/>\n'


def _ensure_1d_array(obj, path: str) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.ndim != 1:
        raise ValueError(f"PDHSink expects 1D arrays only (got shape {arr.shape}) for path: {path}")
    return arr.astype(float, copy=False)


def _format_int_line(values: Sequence[int]) -> str:
    return " ".join(f"{int(value):{_PDH_INT_WIDTH}d}" for value in values) + " "


def _format_float_line(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):{_PDH_FLOAT_WIDTH}.{_PDH_FLOAT_PRECISION}E}" for value in values) + " "


def _format_pdh_header(point_count: int) -> list[str]:
    return [
        "{SAXSquantDirectMeasurement}",
        "SAXS BOX".ljust(80),
        _format_int_line([point_count, 0, 0, 0, 0, 0, 0, 0]),
        _format_float_line([0.0, 0.0, 0.0, 0.0, 0.0]),
        _format_float_line([0.0, 0.0, 0.0, 0.0, 0.0]),
    ]


def _footer_lines(xml_footer: str | None) -> list[str]:
    if xml_footer is None:
        return []
    text = str(xml_footer)
    if text == "":
        return []
    return text.replace("\r\n", "\n").replace("\r", "\n").splitlines()


@define(kw_only=True)
class PDHSink(IoSink):
    """
    Write three 1D ProcessingData leaves to an Anton Paar PDH ASCII file.

    The PDH file layout is intentionally fixed:
    - five header lines
    - line 3, column 1 contains the number of datapoints
    - two zero-valued numeric header lines
    - exactly three numeric data columns
    - optional XML footer
    """

    resource_location: Path = field(converter=Path, validator=validators.instance_of(Path))
    iosink_method_kwargs: dict = field(factory=dict, validator=validators.instance_of(dict))
    logger: MessageHandler = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.logger = MessageHandler(level=self.logging_level, name="PDHSink")

    def write(
        self,
        subpath: str,
        processing_data: ProcessingData,
        data_paths: Sequence[str] | str | None,
        override_resource_location: Path | None = None,
    ) -> Path:
        if subpath not in ("", None) and str(subpath).strip() != "":
            raise ValueError(f"PDHSink does not support subpaths. Use '{self.sink_reference}::' (got '{subpath}').")

        if data_paths is None:
            data_paths = []
        elif isinstance(data_paths, str):
            data_paths = [data_paths]
        else:
            data_paths = list(data_paths)

        if len(data_paths) != 3:
            raise ValueError("PDHSink.write requires exactly three paths in data_paths.")

        for path in data_paths:
            parsed_path = parse_processing_path(path)
            if len(parsed_path.subpath) == 0:
                raise ValueError("PDH export requires explicit leaf paths (e.g. " f"'/sample/Q/signal'). Got: {path}")

        cols = [_ensure_1d_array(resolve_processing_path(processing_data, path), path) for path in data_paths]

        point_count = cols[0].shape[0]
        for path, col in zip(data_paths, cols):
            if col.shape[0] != point_count:
                raise ValueError(
                    f"All columns must have identical length; expected {point_count}, " f"got {col.shape[0]} for {path}"
                )

        out_path = (override_resource_location or self.resource_location).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        xml_footer = self.iosink_method_kwargs.get("xml_footer", _DEFAULT_XML_FOOTER)
        data = np.column_stack(cols)

        self.logger.info(f"PDHSink writing 3 columns x {point_count} rows to {out_path}.")

        with out_path.open("w", encoding="utf-8", newline="") as handle:
            for line in _format_pdh_header(point_count):
                handle.write(line + _PDH_EOL)
            for row in data:
                handle.write(_format_float_line(row) + _PDH_EOL)
            for line in _footer_lines(xml_footer):
                handle.write(line + _PDH_EOL)

        return out_path
