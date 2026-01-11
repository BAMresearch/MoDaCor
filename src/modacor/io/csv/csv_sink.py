# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["CSVSink"]

from pathlib import Path

import numpy as np
from attrs import define, field, validators

from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sink import IoSink
from modacor.io.processing_path import infer_units_for_path, parse_processing_path, resolve_processing_path


def _default_column_name(path: str) -> str:
    pp = parse_processing_path(path)
    return "/".join((pp.databundle_key, pp.basedata_name, *pp.subpath))


def _ensure_1d_array(obj, path: str) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.ndim != 1:
        raise ValueError(f"CSVSink expects 1D arrays only (got shape {arr.shape}) for path: {path}")
    return arr.astype(float, copy=False)


@define(kw_only=True)
class CSVSink(IoSink):
    """
    Write 1D ProcessingData leaves to a delimiter-separated file.

    Deterministic:
    - requires explicit leaf paths (no default signal)
    - no scalar broadcasting
    - does not support sink subpaths (must call as 'sink_ref::')
    - overwrite-only (no streaming/appending here)
    """

    resource_location: Path = field(converter=Path, validator=validators.instance_of(Path))
    iosink_method_kwargs: dict = field(factory=dict, validator=validators.instance_of(dict))
    logger: MessageHandler = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.logger = MessageHandler(level=self.logging_level, name="CSVSink")

    def write(
        self,
        subpath: str,
        processing_data: ProcessingData,
        data_paths: list[str],
        override_resource_location: Path | None = None,  # not sure if this will be usable in normal operation.
    ) -> Path:
        # CSV does not support internal sink locations
        if subpath not in ("", None) and str(subpath).strip() != "":
            raise ValueError(
                f"CSVSink does not support subpaths. Use '{self.sink_reference}::' (got '{subpath}')."  # noqa: E231
            )

        if not data_paths:
            raise ValueError("CSVSink.write requires at least one path in data_paths.")

        out_path = (override_resource_location or self.resource_location).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # enforce explicit leaf: /bundle/basedata/<leaf...>
        for p in data_paths:
            pp = parse_processing_path(p)
            if len(pp.subpath) == 0:
                raise ValueError(
                    "CSV export requires an explicit leaf path (e.g."
                    f" '/{pp.databundle_key}/{pp.basedata_name}/signal'). Got: {p}"
                )

        cols = []
        for p in data_paths:
            obj = resolve_processing_path(processing_data, p)
            cols.append(_ensure_1d_array(obj, p))

        n = cols[0].shape[0]
        for p, c in zip(data_paths, cols):
            if c.shape[0] != n:
                raise ValueError(
                    f"All columns must have identical length; expected {n}, got {c.shape[0]} for {p}"  # noqa: E702
                )

        names = [_default_column_name(p) for p in data_paths]
        units = [infer_units_for_path(processing_data, p) for p in data_paths]

        # delimiter lives in iosink_method_kwargs to keep configuration minimal
        delimiter = self.iosink_method_kwargs.get("delimiter", ";")

        self.logger.info(f"CSVSink writing {len(cols)} columns x {n} rows to {out_path}.")

        data = np.column_stack(cols)  # (n, ncols)

        # ensure deterministic newline + UTF-8
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(delimiter.join(names) + "\n")
            f.write(delimiter.join(units) + "\n")
            np.savetxt(f, data, **self.iosink_method_kwargs)

        return out_path
