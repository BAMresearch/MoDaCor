# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
from attrs import define, field, validators

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.io_sink import IoSink
from modacor.io.processing_path import parse_processing_path, resolve_processing_path

__all__ = ["BufferSink"]


def _join_key(*parts: str) -> str:
    cleaned: list[str] = []
    for part in parts:
        value = str(part).strip().strip("/")
        if value:
            cleaned.extend(item for item in PurePosixPath(value).parts if item != "/")
    return "/".join(cleaned)


def _normalise_data_paths(data_paths: Sequence[str] | str | None) -> list[str]:
    if isinstance(data_paths, str):
        return [data_paths]
    if data_paths is None:
        return []
    return [str(path) for path in data_paths]


@define(kw_only=True)
class BufferSink(IoSink):
    """IoSink that writes selected ProcessingData entries into the runtime buffer."""

    resource_location: Path | str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((Path, str))),
    )
    session_id: str = field(validator=validators.instance_of(str))
    buffer_store: RuntimeBufferStore = field(validator=validators.instance_of(RuntimeBufferStore))
    iosink_method_kwargs: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))

    def _put_array(self, data_key: str, value: Any) -> None:
        self.buffer_store.put_array(self.session_id, "sink", self.sink_reference, data_key, np.asarray(value))

    def _put_basedata(self, prefix: str, bundle_key: str, basedata_name: str, basedata: BaseData) -> list[str]:
        base_key = _join_key(prefix, bundle_key, basedata_name)
        written = []
        signal_key = _join_key(base_key, "signal")
        self._put_array(signal_key, basedata.signal)
        written.append(signal_key)
        self.buffer_store.update_attrs(
            self.session_id,
            "sink",
            self.sink_reference,
            signal_key,
            {
                "units": str(basedata.units),
                "rank_of_data": int(basedata.rank_of_data),
            },
        )

        weights_key = _join_key(base_key, "weights")
        self._put_array(weights_key, basedata.weights)
        written.append(weights_key)
        for name, uncertainty in basedata.uncertainties.items():
            uncertainty_key = _join_key(base_key, "uncertainties", str(name))
            self._put_array(uncertainty_key, uncertainty)
            written.append(uncertainty_key)
        return written

    def write(
        self,
        subpath: str,
        processing_data: ProcessingData,
        data_paths: Sequence[str] | str | None,
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, list[str]]:
        paths = _normalise_data_paths(data_paths)
        if not paths:
            raise ValueError("BufferSink.write requires one or more data_paths.")

        written_arrays: list[str] = []
        written_metadata: list[str] = []
        prefix = _join_key(subpath)

        for path in paths:
            parsed = parse_processing_path(path)
            if not parsed.subpath:
                basedata = processing_data[parsed.databundle_key][parsed.basedata_name]
                if not isinstance(basedata, BaseData):
                    raise TypeError(f"Processing path '{path}' did not resolve to a BaseData instance.")
                written_arrays.extend(self._put_basedata(prefix, parsed.databundle_key, parsed.basedata_name, basedata))
                continue

            value = resolve_processing_path(processing_data, path)
            data_key = _join_key(prefix, parsed.databundle_key, parsed.basedata_name, *parsed.subpath)
            if isinstance(value, BaseData):
                written_arrays.extend(self._put_basedata(prefix, parsed.databundle_key, parsed.basedata_name, value))
            elif isinstance(value, (np.ndarray, int, float, complex, bool, np.number)):
                self._put_array(data_key, value)
                written_arrays.append(data_key)
            else:
                self.buffer_store.put_metadata(
                    self.session_id,
                    "sink",
                    self.sink_reference,
                    data_key,
                    str(value),
                )
                written_metadata.append(data_key)

        return {"arrays": list(dict.fromkeys(written_arrays)), "metadata": list(dict.fromkeys(written_metadata))}
