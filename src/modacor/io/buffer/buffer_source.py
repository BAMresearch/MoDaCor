# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from attrs import define, field, validators

from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.io_source import ArraySlice, IoSource

__all__ = ["BufferSource"]


@define(kw_only=True)
class BufferSource(IoSource):
    """IoSource backed by the runtime session buffer."""

    resource_location: Path | str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((Path, str))),
    )
    session_id: str = field(validator=validators.instance_of(str))
    buffer_store: RuntimeBufferStore = field(validator=validators.instance_of(RuntimeBufferStore))

    def get_data(self, data_key: str, load_slice: ArraySlice = ...) -> np.ndarray:
        array = self.buffer_store.get_array(self.session_id, "source", self.source_reference, data_key)
        if load_slice is None or load_slice is Ellipsis:
            return array
        return array[load_slice]

    def get_data_shape(self, data_key: str) -> tuple[int, ...]:
        return self.buffer_store.get_array_shape(self.session_id, "source", self.source_reference, data_key)

    def get_data_dtype(self, data_key: str) -> np.dtype | None:
        return self.buffer_store.get_array_dtype(self.session_id, "source", self.source_reference, data_key)

    def get_data_attributes(self, data_key: str) -> dict[str, Any]:
        return self.buffer_store.get_attrs(self.session_id, "source", self.source_reference, data_key)

    def get_static_metadata(self, data_key: str) -> Any:
        if "@" in data_key:
            array_key, attr_key = data_key.rsplit("@", 1)
            attrs = self.get_data_attributes(array_key)
            if attr_key not in attrs:
                raise KeyError(
                    f"Buffer attribute '{attr_key}' not found for source '{self.source_reference}' key '{array_key}'."
                )
            return attrs[attr_key]
        return self.buffer_store.get_metadata(self.session_id, "source", self.source_reference, data_key)
