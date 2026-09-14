# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from urllib import parse

import numpy as np

from modacor.io.buffer.codec import encode_npy

if TYPE_CHECKING:
    from .session import SessionClient

__all__ = ["BufferClient", "SinkBufferClient", "SourceBufferClient"]


@dataclass(slots=True)
class _BufferEndpoint:
    """Operations shared by one runtime source or sink buffer."""

    session: SessionClient
    ref: str
    kind: ClassVar[str]

    def _data_path(self, collection: str, data_key: str) -> str:
        key = parse.quote(str(data_key).strip("/"), safe="/")
        ref = parse.quote(self.ref, safe="")
        return f"{self.session._path}/buffers/{self.kind}s/{ref}/{collection}/{key}"

    def manifest(self) -> dict[str, Any]:
        ref = parse.quote(self.ref, safe="")
        return self.session.runtime.request("GET", f"{self.session._path}/buffers/{self.kind}/{ref}/manifest")


class SourceBufferClient(_BufferEndpoint):
    """Array and metadata uploads for one runtime source buffer."""

    kind = "source"

    def put_array(self, data_key: str, values: Any) -> dict[str, Any]:
        return self.session.runtime.request(
            "PUT",
            self._data_path("arrays", data_key),
            data=encode_npy(np.asarray(values)),
            content_type="application/x-npy",
        )

    def put_attrs(self, data_key: str, attrs: Mapping[str, Any]) -> dict[str, Any]:
        return self.session.runtime.request("PUT", self._data_path("attrs", data_key), payload=attrs)

    def put_metadata(self, data_key: str, value: Any) -> dict[str, Any]:
        return self.session.runtime.request("PUT", self._data_path("metadata", data_key), payload={"value": value})


class SinkBufferClient(_BufferEndpoint):
    """Array downloads for one runtime sink buffer."""

    kind = "sink"

    def get_array(self, data_key: str) -> np.ndarray:
        return self.session.runtime.request("GET", self._data_path("arrays", data_key))


@dataclass(slots=True)
class BufferClient(_BufferEndpoint):
    """Backward-compatible generic buffer client; prefer the source/sink clients."""

    kind: str = "source"

    def __post_init__(self) -> None:
        if self.kind not in {"source", "sink"}:
            raise ValueError("kind must be 'source' or 'sink'.")

    def put_array(self, data_key: str, values: Any) -> dict[str, Any]:
        if self.kind != "source":
            raise ValueError("Arrays can only be uploaded to source buffers.")
        return self.session.runtime.request(
            "PUT",
            self._data_path("arrays", data_key),
            data=encode_npy(np.asarray(values)),
            content_type="application/x-npy",
        )

    def put_attrs(self, data_key: str, attrs: Mapping[str, Any]) -> dict[str, Any]:
        if self.kind != "source":
            raise ValueError("Attributes can only be uploaded to source buffers.")
        return self.session.runtime.request("PUT", self._data_path("attrs", data_key), payload=attrs)

    def put_metadata(self, data_key: str, value: Any) -> dict[str, Any]:
        if self.kind != "source":
            raise ValueError("Metadata can only be uploaded to source buffers.")
        return self.session.runtime.request("PUT", self._data_path("metadata", data_key), payload={"value": value})

    def get_array(self, data_key: str) -> np.ndarray:
        if self.kind != "sink":
            raise ValueError("Arrays can only be downloaded from sink buffers.")
        return self.session.runtime.request("GET", self._data_path("arrays", data_key))
