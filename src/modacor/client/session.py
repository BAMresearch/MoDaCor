# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib import parse

from .buffer import SinkBufferClient, SourceBufferClient

if TYPE_CHECKING:
    from .runtime import RuntimeClient

__all__ = ["SessionClient"]


@dataclass(slots=True)
class SessionClient:
    """Operations scoped to one runtime session."""

    runtime: RuntimeClient
    session_id: str
    detail: dict[str, Any] | None = None

    @property
    def _path(self) -> str:
        return f"/v1/sessions/{parse.quote(self.session_id, safe='')}"

    def inspect(self) -> dict[str, Any]:
        self.detail = self.runtime.request("GET", self._path)
        return self.detail

    def delete(self) -> None:
        self.runtime.request("DELETE", self._path)

    def register_sources(self, *sources: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("PUT", f"{self._path}/sources", payload={"sources": list(sources)})

    def register_sinks(self, *sinks: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("PUT", f"{self._path}/sinks", payload={"sinks": list(sinks)})

    def register_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/sources/patch", payload=source)

    def register_sink(self, sink: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/sinks/patch", payload=sink)

    def set_sample(
        self,
        location: str,
        *,
        source_type: str = "hdf",
        kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.runtime.request(
            "POST",
            f"{self._path}/sample",
            payload={"location": str(location), "type": source_type, "kwargs": dict(kwargs or {})},
        )

    def delete_source(self, ref: str) -> None:
        self.runtime.request("DELETE", f"{self._path}/sources/{parse.quote(str(ref), safe='')}")

    def delete_sink(self, ref: str) -> None:
        self.runtime.request("DELETE", f"{self._path}/sinks/{parse.quote(str(ref), safe='')}")

    def source_buffer(self, source_ref: str) -> SourceBufferClient:
        return SourceBufferClient(self, str(source_ref))

    def sink_buffer(self, sink_ref: str) -> SinkBufferClient:
        return SinkBufferClient(self, str(sink_ref))

    def process(
        self,
        *,
        mode: str = "auto",
        changed_sources: Sequence[str] = (),
        changed_keys: Sequence[str] = (),
        write_hdf: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        rollback_snapshot: bool = True,
        chunk_output: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode, "rollback_snapshot": bool(rollback_snapshot)}
        if changed_sources:
            payload["changed_sources"] = list(changed_sources)
        if changed_keys:
            payload["changed_keys"] = list(changed_keys)
        if write_hdf is not None:
            payload["write_hdf"] = dict(write_hdf)
        if run_name is not None:
            payload["run_name"] = run_name
        if chunk_output is not None:
            payload["chunk_output"] = dict(chunk_output)
        return self.runtime.request("POST", f"{self._path}/process", payload=payload)

    def dry_run(
        self,
        *,
        mode: str = "auto",
        changed_sources: Sequence[str] = (),
        changed_keys: Sequence[str] = (),
    ) -> dict[str, Any]:
        payload = {"mode": mode, "changed_sources": list(changed_sources), "changed_keys": list(changed_keys)}
        return self.runtime.request("POST", f"{self._path}/process/dry-run", payload=payload)

    def reset(self, *, mode: str = "full") -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/reset", payload={"mode": mode})

    def recover(self, *, strategy: str, **options: Any) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/recover", payload={"strategy": strategy, **options})

    def runs(self) -> list[dict[str, Any]]:
        return list(self.runtime.request("GET", f"{self._path}/runs").get("runs", []))

    def run(self, run_id: str) -> dict[str, Any]:
        return self.runtime.request("GET", f"{self._path}/runs/{parse.quote(str(run_id), safe='')}")

    def latest_error(self) -> dict[str, Any]:
        return self.runtime.request("GET", f"{self._path}/errors/latest")

    def plot_url(self, sink_ref: str, plot_id: str) -> str:
        sink = parse.quote(str(sink_ref), safe="")
        plot = parse.quote(str(plot_id), safe="")
        return self.runtime.url(f"{self._path}/plots/{sink}/{plot}")
