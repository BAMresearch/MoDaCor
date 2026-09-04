# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Any

from attrs import define, field, validators

from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.io_sink import IoSink

__all__ = ["PlotlyJSONSink"]


@define(kw_only=True)
class PlotlyJSONSink(IoSink):
    """IoSink that stores the latest Plotly figure payload in the runtime buffer."""

    resource_location: Path | str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((Path, str))),
    )
    session_id: str = field(validator=validators.instance_of(str))
    buffer_store: RuntimeBufferStore = field(validator=validators.instance_of(RuntimeBufferStore))
    iosink_method_kwargs: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))

    def write(self, subpath: str, plot_payload: dict[str, Any], **kwargs: Any) -> dict[str, str]:  # noqa: ARG002
        if not isinstance(plot_payload, dict):
            raise TypeError(f"PlotlyJSONSink expects a dict payload, got {type(plot_payload).__name__}.")

        plot_id = str(subpath or self.iosink_method_kwargs.get("default_plot_id", "latest")).strip().strip("/")
        if not plot_id:
            plot_id = "latest"

        self.buffer_store.put_metadata(
            self.session_id,
            "sink",
            self.sink_reference,
            plot_id,
            plot_payload,
        )
        return {"plot_id": plot_id}
