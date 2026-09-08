# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.runtime_support import build_sink_from_spec
from modacor.io.visualization import PlotlyJSONSink


def test_plotly_json_sink_stores_payload_as_sink_metadata():
    store = RuntimeBufferStore()
    sink = PlotlyJSONSink(
        sink_reference="plots",
        resource_location="buffer://session",
        buffer_store=store,
        session_id="s1",
    )

    result = sink.write("corrected", {"data": [], "layout": {"title": {"text": "demo"}}})

    assert result == {"plot_id": "corrected"}
    assert store.get_metadata("s1", "sink", "plots", "corrected")["layout"]["title"]["text"] == "demo"


def test_runtime_support_builds_plotly_json_sink():
    store = RuntimeBufferStore()

    sink = build_sink_from_spec(
        {"ref": "plots", "type": "plotly_json", "location": "buffer://session"},
        buffer_store=store,
        session_id="s1",
    )

    assert isinstance(sink, PlotlyJSONSink)
