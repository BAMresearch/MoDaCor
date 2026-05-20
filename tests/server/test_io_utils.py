# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from modacor.io.buffer import BufferSink, BufferSource
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.runner.pipeline import Pipeline
from modacor.server.io_utils import build_sinks_from_session, build_sources_from_session
from modacor.server.session_manager import SessionManager


def test_build_sinks_from_session_can_opt_into_hdf_runtime_metadata(tmp_path: Path):
    pipeline_yaml = "name: metadata-demo\nsteps: {}\n"
    manager = SessionManager()
    session = manager.create_session(session_id="s1", pipeline_yaml=pipeline_yaml)
    out_file = tmp_path / "out.h5"
    manager.upsert_sinks(
        "s1",
        [
            {
                "ref": "export_hdf",
                "type": "hdf",
                "location": str(out_file),
                "kwargs": {
                    "compression": "gzip",
                    "include_runtime_metadata": {
                        "pipeline_yaml": True,
                        "pipeline_spec": True,
                        "trace_events": True,
                    },
                },
            }
        ],
    )

    sinks = build_sinks_from_session(session, pipeline=Pipeline.from_yaml(pipeline_yaml))

    sink = sinks.get_sink("export_hdf")
    assert isinstance(sink, HDFProcessingSink)
    assert sink.resource_location == out_file
    assert sink.iosink_method_kwargs["compression"] == "gzip"
    assert sink.iosink_method_kwargs["pipeline_yaml"] == pipeline_yaml
    assert sink.iosink_method_kwargs["pipeline_spec"]["name"] == "metadata-demo"
    assert "trace_events" not in sink.iosink_method_kwargs


def test_build_sources_and_sinks_from_session_support_buffer_type():
    manager = SessionManager()
    session = manager.create_session(session_id="s-buffer", pipeline_yaml="name: buffer-demo\nsteps: {}\n")
    manager.upsert_sources(
        "s-buffer",
        [{"ref": "chunk_input", "type": "buffer", "location": "buffer://session"}],
    )
    manager.upsert_sinks(
        "s-buffer",
        [{"ref": "chunk_output", "type": "buffer", "location": "buffer://session"}],
    )

    sources = build_sources_from_session(session, buffer_store=manager.buffer_store)
    sinks = build_sinks_from_session(session, buffer_store=manager.buffer_store)

    assert isinstance(sources.get_source("chunk_input"), BufferSource)
    assert isinstance(sinks.get_sink("chunk_output"), BufferSink)
