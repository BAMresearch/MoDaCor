# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from modacor.io.buffer import BufferSink, BufferSource
from modacor.io.chunking import (
    AxisSelector,
    ChunkArrayLayout,
    ChunkOutputLayout,
    ChunkPlacement,
    ChunkPlan,
    ChunkSourceBinding,
    ChunkSpec,
)
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.tiled import TiledSource
from modacor.runner.pipeline import Pipeline
from modacor.server.io_utils import bind_chunk_source_slices, build_sinks_from_session, build_sources_from_session
from modacor.server.runtime_policy import RuntimePolicy
from modacor.server.session_manager import SessionManager


def _write_hdf(path: Path, *, value: float = 1.0) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=[value])


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


def test_build_sources_and_sinks_from_session_support_custom_aliases():
    manager = SessionManager()
    session = manager.create_session(session_id="s-custom-alias", pipeline_yaml="name: custom-demo\nsteps: {}\n")
    manager.upsert_sources(
        "s-custom-alias",
        [
            {
                "ref": "chunk_input",
                "type": "custom",
                "location": "buffer://session",
                "kwargs": {"class_alias": "runtime-buffer-source"},
            }
        ],
    )
    manager.upsert_sinks(
        "s-custom-alias",
        [
            {
                "ref": "chunk_output",
                "type": "custom",
                "location": "buffer://session",
                "kwargs": {"class_alias": "runtime-buffer-sink"},
            }
        ],
    )
    policy = RuntimePolicy.restricted(
        custom_source_classes={"runtime-buffer-source": BufferSource},
        custom_sink_classes={"runtime-buffer-sink": BufferSink},
    )

    sources = build_sources_from_session(session, buffer_store=manager.buffer_store, runtime_policy=policy)
    sinks = build_sinks_from_session(session, buffer_store=manager.buffer_store, runtime_policy=policy)

    assert isinstance(sources.get_source("chunk_input"), BufferSource)
    assert isinstance(sinks.get_sink("chunk_output"), BufferSink)


def test_build_sources_from_session_reuses_unchanged_hdf_source(tmp_path: Path):
    hdf_file = tmp_path / "sample.h5"
    _write_hdf(hdf_file)

    manager = SessionManager()
    session = manager.create_session(session_id="s-hdf-cache", pipeline_yaml="name: hdf-cache\nsteps: {}\n")
    manager.upsert_sources(
        "s-hdf-cache",
        [{"ref": "sample", "type": "hdf", "location": str(hdf_file)}],
    )

    first_sources = build_sources_from_session(session, buffer_store=manager.buffer_store)
    first_source = first_sources.get_source("sample")
    second_sources = build_sources_from_session(session, buffer_store=manager.buffer_store)

    assert isinstance(first_source, HDFSource)
    assert second_sources.get_source("sample") is first_source
    assert session.source_cache["sample"]["source"] is first_source


def test_build_sources_from_session_rebuilds_hdf_source_when_registration_changes(tmp_path: Path):
    first_file = tmp_path / "first.h5"
    second_file = tmp_path / "second.h5"
    _write_hdf(first_file, value=1.0)
    _write_hdf(second_file, value=2.0)

    manager = SessionManager()
    session = manager.create_session(session_id="s-hdf-cache-change", pipeline_yaml="name: hdf-cache\nsteps: {}\n")
    manager.upsert_sources(
        "s-hdf-cache-change",
        [{"ref": "sample", "type": "hdf", "location": str(first_file)}],
    )
    first_source = build_sources_from_session(session, buffer_store=manager.buffer_store).get_source("sample")

    manager.upsert_sources(
        "s-hdf-cache-change",
        [{"ref": "sample", "type": "hdf", "location": str(second_file)}],
    )
    second_source = build_sources_from_session(session, buffer_store=manager.buffer_store).get_source("sample")

    assert second_source is not first_source
    assert session.source_cache["sample"]["source"] is second_source


def test_build_sources_from_session_rebuilds_hdf_source_when_file_changes_in_place(tmp_path: Path):
    hdf_file = tmp_path / "sample.h5"
    _write_hdf(hdf_file)

    manager = SessionManager()
    session = manager.create_session(session_id="s-hdf-cache-stat", pipeline_yaml="name: hdf-cache\nsteps: {}\n")
    manager.upsert_sources(
        "s-hdf-cache-stat",
        [{"ref": "sample", "type": "hdf", "location": str(hdf_file)}],
    )

    first_source = build_sources_from_session(session, buffer_store=manager.buffer_store).get_source("sample")

    with h5py.File(hdf_file, "a") as h5:
        h5.create_dataset("extra", data=list(range(128)))
    stat = hdf_file.stat()
    os.utime(hdf_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    second_source = build_sources_from_session(session, buffer_store=manager.buffer_store).get_source("sample")

    assert second_source is not first_source
    assert "extra" in second_source._file_datasets_shapes


def test_build_sources_from_session_reuses_unchanged_tiled_source(monkeypatch):
    connections = []

    def connect(resource_location, connection_kwargs):
        connections.append((resource_location, connection_kwargs))
        return {}

    monkeypatch.setattr("modacor.io.tiled.tiled_source.connect_tiled", connect)
    manager = SessionManager()
    session = manager.create_session(session_id="s-tiled-cache", pipeline_yaml="name: tiled\nsteps: {}\n")
    manager.upsert_sources(
        session.session_id,
        [{"ref": "sample", "type": "tiled", "location": "https://example.invalid/catalog"}],
    )

    first_source = build_sources_from_session(session).get_source("sample")
    second_source = build_sources_from_session(session).get_source("sample")

    assert isinstance(first_source, TiledSource)
    assert second_source is first_source
    assert len(connections) == 1


def test_bind_chunk_source_slices_projects_direct_hdf_inputs_without_caching(tmp_path: Path):
    hdf_file = tmp_path / "chunk-input.h5"
    detector = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    monitor = np.arange(4, dtype=np.float32)
    with h5py.File(hdf_file, "w") as h5:
        h5.create_dataset("detector", data=detector)
        h5.create_dataset("monitor", data=monitor)
        h5.create_dataset("calibration", data=np.array([10.0, 20.0]))

    plan = ChunkPlan(
        schema_version="1.0",
        plan_id="direct-hdf",
        total_chunks=2,
        expected_chunk_ids=("c0", "c1"),
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=2,
                arrays=(ChunkArrayLayout("signal", detector.shape, detector.dtype.str),),
            ),
        ),
        driver={"source": "sample::/detector", "full_shape": list(detector.shape)},
        batch_axes=(0,),
        data_axes=(1, 2),
        source_bindings=(
            ChunkSourceBinding("sample", "/detector", "aligned"),
            ChunkSourceBinding("sample", "/monitor", "explicit", axis_map=(0,)),
            ChunkSourceBinding("sample", "/calibration", "static"),
        ),
    )
    chunk = ChunkSpec(
        schema_version=plan.schema_version,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        chunk_id="c1",
        ordinal=1,
        grid_index=(1,),
        source_selection=(AxisSelector.sliced(2, 4), AxisSelector.all(), AxisSelector.all()),
        expected_input_shape=(2, 3, 2),
        placements=(
            ChunkPlacement(
                "signal",
                (AxisSelector.sliced(2, 4), AxisSelector.all(), AxisSelector.all()),
                (2, 3, 2),
            ),
        ),
    )

    manager = SessionManager()
    session = manager.create_session(session_id="direct-hdf", pipeline_yaml="name: direct\nsteps: {}\n")
    manager.upsert_sources(
        session.session_id,
        [{"ref": "sample", "type": "hdf", "location": str(hdf_file)}],
    )
    sources = build_sources_from_session(session, buffer_store=manager.buffer_store)

    resolved = bind_chunk_source_slices(sources, session, plan, chunk)

    np.testing.assert_array_equal(sources.get_data("sample::/detector"), detector[2:4])
    np.testing.assert_array_equal(sources.get_data("sample::/monitor"), monitor[2:4])
    np.testing.assert_array_equal(sources.get_data("sample::/calibration"), np.array([10.0, 20.0]))
    assert resolved[0]["selection"][0] == {"kind": "slice", "start": 2, "stop": 4, "stride": 1}
    assert resolved[2]["selection"] is None
    assert set(sources.get_source("sample")._data_cache) == {"/calibration"}
