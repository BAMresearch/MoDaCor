# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from modacor.server.session_manager import SessionManager


def test_session_manager_run_lifecycle():
    manager = SessionManager()
    manager.create_session(session_id="s1", pipeline_yaml="name: p\nsteps: {}\n")

    run = manager.enqueue_run("s1", mode="partial", effective_mode="full")
    assert run["status"] == "queued"
    assert run["mode"] == "partial"
    assert run["effective_mode"] == "full"

    done = manager.mark_run_succeeded("s1", run["run_id"], details={"num_steps": 3})
    assert done["status"] == "succeeded"
    assert done["num_steps"] == 3

    session = manager.get_session("s1")
    assert session is not None
    assert session.state == "idle"
    assert session.active_run_id is None


def test_session_manager_full_reset_clears_processing_data():
    manager = SessionManager()
    session = manager.create_session(session_id="s2", pipeline_yaml="name: p\nsteps: {}\n")
    session.processing_data = {"dummy": 1}

    manager.reset_session("s2", mode="full")
    session = manager.get_session("s2")
    assert session is not None
    assert session.processing_data is None


def test_session_manager_sink_lifecycle_does_not_disturb_sources():
    manager = SessionManager()
    manager.create_session(session_id="s3", pipeline_yaml="name: p\nsteps: {}\n")

    manager.upsert_sources("s3", [{"ref": "sample", "type": "hdf", "location": "/tmp/sample.nxs"}])
    manager.upsert_sinks(
        "s3",
        [
            {
                "ref": "export_csv",
                "type": "csv",
                "location": "/tmp/export.csv",
                "kwargs": {"delimiter": ","},
            }
        ],
    )

    session = manager.get_session("s3")
    assert session is not None
    assert session.sources["sample"]["type"] == "hdf"
    assert session.sinks["export_csv"]["kwargs"] == {"delimiter": ","}

    assert manager.delete_sink("s3", "export_csv") is True
    assert manager.delete_sink("s3", "export_csv") is False
    assert "sample" in session.sources
    assert session.sinks == {}


def test_session_manager_source_lifecycle_invalidates_source_cache():
    manager = SessionManager()
    session = manager.create_session(session_id="s-source-cache", pipeline_yaml="name: p\nsteps: {}\n")

    manager.upsert_sources("s-source-cache", [{"ref": "sample", "type": "hdf", "location": "/tmp/sample1.nxs"}])
    session.source_cache["sample"] = {"fingerprint": "cached", "source": object()}

    manager.upsert_sources("s-source-cache", [{"ref": "sample", "type": "hdf", "location": "/tmp/sample1.nxs"}])
    assert "sample" in session.source_cache

    manager.upsert_sources("s-source-cache", [{"ref": "sample", "type": "hdf", "location": "/tmp/sample2.nxs"}])
    assert "sample" not in session.source_cache

    session.source_cache["sample"] = {"fingerprint": "cached", "source": object()}
    assert manager.delete_source("s-source-cache", "sample") is True
    assert "sample" not in session.source_cache


def test_session_manager_delete_clears_buffer_store():
    manager = SessionManager()
    manager.create_session(session_id="s-buffer", pipeline_yaml="name: p\nsteps: {}\n")
    manager.buffer_store.put_array("s-buffer", "source", "chunk", "data", [1, 2, 3])

    assert manager.delete_session("s-buffer") is True
    assert manager.buffer_store.manifest("s-buffer", "source", "chunk")["arrays"] == []
