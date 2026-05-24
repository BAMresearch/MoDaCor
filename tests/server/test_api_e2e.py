# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.buffer import decode_npy, encode_npy
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import PipelineRunError, RunResult
from modacor.server.api import create_app
from modacor.server.session_manager import SessionManager

fastapi = pytest.importorskip("fastapi")
testclient_mod = pytest.importorskip("fastapi.testclient")
TestClient = testclient_mod.TestClient


def _post_json(client: TestClient, url: str, payload: dict):
    response = client.post(url, json=payload)
    assert response.status_code in {200, 201, 202}, response.text
    return response.json()


def _put_npy(client: TestClient, url: str, array: np.ndarray):
    response = client.put(url, content=encode_npy(array), headers={"content-type": "application/x-npy"})
    assert response.status_code == 200, response.text
    return response.json()


def test_api_health_and_readiness_expose_runtime_metrics():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    health_response = client.get("/v1/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    ready_empty = client.get("/v1/readiness")
    assert ready_empty.status_code == 200
    assert ready_empty.json() == {
        "status": "ready",
        "ready": True,
        "metrics": {
            "session_count": 0,
            "active_run_count": 0,
            "error_session_count": 0,
            "error_session_ids": [],
            "last_updated_utc": None,
        },
    }

    idle = manager.create_session(session_id="sess-idle", pipeline_yaml="name: idle\nsteps: {}\n")
    running = manager.create_session(session_id="sess-running", pipeline_yaml="name: running\nsteps: {}\n")
    errored = manager.create_session(session_id="sess-error", pipeline_yaml="name: error\nsteps: {}\n")

    idle.updated_utc = "2026-03-15T10:00:00+00:00"
    running.active_run_id = "run-123"
    running.state = "running_full"
    running.updated_utc = "2026-03-15T10:05:00+00:00"
    errored.state = "error_full"
    errored.last_error = {"code": "RUN_FAILED", "message": "synthetic failure", "details": {}}
    errored.updated_utc = "2026-03-15T10:10:00+00:00"

    ready_response = client.get("/v1/readiness")
    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "degraded",
        "ready": True,
        "metrics": {
            "session_count": 3,
            "active_run_count": 1,
            "error_session_count": 1,
            "error_session_ids": ["sess-error"],
            "last_updated_utc": "2026-03-15T10:10:00+00:00",
        },
    }


def test_api_latest_error_diagnostics_returns_current_and_historical_failure():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    manager.create_session(session_id="sess-error-diagnostics", pipeline_yaml="name: diag\nsteps: {}\n")

    empty_response = client.get("/v1/sessions/sess-error-diagnostics/errors/latest")
    assert empty_response.status_code == 200
    assert empty_response.json() == {
        "session_id": "sess-error-diagnostics",
        "state": "idle",
        "active_run_id": None,
        "updated_utc": manager.get_session("sess-error-diagnostics").updated_utc,
        "current_error": None,
        "latest_error": None,
        "latest_failed_run": None,
    }

    first_run = manager.enqueue_run("sess-error-diagnostics", mode="full", effective_mode="full")
    manager.mark_run_failed(
        "sess-error-diagnostics",
        first_run["run_id"],
        code="RUN_FAILED",
        message="synthetic failure",
        details={"exception_type": "RuntimeError", "failed_step_id": "p"},
    )

    failed_response = client.get("/v1/sessions/sess-error-diagnostics/errors/latest")
    assert failed_response.status_code == 200
    failed_payload = failed_response.json()
    assert failed_payload["state"] == "error_full"
    assert failed_payload["current_error"]["code"] == "RUN_FAILED"
    assert failed_payload["current_error"]["run_id"] == first_run["run_id"]
    assert failed_payload["latest_error"]["message"] == "synthetic failure"
    assert failed_payload["latest_failed_run"]["run_id"] == first_run["run_id"]
    assert failed_payload["latest_failed_run"]["error"]["effective_mode"] == "full"

    second_run = manager.enqueue_run("sess-error-diagnostics", mode="full", effective_mode="full")
    manager.mark_run_succeeded("sess-error-diagnostics", second_run["run_id"], details={"num_steps": 0})

    recovered_response = client.get("/v1/sessions/sess-error-diagnostics/errors/latest")
    assert recovered_response.status_code == 200
    recovered_payload = recovered_response.json()
    assert recovered_payload["state"] == "idle"
    assert recovered_payload["current_error"] is None
    assert recovered_payload["latest_error"]["run_id"] == first_run["run_id"]
    assert recovered_payload["latest_failed_run"]["status"] == "failed"


def test_api_process_partial_with_changed_keys_runs_selected_subgraph():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = """
name: e2e_partial
steps:
  p:
    module: PoissonUncertainties
    requires_steps: []
    configuration:
      with_processing_keys:
        - sample
"""
    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-partial",
            "pipeline": {"yaml_text": pipeline_yaml},
            "trace": {"enabled": False},
        },
    )

    # Seed ProcessingData so partial mode can execute directly.
    session = manager.get_session("sess-partial")
    assert session is not None
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(signal=[1.0, 2.0, 3.0], units=ureg.Unit("count"))
    processing_data["sample"] = bundle
    session.processing_data = processing_data

    result = _post_json(
        client,
        "/v1/sessions/sess-partial/process",
        {
            "mode": "partial",
            "changed_keys": ["sample.signal"],
        },
    )
    assert result["status"] == "succeeded"
    assert result["effective_mode"] == "partial"
    assert "run_id" in result

    session_after = manager.get_session("sess-partial")
    assert session_after is not None
    assert "Poisson" in session_after.processing_data["sample"]["signal"].uncertainties
    run_meta = next(item for item in session_after.run_history if item["run_id"] == result["run_id"])
    assert "skipped_steps" in run_meta
    assert "step_durations_s" in run_meta
    assert "elapsed_s" in run_meta
    assert "dirty_steps" in run_meta


def test_api_process_rollback_snapshot_flag_controls_partial_snapshot(monkeypatch):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = """
name: rollback_flag
steps:
  p:
    module: PoissonUncertainties
    requires_steps: []
    configuration:
      with_processing_keys:
        - sample
"""
    for session_id, rollback_snapshot in (("sess-rollback-on", True), ("sess-rollback-off", False)):
        _post_json(
            client,
            "/v1/sessions",
            {
                "session_id": session_id,
                "pipeline": {"yaml_text": pipeline_yaml},
            },
        )
        session = manager.get_session(session_id)
        processing_data = ProcessingData()
        bundle = DataBundle()
        bundle["signal"] = BaseData(signal=np.array([1.0, 2.0]), units=ureg.Unit("count"))
        processing_data["sample"] = bundle
        session.processing_data = processing_data

    calls = {"count": 0}

    def fake_deepcopy(value):
        calls["count"] += 1
        return value

    monkeypatch.setattr("modacor.server.runtime_service.deepcopy", fake_deepcopy)

    _post_json(
        client,
        "/v1/sessions/sess-rollback-on/process",
        {"mode": "partial", "changed_keys": ["sample.signal"]},
    )
    _post_json(
        client,
        "/v1/sessions/sess-rollback-off/process",
        {"mode": "partial", "changed_keys": ["sample.signal"], "rollback_snapshot": False},
    )

    assert calls["count"] == 1
    on_meta = manager.get_session("sess-rollback-on").run_history[-1]
    off_meta = manager.get_session("sess-rollback-off").run_history[-1]
    assert on_meta["rollback_snapshot"] is True
    assert off_meta["rollback_snapshot"] is False


def test_api_sources_patch_upserts_single_source():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = "name: e2e_patch\nsteps: {}\n"
    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-patch",
            "pipeline": {"yaml_text": pipeline_yaml},
        },
    )

    patched = _post_json(
        client,
        "/v1/sessions/sess-patch/sources/patch",
        {
            "ref": "sample",
            "type": "hdf",
            "location": "/tmp/sample1.nxs",
            "kwargs": {"iosource_method_kwargs": {"cache": True}},
        },
    )
    assert patched["session_id"] == "sess-patch"
    assert patched["source"]["ref"] == "sample"
    assert patched["source"]["location"] == "/tmp/sample1.nxs"

    patched2 = _post_json(
        client,
        "/v1/sessions/sess-patch/sources/patch",
        {
            "ref": "sample",
            "type": "hdf",
            "location": "/tmp/sample2.nxs",
        },
    )
    assert patched2["source"]["location"] == "/tmp/sample2.nxs"

    session = manager.get_session("sess-patch")
    assert session is not None
    assert session.sources["sample"]["location"] == "/tmp/sample2.nxs"


def test_api_sinks_patch_upserts_and_deletes_single_sink(tmp_path: Path):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-sinks",
            "pipeline": {"yaml_text": "name: sinks\nsteps: {}\n"},
        },
    )

    out_file = tmp_path / "export.csv"
    patched = _post_json(
        client,
        "/v1/sessions/sess-sinks/sinks/patch",
        {
            "ref": "export_csv",
            "type": "csv",
            "location": str(out_file),
            "kwargs": {"delimiter": ","},
        },
    )
    assert patched["session_id"] == "sess-sinks"
    assert patched["sink"]["ref"] == "export_csv"
    assert patched["sink"]["location"] == str(out_file)

    status = client.get("/v1/sessions/sess-sinks")
    assert status.status_code == 200
    assert status.json()["sinks"][0]["ref"] == "export_csv"

    delete_response = client.delete("/v1/sessions/sess-sinks/sinks/export_csv")
    assert delete_response.status_code == 204
    assert manager.get_session("sess-sinks").sinks == {}


def test_api_process_passes_configured_sinks_to_runner(monkeypatch, tmp_path: Path):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-runner-sinks",
            "pipeline": {"yaml_text": "name: runner_sinks\nsteps: {}\n"},
        },
    )
    manager.upsert_sinks(
        "sess-runner-sinks",
        [{"ref": "export_csv", "type": "csv", "location": str(tmp_path / "out.csv")}],
    )
    captured = {}

    def fake_run_pipeline_job(pipeline, **kwargs):
        captured["sinks"] = kwargs["sinks"]
        return RunResult(
            processing_data=ProcessingData(),
            pipeline=pipeline,
            tracer=None,
            step_durations={},
            executed_steps=[],
            stopped_after_step=None,
        )

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fake_run_pipeline_job)

    result = _post_json(client, "/v1/sessions/sess-runner-sinks/process", {"mode": "full"})

    assert result["status"] == "succeeded"
    assert captured["sinks"].get_sink("export_csv").resource_location == tmp_path / "out.csv"


def test_api_process_stores_trace_report_in_run_metadata(monkeypatch):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-trace-report",
            "pipeline": {"yaml_text": "name: trace_report\nsteps: {}\n"},
            "trace": {"enabled": True},
        },
    )
    captured = {}

    class FakeTracer:
        def last_report(self, n: int, *, renderer=None) -> str:
            captured["n"] = n
            captured["renderer"] = type(renderer).__name__
            return "trace report\nshape: [1065, 1030]"

    def fake_run_pipeline_job(pipeline, **kwargs):  # noqa: ARG001
        return RunResult(
            processing_data=ProcessingData(),
            pipeline=pipeline,
            tracer=FakeTracer(),
            step_durations={},
            executed_steps=[],
            stopped_after_step=None,
        )

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fake_run_pipeline_job)

    result = _post_json(client, "/v1/sessions/sess-trace-report/process", {"mode": "full"})
    runs = client.get("/v1/sessions/sess-trace-report/runs")

    assert result["status"] == "succeeded"
    assert runs.status_code == 200
    assert runs.json()["runs"][-1]["trace_report"] == "trace report\nshape: [1065, 1030]"
    assert captured == {"n": 500, "renderer": "PlainUnicodeRenderer"}


def test_api_process_failure_stores_partial_trace_report(monkeypatch):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-failed-trace-report",
            "pipeline": {"yaml_text": "name: failed_trace_report\nsteps: {}\n"},
            "trace": {"enabled": True},
        },
    )

    class FakeTracer:
        def last_report(self, n: int, *, renderer=None) -> str:  # noqa: ARG002
            return "partial trace report\nshape: [1, 1065, 1030]"

    def fake_run_pipeline_job(pipeline, **kwargs):  # noqa: ARG001
        partial_result = RunResult(
            processing_data=ProcessingData(),
            pipeline=pipeline,
            tracer=FakeTracer(),
            step_durations={},
            executed_steps=["PD_sample"],
            stopped_after_step=None,
        )
        raise PipelineRunError(
            "synthetic failure",
            result=partial_result,
            original_exception=ValueError("synthetic failure"),
            failed_step_id="FA",
        )

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fake_run_pipeline_job)

    response = client.post("/v1/sessions/sess-failed-trace-report/process", json={"mode": "full"})
    latest_error = client.get("/v1/sessions/sess-failed-trace-report/errors/latest").json()

    assert response.status_code == 500
    details = latest_error["latest_error"]["details"]
    assert details["failed_step_id"] == "FA"
    assert details["trace_report"] == "partial trace report\nshape: [1, 1065, 1030]"


def test_api_process_write_hdf_closes_file_for_immediate_reopen(monkeypatch, tmp_path: Path):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = "name: write_hdf_close\nsteps: {}\n"
    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-hdf-close",
            "pipeline": {"yaml_text": pipeline_yaml},
        },
    )

    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(signal=np.array([1.0, 2.0]), units=ureg.Unit("count"))
    processing_data["sample"] = bundle

    def fake_run_pipeline_job(pipeline, **kwargs):  # noqa: ARG001
        return RunResult(
            processing_data=processing_data,
            pipeline=pipeline,
            tracer=None,
            step_durations={},
            executed_steps=[],
            stopped_after_step=None,
        )

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fake_run_pipeline_job)
    out_file = tmp_path / "server-output.h5"

    result = _post_json(
        client,
        "/v1/sessions/sess-hdf-close/process",
        {
            "mode": "full",
            "write_hdf": {
                "path": str(out_file),
                "data_paths": ["/sample/signal/signal"],
            },
        },
    )

    assert result["status"] == "succeeded"
    with h5py.File(out_file, "r+") as h5:
        signal_path = f"processing/result/{result['run_id']}/sample/signal/signal"
        np.testing.assert_allclose(h5[signal_path], [1.0, 2.0])
        h5.attrs["reopened_after_api_run"] = "yes"


def test_api_process_with_api_registered_csv_sink_writes_output(tmp_path: Path):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = """
name: api_sink_write
steps:
  export:
    module: SinkProcessingData
    requires_steps: []
    configuration:
      target: "export_csv::"
      data_paths:
        - /sample/Q/signal
        - /sample/signal/signal
"""
    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-csv-sink",
            "pipeline": {"yaml_text": pipeline_yaml},
        },
    )

    out_file = tmp_path / "export.csv"
    _post_json(
        client,
        "/v1/sessions/sess-csv-sink/sinks/patch",
        {
            "ref": "export_csv",
            "type": "csv",
            "location": str(out_file),
            "kwargs": {"delimiter": ","},
        },
    )

    session = manager.get_session("sess-csv-sink")
    assert session is not None
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["Q"] = BaseData(signal=np.linspace(0.1, 1.0, 5), units=ureg.Unit("1/nm"))
    bundle["signal"] = BaseData(signal=np.array([10, 11, 12, 13, 14], dtype=float), units=ureg.dimensionless)
    processing_data["sample"] = bundle
    session.processing_data = processing_data

    result = _post_json(
        client,
        "/v1/sessions/sess-csv-sink/process",
        {"mode": "partial", "changed_keys": ["sample.signal"]},
    )

    assert result["status"] == "succeeded"
    assert out_file.is_file()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sample/Q/signal,sample/signal/signal"
    assert len(lines) == 7


def test_api_buffer_source_sink_processes_keyed_chunk_data():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = """
name: api_buffer_chunk
steps:
  load_signal:
    module: AppendProcessingData
    requires_steps: []
    configuration:
      processing_key: sample
      databundle_output_key: signal
      signal_location: "chunk_input::sample/signal/signal"
      units_location: "chunk_input::sample/signal/signal@units"
      weights_location: "chunk_input::sample/signal/weights"
      uncertainties_sources:
        poisson: "chunk_input::sample/signal/uncertainties/poisson"
      rank_of_data: 2
  load_mask:
    module: AppendProcessingData
    requires_steps: []
    configuration:
      processing_key: sample
      databundle_output_key: mask
      signal_location: "chunk_input::sample/mask/signal"
      units_override: dimensionless
      rank_of_data: 2
  export:
    module: SinkProcessingData
    requires_steps:
      - load_signal
      - load_mask
    configuration:
      target: "chunk_output::current"
      data_paths:
        - /sample/signal
        - /sample/mask
"""
    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-buffer",
            "pipeline": {"yaml_text": pipeline_yaml},
        },
    )
    _post_json(
        client,
        "/v1/sessions/sess-buffer/sources/patch",
        {"ref": "chunk_input", "type": "buffer", "location": "buffer://session"},
    )
    _post_json(
        client,
        "/v1/sessions/sess-buffer/sinks/patch",
        {"ref": "chunk_output", "type": "buffer", "location": "buffer://session"},
    )

    signal = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    weights = np.full(signal.shape, 0.25, dtype=np.float32)
    poisson = np.ones(signal.shape, dtype=np.float32)
    mask = np.zeros(signal.shape, dtype=np.int16)
    mask[:, 1, 2] = -1

    _put_npy(client, "/v1/sessions/sess-buffer/buffers/sources/chunk_input/arrays/sample/signal/signal", signal)
    _put_npy(client, "/v1/sessions/sess-buffer/buffers/sources/chunk_input/arrays/sample/signal/weights", weights)
    _put_npy(
        client,
        "/v1/sessions/sess-buffer/buffers/sources/chunk_input/arrays/sample/signal/uncertainties/poisson",
        poisson,
    )
    _put_npy(client, "/v1/sessions/sess-buffer/buffers/sources/chunk_input/arrays/sample/mask/signal", mask)

    attrs_response = client.put(
        "/v1/sessions/sess-buffer/buffers/sources/chunk_input/attrs/sample/signal/signal",
        json={"units": "count"},
    )
    assert attrs_response.status_code == 200, attrs_response.text

    result = _post_json(
        client,
        "/v1/sessions/sess-buffer/process",
        {"mode": "full", "rollback_snapshot": False},
    )
    assert result["status"] == "succeeded"

    signal_response = client.get(
        "/v1/sessions/sess-buffer/buffers/sinks/chunk_output/arrays/current/sample/signal/signal"
    )
    assert signal_response.status_code == 200, signal_response.text
    roundtrip_signal = decode_npy(signal_response.content)
    assert roundtrip_signal.dtype == np.dtype("float32")
    np.testing.assert_array_equal(roundtrip_signal, signal)

    mask_response = client.get("/v1/sessions/sess-buffer/buffers/sinks/chunk_output/arrays/current/sample/mask/signal")
    assert mask_response.status_code == 200, mask_response.text
    roundtrip_mask = decode_npy(mask_response.content)
    assert roundtrip_mask.dtype == np.dtype("int16")
    np.testing.assert_array_equal(roundtrip_mask, mask)

    manifest = client.get("/v1/sessions/sess-buffer/buffers/sinks/chunk_output/manifest")
    assert manifest.status_code == 200, manifest.text
    assert "current/sample/signal/uncertainties/poisson" in manifest.json()["arrays"]

    clear_one = client.delete("/v1/sessions/sess-buffer/buffers/sinks/chunk_output/arrays/current/sample/mask/signal")
    assert clear_one.status_code == 200, clear_one.text
    assert clear_one.json()["removed"] >= 1
    missing_mask = client.get("/v1/sessions/sess-buffer/buffers/sinks/chunk_output/arrays/current/sample/mask/signal")
    assert missing_mask.status_code == 404

    clear_sink = client.delete("/v1/sessions/sess-buffer/buffers/sinks/chunk_output")
    assert clear_sink.status_code == 200, clear_sink.text
    assert clear_sink.json()["removed"] >= 1


def test_api_source_templates_and_profile_validation():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    templates_resp = client.get("/v1/source-templates")
    assert templates_resp.status_code == 200
    templates = templates_resp.json()["templates"]
    assert "mouse" in templates

    create_resp = client.post(
        "/v1/sessions",
        json={
            "session_id": "sess-profile",
            "pipeline": {"yaml_text": "name: p\nsteps: {}\n"},
            "source_profile": "mouse",
        },
    )
    assert create_resp.status_code in {200, 201}

    # "mouse" profile requires sample + background, so this should fail until both are set.
    fail_resp = client.post(
        "/v1/sessions/sess-profile/process",
        json={"mode": "full"},
    )
    assert fail_resp.status_code == 422
    fail_detail = fail_resp.json().get("detail", {})
    assert fail_detail.get("code") == "MISSING_REQUIRED_SOURCES"
    assert "sample" in fail_detail.get("details", {}).get("missing_refs", [])

    _post_json(
        client,
        "/v1/sessions/sess-profile/sources/patch",
        {"ref": "sample", "type": "hdf", "location": "/tmp/sample.nxs"},
    )
    _post_json(
        client,
        "/v1/sessions/sess-profile/sources/patch",
        {"ref": "background", "type": "hdf", "location": "/tmp/background.nxs"},
    )

    # Execution may still fail later because files do not exist, but profile validation should pass.
    second_resp = client.post("/v1/sessions/sess-profile/process", json={"mode": "full"})
    assert second_resp.status_code != 422


def test_api_create_session_accepts_yaml_path(tmp_path: Path):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text("name: via-path\nsteps: {}\n", encoding="utf-8")

    response = client.post(
        "/v1/sessions",
        json={
            "session_id": "sess-path",
            "pipeline": {"yaml_path": str(pipeline_path)},
        },
    )
    assert response.status_code in {200, 201}
    payload = response.json()
    assert payload["session_id"] == "sess-path"


def test_api_set_sample_shortcut_upserts_sample_source():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    _post_json(
        client,
        "/v1/sessions",
        {"session_id": "sess-sample-shortcut", "pipeline": {"yaml_text": "name: p\nsteps: {}\n"}},
    )

    first = _post_json(
        client,
        "/v1/sessions/sess-sample-shortcut/sample",
        {"location": "/tmp/sample_a.nxs"},
    )
    assert first["source"]["ref"] == "sample"
    assert first["source"]["type"] == "hdf"
    assert first["source"]["location"] == "/tmp/sample_a.nxs"

    second = _post_json(
        client,
        "/v1/sessions/sess-sample-shortcut/sample",
        {"location": "/tmp/sample_b.nxs", "type": "hdf"},
    )
    assert second["source"]["location"] == "/tmp/sample_b.nxs"


def test_api_process_dry_run_returns_plan_and_missing_profile_sources():
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-dry",
            "pipeline": {
                "yaml_text": (
                    """
name: dry_run_demo
steps:
  p:
    module: PoissonUncertainties
    requires_steps: []
    configuration:
      with_processing_keys:
        - sample
"""
                )
            },
            "source_profile": "mouse",
        },
    )

    # Seed processing data so partial dry-run keeps partial mode.
    session = manager.get_session("sess-dry")
    assert session is not None
    session.processing_data = ProcessingData()

    response = client.post(
        "/v1/sessions/sess-dry/process/dry-run",
        json={"mode": "partial", "changed_keys": ["sample.signal"]},
    )
    assert response.status_code == 200
    plan = response.json()
    assert plan["effective_mode"] == "partial"
    assert "p" in plan["dirty_steps"]
    assert plan["checkpoint_boundary_step"] == "p"
    assert plan["can_process"] is False
    assert "sample" in plan["missing_required_sources"]


def test_api_process_auto_fallback_after_partial_failure(monkeypatch, tmp_path: Path):
    manager = SessionManager()
    app = create_app(session_manager=manager)
    client = TestClient(app)

    pipeline_yaml = """
name: e2e_auto
steps:
  p:
    module: PoissonUncertainties
    requires_steps: []
    configuration:
      with_processing_keys:
        - sample
"""
    _post_json(
        client,
        "/v1/sessions",
        {
            "session_id": "sess-auto",
            "pipeline": {"yaml_text": pipeline_yaml},
            "trace": {"enabled": False},
        },
    )

    session = manager.get_session("sess-auto")
    assert session is not None
    session.processing_data = ProcessingData()  # ensure auto starts with partial attempt
    manager.upsert_sinks(
        "sess-auto",
        [{"ref": "export_csv", "type": "csv", "location": str(tmp_path / "auto.csv")}],
    )

    call_count = {"n": 0}
    seen_sink_locations = []

    def fake_run_pipeline_job(*args, **kwargs):
        call_count["n"] += 1
        seen_sink_locations.append(kwargs["sinks"].get_sink("export_csv").resource_location)
        if call_count["n"] == 1:
            raise RuntimeError("synthetic partial failure")
        return RunResult(
            processing_data=ProcessingData(),
            pipeline=Pipeline.from_yaml("name: fallback\nsteps: {}\n"),
            tracer=None,
            step_durations={},
            executed_steps=[],
            stopped_after_step=None,
        )

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fake_run_pipeline_job)

    result = _post_json(
        client,
        "/v1/sessions/sess-auto/process",
        {
            "mode": "auto",
            "changed_keys": ["sample.signal"],
        },
    )

    assert result["status"] == "succeeded"
    assert result["effective_mode"] == "full"
    assert result["recovered_from_run_id"].startswith("run-")
    assert "fallback_reason" in result
    assert call_count["n"] == 2
    assert seen_sink_locations == [tmp_path / "auto.csv", tmp_path / "auto.csv"]
