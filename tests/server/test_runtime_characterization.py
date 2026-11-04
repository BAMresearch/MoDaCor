# SPDX-License-Identifier: BSD-3-Clause
"""Public runtime behavior around registration, recovery and failed runs."""

from __future__ import annotations

from typing import get_type_hints

import h5py
import numpy as np
import pytest

from modacor.io.buffer import encode_npy
from modacor.io.chunking import ProvisionalChunkOutput, ProvisionalChunkPlan
from modacor.runner import run_pipeline_job
from modacor.server.api import create_app
from modacor.server.runtime_policy import RuntimePolicy
from modacor.server.runtime_service import RuntimeService
from modacor.server.session_manager import SessionManager
from tests.server.test_api_e2e import _server_chunk, _server_chunk_plan

pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

EMPTY_PIPELINE = "name: characterization\nsteps: {}\n"
LOAD_PIPELINE = """name: recovery
steps:
  load:
    module: AppendProcessingData
    configuration:
      processing_key: sample
      signal_location: sample::data
      units_override: count
      rank_of_data: 1
"""


def test_runtime_constructor_annotations_resolve():
    hints = get_type_hints(RuntimeService.__init__)
    assert hints["manager"] is SessionManager
    assert hints["policy"] is RuntimePolicy


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        response = client.post("/v1/sessions", json={"session_id": "s", "pipeline": {"yaml_text": EMPTY_PIPELINE}})
        assert response.status_code == 200
        yield client


@pytest.mark.parametrize("kind", ["source", "sink"])
@pytest.mark.parametrize(
    "payload,detail",
    [
        ({}, "'ref' is required."),
        ({"ref": "data"}, "'type' is required."),
        ({"ref": "data", "type": "buffer"}, "'location' is required."),
        (
            {"ref": "data", "type": "buffer", "location": "buffer://s", "kwargs": [1]},
            "'kwargs' must be an object when provided.",
        ),
    ],
)
def test_patch_validation(client, kind, payload, detail):
    response = client.post(f"/v1/sessions/s/{kind}s/patch", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert client.get("/v1/sessions/s").json()[f"{kind}s"] == []


@pytest.mark.parametrize("kind", ["source", "sink"])
def test_resource_missing_session_and_registration(client, kind):
    registration = {"ref": "data", "type": "buffer", "location": "buffer://s"}
    for method, suffix, payload in [
        ("put", f"{kind}s", {f"{kind}s": [registration]}),
        ("post", f"{kind}s/patch", registration),
    ]:
        response = getattr(client, method)(f"/v1/sessions/absent/{suffix}", json=payload)
        assert response.status_code == 404
        assert "Session 'absent' not found." in response.json()["detail"]
    for session_id, detail in [("absent", "Session 'absent' not found."), ("s", f"{kind.title()} 'data' not found.")]:
        response = client.delete(f"/v1/sessions/{session_id}/{kind}s/data")
        assert response.status_code == 404
        assert detail in response.json()["detail"]
    response = client.put(f"/v1/sessions/s/{kind}s", json={f"{kind}s": {}})
    assert response.status_code == 422
    assert response.json() == {"detail": f"'{kind}s' must be a list."}
    response = client.put(f"/v1/sessions/s/{kind}s", json={f"{kind}s": [{"ref": "data"}]})
    assert response.status_code == 422
    assert "missing required field(s): type, location" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload,detail",
    [
        ({}, "'location' is required."),
        ({"location": "buffer://s", "type": " "}, "'type' must be non-empty."),
        ({"location": "buffer://s", "kwargs": [1]}, "'kwargs' must be an object when provided."),
    ],
)
def test_sample_validation(client, payload, detail):
    response = client.post("/v1/sessions/s/sample", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": detail}


def test_sample_missing_session(client):
    response = client.post("/v1/sessions/absent/sample", json={"location": "buffer://s", "type": "buffer"})
    assert response.status_code == 404
    assert "Session 'absent' not found." in response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["sources", "sinks", "sources/patch", "sinks/patch", "sample"])
def test_resource_policy_rejection(endpoint, tmp_path):
    policy = RuntimePolicy.restricted(source_read_roots=(tmp_path,), sink_write_roots=(tmp_path,))
    with TestClient(create_app(runtime_policy=policy)) as client:
        assert (
            client.post("/v1/sessions", json={"session_id": "s", "pipeline": {"yaml_text": EMPTY_PIPELINE}}).status_code
            == 200
        )
        registration = {"ref": "data", "type": "hdf", "location": str(tmp_path.parent / "outside.h5")}
        if endpoint in {"sources", "sinks"}:
            response = client.put(f"/v1/sessions/s/{endpoint}", json={endpoint: [registration]})
        else:
            response = client.post(f"/v1/sessions/s/{endpoint}", json=registration)
        assert response.status_code == 422
        assert "outside allowed" in response.json()["detail"]
        session = client.get("/v1/sessions/s").json()
        assert session["sources"] == session["sinks"] == []


@pytest.mark.parametrize("mode", ["partial", "full"])
def test_reset_and_run_queries(client, mode):
    run = client.post("/v1/sessions/s/process", json={"mode": "full"}).json()
    response = client.get(f"/v1/sessions/s/runs/{run['run_id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    response = client.post("/v1/sessions/s/reset", json={"mode": mode})
    assert response.status_code == 200
    assert response.json() == {"session_id": "s", "mode": mode, "state": "idle"}
    session = client.get("/v1/sessions/s").json()
    assert session["active_run_id"] is None
    assert session["last_run"]["mode"] == mode
    response = client.get("/v1/sessions/s/runs/absent")
    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found."}


@pytest.mark.parametrize(
    "endpoint,payload,status,detail",
    [
        ("s/reset", {}, 422, "mode must be one of: partial, full."),
        ("absent/reset", {"mode": "full"}, 404, "Session 'absent' not found."),
        ("s/recover", {}, 422, "Invalid recovery strategy."),
        ("absent/recover", {"strategy": "full_reset_only"}, 404, "Session 'absent' not found."),
    ],
)
def test_reset_recover_validation(client, endpoint, payload, status, detail):
    response = client.post(f"/v1/sessions/{endpoint}", json=payload)
    assert response.status_code == status
    assert detail in response.json()["detail"]


@pytest.mark.parametrize("strategy", ["full_reset_only", "full_reset_then_process"])
def test_recover_real_source_failure(client, tmp_path, strategy):
    assert (
        client.post("/v1/sessions", json={"session_id": "load", "pipeline": {"yaml_text": LOAD_PIPELINE}}).status_code
        == 200
    )
    source_file = tmp_path / "arriving.h5"
    assert client.post("/v1/sessions/load/sample", json={"location": str(source_file)}).status_code == 200
    response = client.post("/v1/sessions/load/process", json={"mode": "full"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "SOURCE_UNAVAILABLE"
    assert client.get("/v1/sessions/load").json()["state"] == "error_full"
    with h5py.File(source_file, "w") as handle:
        handle["data"] = [1.0, 2.0]
    output_file = tmp_path / "recovered.h5"
    response = client.post(
        "/v1/sessions/load/recover",
        json={
            "strategy": strategy,
            "write_hdf": {"path": str(output_file), "write_all_processing_data": True},
            "run_name": "recovered",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["strategy"] == strategy
    assert response.json()["state"] == "idle"
    session = client.get("/v1/sessions/load").json()
    assert session["active_run_id"] is None
    error = client.get("/v1/sessions/load/errors/latest").json()
    assert error["current_error"] is None
    assert error["latest_failed_run"]["error"]["code"] == "SOURCE_UNAVAILABLE"
    if strategy == "full_reset_then_process":
        assert session["last_run"]["status"] == "succeeded"
        assert session["last_run"]["hdf_output"] == str(output_file)
        assert output_file.exists()
    else:
        assert session["last_run"]["mode"] == "full"


@pytest.mark.parametrize(
    "payload,detail",
    [
        ({"mode": "invalid"}, "mode must be one of: partial, full, auto."),
        ({"mode": "partial"}, "partial mode requires changed_sources or changed_keys."),
        ({"mode": "full", "chunk_output": []}, "chunk_output must be an object."),
        (
            {"mode": "full", "chunk_output": {}},
            "chunk_output requires output_id and exactly one of chunk_spec or chunk_id.",
        ),
        (
            {"mode": "full", "chunk_output": {"output_id": "missing", "chunk_id": "c0", "chunk_spec": {}}},
            "chunk_output requires output_id and exactly one of chunk_spec or chunk_id.",
        ),
    ],
)
@pytest.mark.parametrize("endpoint", ["process", "process/dry-run"])
def test_process_request_validation(client, payload, detail, endpoint):
    response = client.post(f"/v1/sessions/s/{endpoint}", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert client.get("/v1/sessions/s/runs").json()["runs"] == []


def test_process_unknown_chunk_output(client):
    response = client.post(
        "/v1/sessions/s/process", json={"mode": "full", "chunk_output": {"output_id": "missing", "chunk_id": "c0"}}
    )
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]
    assert client.get("/v1/sessions/s/runs").json()["runs"] == []


@pytest.mark.parametrize("fallback_fails", [False, True])
def test_auto_fallback_run_records_trace_and_chunk_state(client, monkeypatch, tmp_path, fallback_fails):
    assert (
        client.post(
            "/v1/sessions",
            json={"session_id": "load", "pipeline": {"yaml_text": LOAD_PIPELINE}, "trace": {"enabled": True}},
        ).status_code
        == 200
    )
    client.post("/v1/sessions/load/sample", json={"type": "buffer", "location": "buffer://load"})
    assert (
        client.put(
            "/v1/sessions/load/buffers/sources/sample/arrays/data",
            content=encode_npy(np.ones((2, 2), dtype=np.float32)),
        ).status_code
        == 200
    )
    initial = client.post("/v1/sessions/load/process", json={"mode": "full"})
    assert initial.status_code == 200, initial.text
    plan = _server_chunk_plan()
    initialized = client.post(
        "/v1/chunked-outputs",
        json={
            "sink": {"ref": "out", "type": "hdf_chunked", "location": str(tmp_path / "chunks.h5")},
            "subpath": "run",
            "plan": plan.to_dict(),
        },
    )
    assert initialized.status_code == 201, initialized.text
    output_id = initialized.json()["output_id"]

    def fail_partial(pipeline, **kwargs):
        if kwargs.get("selected_step_ids") is not None or fallback_fails:
            # A real pipeline failure supplies the trace and failed step identity.
            kwargs["sources"] = type(kwargs["sources"])()
        return run_pipeline_job(pipeline, **kwargs)

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fail_partial)
    monkeypatch.setattr("modacor.server.process_failure.run_pipeline_job", fail_partial)
    response = client.post(
        "/v1/sessions/load/process",
        json={
            "mode": "auto",
            "changed_sources": ["sample"],
            "write_hdf": {"path": str(tmp_path / "fallback.h5"), "write_all_processing_data": True},
            "run_name": "fallback",
            "chunk_output": {"output_id": output_id, "chunk_spec": _server_chunk(plan, 0).to_dict()},
        },
    )
    assert response.status_code == (500 if fallback_fails else 200), response.text
    runs = client.get("/v1/sessions/load/runs").json()["runs"]
    partial, fallback = runs[-2:]
    assert partial["mode"] == "auto"
    assert partial["effective_mode"] == "partial"
    assert partial["status"] == "failed"
    assert partial["error"]["code"] == "PARTIAL_RUN_FAILED"
    assert partial["error"]["details"]["failed_step_id"] == "load"
    assert partial["error"]["details"]["trace_report"] == ""
    assert fallback["mode"] == fallback["effective_mode"] == "full"
    assert fallback["recovered_from_run_id"] == partial["run_id"]
    assert isinstance(fallback["trace_report"], str)
    session = client.get("/v1/sessions/load").json()
    assert session["active_run_id"] is None
    progress = client.get(f"/v1/chunked-outputs/{output_id}").json()
    if fallback_fails:
        assert fallback["status"] == "failed"
        assert fallback["error"]["code"] == "FULL_RUN_FAILED"
        assert response.json()["detail"]["details"]["run_id"] == fallback["run_id"]
        assert session["state"] == "error_full"
        assert progress["completed_chunks"] == 0
    else:
        assert fallback["status"] == "succeeded"
        assert response.json()["recovered_from_run_id"] == partial["run_id"]
        assert fallback["fallback_reason"] == partial["error"]["message"]
        assert fallback["chunk_output"]["chunk_id"] == "c0"
        assert session["state"] == "idle"
        assert progress["completed_chunks"] == 1


@pytest.mark.parametrize(
    "payload,status,detail",
    [
        ({}, 422, "Provide exactly one of plan or provisional_plan."),
        ({"plan": {}}, 422, "Provide sink or both session_id and sink_ref."),
        ({"plan": {}, "sink": []}, 422, "sink must be an I/O registration object."),
        ({"plan": {}, "session_id": "s", "sink_ref": "missing"}, 404, "is not registered"),
        ({"plan": {}, "sink": {}}, 422, "Invalid complete ChunkPlan:"),
    ],
)
def test_chunk_initialize_validation(client, payload, status, detail):
    response = client.post("/v1/chunked-outputs", json=payload)
    assert response.status_code == status
    assert detail in response.json()["detail"]


def test_chunk_collision_reopen_recover_and_parse_errors(client, tmp_path):
    plan = _server_chunk_plan()
    sink = {"ref": "out", "type": "hdf_chunked", "location": str(tmp_path / "chunks.h5")}
    payload = {"sink": sink, "subpath": "run", "plan": plan.to_dict()}
    initialized = client.post("/v1/chunked-outputs", json=payload)
    assert initialized.status_code == 201
    output_id = initialized.json()["output_id"]
    collision = client.post("/v1/chunked-outputs", json=payload)
    assert collision.status_code == 409
    assert collision.json()["detail"]
    for data, status, detail in [
        ({}, 422, "plan_id is required."),
        ({"plan_id": "missing", "sink": sink}, 404, "missing"),
        ({"plan_id": plan.plan_id, "plan_hash": "wrong", "sink": sink}, 422, "hash"),
    ]:
        response = client.post("/v1/chunked-outputs/reopen", json=data)
        assert response.status_code == status, response.text
        assert detail in response.json()["detail"]
    reopened = client.post("/v1/chunked-outputs/reopen", json={"plan_id": plan.plan_id, "sink": sink})
    assert reopened.status_code == 201
    assert reopened.json()["reopened"] is True
    assert reopened.json()["missing_chunks"] == 2
    for action, status in [("invalid", 422), ("resume", 409)]:
        response = client.post(f"/v1/chunked-outputs/{output_id}/recover", json={"action": action})
        assert response.status_code == status, response.text
        assert response.json()["detail"]
    response = client.post("/v1/chunked-outputs/missing/recover", json={"action": "reconcile"})
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]
    response = client.post(
        "/v1/sessions/s/process",
        json={
            "mode": "full",
            "chunk_output": {
                "output_id": output_id,
                "chunk_spec": {**_server_chunk(plan, 0).to_dict(), "plan_hash": "wrong"},
            },
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    assert client.get("/v1/sessions/s/runs").json()["runs"] == []


def test_provisional_source_validation(client, tmp_path):
    provisional = ProvisionalChunkPlan(
        schema_version="1.0",
        plan_id="provisional",
        driver={"source": "sample::data", "rank_of_data": 1},
        axis_rules=({"axis": 0, "chunk_size": 2},),
        outputs=(ProvisionalChunkOutput("signal", "/sample/signal", "sample/signal"),),
    )
    payload = {
        "sink": {"ref": "out", "type": "hdf_chunked", "location": str(tmp_path / "out.h5")},
        "provisional_plan": provisional.to_dict(),
    }
    for extra, detail in [({}, "session_id is required"), ({"session_id": "s"}, "not registered")]:
        response = client.post("/v1/chunked-outputs", json={**payload, **extra})
        assert response.status_code == 422
        assert detail in response.json()["detail"]
    assert client.post("/v1/sessions/s/sample", json={"type": "buffer", "location": "buffer://s"}).status_code == 200
    response = client.post("/v1/chunked-outputs", json={**payload, "session_id": "s"})
    assert response.status_code == 422
    assert "require driver.full_shape" in response.json()["detail"]
