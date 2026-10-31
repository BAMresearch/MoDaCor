# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import pytest

from modacor.server.api import create_app

pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/sessions",
            json={"session_id": "diagnostics", "pipeline": {"yaml_text": "name: diagnostics\nsteps: {}\n"}},
        )
        assert response.status_code == 200
        yield client


def test_api_4xx_logs_request_and_preserves_detail(client, caplog):
    with caplog.at_level(logging.WARNING, logger="modacor.server.api"):
        response = client.get("/v1/sessions/absent")
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found."}
    record = next(record for record in caplog.records if record.name == "modacor.server.api")
    assert record.levelno == logging.WARNING
    assert record.method == "GET"
    assert record.path == "/v1/sessions/absent"
    assert record.session_id == "absent"
    assert "Session not found." in record.message


def test_unexpected_handler_exception_is_structured_and_logged(client, caplog, monkeypatch):
    def fail(self, session_id):
        raise ValueError("unexpected handler failure")

    monkeypatch.setattr(type(client.app.state.runtime_service), "get_session", fail)
    with caplog.at_level(logging.ERROR):
        response = client.get("/v1/sessions/diagnostics")
    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "INTERNAL_ERROR",
            "message": "ValueError: unexpected handler failure",
            "details": {"method": "GET", "path": "/v1/sessions/diagnostics", "session_id": "diagnostics"},
        }
    }
    record = next(record for record in caplog.records if record.name == "modacor.server.api")
    assert record.exc_info[0] is ValueError
    assert "unexpected handler failure" in caplog.text


@pytest.mark.parametrize("kind", ["source", "sink"])
def test_io_registration_update_and_deletion_logs_identity(client, caplog, kind):
    registration = {"ref": "data", "type": "buffer", "location": "buffer://session"}
    with caplog.at_level(logging.INFO, logger="modacor.server"):
        assert client.put(f"/v1/sessions/diagnostics/{kind}s", json={f"{kind}s": [registration]}).status_code == 200
        assert client.post(f"/v1/sessions/diagnostics/{kind}s/patch", json=registration).status_code == 200
        assert client.delete(f"/v1/sessions/diagnostics/{kind}s/data").status_code == 204
    records = [record for record in caplog.records if record.name == "modacor.server.runtime_service"]
    assert len(records) == 3
    for record, action in zip(records, ["registered", "updated", "deleted"]):
        assert record.levelno == logging.INFO
        assert record.session_id == "diagnostics"
        assert record.ref == "data"
        assert record.io_type == "buffer"
        assert record.location == "buffer://session"
        assert action in record.message


def test_run_failure_logs_traceback_and_keeps_api_error_payload(client, caplog, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("scientific failure")

    monkeypatch.setattr("modacor.server.runtime_service.run_pipeline_job", fail)
    with caplog.at_level(logging.ERROR, logger="modacor.server"):
        response = client.post("/v1/sessions/diagnostics/process", json={"mode": "full"})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "RUN_FAILED"
    assert detail["message"] == "scientific failure"
    assert set(detail["details"]) == {"session_id", "run_id"}
    records = [record for record in caplog.records if record.name.startswith("modacor.server.")]
    assert {record.name for record in records} == {"modacor.server.api", "modacor.server.runtime_service"}
    for record in records:
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None
        assert record.session_id == "diagnostics"
    run_record = next(record for record in records if record.name.endswith("runtime_service"))
    assert run_record.run_id == detail["details"]["run_id"]
    assert run_record.error_code == "RUN_FAILED"
    assert "ValueError: scientific failure" in caplog.text


def _assert_unavailable(client, caplog, *, kind, path, exception_type):
    registration = {"ref": "sample" if kind == "source" else "export", "type": "hdf", "location": str(path)}
    with caplog.at_level(logging.INFO, logger="modacor.server"):
        response = client.put(f"/v1/sessions/diagnostics/{kind}s", json={f"{kind}s": [registration]})
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 1
    assert {key: warnings[0][key] for key in ("ref", "kind", "location")} == {
        "ref": registration["ref"],
        "kind": kind,
        "location": str(path),
    }
    assert warnings[0]["problem"]
    assert any(record.levelno == logging.WARNING and str(path) in record.message for record in caplog.records)
    response = client.post("/v1/sessions/diagnostics/process", json={"mode": "full"})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == f"{kind.upper()}_UNAVAILABLE"
    assert registration["ref"] in detail["message"]
    assert str(path) in detail["message"]
    assert (
        detail["details"].items()
        >= {
            "ref": registration["ref"],
            "type": "hdf",
            "location": str(path),
            "exception_type": exception_type,
        }.items()
    )
    latest = client.get("/v1/sessions/diagnostics/errors/latest").json()["latest_error"]
    assert latest["code"] == detail["code"]
    assert latest["details"]["exception_type"] == exception_type
    assert exception_type in latest["details"]["traceback"]


def test_missing_source_warns_then_fails_with_identity(client, caplog, tmp_path):
    _assert_unavailable(client, caplog, kind="source", path=tmp_path / "missing.h5", exception_type="FileNotFoundError")


@pytest.mark.parametrize("sink_type", ["hdf", "csv"])
def test_sink_creates_new_nested_directories(client, tmp_path, sink_type):
    import h5py
    import numpy as np

    input_file = tmp_path / "input.h5"
    with h5py.File(input_file, "w") as handle:
        handle["data"] = np.array([1.0, 2.0, 3.0])
    pipeline = """name: nested-output
steps:
  load:
    module: AppendProcessingData
    configuration:
      processing_key: sample
      signal_location: sample::data
      units_override: count
      rank_of_data: 1
  export:
    module: SinkProcessingData
    requires_steps: [load]
    configuration:
      target: "export::"
      data_paths: [/sample/signal/signal]
"""
    if sink_type == "hdf":
        pipeline = pipeline.replace("data_paths: [/sample/signal/signal]", "data_paths: [/sample]")
    response = client.post("/v1/sessions", json={"session_id": "nested", "pipeline": {"yaml_text": pipeline}})
    assert response.status_code == 200
    assert (
        client.put(
            "/v1/sessions/nested/sources",
            json={"sources": [{"ref": "sample", "type": "hdf", "location": str(input_file)}]},
        ).status_code
        == 200
    )
    output = tmp_path / "new" / "nested" / ("out.h5" if sink_type == "hdf" else "out.csv")
    response = client.put(
        "/v1/sessions/nested/sinks", json={"sinks": [{"ref": "export", "type": sink_type, "location": str(output)}]}
    )
    assert response.status_code == 200
    assert response.json()["warnings"] == []
    response = client.post("/v1/sessions/nested/process", json={"mode": "full"})
    assert response.status_code == 200, response.text
    assert output.is_file()
    if sink_type == "hdf":
        with h5py.File(output) as handle:
            np.testing.assert_array_equal(handle["processing/result/default/sample/signal/signal"], [1.0, 2.0, 3.0])
    else:
        assert output.stat().st_size > 0


@pytest.mark.parametrize("kind", ["source", "sink"])
def test_permission_failure_warns_then_fails(client, caplog, tmp_path, kind):
    import os
    import sys

    import h5py

    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("chmod permission checks require a non-root POSIX user")
    path = tmp_path / "input.h5" if kind == "source" else tmp_path / "readonly" / "new" / "nested" / "out.h5"
    if kind == "source":
        with h5py.File(path, "w"):
            pass
        protected = path
        mode = 0
    else:
        protected = tmp_path / "readonly"
        protected.mkdir()
        mode = 0o555
    protected.chmod(mode)
    try:
        _assert_unavailable(client, caplog, kind=kind, path=path, exception_type="PermissionError")
    finally:
        protected.chmod(0o700)


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("sources/patch", {"ref": "sample", "type": "hdf"}),
        ("sinks/patch", {"ref": "export", "type": "hdf"}),
        ("sample", {"type": "hdf"}),
    ],
)
def test_patch_and_sample_registration_return_warnings(client, tmp_path, endpoint, payload):
    path = tmp_path if endpoint == "sinks/patch" else tmp_path / "missing" / "data.h5"
    response = client.post(f"/v1/sessions/diagnostics/{endpoint}", json={**payload, "location": str(path)})
    assert response.status_code == 200
    assert response.json()["warnings"][0]["location"] == str(path)


def test_registration_warning_is_rechecked_at_run_time(client, tmp_path):
    import h5py

    path = tmp_path / "arriving.h5"
    response = client.put(
        "/v1/sessions/diagnostics/sources", json={"sources": [{"ref": "sample", "type": "hdf", "location": str(path)}]}
    )
    assert len(response.json()["warnings"]) == 1
    with h5py.File(path, "w"):
        pass
    assert client.post("/v1/sessions/diagnostics/process", json={"mode": "full"}).status_code == 200


@pytest.mark.parametrize(
    "kind,io_type",
    [
        ("source", "buffer"),
        ("source", "tiled"),
        ("source", "custom"),
        ("sink", "buffer"),
        ("sink", "tiled"),
        ("sink", "custom"),
    ],
)
def test_non_file_registration_has_no_location_warning(client, kind, io_type):
    response = client.put(
        f"/v1/sessions/diagnostics/{kind}s",
        json={f"{kind}s": [{"ref": "data", "type": io_type, "location": "not-a-file"}]},
    )
    assert response.status_code == 200
    assert response.json()["warnings"] == []


@pytest.mark.parametrize("kind", ["source", "sink"])
def test_backend_constructor_failure_keeps_cause_and_registration(client, monkeypatch, kind):
    def fail(*args, **kwargs):
        raise ConnectionError("backend unavailable")

    monkeypatch.setattr(f"modacor.server.io_utils.build_{kind}_from_spec", fail)
    response = client.put(
        f"/v1/sessions/diagnostics/{kind}s",
        json={f"{kind}s": [{"ref": "data", "type": "custom", "location": "backend://offline"}]},
    )
    assert response.status_code == 200
    assert response.json()["warnings"] == []
    response = client.post("/v1/sessions/diagnostics/process", json={"mode": "full"})
    detail = response.json()["detail"]
    assert detail["code"] == f"{kind.upper()}_UNAVAILABLE"
    assert detail["details"]["exception_type"] == "ConnectionError"
    assert "data" in detail["message"]
    assert "backend://offline" in detail["message"]
    assert "ConnectionError: backend unavailable" in detail["message"]
