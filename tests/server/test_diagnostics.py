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
