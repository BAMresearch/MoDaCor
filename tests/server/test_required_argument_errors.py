# SPDX-License-Identifier: BSD-3-Clause

import pytest

from modacor.server.api import create_app

pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

INVALID_YAML = "steps: {units: {module: UnitsLabelUpdate}}"


def test_required_arguments_use_existing_registration_and_run_errors(caplog):
    with TestClient(create_app()) as client:
        response = client.post("/v1/sessions", json={"session_id": "invalid", "pipeline": {"yaml_text": INVALID_YAML}})
        assert response.status_code == 409
        message = response.json()["detail"]
        assert "UnitsLabelUpdate (units)" in message and "update_pairs" in message
        assert message in caplog.text
        assert client.app.state.runtime_service.manager.get_session("invalid") is None

        # A stored session can predate enforcement or be modified after registration.
        response = client.post("/v1/sessions", json={"session_id": "stored", "pipeline": {"yaml_text": "steps: {}"}})
        assert response.status_code == 200
        client.app.state.runtime_service.manager.get_session("stored").pipeline_yaml = INVALID_YAML
        response = client.post("/v1/sessions/stored/process", json={"mode": "full"})
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "RUN_FAILED"
        assert response.json()["detail"]["message"] == message
        latest = client.get("/v1/sessions/stored/errors/latest").json()["latest_error"]
        assert latest["code"] == "RUN_FAILED"
        assert message in latest["details"]["traceback"]
