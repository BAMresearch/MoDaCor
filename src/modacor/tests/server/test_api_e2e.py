# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import RunResult
from modacor.server.api import create_app
from modacor.server.session_manager import SessionManager

fastapi = pytest.importorskip("fastapi")
testclient_mod = pytest.importorskip("fastapi.testclient")
TestClient = testclient_mod.TestClient


def _post_json(client: TestClient, url: str, payload: dict):
    response = client.post(url, json=payload)
    assert response.status_code in {200, 201, 202}, response.text
    return response.json()


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


def test_api_process_auto_fallback_after_partial_failure(monkeypatch):
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

    call_count = {"n": 0}

    def fake_run_pipeline_job(*args, **kwargs):
        call_count["n"] += 1
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

    monkeypatch.setattr("modacor.server.api.run_pipeline_job", fake_run_pipeline_job)

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
