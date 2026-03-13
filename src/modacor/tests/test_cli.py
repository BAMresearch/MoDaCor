# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest

from modacor.cli import build_parser, main


def _write_empty_pipeline(path: Path) -> None:
    path.write_text("name: empty\nsteps: {}\n", encoding="utf-8")


def test_cli_run_smoke(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yml"
    _write_empty_pipeline(pipeline_path)

    rc = main(["run", "--pipeline", str(pipeline_path)])

    assert rc == 0


def test_cli_write_hdf_requires_write_path(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yml"
    output_path = tmp_path / "out.h5"
    _write_empty_pipeline(pipeline_path)

    with pytest.raises(
        ValueError, match="--write-hdf requires --write-all-processing-data or at least one --write-path"
    ):
        main(
            [
                "run",
                "--pipeline",
                str(pipeline_path),
                "--write-hdf",
                str(output_path),
            ]
        )


def test_cli_serve_parser_accepts_host_port():
    parser = build_parser()
    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_cli_session_create_calls_api(monkeypatch):
    captured = {}

    def fake_http(base_url, method, path, payload=None):
        captured["base_url"] = base_url
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"session_id": "s1"}

    monkeypatch.setattr("modacor.cli._http_request_json", fake_http)
    rc = main(
        [
            "session",
            "--url",
            "http://127.0.0.1:8000",
            "create",
            "--session-id",
            "s1",
            "--pipeline-yaml-text",
            "name: demo\nsteps: {}\n",
            "--trace",
            "--trace-watch",
            "sample:signal",
        ]
    )
    assert rc == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/sessions"
    assert captured["payload"]["session_id"] == "s1"
    assert captured["payload"]["trace"]["enabled"] is True
    assert captured["payload"]["trace"]["watch"] == {"sample": ["signal"]}


def test_cli_session_create_with_source_template(monkeypatch):
    captured = {}

    def fake_http(base_url, method, path, payload=None):
        captured["payload"] = payload
        return {"session_id": "s2"}

    monkeypatch.setattr("modacor.cli._http_request_json", fake_http)
    rc = main(
        [
            "session",
            "create",
            "--session-id",
            "s2",
            "--pipeline-yaml-text",
            "name: demo\nsteps: {}\n",
            "--source-template",
            "mouse",
        ]
    )
    assert rc == 0
    assert captured["payload"]["source_profile"] == "mouse"


def test_cli_session_process_builds_write_hdf_payload(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_http(base_url, method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"run_id": "r1"}

    monkeypatch.setattr("modacor.cli._http_request_json", fake_http)
    out_path = tmp_path / "out.h5"
    rc = main(
        [
            "session",
            "process",
            "--session-id",
            "s1",
            "--mode",
            "partial",
            "--changed-source",
            "sample",
            "--changed-key",
            "sample.signal",
            "--write-hdf-path",
            str(out_path),
            "--write-all-processing-data",
        ]
    )
    assert rc == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/sessions/s1/process"
    assert captured["payload"]["changed_keys"] == ["sample.signal"]
    assert captured["payload"]["write_hdf"]["path"] == str(out_path)
    assert captured["payload"]["write_hdf"]["write_all_processing_data"] is True


def test_cli_session_templates_calls_api(monkeypatch):
    captured = {}

    def fake_http(base_url, method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        return {"templates": {"mouse": {}}}

    monkeypatch.setattr("modacor.cli._http_request_json", fake_http)
    rc = main(["session", "templates"])
    assert rc == 0
    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/source-templates"


def test_cli_session_dry_run_calls_api(monkeypatch):
    captured = {}

    def fake_http(base_url, method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"dirty_steps": ["p"]}

    monkeypatch.setattr("modacor.cli._http_request_json", fake_http)
    rc = main(
        [
            "session",
            "dry-run",
            "--session-id",
            "s1",
            "--mode",
            "partial",
            "--changed-key",
            "sample.signal",
        ]
    )
    assert rc == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/sessions/s1/process/dry-run"
    assert captured["payload"]["changed_keys"] == ["sample.signal"]
