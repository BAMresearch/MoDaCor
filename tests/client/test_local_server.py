# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from modacor.client import LocalRuntimeServer


def test_local_server_reuses_ready_process(monkeypatch):
    server = LocalRuntimeServer(port=8123)
    monkeypatch.setattr(server.client, "is_ready", lambda **kwargs: True)

    client = server.start()

    assert client is server.client
    assert server.launched is False


def test_local_server_stops_only_process_it_launched(monkeypatch):
    events = []
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: events.append("terminate"),
        wait=lambda timeout: events.append(("wait", timeout)),
        kill=lambda: events.append("kill"),
    )
    server = LocalRuntimeServer(shutdown_timeout=3)
    server.process = process

    server.stop()

    assert events == ["terminate", ("wait", 3.0)]
    assert server.launched is False


def test_local_server_validates_configuration():
    with pytest.raises(ValueError, match="port"):
        LocalRuntimeServer(port=0)
    with pytest.raises(ValueError, match="timeouts"):
        LocalRuntimeServer(startup_timeout=0)


def test_local_server_command_contains_roots_and_limits(tmp_path):
    server = LocalRuntimeServer(
        host="127.0.0.2",
        port=8123,
        read_roots=[tmp_path / "read"],
        write_roots=[tmp_path / "write"],
        runtime_policy="restricted",
        max_sessions=2,
        max_pipeline_yaml_bytes=100,
        max_buffer_upload_bytes=200,
    )

    assert server._command()[-10:] == [
        "--read-root",
        str(tmp_path / "read"),
        "--write-root",
        str(tmp_path / "write"),
        "--max-sessions",
        "2",
        "--max-pipeline-yaml-bytes",
        "100",
        "--max-buffer-upload-bytes",
        "200",
    ]


def test_local_server_kills_process_that_ignores_terminate():
    events = []

    def wait(timeout):
        events.append(("wait", timeout))
        if events.count(("wait", timeout)) == 1:
            raise subprocess.TimeoutExpired("runtime", timeout)

    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: events.append("terminate"),
        wait=wait,
        kill=lambda: events.append("kill"),
    )
    server = LocalRuntimeServer(shutdown_timeout=3)
    server.process = process

    server.stop()

    assert events == ["terminate", ("wait", 3.0), "kill", ("wait", 3.0)]


def test_local_server_closes_log_when_process_launch_fails(monkeypatch, tmp_path):
    server = LocalRuntimeServer(log_path=tmp_path / "runtime.log")
    monkeypatch.setattr(server.client, "is_ready", lambda **kwargs: False)

    def fail_launch(*args, **kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr(subprocess, "Popen", fail_launch)

    with pytest.raises(OSError, match="cannot launch"):
        server.start()

    assert server.process is None
    assert server._log_file is None


def test_local_server_discards_stale_process_before_reusing_ready_server(monkeypatch, tmp_path):
    server = LocalRuntimeServer(log_path=tmp_path / "runtime.log")
    server.process = SimpleNamespace(poll=lambda: 1)
    server._open_log()
    monkeypatch.setattr(server.client, "is_ready", lambda **kwargs: True)

    client = server.start()

    assert client is server.client
    assert server.process is None
    assert server._log_file is None


def test_local_server_stops_owned_process_when_readiness_wait_fails(monkeypatch):
    events = []
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: events.append("terminate"),
        wait=lambda timeout: events.append(("wait", timeout)),
        kill=lambda: events.append("kill"),
    )
    server = LocalRuntimeServer(shutdown_timeout=2)
    server.process = process
    monkeypatch.setattr(server.client, "is_ready", lambda **kwargs: False)

    def fail_wait(**kwargs):
        raise TimeoutError("not ready")

    monkeypatch.setattr(server.client, "wait_until_ready", fail_wait)

    with pytest.raises(TimeoutError, match="not ready"):
        server.start()

    assert events == ["terminate", ("wait", 2.0)]
    assert server.process is None
