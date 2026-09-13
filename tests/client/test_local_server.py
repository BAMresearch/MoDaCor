# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

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
