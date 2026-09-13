# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from modacor.client import RuntimeAPIError, RuntimeClient
from modacor.io.buffer.codec import encode_npy


@dataclass
class StubTransport:
    responses: list[tuple[int, bytes, dict[str, str]]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request(self, method, url, *, data, headers, timeout):
        self.calls.append({"method": method, "url": url, "data": data, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


def response(status: int, payload: Any = None, *, content_type: str = "application/json"):
    raw = b"" if payload is None else json.dumps(payload).encode()
    return status, raw, {"Content-Type": content_type}


def test_structured_api_error_exposes_runtime_fields():
    transport = StubTransport(
        [response(409, {"detail": {"code": "CONFLICT", "message": "already exists", "plan_id": "p1"}})]
    )
    client = RuntimeClient("http://runtime", transport=transport)

    with pytest.raises(RuntimeAPIError) as caught:
        client.request("POST", "/v1/chunked-outputs", payload={})

    assert caught.value.status == 409
    assert caught.value.status_code == 409
    assert caught.value.code == "CONFLICT"
    assert caught.value.message == "already exists"
    assert caught.value.details == {"plan_id": "p1"}
    assert caught.value.endpoint == "/v1/chunked-outputs"


def test_replace_session_ignores_missing_session_and_returns_scoped_client():
    transport = StubTransport(
        [
            response(404, {"detail": "Session not found."}),
            response(200, {"session_id": "demo", "state": "idle"}),
        ]
    )
    client = RuntimeClient("http://runtime/", transport=transport)

    session = client.replace_session("demo", pipeline_yaml="name: demo\nsteps: {}\n")

    assert session.session_id == "demo"
    assert session.detail["state"] == "idle"
    assert [call["method"] for call in transport.calls] == ["DELETE", "POST"]
    create_payload = json.loads(transport.calls[1]["data"])
    assert create_payload["pipeline"]["yaml_text"].startswith("name: demo")


def test_session_and_buffer_clients_build_payloads_and_decode_arrays():
    values = np.arange(6).reshape(2, 3)
    transport = StubTransport(
        [
            response(200, {"sources": []}),
            response(200, {"shape": [2, 3]}),
            (200, encode_npy(values), {"Content-Type": "application/x-npy"}),
        ]
    )
    session = RuntimeClient("http://runtime", transport=transport).session("a session")

    session.register_sources({"ref": "sample", "type": "buffer", "location": "buffer://session"})
    session.source_buffer("sample").put_array("/entry/data", values)
    actual = session.sink_buffer("result").get_array("/sample/signal")

    assert np.array_equal(actual, values)
    assert transport.calls[0]["url"].endswith("/v1/sessions/a%20session/sources")
    assert transport.calls[1]["headers"]["Content-Type"] == "application/x-npy"


def test_chunked_output_handle_wraps_create_process_and_finalize():
    transport = StubTransport(
        [
            response(201, {"output_id": "out-1", "status": "open"}),
            response(200, {"status": "complete", "completed_chunks": 1}),
        ]
    )
    client = RuntimeClient("http://runtime", transport=transport)
    plan = {"plan_id": "p1", "plan_hash": "hash-1", "outputs": [], "chunks": []}
    output = client.chunked_outputs.create(
        sink={"ref": "result", "type": "hdf_chunked", "location": "/tmp/out.h5"},
        subpath="run",
        plan=plan,
    )
    spec = {"chunk_id": "c0", "source_selection": [], "placements": []}

    chunk_payload = output.chunk(spec)
    finalized = output.finalize()

    assert chunk_payload["output_id"] == "out-1"
    assert chunk_payload["chunk_spec"]["chunk_id"] == "c0"
    assert finalized["status"] == "complete"
    assert json.loads(transport.calls[1]["data"]) == {"plan_hash": "hash-1"}
