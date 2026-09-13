# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any, Protocol
from urllib import error, parse, request

import numpy as np

from modacor.io.buffer.codec import decode_npy, encode_npy

__all__ = [
    "BufferClient",
    "ChunkedOutputHandle",
    "ChunkedOutputsClient",
    "RuntimeAPIError",
    "RuntimeClient",
    "SessionClient",
]


class RuntimeAPIError(RuntimeError):
    """A structured runtime-service or transport failure."""

    def __init__(
        self,
        *,
        status: int | None,
        code: str | None,
        message: str,
        details: Any = None,
        endpoint: str,
        method: str,
    ) -> None:
        self.status = status
        self.status_code = status
        self.code = code
        self.message = str(message)
        self.details = details
        self.endpoint = endpoint
        self.method = method.upper()
        status_text = "transport error" if status is None else f"HTTP {status}"
        code_text = f" [{code}]" if code else ""
        super().__init__(f"{self.method} {endpoint}: {status_text}{code_text}: {self.message}")


class RuntimeTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        raise NotImplementedError


class _UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        http_request = request.Request(url, method=method.upper(), data=data, headers=dict(headers))
        try:
            with request.urlopen(http_request, timeout=timeout) as response:  # noqa: S310
                return response.status, response.read(), dict(response.headers.items())
        except error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers.items())
        except (error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeAPIError(
                status=None,
                code="TRANSPORT_ERROR",
                message=str(reason),
                endpoint=url,
                method=method,
            ) from exc


def _as_json_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError(f"{name} must be a mapping or provide to_dict().")


def _error_fields(payload: Any, fallback: str) -> tuple[str | None, str, Any]:
    detail = payload.get("detail", payload) if isinstance(payload, Mapping) else payload
    if isinstance(detail, Mapping):
        code = detail.get("code")
        message = detail.get("message") or detail.get("detail") or fallback
        details = detail.get("details")
        if details is None:
            details = {key: value for key, value in detail.items() if key not in {"code", "message", "detail"}}
            if not details:
                details = None
        return None if code is None else str(code), str(message), details
    if detail not in (None, ""):
        return None, str(detail), None
    return None, fallback, None


class RuntimeClient:
    """Small synchronous façade over the MoDaCor runtime HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        transport: RuntimeTransport | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must be non-empty.")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        self._transport = transport or _UrllibTransport()
        self.chunked_outputs = ChunkedOutputsClient(self)

    def url(self, path: str) -> str:
        return self.base_url + "/" + str(path).lstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        data: bytes | None = None,
        content_type: str | None = None,
        timeout: float | None = None,
        query: Mapping[str, Any] | None = None,
        expected: Sequence[int] = (200, 201, 202, 204),
    ) -> Any:
        if payload is not None and data is not None:
            raise ValueError("Provide payload or data, not both.")
        endpoint = "/" + str(path).lstrip("/")
        if query:
            endpoint += "?" + parse.urlencode({key: value for key, value in query.items() if value is not None})
        headers: dict[str, str] = {"Accept": "application/json"}
        body = data
        if payload is not None:
            body = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif content_type is not None:
            headers["Content-Type"] = content_type

        try:
            status, raw, response_headers = self._transport.request(
                method,
                self.url(endpoint),
                data=body,
                headers=headers,
                timeout=self.timeout if timeout is None else float(timeout),
            )
        except RuntimeAPIError as exc:
            raise RuntimeAPIError(
                status=exc.status,
                code=exc.code,
                message=exc.message,
                details=exc.details,
                endpoint=endpoint,
                method=method,
            ) from exc
        decoded: Any = None
        if raw:
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = raw
        if status not in expected:
            fallback = raw.decode("utf-8", errors="replace") if raw else "Request failed."
            code, message, details = _error_fields(decoded, fallback)
            raise RuntimeAPIError(
                status=status,
                code=code,
                message=message,
                details=details,
                endpoint=endpoint,
                method=method,
            )
        content_type_header = next(
            (str(value) for key, value in response_headers.items() if key.lower() == "content-type"),
            "",
        )
        if raw and "application/x-npy" in content_type_header:
            return decode_npy(raw)
        return decoded

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/v1/health")

    def readiness(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self.request("GET", "/v1/readiness", timeout=timeout)

    def is_ready(self, *, timeout: float = 1.0) -> bool:
        try:
            return bool(self.readiness(timeout=timeout).get("ready", False))
        except RuntimeAPIError:
            return False

    def wait_until_ready(self, *, timeout: float = 45.0, interval: float = 0.25) -> dict[str, Any]:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            try:
                readiness = self.readiness(timeout=min(1.0, max(0.1, deadline - monotonic())))
            except RuntimeAPIError:
                readiness = None
            if readiness and readiness.get("ready"):
                return readiness
            sleep(min(interval, max(0.0, deadline - monotonic())))
        raise TimeoutError(f"MoDaCor runtime did not become ready within {timeout:g} seconds at {self.base_url}.")

    def list_sessions(self) -> list[dict[str, Any]]:
        return list(self.request("GET", "/v1/sessions").get("sessions", []))

    def session(self, session_id: str) -> SessionClient:
        return SessionClient(self, str(session_id))

    def create_session(
        self,
        session_id: str,
        *,
        pipeline_yaml: str | None = None,
        pipeline_yaml_path: str | None = None,
        name: str | None = None,
        trace: Mapping[str, Any] | None = None,
        source_profile: str | None = None,
        auto_full_reset_on_partial_error: bool = True,
    ) -> SessionClient:
        if bool(pipeline_yaml) == bool(pipeline_yaml_path):
            raise ValueError("Provide exactly one of pipeline_yaml or pipeline_yaml_path.")
        pipeline = {"yaml_text": pipeline_yaml} if pipeline_yaml is not None else {"yaml_path": pipeline_yaml_path}
        payload: dict[str, Any] = {
            "session_id": str(session_id),
            "pipeline": pipeline,
            "auto_full_reset_on_partial_error": bool(auto_full_reset_on_partial_error),
        }
        if name is not None:
            payload["name"] = name
        if trace is not None:
            payload["trace"] = dict(trace)
        if source_profile is not None:
            payload["source_profile"] = source_profile
        response = self.request("POST", "/v1/sessions", payload=payload)
        return SessionClient(self, str(session_id), detail=response)

    def replace_session(self, session_id: str, **kwargs: Any) -> SessionClient:
        try:
            self.session(session_id).delete()
        except RuntimeAPIError as exc:
            if exc.status != 404:
                raise
        return self.create_session(session_id, **kwargs)


@dataclass(slots=True)
class SessionClient:
    """Operations scoped to one runtime session."""

    runtime: RuntimeClient
    session_id: str
    detail: dict[str, Any] | None = None

    @property
    def _path(self) -> str:
        return f"/v1/sessions/{parse.quote(self.session_id, safe='')}"

    def inspect(self) -> dict[str, Any]:
        self.detail = self.runtime.request("GET", self._path)
        return self.detail

    def delete(self) -> None:
        self.runtime.request("DELETE", self._path)

    def register_sources(self, *sources: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("PUT", f"{self._path}/sources", payload={"sources": list(sources)})

    def register_sinks(self, *sinks: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("PUT", f"{self._path}/sinks", payload={"sinks": list(sinks)})

    def register_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/sources/patch", payload=source)

    def register_sink(self, sink: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/sinks/patch", payload=sink)

    def set_sample(self, location: str, *, source_type: str = "hdf", kwargs: Mapping[str, Any] | None = None):
        return self.runtime.request(
            "POST",
            f"{self._path}/sample",
            payload={"location": str(location), "type": source_type, "kwargs": dict(kwargs or {})},
        )

    def delete_source(self, ref: str) -> None:
        self.runtime.request("DELETE", f"{self._path}/sources/{parse.quote(str(ref), safe='')}")

    def delete_sink(self, ref: str) -> None:
        self.runtime.request("DELETE", f"{self._path}/sinks/{parse.quote(str(ref), safe='')}")

    def source_buffer(self, source_ref: str) -> BufferClient:
        return BufferClient(self, str(source_ref), kind="source")

    def sink_buffer(self, sink_ref: str) -> BufferClient:
        return BufferClient(self, str(sink_ref), kind="sink")

    def process(
        self,
        *,
        mode: str = "auto",
        changed_sources: Sequence[str] = (),
        changed_keys: Sequence[str] = (),
        write_hdf: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        rollback_snapshot: bool = True,
        chunk_output: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode, "rollback_snapshot": bool(rollback_snapshot)}
        if changed_sources:
            payload["changed_sources"] = list(changed_sources)
        if changed_keys:
            payload["changed_keys"] = list(changed_keys)
        if write_hdf is not None:
            payload["write_hdf"] = dict(write_hdf)
        if run_name is not None:
            payload["run_name"] = run_name
        if chunk_output is not None:
            payload["chunk_output"] = dict(chunk_output)
        return self.runtime.request("POST", f"{self._path}/process", payload=payload)

    def dry_run(
        self,
        *,
        mode: str = "auto",
        changed_sources: Sequence[str] = (),
        changed_keys: Sequence[str] = (),
    ) -> dict[str, Any]:
        payload = {"mode": mode, "changed_sources": list(changed_sources), "changed_keys": list(changed_keys)}
        return self.runtime.request("POST", f"{self._path}/process/dry-run", payload=payload)

    def reset(self, *, mode: str = "full") -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/reset", payload={"mode": mode})

    def recover(self, *, strategy: str, **options: Any) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/recover", payload={"strategy": strategy, **options})

    def runs(self) -> list[dict[str, Any]]:
        return list(self.runtime.request("GET", f"{self._path}/runs").get("runs", []))

    def run(self, run_id: str) -> dict[str, Any]:
        return self.runtime.request("GET", f"{self._path}/runs/{parse.quote(str(run_id), safe='')}")

    def latest_error(self) -> dict[str, Any]:
        return self.runtime.request("GET", f"{self._path}/errors/latest")

    def plot_url(self, sink_ref: str, plot_id: str) -> str:
        sink = parse.quote(str(sink_ref), safe="")
        plot = parse.quote(str(plot_id), safe="")
        return self.runtime.url(f"{self._path}/plots/{sink}/{plot}")


@dataclass(slots=True)
class BufferClient:
    """Array and metadata transfer for one runtime buffer."""

    session: SessionClient
    ref: str
    kind: str = "source"

    def __post_init__(self) -> None:
        if self.kind not in {"source", "sink"}:
            raise ValueError("kind must be 'source' or 'sink'.")

    def _data_path(self, collection: str, data_key: str) -> str:
        key = parse.quote(str(data_key).strip("/"), safe="/")
        ref = parse.quote(self.ref, safe="")
        return f"{self.session._path}/buffers/{self.kind}s/{ref}/{collection}/{key}"

    def put_array(self, data_key: str, values: Any) -> dict[str, Any]:
        if self.kind != "source":
            raise ValueError("Arrays can only be uploaded to source buffers.")
        return self.session.runtime.request(
            "PUT",
            self._data_path("arrays", data_key),
            data=encode_npy(np.asarray(values)),
            content_type="application/x-npy",
        )

    def get_array(self, data_key: str) -> np.ndarray:
        if self.kind != "sink":
            raise ValueError("Arrays can only be downloaded from sink buffers.")
        return self.session.runtime.request("GET", self._data_path("arrays", data_key))

    def put_attrs(self, data_key: str, attrs: Mapping[str, Any]) -> dict[str, Any]:
        if self.kind != "source":
            raise ValueError("Attributes can only be uploaded to source buffers.")
        return self.session.runtime.request("PUT", self._data_path("attrs", data_key), payload=attrs)

    def put_metadata(self, data_key: str, value: Any) -> dict[str, Any]:
        if self.kind != "source":
            raise ValueError("Metadata can only be uploaded to source buffers.")
        return self.session.runtime.request("PUT", self._data_path("metadata", data_key), payload={"value": value})

    def manifest(self) -> dict[str, Any]:
        ref = parse.quote(self.ref, safe="")
        return self.session.runtime.request("GET", f"{self.session._path}/buffers/{self.kind}/{ref}/manifest")


class ChunkedOutputsClient:
    def __init__(self, runtime: RuntimeClient) -> None:
        self.runtime = runtime

    def create(
        self,
        *,
        sink: Mapping[str, Any] | None = None,
        session: SessionClient | None = None,
        sink_ref: str | None = None,
        subpath: str,
        plan: Any,
        collision: str = "error",
    ) -> ChunkedOutputHandle:
        payload = self._destination(sink=sink, session=session, sink_ref=sink_ref)
        plan_dict = _as_json_dict(plan, "plan")
        payload.update({"subpath": subpath, "plan": plan_dict, "collision": collision})
        response = self.runtime.request("POST", "/v1/chunked-outputs", payload=payload)
        return ChunkedOutputHandle(self.runtime, response, plan_hash=plan_dict.get("plan_hash"))

    def create_provisional(
        self,
        *,
        session: SessionClient,
        subpath: str,
        plan: Any,
        sink: Mapping[str, Any] | None = None,
        sink_ref: str | None = None,
        collision: str = "error",
    ) -> ChunkedOutputHandle:
        payload = self._destination(sink=sink, session=session, sink_ref=sink_ref)
        plan_dict = _as_json_dict(plan, "plan")
        payload.update({"subpath": subpath, "provisional_plan": plan_dict, "collision": collision})
        response = self.runtime.request("POST", "/v1/chunked-outputs", payload=payload)
        return ChunkedOutputHandle(self.runtime, response)

    def reopen(
        self,
        *,
        plan_id: str,
        sink: Mapping[str, Any] | None = None,
        session: SessionClient | None = None,
        sink_ref: str | None = None,
        plan_hash: str | None = None,
    ) -> ChunkedOutputHandle:
        payload = self._destination(sink=sink, session=session, sink_ref=sink_ref)
        payload["plan_id"] = plan_id
        if plan_hash is not None:
            payload["plan_hash"] = plan_hash
        response = self.runtime.request("POST", "/v1/chunked-outputs/reopen", payload=payload)
        return ChunkedOutputHandle(self.runtime, response, plan_hash=plan_hash or response.get("plan_hash"))

    @staticmethod
    def _destination(
        *,
        sink: Mapping[str, Any] | None,
        session: SessionClient | None,
        sink_ref: str | None,
    ) -> dict[str, Any]:
        if sink is not None:
            if session is not None or sink_ref is not None:
                raise ValueError("Provide sink or session and sink_ref, not both.")
            return {"sink": dict(sink)}
        if session is None or sink_ref is None:
            raise ValueError("Provide sink or both session and sink_ref.")
        return {"session_id": session.session_id, "sink_ref": sink_ref}


@dataclass(slots=True)
class ChunkedOutputHandle:
    """Client-side handle for one server-owned chunked output lifecycle."""

    runtime: RuntimeClient
    creation: dict[str, Any]
    plan_hash: str | None = None

    @property
    def output_id(self) -> str:
        return str(self.creation["output_id"])

    def chunk(self, spec: Any) -> dict[str, Any]:
        if isinstance(spec, str):
            return {"output_id": self.output_id, "chunk_id": spec}
        return {"output_id": self.output_id, "chunk_spec": _as_json_dict(spec, "chunk spec")}

    def inspect(self, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        return self.runtime.request(
            "GET",
            f"/v1/chunked-outputs/{parse.quote(self.output_id, safe='')}",
            query={"offset": offset, "limit": limit},
        )

    def finalize(self, *, plan_hash: str | None = None) -> dict[str, Any]:
        active_hash = plan_hash or self.plan_hash
        if active_hash is None:
            active_hash = self.inspect(limit=1).get("plan_hash")
        if not active_hash:
            raise ValueError("The resolved plan hash is not available yet.")
        return self.runtime.request(
            "POST",
            f"/v1/chunked-outputs/{parse.quote(self.output_id, safe='')}/finalize",
            payload={"plan_hash": active_hash},
        )

    def recover(self, action: str) -> dict[str, Any]:
        return self.runtime.request(
            "POST",
            f"/v1/chunked-outputs/{parse.quote(self.output_id, safe='')}/recover",
            payload={"action": action},
        )

    def detach(self) -> None:
        self.runtime.request("DELETE", f"/v1/chunked-outputs/{parse.quote(self.output_id, safe='')}")
