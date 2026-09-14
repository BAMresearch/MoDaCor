# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib import parse

from modacor.io.buffer.codec import decode_npy

from .buffer import BufferClient
from .chunked import ChunkedOutputHandle, ChunkedOutputsClient
from .errors import RuntimeAPIError, error_fields
from .session import SessionClient
from .transport import RuntimeTransport, UrllibRuntimeTransport

__all__ = [
    "BufferClient",
    "ChunkedOutputHandle",
    "ChunkedOutputsClient",
    "RuntimeAPIError",
    "RuntimeClient",
    "RuntimeTransport",
    "SessionClient",
]


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
        self._transport = transport or UrllibRuntimeTransport()
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
        endpoint = self._endpoint(path, query)
        body, headers = self._request_body(payload=payload, data=data, content_type=content_type)
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
        decoded = self._decode_body(raw)
        if status not in expected:
            fallback = raw.decode("utf-8", errors="replace") if raw else "Request failed."
            code, message, details = error_fields(decoded, fallback)
            raise RuntimeAPIError(
                status=status,
                code=code,
                message=message,
                details=details,
                endpoint=endpoint,
                method=method,
            )
        if raw and "application/x-npy" in self._content_type(response_headers):
            return decode_npy(raw)
        return decoded

    @staticmethod
    def _endpoint(path: str, query: Mapping[str, Any] | None) -> str:
        endpoint = "/" + str(path).lstrip("/")
        if query:
            endpoint += "?" + parse.urlencode({key: value for key, value in query.items() if value is not None})
        return endpoint

    @staticmethod
    def _request_body(
        *,
        payload: Mapping[str, Any] | None,
        data: bytes | None,
        content_type: str | None,
    ) -> tuple[bytes | None, dict[str, str]]:
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
            return json.dumps(dict(payload)).encode("utf-8"), headers
        if content_type is not None:
            headers["Content-Type"] = content_type
        return data, headers

    @staticmethod
    def _decode_body(raw: bytes) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw

    @staticmethod
    def _content_type(headers: Mapping[str, str]) -> str:
        return next((str(value) for key, value in headers.items() if key.lower() == "content-type"), "")

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
        pipeline_yaml_path: str | Path | None = None,
        name: str | None = None,
        trace: Mapping[str, Any] | None = None,
        source_profile: str | None = None,
        auto_full_reset_on_partial_error: bool = True,
    ) -> SessionClient:
        if bool(pipeline_yaml) == bool(pipeline_yaml_path):
            raise ValueError("Provide exactly one of pipeline_yaml or pipeline_yaml_path.")
        pipeline = {"yaml_text": pipeline_yaml} if pipeline_yaml is not None else {"yaml_path": str(pipeline_yaml_path)}
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
