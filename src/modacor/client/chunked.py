# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib import parse

if TYPE_CHECKING:
    from .runtime import RuntimeClient
    from .session import SessionClient

__all__ = ["ChunkedOutputHandle", "ChunkedOutputsClient"]


def _as_json_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError(f"{name} must be a mapping or provide to_dict().")


def _chunk_id(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping) and "chunk_id" in spec:
        return str(spec["chunk_id"])
    value = getattr(spec, "chunk_id", None)
    if value is None:
        raise TypeError("A provisional chunk must be a chunk id or expose chunk_id.")
    return str(value)


class ChunkedOutputsClient:
    """Create and reopen server-owned chunked output resources."""

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
        return ChunkedOutputHandle(self.runtime, response, provisional=True)

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
        provisional = response.get("status") == "awaiting_schema" or (
            "provisional_hash" in response and "plan_hash" not in response
        )
        return ChunkedOutputHandle(
            self.runtime,
            response,
            plan_hash=plan_hash or response.get("plan_hash"),
            provisional=provisional,
        )

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
    provisional: bool = False

    @property
    def output_id(self) -> str:
        return str(self.creation["output_id"])

    @property
    def _path(self) -> str:
        return f"/v1/chunked-outputs/{parse.quote(self.output_id, safe='')}"

    def chunk_id(self, chunk_id: str) -> dict[str, Any]:
        return {"output_id": self.output_id, "chunk_id": str(chunk_id)}

    def chunk_spec(self, spec: Any) -> dict[str, Any]:
        return {"output_id": self.output_id, "chunk_spec": _as_json_dict(spec, "chunk spec")}

    def chunk(self, spec: Any) -> dict[str, Any]:
        """Build a process payload for a complete or provisional chunk."""

        if self.provisional or isinstance(spec, str):
            return self.chunk_id(_chunk_id(spec))
        return self.chunk_spec(spec)

    def inspect(self, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        return self.runtime.request("GET", self._path, query={"offset": offset, "limit": limit})

    def finalize(self, *, plan_hash: str | None = None) -> dict[str, Any]:
        active_hash = plan_hash or self.plan_hash
        if active_hash is None:
            active_hash = self.inspect(limit=1).get("plan_hash")
        if not active_hash:
            raise ValueError("The resolved plan hash is not available yet.")
        self.plan_hash = str(active_hash)
        return self.runtime.request("POST", f"{self._path}/finalize", payload={"plan_hash": self.plan_hash})

    def recover(self, action: str) -> dict[str, Any]:
        return self.runtime.request("POST", f"{self._path}/recover", payload={"action": action})

    def detach(self) -> None:
        self.runtime.request("DELETE", self._path)
