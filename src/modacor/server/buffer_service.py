# SPDX-License-Identifier: BSD-3-Clause
"""BufferService implementation for the runtime API."""

from __future__ import annotations

__all__ = ["BufferService"]

import logging
from dataclasses import dataclass
from typing import Any

from modacor.io.buffer.codec import decode_npy, encode_npy

from .errors import ApiError
from .runtime_policy import RuntimePolicy
from .session_service import SessionService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BufferService(SessionService):
    """BufferService operations for the runtime facade."""

    policy: RuntimePolicy

    def put_buffer_source_array(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: bytes,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        try:
            self.policy.validate_buffer_upload(payload)
        except ValueError as exc:
            raise ApiError(status_code=413, detail=str(exc)) from exc
        try:
            array = decode_npy(payload)
            self.manager.buffer_store.put_array(session_id, "source", source_ref, data_key, array)
        except Exception as exc:
            # API boundary: arbitrary decoder/store failures retain the existing 422 response.
            logger.warning("Invalid .npy buffer payload", exc_info=True, extra={"session_id": session_id})
            raise ApiError(status_code=422, detail=f"Invalid .npy buffer payload: {exc}") from exc
        return {
            "session_id": session_id,
            "kind": "source",
            "ref": source_ref,
            "data_key": data_key.strip("/"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    def put_buffer_source_attrs(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_session(session_id)
        if not isinstance(payload, dict):
            raise ApiError(status_code=422, detail="Buffer attrs payload must be an object.")
        self.manager.buffer_store.put_attrs(session_id, "source", source_ref, data_key, payload)
        return {
            "session_id": session_id,
            "kind": "source",
            "ref": source_ref,
            "data_key": data_key.strip("/"),
            "attrs": dict(payload),
        }

    def put_buffer_source_metadata(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_session(session_id)
        if not isinstance(payload, dict) or "value" not in payload:
            raise ApiError(status_code=422, detail="Buffer metadata payload must be an object containing 'value'.")
        self.manager.buffer_store.put_metadata(session_id, "source", source_ref, data_key, payload["value"])
        return {
            "session_id": session_id,
            "kind": "source",
            "ref": source_ref,
            "data_key": data_key.strip("/"),
            "value": payload["value"],
        }

    def get_buffer_sink_array(self, session_id: str, sink_ref: str, data_key: str) -> bytes:
        self._require_session(session_id)
        try:
            array = self.manager.buffer_store.get_array(session_id, "sink", sink_ref, data_key)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        return encode_npy(array)

    def get_buffer_manifest(self, session_id: str, kind: str, ref: str) -> dict[str, Any]:
        self._require_session(session_id)
        try:
            return self.manager.buffer_store.manifest(session_id, kind, ref)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

    def get_plot_payload(self, session_id: str, sink_ref: str, plot_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        try:
            plot = self.manager.buffer_store.get_metadata(session_id, "sink", sink_ref, plot_id)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not isinstance(plot, dict):
            raise ApiError(status_code=500, detail=f"Stored plot payload {sink_ref!r}/{plot_id!r} is not an object.")
        return {
            "session_id": session_id,
            "sink_ref": sink_ref,
            "plot_id": plot_id,
            "state": session.state,
            "updated_utc": session.updated_utc,
            "latest_run": session.run_history[-1] if session.run_history else None,
            "plot": plot,
        }

    def clear_buffers(
        self,
        session_id: str,
        *,
        kind: str | None = None,
        ref: str | None = None,
        data_key: str | None = None,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        try:
            removed = self.manager.buffer_store.clear(session_id, kind=kind, ref=ref, data_key=data_key)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        return {"session_id": session_id, "removed": removed}

    def clear_buffer_sink(self, session_id: str, sink_ref: str) -> dict[str, Any]:
        return self.clear_buffers(session_id, kind="sink", ref=sink_ref)

    def clear_buffer_sink_array(self, session_id: str, sink_ref: str, data_key: str) -> dict[str, Any]:
        return self.clear_buffers(session_id, kind="sink", ref=sink_ref, data_key=data_key)
