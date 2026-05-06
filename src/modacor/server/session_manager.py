# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

__all__ = ["PipelineSession", "SessionManager"]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _normalize_io_registration(registration: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if not isinstance(registration, dict):
        raise ValueError(f"{kind} registration must be an object.")

    missing = [key for key in ("ref", "type", "location") if key not in registration]
    if missing:
        raise ValueError(f"{kind} registration missing required field(s): {', '.join(missing)}.")

    ref = str(registration["ref"]).strip()
    reg_type = str(registration["type"]).strip()
    location = str(registration["location"]).strip()
    kwargs = registration.get("kwargs", {}) or {}

    if not ref:
        raise ValueError(f"{kind} registration ref must be non-empty.")
    if not reg_type:
        raise ValueError(f"{kind} registration type must be non-empty.")
    if not location:
        raise ValueError(f"{kind} registration location must be non-empty.")
    if not isinstance(kwargs, dict):
        raise ValueError(f"{kind} registration kwargs must be an object when provided.")

    return {
        "ref": ref,
        "type": reg_type,
        "location": location,
        "kwargs": dict(kwargs),
    }


@dataclass(slots=True)
class PipelineSession:
    session_id: str
    name: str | None = None
    pipeline_yaml: str | None = None
    trace_enabled: bool = False
    trace_watch: dict[str, list[str]] = field(default_factory=dict)
    auto_full_reset_on_partial_error: bool = True
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    sinks: dict[str, dict[str, Any]] = field(default_factory=dict)
    state: str = "idle"
    active_run_id: str | None = None
    updated_utc: str = field(default_factory=_utc_now_iso)
    run_history: list[dict[str, Any]] = field(default_factory=list)
    processing_data: Any | None = None
    last_error: dict[str, Any] | None = None
    source_profile: str | None = None
    required_source_refs: list[str] = field(default_factory=list)


class SessionManager:
    """In-memory session registry for the runtime API scaffold."""

    def __init__(self) -> None:
        self._sessions: dict[str, PipelineSession] = {}
        self._lock = RLock()

    def list_sessions(self) -> list[PipelineSession]:
        with self._lock:
            return list(self._sessions.values())

    def get_session(self, session_id: str) -> PipelineSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def create_session(
        self,
        *,
        session_id: str,
        name: str | None = None,
        pipeline_yaml: str | None = None,
        trace_enabled: bool = False,
        trace_watch: dict[str, list[str]] | None = None,
        auto_full_reset_on_partial_error: bool = True,
        source_profile: str | None = None,
        required_source_refs: list[str] | None = None,
    ) -> PipelineSession:
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session '{session_id}' already exists.")
            session = PipelineSession(
                session_id=session_id,
                name=name,
                pipeline_yaml=pipeline_yaml,
                trace_enabled=trace_enabled,
                trace_watch=trace_watch or {},
                auto_full_reset_on_partial_error=auto_full_reset_on_partial_error,
                source_profile=source_profile,
                required_source_refs=list(required_source_refs or []),
            )
            self._sessions[session_id] = session
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def upsert_sources(self, session_id: str, sources: list[dict[str, Any]]) -> PipelineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            for source in sources:
                normalized = _normalize_io_registration(source, kind="Source")
                session.sources[normalized["ref"]] = normalized
            session.updated_utc = _utc_now_iso()
            return session

    def upsert_sinks(self, session_id: str, sinks: list[dict[str, Any]]) -> PipelineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            for sink in sinks:
                normalized = _normalize_io_registration(sink, kind="Sink")
                session.sinks[normalized["ref"]] = normalized
            session.updated_utc = _utc_now_iso()
            return session

    def delete_source(self, session_id: str, ref: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            existed = ref in session.sources
            session.sources.pop(ref, None)
            session.updated_utc = _utc_now_iso()
            return existed

    def delete_sink(self, session_id: str, ref: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            existed = ref in session.sinks
            session.sinks.pop(ref, None)
            session.updated_utc = _utc_now_iso()
            return existed

    def enqueue_run(
        self,
        session_id: str,
        *,
        mode: str,
        changed_sources: list[str] | None = None,
        effective_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            if session.active_run_id is not None:
                raise RuntimeError(f"Session '{session_id}' is busy.")

            run_id = f"run-{uuid4().hex[:12]}"
            run_mode = effective_mode or mode
            state = "running_partial" if run_mode == "partial" else "running_full"
            session.state = state
            session.active_run_id = run_id
            session.updated_utc = _utc_now_iso()
            session.last_error = None

            run_meta = {
                "run_id": run_id,
                "mode": mode,
                "effective_mode": run_mode,
                "status": "queued",
                "changed_sources": changed_sources or [],
                "started_utc": _utc_now_iso(),
                "finished_utc": None,
                "failed_step_id": None,
            }
            session.run_history.append(run_meta)
            return run_meta

    def mark_run_succeeded(
        self, session_id: str, run_id: str, *, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")

            run_meta = next((item for item in session.run_history if item.get("run_id") == run_id), None)
            if run_meta is None:
                raise KeyError(f"Run '{run_id}' not found in session '{session_id}'.")

            run_meta["status"] = "succeeded"
            run_meta["finished_utc"] = _utc_now_iso()
            if details:
                run_meta.update(details)

            session.active_run_id = None
            session.state = "idle"
            session.updated_utc = _utc_now_iso()
            session.last_error = None
            return run_meta

    def mark_run_failed(
        self,
        session_id: str,
        run_id: str,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")

            run_meta = next((item for item in session.run_history if item.get("run_id") == run_id), None)
            if run_meta is None:
                raise KeyError(f"Run '{run_id}' not found in session '{session_id}'.")

            run_meta["status"] = "failed"
            run_meta["finished_utc"] = _utc_now_iso()
            if details:
                run_meta.update(details)

            failed_mode = run_meta.get("effective_mode") or run_meta.get("mode")
            error_payload = {
                "code": code,
                "message": message,
                "details": details or {},
                "run_id": run_id,
                "recorded_utc": run_meta["finished_utc"],
                "effective_mode": failed_mode,
            }
            run_meta["error"] = error_payload
            session.last_error = error_payload
            session.active_run_id = None
            session.state = "error_partial" if failed_mode == "partial" else "error_full"
            session.updated_utc = _utc_now_iso()
            return run_meta

    def reset_session(self, session_id: str, *, mode: str) -> PipelineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            session.active_run_id = None
            session.state = "idle"
            if mode == "full":
                session.processing_data = None
                session.last_error = None
            session.updated_utc = _utc_now_iso()
            session.run_history.append(
                {
                    "run_id": f"reset-{uuid4().hex[:10]}",
                    "mode": mode,
                    "status": "queued",
                    "started_utc": _utc_now_iso(),
                }
            )
            return session
