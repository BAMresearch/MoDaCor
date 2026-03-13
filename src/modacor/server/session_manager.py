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


@dataclass(slots=True)
class PipelineSession:
    session_id: str
    name: str | None = None
    pipeline_yaml: str | None = None
    trace_enabled: bool = False
    trace_watch: dict[str, list[str]] = field(default_factory=dict)
    auto_full_reset_on_partial_error: bool = True
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    state: str = "idle"
    active_run_id: str | None = None
    updated_utc: str = field(default_factory=_utc_now_iso)
    run_history: list[dict[str, Any]] = field(default_factory=list)


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
                ref = str(source["ref"])
                session.sources[ref] = {
                    "ref": ref,
                    "type": str(source["type"]),
                    "location": str(source["location"]),
                    "kwargs": dict(source.get("kwargs", {})),
                }
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

    def enqueue_run(self, session_id: str, *, mode: str, changed_sources: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            if session.active_run_id is not None:
                raise RuntimeError(f"Session '{session_id}' is busy.")

            run_id = f"run-{uuid4().hex[:12]}"
            state = "running_partial" if mode == "partial" else "running_full"
            session.state = state
            session.active_run_id = run_id
            session.updated_utc = _utc_now_iso()

            run_meta = {
                "run_id": run_id,
                "mode": mode,
                "status": "queued",
                "changed_sources": changed_sources or [],
                "started_utc": _utc_now_iso(),
            }
            session.run_history.append(run_meta)
            return run_meta

    def reset_session(self, session_id: str, *, mode: str) -> PipelineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' not found.")
            session.active_run_id = None
            session.state = "idle"
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
