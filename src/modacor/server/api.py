# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

from .session_manager import PipelineSession, SessionManager

__all__ = ["create_app"]


def _session_summary(session: PipelineSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "name": session.name,
        "state": session.state,
        "active_run_id": session.active_run_id,
        "updated_utc": session.updated_utc,
    }


def _session_detail(session: PipelineSession) -> dict[str, Any]:
    out = _session_summary(session)
    out.update(
        {
            "sources": list(session.sources.values()),
            "trace": {
                "enabled": session.trace_enabled,
                "watch": session.trace_watch,
                "record_only_on_change": True,
            },
            "last_run": session.run_history[-1] if session.run_history else None,
        }
    )
    return out


def create_app(session_manager: SessionManager | None = None):  # noqa: C901
    """
    Build and return the FastAPI app.

    Imports FastAPI lazily so importing this module does not require server
    dependencies in non-server environments.
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "FastAPI is not installed. Install server extras, e.g. 'pip install modacor[server]'."
        ) from exc

    manager = session_manager or SessionManager()
    app = FastAPI(
        title="MoDaCor Runtime Service",
        version="0.1.0-draft",
        description="Scaffold API for long-lived MoDaCor pipeline sessions.",
    )

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": [_session_summary(s) for s in manager.list_sessions()]}

    @app.post("/v1/sessions")
    def create_session(payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id is required.")

        pipeline = payload.get("pipeline", {}) or {}
        yaml_text = pipeline.get("yaml_text")
        yaml_path = pipeline.get("yaml_path")

        if bool(yaml_text) == bool(yaml_path):
            raise HTTPException(
                status_code=422, detail="Exactly one of pipeline.yaml_text or pipeline.yaml_path required."
            )

        if yaml_path:
            try:
                pipeline_yaml = Path(str(yaml_path)).read_text(encoding="utf-8")
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Failed to read pipeline yaml_path: {exc}") from exc
        else:
            pipeline_yaml = str(yaml_text)

        trace = payload.get("trace", {}) or {}
        try:
            session = manager.create_session(
                session_id=session_id,
                name=payload.get("name"),
                pipeline_yaml=pipeline_yaml,
                trace_enabled=bool(trace.get("enabled", False)),
                trace_watch=dict(trace.get("watch", {}) or {}),
                auto_full_reset_on_partial_error=bool(payload.get("auto_full_reset_on_partial_error", True)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _session_detail(session)

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return _session_detail(session)

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        if not manager.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")

    @app.put("/v1/sessions/{session_id}/sources")
    def upsert_sources(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sources = payload.get("sources", [])
        if not isinstance(sources, list):
            raise HTTPException(status_code=422, detail="'sources' must be a list.")
        try:
            session = manager.upsert_sources(session_id, sources=sources)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"session_id": session.session_id, "sources": list(session.sources.values())}

    @app.delete("/v1/sessions/{session_id}/sources/{ref}", status_code=204)
    def delete_source(session_id: str, ref: str) -> None:
        try:
            existed = manager.delete_source(session_id, ref)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not existed:
            raise HTTPException(status_code=404, detail=f"Source '{ref}' not found.")

    @app.post("/v1/sessions/{session_id}/process")
    def process(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"partial", "full", "auto"}:
            raise HTTPException(status_code=422, detail="mode must be one of: partial, full, auto.")
        changed_sources = payload.get("changed_sources") or []
        if mode == "partial" and not changed_sources:
            raise HTTPException(status_code=422, detail="changed_sources is required for partial mode.")
        try:
            run_meta = manager.enqueue_run(session_id, mode=mode, changed_sources=list(changed_sources))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"session_id": session_id, "run_id": run_meta["run_id"], "state": "accepted"}

    @app.post("/v1/sessions/{session_id}/reset")
    def reset(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"partial", "full"}:
            raise HTTPException(status_code=422, detail="mode must be one of: partial, full.")
        try:
            session = manager.reset_session(session_id, mode=mode)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"session_id": session.session_id, "mode": mode, "state": session.state}

    @app.post("/v1/sessions/{session_id}/recover")
    def recover(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        strategy = str(payload.get("strategy", "")).strip()
        if strategy not in {"full_reset_then_process", "full_reset_only"}:
            raise HTTPException(status_code=422, detail="Invalid recovery strategy.")
        try:
            manager.reset_session(session_id, mode="full")
            if strategy == "full_reset_only":
                session = manager.get_session(session_id)
                assert session is not None
                return {"session_id": session_id, "state": session.state, "strategy": strategy}

            run_meta = manager.enqueue_run(
                session_id,
                mode="full",
                changed_sources=list(payload.get("changed_sources", []) or []),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {"session_id": session_id, "run_id": run_meta["run_id"], "state": "accepted", "strategy": strategy}

    @app.get("/v1/sessions/{session_id}/runs")
    def list_runs(session_id: str) -> dict[str, Any]:
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"session_id": session_id, "runs": session.run_history}

    @app.get("/v1/sessions/{session_id}/runs/{run_id}")
    def get_run(session_id: str, run_id: str) -> dict[str, Any]:
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        for run_meta in session.run_history:
            if run_meta.get("run_id") == run_id:
                return run_meta
        raise HTTPException(status_code=404, detail="Run not found.")

    @app.websocket("/v1/sessions/{session_id}/events")
    async def events(session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        session = manager.get_session(session_id)
        if session is None:
            await websocket.send_json({"event": "error", "payload": {"code": "SESSION_NOT_FOUND"}})
            await websocket.close(code=1008)
            return
        await websocket.send_json(
            {
                "event": "session_state_changed",
                "session_id": session_id,
                "payload": {"state": session.state, "active_run_id": session.active_run_id},
            }
        )
        await websocket.close(code=1000)

    return app
