# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

from .errors import ApiError
from .runtime_service import RuntimeService
from .session_manager import SessionManager

__all__ = ["create_app"]


def create_app(  # noqa: C901
    session_manager: SessionManager | None = None,
):
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

    service = RuntimeService(manager=session_manager or SessionManager())
    app = FastAPI(
        title="MoDaCor Runtime Service",
        version="0.1.0-draft",
        description="Scaffold API for long-lived MoDaCor pipeline sessions.",
    )

    def _call(handler, *args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return _call(service.health)

    @app.get("/v1/readiness")
    def readiness() -> dict[str, Any]:
        return _call(service.readiness)

    @app.get("/v1/source-templates")
    def source_templates() -> dict[str, Any]:
        return _call(service.source_templates)

    @app.get("/v1/sessions")
    def list_sessions() -> dict[str, Any]:
        return _call(service.list_sessions)

    @app.post("/v1/sessions")
    def create_session(payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.create_session, payload)

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return _call(service.get_session, session_id)

    @app.get("/v1/sessions/{session_id}/errors/latest")
    def latest_error(session_id: str) -> dict[str, Any]:
        return _call(service.latest_error, session_id)

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        _call(service.delete_session, session_id)

    @app.put("/v1/sessions/{session_id}/sources")
    def upsert_sources(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.upsert_sources, session_id, payload)

    @app.post("/v1/sessions/{session_id}/sources/patch")
    def patch_source(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.patch_source, session_id, payload)

    @app.post("/v1/sessions/{session_id}/sample")
    def set_sample_source(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.set_sample_source, session_id, payload)

    @app.delete("/v1/sessions/{session_id}/sources/{ref}", status_code=204)
    def delete_source(session_id: str, ref: str) -> None:
        _call(service.delete_source, session_id, ref)

    @app.post("/v1/sessions/{session_id}/process")
    def process(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.process, session_id, payload)

    @app.post("/v1/sessions/{session_id}/process/dry-run")
    def process_dry_run(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.process_dry_run, session_id, payload)

    @app.post("/v1/sessions/{session_id}/reset")
    def reset(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.reset, session_id, payload)

    @app.post("/v1/sessions/{session_id}/recover")
    def recover(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.recover, session_id, payload)

    @app.get("/v1/sessions/{session_id}/runs")
    def list_runs(session_id: str) -> dict[str, Any]:
        return _call(service.list_runs, session_id)

    @app.get("/v1/sessions/{session_id}/runs/{run_id}")
    def get_run(session_id: str, run_id: str) -> dict[str, Any]:
        return _call(service.get_run, session_id, run_id)

    @app.websocket("/v1/sessions/{session_id}/events")
    async def events(session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            payload = service.session_state_event(session_id)
        except ApiError:
            await websocket.send_json({"event": "error", "payload": {"code": "SESSION_NOT_FOUND"}})
            await websocket.close(code=1008)
            return
        await websocket.send_json(payload)
        await websocket.close(code=1000)

    return app
