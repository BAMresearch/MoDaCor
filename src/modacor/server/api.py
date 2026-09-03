# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from .errors import ApiError
from .runtime_policy import RuntimePolicy
from .runtime_service import RuntimeService
from .session_manager import SessionManager

__all__ = ["create_app"]


@lru_cache(maxsize=1)
def _plotly_js_bytes() -> bytes:
    try:
        return resources.files("plotly").joinpath("package_data", "plotly.min.js").read_bytes()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plotly assets are not installed. Install them with 'pip install modacor[plotting]'."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError("Plotly is installed, but package_data/plotly.min.js was not found.") from exc


def _plot_page_html(*, session_id: str, sink_ref: str, plot_id: str) -> str:
    json_url = f"/v1/sessions/{session_id}/plots/{sink_ref}/{plot_id}/json"
    json_url_literal = json.dumps(json_url)
    title = f"{session_id} / {sink_ref} / {plot_id}"
    title_literal = json.dumps(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="/v1/assets/plotting/plotly.min.js" onerror="window.__plotlyLoadError = true"></script>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fa;
      color: #17202a;
    }}
    header {{
      height: 42px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      border-bottom: 1px solid #d8dde6;
      background: #ffffff;
      font-size: 13px;
    }}
    #plot {{
      width: 100%;
      height: calc(100% - 42px);
    }}
    #status {{
      color: #5b6472;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <header>
    <strong id="title"></strong>
    <span id="status"></span>
  </header>
  <div id="plot"></div>
  <script>
    const jsonUrl = {json_url_literal};
    const fallbackTitle = {title_literal};
    const titleNode = document.getElementById("title");
    const statusNode = document.getElementById("status");
    titleNode.textContent = fallbackTitle;

    async function refreshPlot() {{
      try {{
        if (!window.Plotly) {{
          statusNode.textContent = window.__plotlyLoadError
            ? "Plotly asset unavailable; install modacor[plotting] and restart the server"
            : "loading Plotly";
          return;
        }}
        const response = await fetch(jsonUrl, {{cache: "no-store"}});
        if (!response.ok) {{
          statusNode.textContent = response.status === 404 ? "waiting for plot" : `HTTP ${{response.status}}`;
          return;
        }}
        const payload = await response.json();
        const plot = payload.plot || {{}};
        const layout = plot.layout || {{}};
        const metadata = plot.metadata || {{}};
        titleNode.textContent = layout.title && layout.title.text ? layout.title.text : fallbackTitle;
        statusNode.textContent = [
          payload.state || "",
          payload.latest_run && payload.latest_run.status ? payload.latest_run.status : "",
          metadata.valid_points !== undefined ? `${{metadata.valid_points}} pts` : "",
        ].filter(Boolean).join(" | ");
        await Plotly.react("plot", plot.data || [], layout, {{responsive: true, displaylogo: false}});
      }} catch (err) {{
        statusNode.textContent = String(err);
      }}
    }}

    refreshPlot();
    setInterval(refreshPlot, 1500);
  </script>
</body>
</html>
"""


def create_app(  # noqa: C901
    session_manager: SessionManager | None = None,
    runtime_policy: RuntimePolicy | None = None,
):
    """
    Build and return the FastAPI app.

    Imports FastAPI lazily so importing this module does not require server
    dependencies in non-server environments.
    """
    try:
        from fastapi import Body, FastAPI, HTTPException, Response, WebSocket
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "FastAPI is not installed. Install server extras, e.g. 'pip install modacor[server]'."
        ) from exc

    policy = runtime_policy or RuntimePolicy.trusted()
    service = RuntimeService(manager=session_manager or SessionManager(max_sessions=policy.max_sessions), policy=policy)
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

    @app.get("/v1/assets/plotting/plotly.min.js")
    def get_plotly_js() -> Response:
        try:
            payload = _plotly_js_bytes()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=payload,
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

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

    @app.put("/v1/sessions/{session_id}/sinks")
    def upsert_sinks(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.upsert_sinks, session_id, payload)

    @app.post("/v1/sessions/{session_id}/sinks/patch")
    def patch_sink(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.patch_sink, session_id, payload)

    @app.post("/v1/sessions/{session_id}/sample")
    def set_sample_source(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _call(service.set_sample_source, session_id, payload)

    @app.delete("/v1/sessions/{session_id}/sources/{ref}", status_code=204)
    def delete_source(session_id: str, ref: str) -> None:
        _call(service.delete_source, session_id, ref)

    @app.delete("/v1/sessions/{session_id}/sinks/{ref}", status_code=204)
    def delete_sink(session_id: str, ref: str) -> None:
        _call(service.delete_sink, session_id, ref)

    @app.put("/v1/sessions/{session_id}/buffers/sources/{source_ref}/arrays/{data_key:path}")
    def put_buffer_source_array(
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: bytes = Body(...),
    ) -> dict[str, Any]:
        return _call(service.put_buffer_source_array, session_id, source_ref, data_key, payload)

    @app.put("/v1/sessions/{session_id}/buffers/sources/{source_ref}/attrs/{data_key:path}")
    def put_buffer_source_attrs(
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return _call(service.put_buffer_source_attrs, session_id, source_ref, data_key, payload)

    @app.put("/v1/sessions/{session_id}/buffers/sources/{source_ref}/metadata/{data_key:path}")
    def put_buffer_source_metadata(
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return _call(service.put_buffer_source_metadata, session_id, source_ref, data_key, payload)

    @app.get("/v1/sessions/{session_id}/buffers/sinks/{sink_ref}/arrays/{data_key:path}")
    def get_buffer_sink_array(session_id: str, sink_ref: str, data_key: str) -> Response:
        payload = _call(service.get_buffer_sink_array, session_id, sink_ref, data_key)
        return Response(content=payload, media_type="application/x-npy")

    @app.get("/v1/sessions/{session_id}/buffers/{kind}/{ref}/manifest")
    def get_buffer_manifest(session_id: str, kind: str, ref: str) -> dict[str, Any]:
        return _call(service.get_buffer_manifest, session_id, kind, ref)

    @app.get("/v1/sessions/{session_id}/plots/{sink_ref}/{plot_id}/json")
    def get_plot_payload(session_id: str, sink_ref: str, plot_id: str) -> dict[str, Any]:
        return _call(service.get_plot_payload, session_id, sink_ref, plot_id)

    @app.get("/v1/sessions/{session_id}/plots/{sink_ref}/{plot_id}")
    def get_plot_page(session_id: str, sink_ref: str, plot_id: str) -> Response:
        return Response(
            content=_plot_page_html(session_id=session_id, sink_ref=sink_ref, plot_id=plot_id),
            media_type="text/html",
        )

    @app.delete("/v1/sessions/{session_id}/buffers/sinks/{sink_ref}")
    def clear_buffer_sink(session_id: str, sink_ref: str) -> dict[str, Any]:
        return _call(service.clear_buffer_sink, session_id, sink_ref)

    @app.delete("/v1/sessions/{session_id}/buffers/sinks/{sink_ref}/arrays/{data_key:path}")
    def clear_buffer_sink_array(session_id: str, sink_ref: str, data_key: str) -> dict[str, Any]:
        return _call(service.clear_buffer_sink_array, session_id, sink_ref, data_key)

    @app.delete("/v1/sessions/{session_id}/buffers")
    def clear_buffers(
        session_id: str,
        kind: str | None = None,
        ref: str | None = None,
        data_key: str | None = None,
    ) -> dict[str, Any]:
        return _call(service.clear_buffers, session_id, kind=kind, ref=ref, data_key=data_key)

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
