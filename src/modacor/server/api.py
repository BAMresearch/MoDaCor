# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from graphlib import TopologicalSorter
from importlib import import_module
from pathlib import Path
from typing import Any

from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sources import IoSources
from modacor.io.yaml.yaml_source import YAMLSource
from modacor.runner import run_pipeline_job
from modacor.runner.pipeline import Pipeline

from .execution import find_dirty_step_ids
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


def _resolve_effective_mode(requested_mode: str) -> tuple[str, str | None]:
    if requested_mode == "partial":
        return "partial", None
    if requested_mode == "auto":
        return "partial", "auto mode: partial first, full fallback on failure"
    return "full", None


def _ordered_step_ids(pipeline: Pipeline) -> list[str]:
    return [str(node.step_id) for node in TopologicalSorter(pipeline.graph).static_order()]


def _load_custom_source_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def _build_sources_from_session(session: PipelineSession) -> IoSources:
    from modacor.io.csv.csv_source import CSVSource

    sources = IoSources()
    type_map: dict[str, Any] = {
        "hdf": HDFSource,
        "yaml": YAMLSource,
        "csv": CSVSource,
    }

    for ref in sorted(session.sources.keys()):
        reg = session.sources[ref]
        source_type = str(reg["type"]).strip().lower()
        location = Path(str(reg["location"]))
        kwargs = dict(reg.get("kwargs", {}) or {})

        if source_type == "custom":
            class_path = kwargs.pop("class_path", None)
            if not class_path:
                raise ValueError(f"Custom source '{ref}' requires kwargs.class_path.")
            source_cls = _load_custom_source_class(str(class_path))
        else:
            if source_type not in type_map:
                raise ValueError(f"Unsupported source type '{source_type}' for ref '{ref}'.")
            source_cls = type_map[source_type]

        source = source_cls(
            source_reference=ref,
            resource_location=location,
            iosource_method_kwargs=kwargs.get("iosource_method_kwargs", kwargs),
        )
        sources.register_source(source)
    return sources


def _maybe_write_hdf_output(
    write_hdf: dict[str, Any] | None,
    *,
    run_name: str,
    result: Any,
    pipeline_yaml: str,
) -> str | None:
    if not write_hdf:
        return None

    out_path_raw = write_hdf.get("path")
    if not out_path_raw:
        raise ValueError("write_hdf.path is required when write_hdf is provided.")

    data_paths = list(write_hdf.get("data_paths", []) or [])
    write_all = bool(write_hdf.get("write_all_processing_data", False))
    if not data_paths and not write_all:
        raise ValueError("write_hdf requires data_paths or write_all_processing_data=true.")

    out_path = Path(str(out_path_raw))
    sink = HDFProcessingSink(resource_location=out_path)
    sink.write(
        run_name,
        result.processing_data,
        data_paths=data_paths or None,
        write_all_processing_data=write_all,
        pipeline_spec=result.pipeline.to_spec(),
        pipeline_yaml=pipeline_yaml,
        trace_events=result.tracer.events if result.tracer is not None else None,
    )
    return str(out_path)


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
        changed_keys = payload.get("changed_keys") or []
        if mode == "partial" and not changed_sources and not changed_keys:
            raise HTTPException(status_code=422, detail="partial mode requires changed_sources or changed_keys.")
        session = manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        effective_mode, mode_note = _resolve_effective_mode(mode)
        if effective_mode == "partial" and session.processing_data is None:
            effective_mode = "full"
            mode_note = "No previous ProcessingData snapshot available; executed full rerun."
        if mode == "auto" and not changed_sources and not changed_keys:
            effective_mode = "full"
            mode_note = "Auto mode without changed_sources/changed_keys defaults to full rerun."
        try:
            run_meta = manager.enqueue_run(
                session_id,
                mode=mode,
                changed_sources=list(changed_sources),
                effective_mode=effective_mode,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        run_id = str(run_meta["run_id"])
        try:
            pipeline = Pipeline.from_yaml(session.pipeline_yaml or "")
            sources = _build_sources_from_session(session)

            reuse_processing_data = effective_mode == "partial" and session.processing_data is not None
            selected_step_ids: set[str] | None = None
            snapshot_before_partial = None
            dirty_step_ids_ordered: list[str] = []
            boundary_step_id: str | None = None
            if effective_mode == "partial":
                selected_step_ids = find_dirty_step_ids(
                    pipeline,
                    changed_sources=list(changed_sources),
                    changed_keys=list(changed_keys),
                )
                topo_ids = _ordered_step_ids(pipeline)
                dirty_step_ids_ordered = [step_id for step_id in topo_ids if step_id in selected_step_ids]
                boundary_step_id = dirty_step_ids_ordered[0] if dirty_step_ids_ordered else None
                if not selected_step_ids:
                    run_meta = manager.mark_run_succeeded(
                        session_id,
                        run_id,
                        details={
                            "status": "succeeded",
                            "executed_steps": [],
                            "num_steps": 0,
                            "note": "No pipeline steps matched changed_sources.",
                            "changed_keys": list(changed_keys),
                        },
                    )
                    return {
                        "session_id": session_id,
                        "run_id": run_id,
                        "state": "idle",
                        "status": run_meta.get("status"),
                        "effective_mode": run_meta.get("effective_mode"),
                        "note": run_meta.get("note"),
                    }

                if session.processing_data is not None:
                    # Boundary checkpoint: restore this snapshot if partial chain fails.
                    snapshot_before_partial = deepcopy(session.processing_data)

            result = run_pipeline_job(
                pipeline,
                sources=sources,
                processing_data=session.processing_data if reuse_processing_data else None,
                trace=session.trace_enabled,
                trace_watch=session.trace_watch,
                selected_step_ids=selected_step_ids,
            )
            session.processing_data = result.processing_data

            write_hdf = payload.get("write_hdf", None)
            hdf_out_path = _maybe_write_hdf_output(
                write_hdf if isinstance(write_hdf, dict) else None,
                run_name=str(payload.get("run_name") or run_id),
                result=result,
                pipeline_yaml=session.pipeline_yaml or "",
            )

            details = {
                "status": "succeeded",
                "executed_steps": result.executed_steps,
                "num_steps": len(result.executed_steps),
                "changed_sources": list(changed_sources),
                "changed_keys": list(changed_keys),
            }
            if mode_note:
                details["note"] = mode_note
            if dirty_step_ids_ordered:
                details["dirty_steps"] = dirty_step_ids_ordered
            if boundary_step_id is not None:
                details["checkpoint_boundary_step"] = boundary_step_id
            if hdf_out_path is not None:
                details["hdf_output"] = hdf_out_path

            run_meta = manager.mark_run_succeeded(session_id, run_id, details=details)
            return {
                "session_id": session_id,
                "run_id": run_id,
                "state": "idle",
                "status": run_meta.get("status"),
                "effective_mode": run_meta.get("effective_mode"),
                "note": mode_note,
                "hdf_output": run_meta.get("hdf_output"),
            }
        except Exception as exc:
            if effective_mode == "partial" and snapshot_before_partial is not None:
                session.processing_data = snapshot_before_partial
            manager.mark_run_failed(
                session_id,
                run_id,
                code="PARTIAL_RUN_FAILED" if effective_mode == "partial" else "RUN_FAILED",
                message=str(exc),
                details={"exception_type": type(exc).__name__},
            )
            # "auto" mode: fallback to a full rerun after partial failure.
            if mode == "auto":
                fallback_run = manager.enqueue_run(
                    session_id,
                    mode="full",
                    changed_sources=list(changed_sources),
                    effective_mode="full",
                )
                fallback_id = str(fallback_run["run_id"])
                try:
                    fallback_result = run_pipeline_job(
                        Pipeline.from_yaml(session.pipeline_yaml or ""),
                        sources=sources,
                        processing_data=None,
                        trace=session.trace_enabled,
                        trace_watch=session.trace_watch,
                    )
                    session.processing_data = fallback_result.processing_data

                    write_hdf = payload.get("write_hdf", None)
                    hdf_out_path = _maybe_write_hdf_output(
                        write_hdf if isinstance(write_hdf, dict) else None,
                        run_name=str(payload.get("run_name") or fallback_id),
                        result=fallback_result,
                        pipeline_yaml=session.pipeline_yaml or "",
                    )

                    done = manager.mark_run_succeeded(
                        session_id,
                        fallback_id,
                        details={
                            "status": "succeeded",
                            "executed_steps": fallback_result.executed_steps,
                            "num_steps": len(fallback_result.executed_steps),
                            "note": "Auto fallback succeeded after partial failure.",
                            "recovered_from_run_id": run_id,
                            "changed_sources": list(changed_sources),
                            "changed_keys": list(changed_keys),
                            **({"hdf_output": hdf_out_path} if hdf_out_path else {}),
                        },
                    )
                    return {
                        "session_id": session_id,
                        "run_id": fallback_id,
                        "state": "idle",
                        "status": done.get("status"),
                        "effective_mode": done.get("effective_mode"),
                        "note": done.get("note"),
                        "recovered_from_run_id": run_id,
                        "hdf_output": done.get("hdf_output"),
                    }
                except Exception as fallback_exc:
                    manager.mark_run_failed(
                        session_id,
                        fallback_id,
                        code="FULL_RUN_FAILED",
                        message=str(fallback_exc),
                        details={"exception_type": type(fallback_exc).__name__, "recovered_from_run_id": run_id},
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "code": "FULL_RUN_FAILED",
                            "message": str(fallback_exc),
                            "details": {"session_id": session_id, "run_id": fallback_id},
                        },
                    ) from fallback_exc

            raise HTTPException(
                status_code=500,
                detail={
                    "code": "PARTIAL_RUN_FAILED" if effective_mode == "partial" else "RUN_FAILED",
                    "message": str(exc),
                    "details": {"session_id": session_id, "run_id": run_id},
                },
            ) from exc

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
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # Reuse /process execution path for actual rerun.
        process_payload: dict[str, Any] = {"mode": "full", "changed_sources": list(payload.get("changed_sources", []))}
        if "write_hdf" in payload:
            process_payload["write_hdf"] = payload["write_hdf"]
        if "run_name" in payload:
            process_payload["run_name"] = payload["run_name"]
        process_response = process(session_id, process_payload)
        process_response["strategy"] = strategy
        return process_response

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
