# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.runner import run_pipeline_job
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import RunResult

from .errors import ApiError
from .execution import find_dirty_step_ids
from .io_utils import build_sources_from_session, write_hdf_output
from .planning import build_dry_run_plan, missing_required_source_refs, ordered_step_ids, resolve_effective_mode
from .session_manager import PipelineSession, SessionManager
from .source_profiles import get_source_profile, list_source_profiles

__all__ = ["RuntimeService"]


@dataclass(slots=True)
class ProcessRequest:
    """Normalized process request payload used by runtime service methods."""

    mode: str
    changed_sources: list[str]
    changed_keys: list[str]
    write_hdf: dict[str, Any] | None = None
    run_name: str | None = None


@dataclass(slots=True)
class ProcessPreparation:
    """Execution preparation details for a process request."""

    selected_step_ids: set[str] | None = None
    snapshot_before_partial: ProcessingData | None = None
    dirty_step_ids_ordered: list[str] = field(default_factory=list)
    boundary_step_id: str | None = None
    early_response: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeService:
    """Application service for the MoDaCor runtime API."""

    manager: SessionManager

    def session_summary(self, session: PipelineSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "name": session.name,
            "state": session.state,
            "active_run_id": session.active_run_id,
            "updated_utc": session.updated_utc,
        }

    def session_detail(self, session: PipelineSession) -> dict[str, Any]:
        out = self.session_summary(session)
        out.update(
            {
                "sources": list(session.sources.values()),
                "trace": {
                    "enabled": session.trace_enabled,
                    "watch": session.trace_watch,
                    "record_only_on_change": True,
                },
                "last_run": session.run_history[-1] if session.run_history else None,
                "source_profile": session.source_profile,
                "required_source_refs": list(session.required_source_refs),
            }
        )
        return out

    def source_templates(self) -> dict[str, Any]:
        return {"templates": list_source_profiles()}

    def list_sessions(self) -> dict[str, Any]:
        return {"sessions": [self.session_summary(session) for session in self.manager.list_sessions()]}

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ApiError(status_code=422, detail="session_id is required.")

        pipeline = payload.get("pipeline", {}) or {}
        yaml_text = pipeline.get("yaml_text")
        yaml_path = pipeline.get("yaml_path")
        if bool(yaml_text) == bool(yaml_path):
            raise ApiError(status_code=422, detail="Exactly one of pipeline.yaml_text or pipeline.yaml_path required.")

        if yaml_path:
            try:
                pipeline_yaml = Path(str(yaml_path)).read_text(encoding="utf-8")
            except Exception as exc:
                raise ApiError(status_code=422, detail=f"Failed to read pipeline yaml_path: {exc}") from exc
        else:
            pipeline_yaml = str(yaml_text)

        source_profile_name = payload.get("source_profile")
        required_source_refs: list[str] = []
        normalized_profile: str | None = None
        if source_profile_name is not None:
            profile = get_source_profile(str(source_profile_name))
            if profile is None:
                raise ApiError(status_code=422, detail=f"Unknown source_profile: {source_profile_name!r}")
            normalized_profile = str(source_profile_name).strip().lower()
            required_source_refs = [str(item["ref"]) for item in profile.get("required_sources", [])]

        trace = payload.get("trace", {}) or {}
        try:
            session = self.manager.create_session(
                session_id=session_id,
                name=payload.get("name"),
                pipeline_yaml=pipeline_yaml,
                trace_enabled=bool(trace.get("enabled", False)),
                trace_watch=dict(trace.get("watch", {}) or {}),
                auto_full_reset_on_partial_error=bool(payload.get("auto_full_reset_on_partial_error", True)),
                source_profile=normalized_profile,
                required_source_refs=required_source_refs,
            )
        except ValueError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        return self.session_detail(session)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.session_detail(self._require_session(session_id))

    def delete_session(self, session_id: str) -> None:
        if not self.manager.delete_session(session_id):
            raise ApiError(status_code=404, detail="Session not found.")

    def upsert_sources(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sources = payload.get("sources", [])
        if not isinstance(sources, list):
            raise ApiError(status_code=422, detail="'sources' must be a list.")
        try:
            session = self.manager.upsert_sources(session_id, sources=sources)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        return {"session_id": session.session_id, "sources": list(session.sources.values())}

    def patch_source(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ref = str(payload.get("ref", "")).strip()
        source_type = str(payload.get("type", "")).strip()
        location = str(payload.get("location", "")).strip()
        kwargs = payload.get("kwargs", {}) or {}

        if not ref:
            raise ApiError(status_code=422, detail="'ref' is required.")
        if not source_type:
            raise ApiError(status_code=422, detail="'type' is required.")
        if not location:
            raise ApiError(status_code=422, detail="'location' is required.")
        if not isinstance(kwargs, dict):
            raise ApiError(status_code=422, detail="'kwargs' must be an object when provided.")

        try:
            session = self.manager.upsert_sources(
                session_id,
                sources=[{"ref": ref, "type": source_type, "location": location, "kwargs": kwargs}],
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc

        return {
            "session_id": session.session_id,
            "source": session.sources.get(ref),
            "sources": list(session.sources.values()),
        }

    def set_sample_source(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        location = str(payload.get("location", "")).strip()
        source_type = str(payload.get("type", "hdf")).strip()
        kwargs = payload.get("kwargs", {}) or {}

        if not location:
            raise ApiError(status_code=422, detail="'location' is required.")
        if not source_type:
            raise ApiError(status_code=422, detail="'type' must be non-empty.")
        if not isinstance(kwargs, dict):
            raise ApiError(status_code=422, detail="'kwargs' must be an object when provided.")

        try:
            session = self.manager.upsert_sources(
                session_id,
                sources=[{"ref": "sample", "type": source_type, "location": location, "kwargs": kwargs}],
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc

        return {
            "session_id": session.session_id,
            "source": session.sources.get("sample"),
            "sources": list(session.sources.values()),
        }

    def delete_source(self, session_id: str, ref: str) -> None:
        try:
            existed = self.manager.delete_source(session_id, ref)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not existed:
            raise ApiError(status_code=404, detail=f"Source '{ref}' not found.")

    def process(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._parse_process_request(payload)
        session = self._require_session(session_id)
        self._ensure_required_sources(session)

        effective_mode, mode_note = self._resolve_process_mode(session, request)
        run_id = self._enqueue_process_run(
            session_id,
            mode=request.mode,
            changed_sources=request.changed_sources,
            effective_mode=effective_mode,
        )

        preparation = ProcessPreparation()
        sources: IoSources | None = None
        try:
            pipeline = Pipeline.from_yaml(session.pipeline_yaml or "")
            sources = build_sources_from_session(session)
            preparation = self._prepare_process_execution(
                session_id=session_id,
                run_id=run_id,
                session=session,
                pipeline=pipeline,
                request=request,
                effective_mode=effective_mode,
            )
            if preparation.early_response is not None:
                return preparation.early_response

            result, elapsed_s = self._execute_process_run(
                session=session,
                pipeline=pipeline,
                sources=sources,
                effective_mode=effective_mode,
                preparation=preparation,
            )
            return self._finalize_process_run(
                session_id=session_id,
                run_id=run_id,
                session=session,
                pipeline=pipeline,
                request=request,
                effective_mode=effective_mode,
                mode_note=mode_note,
                preparation=preparation,
                result=result,
                elapsed_s=elapsed_s,
            )
        except Exception as exc:
            return self._handle_process_failure(
                session=session,
                session_id=session_id,
                request=request,
                effective_mode=effective_mode,
                run_id=run_id,
                preparation=preparation,
                sources=sources,
                exc=exc,
            )

    def process_dry_run(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._parse_process_request(payload)
        session = self._require_session(session_id)
        try:
            return build_dry_run_plan(
                session,
                mode=request.mode,
                changed_sources=request.changed_sources,
                changed_keys=request.changed_keys,
            )
        except Exception as exc:
            raise ApiError(
                status_code=500,
                detail={
                    "code": "DRY_RUN_FAILED",
                    "message": str(exc),
                    "details": {"session_id": session_id},
                },
            ) from exc

    def reset(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"partial", "full"}:
            raise ApiError(status_code=422, detail="mode must be one of: partial, full.")
        try:
            session = self.manager.reset_session(session_id, mode=mode)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        return {"session_id": session.session_id, "mode": mode, "state": session.state}

    def recover(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        strategy = str(payload.get("strategy", "")).strip()
        if strategy not in {"full_reset_then_process", "full_reset_only"}:
            raise ApiError(status_code=422, detail="Invalid recovery strategy.")
        try:
            self.manager.reset_session(session_id, mode="full")
            if strategy == "full_reset_only":
                session = self._require_session(session_id)
                return {"session_id": session_id, "state": session.state, "strategy": strategy}
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc

        process_payload: dict[str, Any] = {"mode": "full", "changed_sources": list(payload.get("changed_sources", []))}
        if "write_hdf" in payload:
            process_payload["write_hdf"] = payload["write_hdf"]
        if "run_name" in payload:
            process_payload["run_name"] = payload["run_name"]
        process_response = self.process(session_id, process_payload)
        process_response["strategy"] = strategy
        return process_response

    def list_runs(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        return {"session_id": session_id, "runs": session.run_history}

    def get_run(self, session_id: str, run_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        for run_meta in session.run_history:
            if run_meta.get("run_id") == run_id:
                return run_meta
        raise ApiError(status_code=404, detail="Run not found.")

    def session_state_event(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        return {
            "event": "session_state_changed",
            "session_id": session_id,
            "payload": {"state": session.state, "active_run_id": session.active_run_id},
        }

    def _require_session(self, session_id: str) -> PipelineSession:
        session = self.manager.get_session(session_id)
        if session is None:
            raise ApiError(status_code=404, detail="Session not found.")
        return session

    def _ensure_required_sources(self, session: PipelineSession) -> None:
        missing_refs = missing_required_source_refs(session)
        if not missing_refs:
            return
        raise ApiError(
            status_code=422,
            detail={
                "code": "MISSING_REQUIRED_SOURCES",
                "message": "Session source profile requirements are not satisfied.",
                "details": {
                    "source_profile": session.source_profile,
                    "missing_refs": missing_refs,
                    "required_refs": session.required_source_refs,
                },
            },
        )

    def _parse_process_request(self, payload: dict[str, Any]) -> ProcessRequest:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"partial", "full", "auto"}:
            raise ApiError(status_code=422, detail="mode must be one of: partial, full, auto.")

        changed_sources = list(payload.get("changed_sources") or [])
        changed_keys = list(payload.get("changed_keys") or [])
        if mode == "partial" and not changed_sources and not changed_keys:
            raise ApiError(status_code=422, detail="partial mode requires changed_sources or changed_keys.")

        write_hdf_raw = payload.get("write_hdf")
        write_hdf = dict(write_hdf_raw) if isinstance(write_hdf_raw, dict) else None
        run_name_raw = payload.get("run_name")
        run_name = str(run_name_raw) if run_name_raw is not None else None
        return ProcessRequest(
            mode=mode,
            changed_sources=changed_sources,
            changed_keys=changed_keys,
            write_hdf=write_hdf,
            run_name=run_name,
        )

    def _resolve_process_mode(self, session: PipelineSession, request: ProcessRequest) -> tuple[str, str | None]:
        effective_mode, mode_note = resolve_effective_mode(request.mode)
        if effective_mode == "partial" and session.processing_data is None:
            effective_mode = "full"
            mode_note = "No previous ProcessingData snapshot available; executed full rerun."
        if request.mode == "auto" and not request.changed_sources and not request.changed_keys:
            effective_mode = "full"
            mode_note = "Auto mode without changed_sources/changed_keys defaults to full rerun."
        return effective_mode, mode_note

    def _enqueue_process_run(
        self,
        session_id: str,
        *,
        mode: str,
        changed_sources: list[str],
        effective_mode: str,
    ) -> str:
        try:
            run_meta = self.manager.enqueue_run(
                session_id,
                mode=mode,
                changed_sources=changed_sources,
                effective_mode=effective_mode,
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        return str(run_meta["run_id"])

    def _prepare_process_execution(
        self,
        *,
        session_id: str,
        run_id: str,
        session: PipelineSession,
        pipeline: Pipeline,
        request: ProcessRequest,
        effective_mode: str,
    ) -> ProcessPreparation:
        if effective_mode != "partial":
            return ProcessPreparation()

        selected_step_ids = find_dirty_step_ids(
            pipeline,
            changed_sources=request.changed_sources,
            changed_keys=request.changed_keys,
        )
        topo_ids = ordered_step_ids(pipeline)
        dirty_step_ids_ordered = [step_id for step_id in topo_ids if step_id in selected_step_ids]
        boundary_step_id = dirty_step_ids_ordered[0] if dirty_step_ids_ordered else None

        if not selected_step_ids:
            return ProcessPreparation(
                early_response=self._mark_noop_process_run(
                    session_id=session_id,
                    run_id=run_id,
                    changed_keys=request.changed_keys,
                    topo_ids=topo_ids,
                )
            )

        snapshot_before_partial = deepcopy(session.processing_data) if session.processing_data is not None else None
        return ProcessPreparation(
            selected_step_ids=selected_step_ids,
            snapshot_before_partial=snapshot_before_partial,
            dirty_step_ids_ordered=dirty_step_ids_ordered,
            boundary_step_id=boundary_step_id,
        )

    def _mark_noop_process_run(
        self,
        *,
        session_id: str,
        run_id: str,
        changed_keys: list[str],
        topo_ids: list[str],
    ) -> dict[str, Any]:
        run_meta = self.manager.mark_run_succeeded(
            session_id,
            run_id,
            details={
                "status": "succeeded",
                "executed_steps": [],
                "num_steps": 0,
                "note": "No pipeline steps matched changed_sources.",
                "changed_keys": changed_keys,
                "dirty_steps": [],
                "skipped_steps": topo_ids,
                "step_durations_s": {},
                "elapsed_s": 0.0,
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

    def _execute_process_run(
        self,
        *,
        session: PipelineSession,
        pipeline: Pipeline,
        sources: IoSources,
        effective_mode: str,
        preparation: ProcessPreparation,
    ) -> tuple[RunResult, float]:
        reuse_processing_data = effective_mode == "partial" and session.processing_data is not None
        run_t0 = perf_counter()
        result = run_pipeline_job(
            pipeline,
            sources=sources,
            processing_data=session.processing_data if reuse_processing_data else None,
            trace=session.trace_enabled,
            trace_watch=session.trace_watch,
            selected_step_ids=preparation.selected_step_ids,
        )
        elapsed_s = perf_counter() - run_t0
        session.processing_data = result.processing_data
        return result, elapsed_s

    def _finalize_process_run(
        self,
        *,
        session_id: str,
        run_id: str,
        session: PipelineSession,
        pipeline: Pipeline,
        request: ProcessRequest,
        effective_mode: str,
        mode_note: str | None,
        preparation: ProcessPreparation,
        result: RunResult,
        elapsed_s: float,
    ) -> dict[str, Any]:
        hdf_out_path = write_hdf_output(
            request.write_hdf,
            run_name=request.run_name or run_id,
            result=result,
            pipeline_yaml=session.pipeline_yaml or "",
        )

        topo_ids = ordered_step_ids(pipeline)
        executed_set = set(result.executed_steps)
        skipped_steps = [step_id for step_id in topo_ids if step_id not in executed_set]
        details: dict[str, Any] = {
            "status": "succeeded",
            "executed_steps": result.executed_steps,
            "num_steps": len(result.executed_steps),
            "changed_sources": request.changed_sources,
            "changed_keys": request.changed_keys,
            "skipped_steps": skipped_steps,
            "step_durations_s": result.step_durations,
            "elapsed_s": elapsed_s,
        }
        if mode_note:
            details["note"] = mode_note
        if preparation.dirty_step_ids_ordered:
            details["dirty_steps"] = preparation.dirty_step_ids_ordered
        if preparation.boundary_step_id is not None:
            details["checkpoint_boundary_step"] = preparation.boundary_step_id
        if hdf_out_path is not None:
            details["hdf_output"] = hdf_out_path

        run_meta = self.manager.mark_run_succeeded(session_id, run_id, details=details)
        return {
            "session_id": session_id,
            "run_id": run_id,
            "state": "idle",
            "status": run_meta.get("status"),
            "effective_mode": effective_mode,
            "note": run_meta.get("note"),
            "hdf_output": run_meta.get("hdf_output"),
        }

    def _handle_process_failure(
        self,
        *,
        session: PipelineSession,
        session_id: str,
        request: ProcessRequest,
        effective_mode: str,
        run_id: str,
        preparation: ProcessPreparation,
        sources: IoSources | None,
        exc: Exception,
    ) -> dict[str, Any]:
        if effective_mode == "partial" and preparation.snapshot_before_partial is not None:
            session.processing_data = preparation.snapshot_before_partial

        error_code = "PARTIAL_RUN_FAILED" if effective_mode == "partial" else "RUN_FAILED"
        self.manager.mark_run_failed(
            session_id,
            run_id,
            code=error_code,
            message=str(exc),
            details={"exception_type": type(exc).__name__},
        )
        if request.mode == "auto" and effective_mode == "partial" and sources is not None:
            return self._run_auto_fallback(
                session=session,
                session_id=session_id,
                request=request,
                sources=sources,
                recovered_from_run_id=run_id,
                fallback_reason=exc,
            )
        raise ApiError(
            status_code=500,
            detail={
                "code": error_code,
                "message": str(exc),
                "details": {"session_id": session_id, "run_id": run_id},
            },
        ) from exc

    def _run_auto_fallback(
        self,
        *,
        session: PipelineSession,
        session_id: str,
        request: ProcessRequest,
        sources: IoSources,
        recovered_from_run_id: str,
        fallback_reason: Exception,
    ) -> dict[str, Any]:
        fallback_run = self.manager.enqueue_run(
            session_id,
            mode="full",
            changed_sources=request.changed_sources,
            effective_mode="full",
        )
        fallback_id = str(fallback_run["run_id"])
        try:
            fallback_pipeline = Pipeline.from_yaml(session.pipeline_yaml or "")
            fallback_t0 = perf_counter()
            fallback_result = run_pipeline_job(
                fallback_pipeline,
                sources=sources,
                processing_data=None,
                trace=session.trace_enabled,
                trace_watch=session.trace_watch,
            )
            fallback_elapsed = perf_counter() - fallback_t0
            session.processing_data = fallback_result.processing_data

            hdf_out_path = write_hdf_output(
                request.write_hdf,
                run_name=request.run_name or fallback_id,
                result=fallback_result,
                pipeline_yaml=session.pipeline_yaml or "",
            )

            fallback_topo_ids = ordered_step_ids(fallback_result.pipeline)
            fallback_executed_set = set(fallback_result.executed_steps)
            fallback_skipped = [step_id for step_id in fallback_topo_ids if step_id not in fallback_executed_set]
            done = self.manager.mark_run_succeeded(
                session_id,
                fallback_id,
                details={
                    "status": "succeeded",
                    "executed_steps": fallback_result.executed_steps,
                    "num_steps": len(fallback_result.executed_steps),
                    "note": "Auto fallback succeeded after partial failure.",
                    "recovered_from_run_id": recovered_from_run_id,
                    "fallback_reason": str(fallback_reason),
                    "changed_sources": request.changed_sources,
                    "changed_keys": request.changed_keys,
                    "skipped_steps": fallback_skipped,
                    "step_durations_s": fallback_result.step_durations,
                    "elapsed_s": fallback_elapsed,
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
                "recovered_from_run_id": recovered_from_run_id,
                "fallback_reason": done.get("fallback_reason"),
                "hdf_output": done.get("hdf_output"),
            }
        except Exception as fallback_exc:
            self.manager.mark_run_failed(
                session_id,
                fallback_id,
                code="FULL_RUN_FAILED",
                message=str(fallback_exc),
                details={
                    "exception_type": type(fallback_exc).__name__,
                    "recovered_from_run_id": recovered_from_run_id,
                },
            )
            raise ApiError(
                status_code=500,
                detail={
                    "code": "FULL_RUN_FAILED",
                    "message": str(fallback_exc),
                    "details": {"session_id": session_id, "run_id": fallback_id},
                },
            ) from fallback_exc
