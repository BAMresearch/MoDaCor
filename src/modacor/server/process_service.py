# SPDX-License-Identifier: BSD-3-Clause
"""ProcessService implementation for the runtime API."""

from __future__ import annotations

__all__ = ["ProcessService"]

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modacor.io.chunking import ChunkSpec, ProvisionalChunkSpec
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.runner import run_pipeline_job
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import RunResult
from modacor.runner.process_step_registry import ProcessStepRegistry

from .chunked_outputs import ChunkedOutputManager
from .errors import ApiError
from .execution import find_dirty_step_ids
from .io_utils import bind_chunk_source_slices, build_sinks_from_session, build_sources_from_session, write_hdf_output
from .planning import build_dry_run_plan, missing_required_source_refs, ordered_step_ids, resolve_effective_mode
from .process_failure import ProcessFailureHandler
from .process_requests import (
    ChunkPublishRequest,
    ProcessPreparation,
    ProcessRequest,
    _ChunkPublicationError,
    _pipeline_trace_events,
    _source_provenance,
    _trace_report,
)
from .runtime_policy import RuntimePolicy
from .session_manager import PipelineSession
from .session_service import SessionService


def _chunk_spec_from_request(spec_raw: dict[str, Any]) -> ChunkSpec:
    try:
        return ChunkSpec.from_dict(spec_raw)
    except KeyError as exc:
        # A missing field is a malformed request, not an unknown output (404).
        raise ValueError(f"Invalid chunk_spec: missing field {exc}.") from exc


@dataclass(slots=True)
class ProcessService(SessionService):
    """ProcessService operations for the runtime facade."""

    policy: RuntimePolicy
    process_step_registry: ProcessStepRegistry
    chunked_outputs: ChunkedOutputManager
    _failures: ProcessFailureHandler

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
        sinks: IoSinks | None = None
        try:
            pipeline = Pipeline.from_yaml(
                session.pipeline_yaml or "",
                registry=self.process_step_registry,
                max_expanded_steps=self.policy.max_expanded_pipeline_steps,
            )
            sources = build_sources_from_session(
                session,
                buffer_store=self.manager.buffer_store,
                runtime_policy=self.policy,
            )
            if request.chunk_output is not None:
                resource = self.chunked_outputs.get(request.chunk_output.output_id)
                active_plan = resource.plan or resource.provisional_plan
                assert active_plan is not None
                request.resolved_source_slices = bind_chunk_source_slices(
                    sources,
                    session,
                    active_plan,
                    request.chunk_output.chunk_spec,
                )
            sinks = build_sinks_from_session(
                session,
                pipeline=pipeline,
                buffer_store=self.manager.buffer_store,
                runtime_policy=self.policy,
            )
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
                sinks=sinks,
                effective_mode=effective_mode,
                preparation=preparation,
                chunk_spec=None if request.chunk_output is None else request.chunk_output.chunk_spec,
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
            # Run boundary: record and log every failure, including plugin/programming errors.
            return self._failures._handle_process_failure(
                session=session,
                session_id=session_id,
                request=request,
                effective_mode=effective_mode,
                run_id=run_id,
                preparation=preparation,
                sources=sources,
                sinks=sinks,
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
                registry=self.process_step_registry,
                max_expanded_steps=self.policy.max_expanded_pipeline_steps,
            )
        except Exception as exc:
            # API boundary: the chained cause is logged by the API with the unchanged error code.
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

        write_hdf_raw = payload.get("write_hdf")
        write_hdf = dict(write_hdf_raw) if isinstance(write_hdf_raw, dict) else None
        if write_hdf is not None and write_hdf.get("path"):
            try:
                self.policy.validate_write_hdf_path(write_hdf["path"])
            except ValueError as exc:
                raise ApiError(status_code=422, detail=str(exc)) from exc
        run_name_raw = payload.get("run_name")
        run_name = str(run_name_raw) if run_name_raw is not None else None
        chunk_output: ChunkPublishRequest | None = None
        chunk_output_raw = payload.get("chunk_output")
        if chunk_output_raw is not None:
            if not isinstance(chunk_output_raw, dict):
                raise ApiError(status_code=422, detail="chunk_output must be an object.")
            output_id = str(chunk_output_raw.get("output_id", "")).strip()
            spec_raw = chunk_output_raw.get("chunk_spec")
            chunk_id = str(chunk_output_raw.get("chunk_id", "")).strip()
            if not output_id or (isinstance(spec_raw, dict) == bool(chunk_id)):
                raise ApiError(
                    status_code=422,
                    detail="chunk_output requires output_id and exactly one of chunk_spec or chunk_id.",
                )
            try:
                resource = self.chunked_outputs.get(output_id)
                if isinstance(spec_raw, dict):
                    if resource.plan is None:
                        raise ValueError("An awaiting-schema output accepts chunk_id, not a complete chunk_spec.")
                    chunk_spec: ChunkSpec | ProvisionalChunkSpec = _chunk_spec_from_request(spec_raw)
                    chunk_spec.validate_for_plan(resource.plan)
                else:
                    chunk_spec = self.chunked_outputs.execution_chunk(output_id, chunk_id)
            except KeyError as exc:
                raise ApiError(status_code=404, detail=str(exc)) from exc
            except (TypeError, ValueError) as exc:
                raise ApiError(status_code=422, detail=str(exc)) from exc
            chunk_output = ChunkPublishRequest(output_id=output_id, chunk_spec=chunk_spec)
            active_plan = resource.plan or resource.provisional_plan
            assert active_plan is not None
            for binding in active_plan.source_bindings:
                if binding.role != "static" and binding.source_ref not in changed_sources:
                    changed_sources.append(binding.source_ref)
            if not active_plan.source_bindings:
                driver_ref = str(active_plan.driver.get("source", "")).partition("::")[0].strip()
                if driver_ref and driver_ref not in changed_sources:
                    changed_sources.append(driver_ref)
        if mode == "partial" and not changed_sources and not changed_keys:
            raise ApiError(status_code=422, detail="partial mode requires changed_sources or changed_keys.")
        return ProcessRequest(
            mode=mode,
            changed_sources=changed_sources,
            changed_keys=changed_keys,
            write_hdf=write_hdf,
            run_name=run_name,
            rollback_snapshot=bool(payload.get("rollback_snapshot", True)),
            chunk_output=chunk_output,
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
            if request.chunk_output is not None:
                raise RuntimeError("A chunk_output request cannot publish a partial run with no dirty steps.")
            return ProcessPreparation(
                early_response=self._mark_noop_process_run(
                    session_id=session_id,
                    run_id=run_id,
                    changed_keys=request.changed_keys,
                    topo_ids=topo_ids,
                )
            )

        snapshot_before_partial = (
            deepcopy(session.processing_data)
            if request.rollback_snapshot and session.processing_data is not None
            else None
        )
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
        sinks: IoSinks,
        effective_mode: str,
        preparation: ProcessPreparation,
        chunk_spec: ChunkSpec | ProvisionalChunkSpec | None = None,
    ) -> tuple[RunResult, float]:
        reuse_processing_data = effective_mode == "partial" and session.processing_data is not None
        run_t0 = perf_counter()
        result = run_pipeline_job(
            pipeline,
            sources=sources,
            sinks=sinks,
            processing_data=session.processing_data if reuse_processing_data else None,
            trace=session.trace_enabled,
            trace_watch=session.trace_watch,
            tracer_kwargs={
                "snapshot_processing_data": session.trace_snapshot_processing_data,
                "snapshot_step_ids": set(session.trace_snapshot_step_ids),
            },
            selected_step_ids=preparation.selected_step_ids,
            capture_partial_on_error=True,
            chunk_spec=chunk_spec,
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
            runtime_policy=self.policy,
        )

        chunk_write = None
        published_chunk_spec: ChunkSpec | None = None
        if request.chunk_output is not None:
            try:
                chunk_write = self.chunked_outputs.write_chunk(
                    request.chunk_output.output_id,
                    result.processing_data,
                    chunk=request.chunk_output.chunk_spec,
                    execution_metadata={
                        "session_id": session_id,
                        "run_id": run_id,
                        "effective_mode": effective_mode,
                        "sources": _source_provenance(session),
                        "source_slices": request.resolved_source_slices,
                        **request.chunk_output.chunk_spec.identity_dict(),
                    },
                    pipeline_provenance=result.pipeline.provenance(),
                    trace_events=_pipeline_trace_events(result.pipeline) if session.trace_enabled else None,
                )
                if isinstance(request.chunk_output.chunk_spec, ProvisionalChunkSpec):
                    resolved = self.chunked_outputs.execution_chunk(
                        request.chunk_output.output_id,
                        request.chunk_output.chunk_spec.chunk_id,
                    )
                    assert isinstance(resolved, ChunkSpec)
                    published_chunk_spec = resolved
                else:
                    published_chunk_spec = request.chunk_output.chunk_spec
            except Exception as exc:
                # Publication boundary: preserve every backend failure for the run-error record.
                raise _ChunkPublicationError(
                    request.chunk_output.output_id,
                    request.chunk_output.chunk_spec,
                    exc,
                ) from exc

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
            "rollback_snapshot": request.rollback_snapshot,
        }
        if mode_note:
            details["note"] = mode_note
        if preparation.dirty_step_ids_ordered:
            details["dirty_steps"] = preparation.dirty_step_ids_ordered
        if preparation.boundary_step_id is not None:
            details["checkpoint_boundary_step"] = preparation.boundary_step_id
        if hdf_out_path is not None:
            details["hdf_output"] = hdf_out_path
        if chunk_write is not None:
            assert published_chunk_spec is not None
            details["chunk_output"] = {
                "output_id": request.chunk_output.output_id,
                **published_chunk_spec.identity_dict(),
                **chunk_write.to_dict(),
            }
            if isinstance(request.chunk_output.chunk_spec, ProvisionalChunkSpec):
                details["chunk_output"]["provisional_hash"] = request.chunk_output.chunk_spec.provisional_hash
        if request.resolved_source_slices:
            details["source_slices"] = request.resolved_source_slices
        trace_report = _trace_report(result)
        if trace_report is not None:
            details["trace_report"] = trace_report

        run_meta = self.manager.mark_run_succeeded(session_id, run_id, details=details)
        response = {
            "session_id": session_id,
            "run_id": run_id,
            "state": "idle",
            "status": run_meta.get("status"),
            "effective_mode": effective_mode,
            "note": run_meta.get("note"),
            "hdf_output": run_meta.get("hdf_output"),
        }
        if chunk_write is not None:
            response["chunk_output"] = run_meta["chunk_output"]
        return response
