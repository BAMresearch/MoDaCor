# SPDX-License-Identifier: BSD-3-Clause
"""ProcessFailureHandler implementation for the runtime API."""

from __future__ import annotations

__all__ = ["ProcessFailureHandler"]

import logging
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modacor.io.chunking import ChunkSpec, ProvisionalChunkSpec
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.runner import run_pipeline_job
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import PipelineRunError
from modacor.runner.process_step_registry import ProcessStepRegistry

from .chunked_outputs import ChunkedOutputManager
from .errors import ApiError, IOBuildError
from .io_utils import write_hdf_output
from .planning import ordered_step_ids
from .process_requests import (
    ProcessPreparation,
    ProcessRequest,
    _ChunkPublicationError,
    _source_provenance,
    _trace_report,
)
from .runtime_policy import RuntimePolicy
from .session_manager import PipelineSession
from .session_service import SessionService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessFailureHandler(SessionService):
    """ProcessFailureHandler operations for the runtime facade."""

    policy: RuntimePolicy
    process_step_registry: ProcessStepRegistry
    chunked_outputs: ChunkedOutputManager

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
        sinks: IoSinks | None,
        exc: Exception,
    ) -> dict[str, Any]:
        publication_failure = isinstance(exc, _ChunkPublicationError)
        if not publication_failure and effective_mode == "partial" and preparation.snapshot_before_partial is not None:
            session.processing_data = preparation.snapshot_before_partial

        original_exception = (
            exc.original_exception if isinstance(exc, (PipelineRunError, _ChunkPublicationError, IOBuildError)) else exc
        )
        error_code = (
            "CHUNK_WRITE_FAILED"
            if publication_failure
            else ("PARTIAL_RUN_FAILED" if effective_mode == "partial" else "RUN_FAILED")
        )
        error_details: dict[str, Any] = {
            "exception_type": type(original_exception).__name__,
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, IOBuildError):
            error_code = "SOURCE_UNAVAILABLE" if exc.kind == "source" else "SINK_UNAVAILABLE"
            error_details.update(exc.details())
        if isinstance(exc, PipelineRunError):
            if exc.failed_step_id is not None:
                error_details["failed_step_id"] = exc.failed_step_id
            trace_report = _trace_report(exc.result)
            if trace_report is not None:
                error_details["trace_report"] = trace_report
        if isinstance(exc, _ChunkPublicationError):
            error_details.update(
                {
                    "processing_succeeded": True,
                    "output_id": exc.output_id,
                    "chunk_identity": exc.chunk_spec.identity_dict(),
                }
            )
        elif request.chunk_output is not None:
            error_details.update(
                {
                    "output_id": request.chunk_output.output_id,
                    "chunk_identity": request.chunk_output.chunk_spec.identity_dict(),
                }
            )

        logger.error(
            "Run failed session_id=%s run_id=%s code=%s failed_step_id=%s message=%s",
            session_id,
            run_id,
            error_code,
            error_details.get("failed_step_id"),
            str(exc),
            extra={
                "session_id": session_id,
                "run_id": run_id,
                "error_code": error_code,
                "failed_step_id": error_details.get("failed_step_id"),
            },
            exc_info=exc,
        )
        self.manager.mark_run_failed(
            session_id,
            run_id,
            code=error_code,
            message=str(exc),
            details=error_details,
        )
        if (
            not publication_failure
            and request.mode == "auto"
            and effective_mode == "partial"
            and sources is not None
            and sinks is not None
        ):
            return self._run_auto_fallback(
                session=session,
                session_id=session_id,
                request=request,
                sources=sources,
                sinks=sinks,
                recovered_from_run_id=run_id,
                fallback_reason=exc,
            )
        raise ApiError(
            status_code=500,
            detail={
                "code": error_code,
                "message": str(exc),
                "details": {
                    "session_id": session_id,
                    "run_id": run_id,
                    **(exc.details() if isinstance(exc, IOBuildError) else {}),
                },
            },
        ) from exc

    def _run_auto_fallback(
        self,
        *,
        session: PipelineSession,
        session_id: str,
        request: ProcessRequest,
        sources: IoSources,
        sinks: IoSinks,
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
            fallback_pipeline = Pipeline.from_yaml(
                session.pipeline_yaml or "",
                registry=self.process_step_registry,
                max_expanded_steps=self.policy.max_expanded_pipeline_steps,
            )
            fallback_t0 = perf_counter()
            fallback_result = run_pipeline_job(
                fallback_pipeline,
                sources=sources,
                sinks=sinks,
                processing_data=None,
                trace=session.trace_enabled,
                trace_watch=session.trace_watch,
                tracer_kwargs={
                    "snapshot_processing_data": session.trace_snapshot_processing_data,
                    "snapshot_step_ids": set(session.trace_snapshot_step_ids),
                },
                capture_partial_on_error=True,
                chunk_spec=None if request.chunk_output is None else request.chunk_output.chunk_spec,
            )
            fallback_elapsed = perf_counter() - fallback_t0
            session.processing_data = fallback_result.processing_data

            hdf_out_path = write_hdf_output(
                request.write_hdf,
                run_name=request.run_name or fallback_id,
                result=fallback_result,
                runtime_policy=self.policy,
            )

            chunk_write = None
            published_chunk_spec: ChunkSpec | None = None
            if request.chunk_output is not None:
                try:
                    chunk_write = self.chunked_outputs.write_chunk(
                        request.chunk_output.output_id,
                        fallback_result.processing_data,
                        chunk=request.chunk_output.chunk_spec,
                        execution_metadata={
                            "session_id": session_id,
                            "run_id": fallback_id,
                            "effective_mode": "full",
                            "sources": _source_provenance(session),
                            "source_slices": request.resolved_source_slices,
                            **request.chunk_output.chunk_spec.identity_dict(),
                        },
                        pipeline_provenance=fallback_result.pipeline.provenance(),
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

            fallback_topo_ids = ordered_step_ids(fallback_result.pipeline)
            fallback_executed_set = set(fallback_result.executed_steps)
            fallback_skipped = [step_id for step_id in fallback_topo_ids if step_id not in fallback_executed_set]
            trace_report = _trace_report(fallback_result)
            details: dict[str, Any] = {
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
            }
            if trace_report is not None:
                details["trace_report"] = trace_report
            if hdf_out_path:
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

            done = self.manager.mark_run_succeeded(
                session_id,
                fallback_id,
                details=details,
            )
            response = {
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
            if chunk_write is not None:
                response["chunk_output"] = done["chunk_output"]
            return response
        except Exception as fallback_exc:
            # Recovery boundary: retain the traceback of any failed fallback in the run record.
            original_exception = (
                fallback_exc.original_exception
                if isinstance(fallback_exc, (PipelineRunError, _ChunkPublicationError))
                else fallback_exc
            )
            error_details: dict[str, Any] = {
                "exception_type": type(original_exception).__name__,
                "traceback": traceback.format_exc(),
                "recovered_from_run_id": recovered_from_run_id,
            }
            if isinstance(fallback_exc, PipelineRunError):
                if fallback_exc.failed_step_id is not None:
                    error_details["failed_step_id"] = fallback_exc.failed_step_id
                trace_report = _trace_report(fallback_exc.result)
                if trace_report is not None:
                    error_details["trace_report"] = trace_report

            publication_failure = isinstance(fallback_exc, _ChunkPublicationError)
            if publication_failure:
                error_details.update(
                    {
                        "processing_succeeded": True,
                        "output_id": fallback_exc.output_id,
                        "chunk_identity": fallback_exc.chunk_spec.identity_dict(),
                    }
                )
            failure_code = "CHUNK_WRITE_FAILED" if publication_failure else "FULL_RUN_FAILED"
            self.manager.mark_run_failed(
                session_id,
                fallback_id,
                code=failure_code,
                message=str(fallback_exc),
                details=error_details,
            )
            raise ApiError(
                status_code=500,
                detail={
                    "code": failure_code,
                    "message": str(fallback_exc),
                    "details": {"session_id": session_id, "run_id": fallback_id},
                },
            ) from fallback_exc
