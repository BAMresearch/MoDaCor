# SPDX-License-Identifier: BSD-3-Clause
"""Process request containers and shared trace/provenance helpers."""

from __future__ import annotations

__all__ = ["ProcessRequest", "ChunkPublishRequest", "ProcessPreparation", "TRACE_REPORT_LINES"]

from dataclasses import dataclass, field
from typing import Any

from modacor.dataclasses.processing_data import ProcessingData
from modacor.debug.pipeline_tracer import PlainUnicodeRenderer
from modacor.io.chunking import ChunkSpec, ProvisionalChunkSpec
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import RunResult

from .session_manager import PipelineSession

TRACE_REPORT_LINES = 500


def _source_provenance(session: PipelineSession) -> list[dict[str, str]]:
    """Record source identity without persisting constructor kwargs or credentials."""

    return [
        {name: str(registration[name]) for name in ("ref", "type", "location")}
        for registration in session.sources.values()
    ]


def _pipeline_trace_events(pipeline: Pipeline) -> list[Any]:
    """Flatten lightweight, array-free step events in execution order."""

    return [event for step_events in pipeline.trace_events.values() for event in step_events]


@dataclass(frozen=True, slots=True)
class ChunkPublishRequest:
    output_id: str
    chunk_spec: ChunkSpec | ProvisionalChunkSpec


@dataclass(slots=True)
class ProcessRequest:
    """Normalized process request payload used by runtime service methods."""

    mode: str
    changed_sources: list[str]
    changed_keys: list[str]
    write_hdf: dict[str, Any] | None = None
    run_name: str | None = None
    rollback_snapshot: bool = True
    chunk_output: ChunkPublishRequest | None = None
    resolved_source_slices: list[dict[str, Any]] = field(default_factory=list)


class _ChunkPublicationError(RuntimeError):
    def __init__(
        self,
        output_id: str,
        chunk_spec: ChunkSpec | ProvisionalChunkSpec,
        original_exception: Exception,
    ) -> None:
        super().__init__(str(original_exception))
        self.output_id = output_id
        self.chunk_spec = chunk_spec
        self.original_exception = original_exception


@dataclass(slots=True)
class ProcessPreparation:
    """Execution preparation details for a process request."""

    selected_step_ids: set[str] | None = None
    snapshot_before_partial: ProcessingData | None = None
    dirty_step_ids_ordered: list[str] = field(default_factory=list)
    boundary_step_id: str | None = None
    early_response: dict[str, Any] | None = None


def _trace_report(result: RunResult) -> str | None:
    if result.tracer is None:
        return None
    return result.tracer.last_report(
        TRACE_REPORT_LINES,
        renderer=PlainUnicodeRenderer(wrap_in_markdown_codeblock=False),
    )
