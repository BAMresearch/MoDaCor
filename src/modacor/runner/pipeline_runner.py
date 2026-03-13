# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from modacor.dataclasses.processing_data import ProcessingData
from modacor.debug.pipeline_tracer import PipelineTracer
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.runner.pipeline import Pipeline

__all__ = ["RunResult", "run_pipeline_job"]


@dataclass(slots=True)
class RunResult:
    """Container returned by :func:`run_pipeline_job`."""

    processing_data: ProcessingData
    pipeline: Pipeline
    tracer: PipelineTracer | None
    step_durations: dict[str, float]
    executed_steps: list[str]
    stopped_after_step: str | None


def _ensure_pipeline(pipeline: Pipeline | Path | str) -> Pipeline:
    if isinstance(pipeline, Pipeline):
        return pipeline
    return Pipeline.from_yaml_file(Path(pipeline))


def run_pipeline_job(
    pipeline: Pipeline | Path | str,
    *,
    sources: IoSources | None = None,
    sinks: IoSinks | None = None,
    processing_data: ProcessingData | None = None,
    trace: bool = False,
    trace_watch: dict[str, list[str]] | None = None,
    tracer_kwargs: dict[str, Any] | None = None,
    stop_after: str | None = None,
    selected_step_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> RunResult:
    """
    Execute a pipeline end-to-end.

    This helper is intentionally shared between notebooks and CLI usage so both
    execution modes follow the exact same scheduler path.
    """
    pipeline_obj = _ensure_pipeline(pipeline)
    processing_data_obj = processing_data if processing_data is not None else ProcessingData()
    sources_obj = sources if sources is not None else IoSources()
    sinks_obj = sinks if sinks is not None else IoSinks()

    tracer: PipelineTracer | None = None
    if trace:
        _kwargs = dict(tracer_kwargs or {})
        _kwargs.setdefault("watch", trace_watch or {})
        _kwargs.setdefault("record_only_on_change", True)
        _kwargs.setdefault("record_empty_step_events", True)
        tracer = PipelineTracer(**_kwargs)

    pipeline_obj.clear_trace_events()
    pipeline_obj._reinitialize()
    pipeline_obj.prepare()

    stop_token = str(stop_after) if stop_after is not None else None
    selected_steps = {str(step_id) for step_id in selected_step_ids} if selected_step_ids is not None else None
    step_durations: dict[str, float] = {}
    executed_steps: list[str] = []
    stopped_after_step: str | None = None

    while pipeline_obj.is_active():
        should_stop = False
        for node in pipeline_obj.get_ready():
            node.processing_data = processing_data_obj
            node.io_sources = sources_obj
            node.io_sinks = sinks_obj

            step_id = str(node.step_id)
            if selected_steps is None or step_id in selected_steps:
                t0 = perf_counter()
                node.execute(processing_data_obj)
                duration = perf_counter() - t0

                step_durations[step_id] = duration
                executed_steps.append(step_id)

                if tracer is not None:
                    tracer.after_step(node, processing_data_obj, duration_s=duration)
                    pipeline_obj.attach_tracer_event(
                        node,
                        tracer,
                        include_rendered_trace=True,
                        include_rendered_config=True,
                        rendered_format="text/html",
                    )

            pipeline_obj.done(node)

            if stop_token is not None and step_id == stop_token:
                stopped_after_step = step_id
                should_stop = True
                break

        if should_stop:
            break

    return RunResult(
        processing_data=processing_data_obj,
        pipeline=pipeline_obj,
        tracer=tracer,
        step_durations=step_durations,
        executed_steps=executed_steps,
        stopped_after_step=stopped_after_step,
    )
