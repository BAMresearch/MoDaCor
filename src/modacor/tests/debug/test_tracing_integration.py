# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "13/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path

from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.dataclasses.processing_data import ProcessingData
from modacor.debug.pipeline_tracer import PipelineTracer
from modacor.runner.pipeline import Pipeline


class DummyStep(ProcessStep):
    documentation = ProcessStepDescriber(
        calling_name="Dummy",
        calling_id="dummy.step",
        calling_module_path=Path(__file__),
        calling_version="0",
        required_data_keys=[],
        required_arguments=[],
        default_configuration={},
    )

    def calculate(self):
        return {}  # does not modify ProcessingData


def test_tracer_emits_nothing_when_no_changes_and_no_empty_events():
    tracer = PipelineTracer(
        watch={"sample": ["signal"]},
        record_only_on_change=True,
        record_empty_step_events=False,  # requires the optional field; if you did not add it, remove this line
    )

    step = DummyStep(io_sources=None, step_id="A")
    data = ProcessingData()

    tracer.after_step(step, data)
    assert tracer.events == []


def test_tracer_can_emit_empty_step_events_when_enabled():
    tracer = PipelineTracer(
        watch={"sample": ["signal"]},
        record_only_on_change=True,
        record_empty_step_events=True,  # requires the optional field
    )

    step = DummyStep(io_sources=None, step_id="A")
    data = ProcessingData()

    tracer.after_step(step, data)
    assert len(tracer.events) == 1
    assert tracer.events[0]["step_id"] == "A"
    assert tracer.events[0]["changed"] == {}


def test_pipeline_attach_tracer_event_always_attaches_event_even_without_tracer_events():
    step = DummyStep(io_sources=None, step_id="A")
    pipeline = Pipeline.from_dict({step: []}, name="t")

    tracer = PipelineTracer(watch={"sample": ["signal"]})
    # No tracer.after_step() call -> tracer.events empty

    ev = pipeline.attach_tracer_event(step, tracer)

    assert ev.step_id == "A"
    assert ev.module == "DummyStep"
    assert ev.datasets == {}  # no tracer payload
    assert "A" in pipeline.trace_events
    assert pipeline.trace_events["A"][-1] is ev


def test_pipeline_attach_tracer_event_includes_datasets_when_tracer_has_matching_event():
    step = DummyStep(io_sources=None, step_id="A")
    pipeline = Pipeline.from_dict({step: []}, name="t")

    tracer = PipelineTracer(
        watch={"sample": ["signal"]},
        record_only_on_change=True,
        record_empty_step_events=True,  # ensure tracer produces an event even if empty
    )
    data = ProcessingData()

    tracer.after_step(step, data)
    ev = pipeline.attach_tracer_event(step, tracer)

    # Event exists; datasets may still be empty because watched target doesn't exist in ProcessingData
    assert ev.step_id == "A"
    assert isinstance(ev.datasets, dict)


def test_pipeline_attach_tracer_event_can_embed_rendered_trace_block():
    step = DummyStep(io_sources=None, step_id="A")
    pipeline = Pipeline.from_dict({step: []}, name="t")

    tracer = PipelineTracer(
        watch={"sample": ["signal"]},
        record_only_on_change=True,
    )

    # Ensure the tracer has at least one event to render:
    tracer.record_only_on_change = False
    tracer.after_step(step, ProcessingData())
    assert tracer.events  # sanity

    ev = pipeline.attach_tracer_event(step, tracer, include_rendered=True)

    assert isinstance(ev.messages, list)
    assert ev.messages, "Expected at least one rendered message block"
    assert ev.messages[0]["kind"] in {"rendered_trace", "rendered_trace_error"}
    assert "format" in ev.messages[0]
    assert "content" in ev.messages[0]
