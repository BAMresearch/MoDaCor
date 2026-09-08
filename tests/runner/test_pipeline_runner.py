# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.processing_data import ProcessingData
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import PipelineRunError, run_pipeline_job


class SeedSignal(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        out = DataBundle()
        out["signal"] = BaseData(signal=np.array([1.0]), units=ureg.dimensionless)
        self.processing_data["sample"] = out
        return {"sample": out}


class AddOne(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        current = self.processing_data["sample"]["signal"]
        out = DataBundle()
        out["signal"] = BaseData(signal=current.signal + 1.0, units=current.units)
        self.processing_data["sample"] = out
        return {"sample": out}


class FailStep(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        raise ValueError("synthetic failure")


def test_run_pipeline_job_executes_and_traces():
    s1 = SeedSignal(step_id="s1")
    s2 = AddOne(step_id="s2")
    pipeline = Pipeline.from_dict({s1: set(), s2: {s1}}, name="unit-test")

    result = run_pipeline_job(pipeline, trace=True, trace_watch={"sample": ["signal"]})

    assert result.executed_steps == ["s1", "s2"]
    assert "s1" in result.step_durations
    assert "s2" in result.step_durations
    assert np.allclose(result.processing_data["sample"]["signal"].signal, np.array([2.0]))
    assert result.tracer is not None
    assert "s1" in result.pipeline.trace_events
    assert "s2" in result.pipeline.trace_events


def test_run_pipeline_job_can_reuse_pipeline_instance_without_scheduler_reset():
    s1 = SeedSignal(step_id="s1")
    s2 = AddOne(step_id="s2")
    pipeline = Pipeline.from_dict({s1: set(), s2: {s1}}, name="unit-test-reuse")

    first = run_pipeline_job(pipeline)
    second = run_pipeline_job(pipeline)

    assert first.executed_steps == ["s1", "s2"]
    assert second.executed_steps == ["s1", "s2"]
    assert np.allclose(second.processing_data["sample"]["signal"].signal, np.array([2.0]))


def test_run_pipeline_job_error_keeps_partial_trace_context():
    s1 = SeedSignal(step_id="seed")
    s2 = FailStep(step_id="fail")
    pipeline = Pipeline.from_dict({s1: set(), s2: {s1}}, name="unit-test-fail")

    with pytest.raises(PipelineRunError) as excinfo:
        run_pipeline_job(
            pipeline,
            trace=True,
            trace_watch={"sample": ["signal"]},
            capture_partial_on_error=True,
        )

    err = excinfo.value
    assert err.failed_step_id == "fail"
    assert isinstance(err.original_exception, ValueError)
    assert err.result.executed_steps == ["seed"]
    assert err.result.tracer is not None
    assert "seed" in err.result.tracer.last_report(10)


def test_run_pipeline_job_raises_original_error_by_default():
    s1 = SeedSignal(step_id="seed")
    s2 = FailStep(step_id="fail")
    pipeline = Pipeline.from_dict({s1: set(), s2: {s1}}, name="unit-test-fail-default")

    with pytest.raises(ValueError, match="synthetic failure"):
        run_pipeline_job(pipeline)


def test_run_pipeline_job_can_stop_after_step():
    s1 = SeedSignal(step_id="first")
    s2 = AddOne(step_id="second")
    pipeline = Pipeline.from_dict({s1: set(), s2: {s1}}, name="unit-test-stop")

    result = run_pipeline_job(pipeline, stop_after="first")

    assert result.executed_steps == ["first"]
    assert result.stopped_after_step == "first"
    assert np.allclose(result.processing_data["sample"]["signal"].signal, np.array([1.0]))


def test_run_pipeline_job_can_execute_selected_steps_only():
    s1 = SeedSignal(step_id="seed")
    s2 = AddOne(step_id="add")
    pipeline = Pipeline.from_dict({s1: set(), s2: {s1}}, name="unit-test-select")

    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(signal=np.array([10.0]), units=ureg.dimensionless)
    processing_data["sample"] = bundle

    result = run_pipeline_job(
        pipeline,
        processing_data=processing_data,
        selected_step_ids={"add"},
    )

    assert result.executed_steps == ["add"]
    assert np.allclose(result.processing_data["sample"]["signal"].signal, np.array([11.0]))
