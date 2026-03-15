# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.processing_data import ProcessingData
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import run_pipeline_job


class SeedSignal(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        out = DataBundle()
        out["signal"] = BaseData(signal=np.array([1.0]), units=ureg.dimensionless)
        return {"sample": out}


class AddOne(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        current = self.processing_data["sample"]["signal"]
        out = DataBundle()
        out["signal"] = BaseData(signal=current.signal + 1.0, units=current.units)
        return {"sample": out}


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
