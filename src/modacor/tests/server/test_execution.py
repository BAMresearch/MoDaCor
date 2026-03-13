# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.runner.pipeline import Pipeline
from modacor.server.execution import find_dirty_step_ids


class DummyStep(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        return {}


def test_find_dirty_step_ids_from_changed_source_and_descendants():
    load = DummyStep(step_id="load")
    corr = DummyStep(step_id="corr")
    out = DummyStep(step_id="out")

    load.configuration["signal_location"] = "sample::/entry/data"
    corr.configuration["with_processing_keys"] = ["sample"]
    out.configuration["target"] = "export_csv::"

    pipeline = Pipeline.from_dict(
        {
            load: set(),
            corr: {load},
            out: {corr},
        },
        name="dirty-test",
    )

    dirty = find_dirty_step_ids(pipeline, changed_sources=["sample"])
    assert dirty == {"load", "corr", "out"}


def test_find_dirty_step_ids_handles_append_source_style_identifier():
    add_source = DummyStep(step_id="src")
    downstream = DummyStep(step_id="down")

    add_source.configuration["source_identifier"] = ["background"]

    pipeline = Pipeline.from_dict({add_source: set(), downstream: {add_source}}, name="dirty-test-src-id")

    dirty = find_dirty_step_ids(pipeline, changed_sources=["background"])
    assert dirty == {"src", "down"}
