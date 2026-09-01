# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.modules.base_modules.append_processing_data import AppendProcessingData
from modacor.modules.base_modules.append_sink import AppendSink
from modacor.modules.base_modules.append_source import AppendSource
from modacor.modules.base_modules.sink_processing_data import SinkProcessingData
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
    add_source = AppendSource(step_id="src")
    downstream = DummyStep(step_id="down")

    add_source.modify_config_by_dict(
        {
            "source_identifier": ["background"],
            "source_location": ["/tmp/background.nxs"],
            "iosource_module": "modacor.io.hdf.hdf_source.HDFSource",
        }
    )

    pipeline = Pipeline.from_dict({add_source: set(), downstream: {add_source}}, name="dirty-test-src-id")

    dirty = find_dirty_step_ids(pipeline, changed_sources=["background"])
    assert dirty == {"src", "down"}


def test_find_dirty_step_ids_with_changed_keys_matches_processing_patterns():
    load = DummyStep(step_id="load")
    corr = DummyStep(step_id="corr")

    load.configuration["processing_key"] = "sample"
    load.configuration["databundle_output_key"] = "signal"
    corr.configuration["with_processing_keys"] = ["sample"]

    pipeline = Pipeline.from_dict({load: set(), corr: {load}}, name="dirty-test-keys")

    dirty = find_dirty_step_ids(pipeline, changed_keys=["sample.signal"])
    assert dirty == {"load", "corr"}


def test_append_processing_data_dependency_contract_is_explicit():
    step = AppendProcessingData(step_id="load")
    step.modify_config_by_dict(
        {
            "processing_key": "sample",
            "signal_location": "sample_file::/entry/data",
            "rank_of_data": "metadata::/rank",
            "databundle_output_key": "signal",
            "units_location": "metadata::/units",
            "weights_location": "sample_file::/entry/weights",
            "uncertainties_sources": {"poisson": "sample_file::/entry/sigma"},
        }
    )

    contract = step.dependency_contract()

    assert contract.source_refs == {"sample_file", "metadata"}
    assert contract.processing_reads == frozenset()
    assert contract.processing_writes == {"sample.signal"}


def test_append_sink_dependency_contract_is_not_processing_wildcard():
    add_sink = AppendSink(step_id="sink")
    add_sink.modify_config_by_dict(
        {
            "sink_identifier": "csv",
            "sink_location": "/tmp/out.csv",
            "iosink_module": "modacor.io.csv.csv_sink.CSVSink",
        }
    )

    pipeline = Pipeline.from_dict({add_sink: set()}, name="append-sink-contract")

    assert find_dirty_step_ids(pipeline, changed_keys=["sample.signal"]) == set()


def test_sink_processing_data_dependency_contract_reads_exported_paths():
    export = SinkProcessingData(step_id="export")
    export.modify_config_by_dict(
        {
            "target": "csv::",
            "data_paths": ["/sample/signal/signal", "/sample/Q/signal"],
        }
    )

    contract = export.dependency_contract()

    assert contract.processing_reads == {"sample.signal", "sample.Q"}
    assert contract.processing_writes == frozenset()
    assert find_dirty_step_ids(Pipeline.from_dict({export: set()}), changed_keys=["sample.Q"]) == {"export"}
    assert find_dirty_step_ids(Pipeline.from_dict({export: set()}), changed_keys=["other.signal"]) == set()
