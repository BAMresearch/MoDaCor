# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from modacor import __version__, ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunking import (
    AxisSelector,
    ChunkArrayLayout,
    ChunkOutputLayout,
    ChunkPlacement,
    ChunkPlan,
    ChunkSpec,
    UnsupportedSinkCapability,
)
from modacor.io.hdf import HDFChunkedProcessingSink, HDFProcessingSink
from modacor.io.io_sinks import IoSinks


def _read_text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _processing_data(values: np.ndarray) -> ProcessingData:
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(signal=values, units=ureg.Unit("count"), rank_of_data=1)
    bundle.default_plot = "signal"
    processing_data["sample"] = bundle
    return processing_data


def _plan() -> ChunkPlan:
    return ChunkPlan(
        schema_version="1.0",
        plan_id="assembly-1",
        total_chunks=3,
        expected_chunk_ids=("c000000", "c000001", "c000002"),
        outputs=(
            ChunkOutputLayout(
                output_id="corrected_signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(ChunkArrayLayout(component="signal", final_shape=(5, 2), dtype="float32"),),
            ),
        ),
        driver={"source": "sample::/entry/data", "full_shape": [5, 2], "dtype": "float32"},
        batch_axes=(0,),
        data_axes=(1,),
        axis_rules=({"axis": 0, "chunk_size": 2},),
    )


def _chunk(plan: ChunkPlan, ordinal: int, start: int, stop: int) -> ChunkSpec:
    return ChunkSpec(
        schema_version=plan.schema_version,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        chunk_id=plan.expected_chunk_ids[ordinal],
        ordinal=ordinal,
        grid_index=(ordinal,),
        source_selection=(AxisSelector.sliced(start, stop), AxisSelector.all()),
        expected_input_shape=(stop - start, 2),
        placements=(
            ChunkPlacement(
                output_id="corrected_signal",
                destination_selection=(AxisSelector.sliced(start, stop), AxisSelector.all()),
                expected_shape=(stop - start, 2),
            ),
        ),
    )


def test_chunk_capability_routing_rejects_ordinary_sink(tmp_path: Path):
    sinks = IoSinks()
    sinks.register_sink(HDFProcessingSink(sink_reference="ordinary", resource_location=tmp_path / "out.h5"))

    assert sinks.get_sink("ordinary").supports_chunked_writes is False
    with pytest.raises(UnsupportedSinkCapability, match="HDFProcessingSink"):
        sinks.initialize_chunked("ordinary::run", _plan())


def test_hdf_chunked_sink_writes_edge_chunks_out_of_order_and_finalizes(tmp_path: Path):
    chunked_file = tmp_path / "chunked.h5"
    ordinary_file = tmp_path / "ordinary.h5"
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=chunked_file)

    with h5py.File(chunked_file, "w") as h5:
        h5.create_dataset("raw/frames", data=np.arange(3))

    initialized = sink.initialize_chunked("run1", plan)
    assert initialized.status == "writing"
    assert initialized.completed_chunks == 0

    chunks = [
        (_chunk(plan, 0, 0, 2), np.arange(4, dtype=np.float32).reshape(2, 2)),
        (_chunk(plan, 1, 2, 4), np.arange(4, 8, dtype=np.float32).reshape(2, 2)),
        (_chunk(plan, 2, 4, 5), np.arange(8, 10, dtype=np.float32).reshape(1, 2)),
    ]
    for chunk, values in (chunks[2], chunks[0], chunks[1]):
        result = sink.write_chunk(
            "run1",
            _processing_data(values),
            plan=plan,
            chunk=chunk,
            execution_metadata={"attempt": 1},
        )
        assert result.chunk_id == chunk.chunk_id

    duplicate = sink.write_chunk("run1", _processing_data(chunks[1][1]), plan=plan, chunk=chunks[1][0])
    assert duplicate.completed_chunks == 3

    finalized = sink.finalize_chunked(
        "run1",
        plan=plan,
        pipeline_spec={"name": "chunk-test"},
        pipeline_yaml="name: chunk-test\nsteps: {}\n",
    )
    assert finalized.status == "complete"
    assert sink.finalize_chunked("run1", plan=plan).status == "complete"

    expected = np.arange(10, dtype=np.float32).reshape(5, 2)
    HDFProcessingSink(resource_location=ordinary_file).write(
        "run1",
        _processing_data(expected),
        data_paths=["/sample/signal"],
        pipeline_spec={"name": "chunk-test"},
        pipeline_yaml="name: chunk-test\nsteps: {}\n",
    )

    with h5py.File(chunked_file, "r") as chunked, h5py.File(ordinary_file, "r") as ordinary:
        chunked_signal = chunked["processing/result/run1/sample/signal/signal"]
        ordinary_signal = ordinary["processing/result/run1/sample/signal/signal"]
        np.testing.assert_array_equal(chunked_signal, ordinary_signal)
        assert chunked_signal.dtype == ordinary_signal.dtype
        assert dict(chunked_signal.attrs) == dict(ordinary_signal.attrs)
        assert chunked.attrs["default"] == ordinary.attrs["default"] == "processing"
        np.testing.assert_array_equal(chunked["raw/frames"], np.arange(3))
        assert chunked["processing/result/run1"].attrs["modacor_version"] == __version__
        assert _read_text(chunked["processing/program_version"]) == __version__

        plan_group = chunked["processing/chunk_plans/assembly-1"]
        assert _read_text(plan_group["status"]) == "complete"
        assert ChunkPlan.from_dict(json.loads(_read_text(plan_group["plan_json"]))) == plan
        assert _read_text(plan_group["chunks/c000002/status"]) == "complete"
        assert json.loads(_read_text(plan_group["chunks/c000002/execution_json"])) == {"attempt": 1}


def test_hdf_chunked_sink_rejects_incomplete_finalize_and_resumes(tmp_path: Path):
    out_file = tmp_path / "resume.h5"
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run1", plan)
    chunk = _chunk(plan, 0, 0, 2)
    sink.write_chunk("run1", _processing_data(np.ones((2, 2), dtype=np.float32)), plan=plan, chunk=chunk)

    with pytest.raises(RuntimeError, match="incomplete chunks"):
        sink.finalize_chunked("run1", plan=plan)

    resumed = sink.initialize_chunked("run1", plan, collision="resume")
    assert resumed.status == "writing"
    assert resumed.completed_chunks == 1


def test_hdf_chunked_sink_rejects_overlap_and_conflicting_retry(tmp_path: Path):
    out_file = tmp_path / "overlap.h5"
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run1", plan)
    first = _chunk(plan, 0, 0, 2)
    sink.write_chunk("run1", _processing_data(np.ones((2, 2), dtype=np.float32)), plan=plan, chunk=first)

    overlapping = _chunk(plan, 1, 1, 3)
    with pytest.raises(ValueError, match="overlaps"):
        sink.write_chunk(
            "run1",
            _processing_data(np.ones((2, 2), dtype=np.float32)),
            plan=plan,
            chunk=overlapping,
        )

    conflicting = ChunkSpec(
        schema_version=first.schema_version,
        plan_id=first.plan_id,
        plan_hash=first.plan_hash,
        chunk_id=first.chunk_id,
        ordinal=first.ordinal,
        grid_index=first.grid_index,
        source_selection=first.source_selection,
        expected_input_shape=first.expected_input_shape,
        placements=(
            ChunkPlacement(
                output_id="corrected_signal",
                destination_selection=(AxisSelector.sliced(3, 5), AxisSelector.all()),
                expected_shape=(2, 2),
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicting ChunkSpec"):
        sink.write_chunk(
            "run1",
            _processing_data(np.ones((2, 2), dtype=np.float32)),
            plan=plan,
            chunk=conflicting,
        )


def test_hdf_chunked_sink_rejects_non_signal_components(tmp_path: Path):
    signal = ChunkArrayLayout(component="signal", final_shape=(2,), dtype="float64")
    weights = ChunkArrayLayout(component="weights", final_shape=(2,), dtype="float64")
    plan = ChunkPlan(
        schema_version="1.0",
        plan_id="unsupported-components",
        total_chunks=1,
        expected_chunk_ids=("c0",),
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(signal, weights),
            ),
        ),
    )

    with pytest.raises(NotImplementedError, match="signal arrays only"):
        HDFChunkedProcessingSink(resource_location=tmp_path / "out.h5").initialize_chunked("run", plan)


@pytest.mark.parametrize(
    ("processing_data", "message"),
    [
        (_processing_data(np.ones((2, 2), dtype=np.float64)), "dtype"),
        (
            ProcessingData(
                {
                    "sample": DataBundle(
                        {
                            "signal": BaseData(
                                signal=np.ones((2, 2), dtype=np.float32),
                                units=ureg.Unit("meter"),
                                rank_of_data=1,
                            )
                        }
                    )
                }
            ),
            "units",
        ),
        (
            ProcessingData(
                {
                    "sample": DataBundle(
                        {
                            "signal": BaseData(
                                signal=np.ones((2, 2), dtype=np.float32),
                                units=ureg.Unit("count"),
                                rank_of_data=0,
                            )
                        }
                    )
                }
            ),
            "rank_of_data",
        ),
    ],
)
def test_hdf_chunked_sink_validates_chunk_metadata(
    tmp_path: Path,
    processing_data: ProcessingData,
    message: str,
):
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=tmp_path / f"{message}.h5")
    sink.initialize_chunked("run", plan)

    with pytest.raises(ValueError, match=message):
        sink.write_chunk("run", processing_data, plan=plan, chunk=_chunk(plan, 0, 0, 2))


def test_hdf_chunked_sink_places_chunks_across_multiple_batch_axes(tmp_path: Path):
    out_file = tmp_path / "multi-axis.h5"
    chunk_ids = ("c00", "c01", "c10", "c11")
    plan = ChunkPlan(
        schema_version="1.0",
        plan_id="multi-axis",
        total_chunks=4,
        expected_chunk_ids=chunk_ids,
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(ChunkArrayLayout(component="signal", final_shape=(3, 4, 2), dtype="float32"),),
            ),
        ),
        driver={"full_shape": [3, 4, 2]},
        batch_axes=(0, 1),
        data_axes=(2,),
    )
    ranges = ((0, 2, 0, 2), (0, 2, 2, 4), (2, 3, 0, 2), (2, 3, 2, 4))
    full = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run", plan)

    chunks = []
    for ordinal, (start0, stop0, start1, stop1) in enumerate(ranges):
        selectors = (
            AxisSelector.sliced(start0, stop0),
            AxisSelector.sliced(start1, stop1),
            AxisSelector.all(),
        )
        expected_shape = (stop0 - start0, stop1 - start1, 2)
        chunks.append(
            (
                ChunkSpec(
                    schema_version="1.0",
                    plan_id=plan.plan_id,
                    plan_hash=plan.plan_hash,
                    chunk_id=chunk_ids[ordinal],
                    ordinal=ordinal,
                    grid_index=(ordinal // 2, ordinal % 2),
                    source_selection=selectors,
                    expected_input_shape=expected_shape,
                    placements=(ChunkPlacement("signal", selectors, expected_shape),),
                ),
                full[start0:stop0, start1:stop1, :],
            )
        )

    for chunk, values in reversed(chunks):
        sink.write_chunk("run", _processing_data(values), plan=plan, chunk=chunk)
    sink.finalize_chunked("run", plan=plan)

    with h5py.File(out_file, "r") as h5:
        np.testing.assert_array_equal(h5["processing/result/run/sample/signal/signal"], full)


def test_hdf_chunked_sink_rejects_incomplete_destination_coverage(tmp_path: Path):
    out_file = tmp_path / "coverage-gap.h5"
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run", plan)

    for ordinal in range(3):
        selectors = (AxisSelector.sliced(ordinal, ordinal + 1), AxisSelector.all())
        chunk = ChunkSpec(
            schema_version="1.0",
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            chunk_id=plan.expected_chunk_ids[ordinal],
            ordinal=ordinal,
            grid_index=(ordinal,),
            source_selection=selectors,
            expected_input_shape=(1, 2),
            placements=(ChunkPlacement("corrected_signal", selectors, (1, 2)),),
        )
        sink.write_chunk(
            "run",
            _processing_data(np.ones((1, 2), dtype=np.float32)),
            plan=plan,
            chunk=chunk,
        )

    with pytest.raises(ValueError, match="cover 6 elements.*expected 10"):
        sink.finalize_chunked("run", plan=plan)
