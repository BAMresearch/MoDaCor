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
from modacor.io.chunk_planning import resolve_chunk_input_plan, resolve_provisional_chunk_plan
from modacor.io.chunking import (
    AxisSelector,
    ChunkArrayLayout,
    ChunkOutputLayout,
    ChunkPlacement,
    ChunkPlan,
    ChunkSpec,
    PlacementBinding,
    ProvisionalChunkOutput,
    ProvisionalChunkPlan,
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


def _provisional_plan() -> ProvisionalChunkPlan:
    return ProvisionalChunkPlan(
        schema_version="1.0",
        plan_id="provisional-assembly",
        driver={"source": "sample::/entry/data", "rank_of_data": 1},
        axis_rules=({"axis": 0, "chunk_size": 2},),
        outputs=(ProvisionalChunkOutput("signal", "/sample/signal", "sample/signal"),),
    )


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
    with pytest.raises(UnsupportedSinkCapability, match="HDFProcessingSink"):
        sinks.load_chunked_plan("ordinary", "assembly-1")
    with pytest.raises(UnsupportedSinkCapability, match="HDFProcessingSink"):
        sinks.recover_chunked("ordinary::run", plan=_plan(), action="reconcile")


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
        trace_events = [
            {
                "step_id": "reduce",
                "module": "ReduceDimensionality",
                "duration_s": 0.01,
                "datasets": {},
                "chunk_identity": chunk.identity_dict(),
            }
        ]
        result = sink.write_chunk(
            "run1",
            _processing_data(values),
            plan=plan,
            chunk=chunk,
            execution_metadata={"attempt": 1},
            trace_events=trace_events,
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
        for chunk, _values in chunks:
            trace_group = chunked[f"processing/tracer/run1/chunks/{chunk.chunk_id}"]
            events = json.loads(_read_text(trace_group["events"]))
            assert events[0]["step_id"] == "reduce"
            assert events[0]["chunk_identity"] == chunk.identity_dict()
            assert trace_group["steps/0001_reduce"].attrs["module"] == "ReduceDimensionality"


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


def test_hdf_chunked_sink_replace_clears_prior_run_trace(tmp_path: Path):
    out_file = tmp_path / "replace-trace.h5"
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run1", plan)
    chunk = _chunk(plan, 0, 0, 2)
    sink.write_chunk(
        "run1",
        _processing_data(np.ones((2, 2), dtype=np.float32)),
        plan=plan,
        chunk=chunk,
        trace_events=[{"step_id": "load", "module": "AppendProcessingData"}],
    )

    sink.initialize_chunked("run1", plan, collision="replace")

    with h5py.File(out_file, "r") as h5:
        assert "processing/tracer/run1" not in h5


def test_hdf_chunked_sink_inspects_failed_write_and_retries(monkeypatch, tmp_path: Path):
    from modacor.io.hdf import hdf_chunked_processing_sink as sink_module

    out_file = tmp_path / "failed-retry.h5"
    plan = _plan()
    chunk = _chunk(plan, 0, 0, 2)
    data = _processing_data(np.ones((2, 2), dtype=np.float32))
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run1", plan)

    original_write = sink_module._PendingWrite.write

    def fail_write(self):  # noqa: ANN001
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(sink_module._PendingWrite, "write", fail_write)
    with pytest.raises(OSError, match="synthetic storage failure"):
        sink.write_chunk("run1", data, plan=plan, chunk=chunk)

    failed = sink.inspect_chunked("run1", plan=plan, offset=0, limit=1)
    assert failed.failed_chunks == 1
    assert failed.missing_chunks == 2
    assert failed.chunks == ({"chunk_id": "c000000", "ordinal": 0, "status": "failed"},)

    monkeypatch.setattr(sink_module._PendingWrite, "write", original_write)
    retried = sink.write_chunk("run1", data, plan=plan, chunk=chunk)
    assert retried.completed_chunks == 1
    inspected = sink.inspect_chunked("run1", plan=plan)
    assert inspected.completed_chunks == 1
    assert inspected.failed_chunks == 0


def test_hdf_chunked_sink_loads_and_recovers_persisted_plan(tmp_path: Path):
    out_file = tmp_path / "recover.h5"
    plan = _plan()
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sinks = IoSinks()
    sinks.register_sink(sink, "chunked")
    sink.initialize_chunked("run1", plan)

    with h5py.File(out_file, "r+") as h5:
        h5["processing/chunk_plans/assembly-1/chunks/c000000/status"][()] = "writing"
        h5["processing/chunk_plans/assembly-1/status"][()] = "finalizing"

    subpath, loaded = sinks.load_chunked_plan("chunked", "assembly-1")
    assert subpath == "run1"
    assert loaded == plan

    reconciled = sinks.recover_chunked("chunked::run1", plan=loaded, action="reconcile")
    assert reconciled.status == "writing"
    assert reconciled.failed_chunks == 1

    abandoned = sink.recover_chunked("run1", plan=loaded, action="abandon")
    assert abandoned.status == "abandoned"
    with pytest.raises(RuntimeError, match="not writable"):
        sink.write_chunk(
            "run1",
            _processing_data(np.ones((2, 2), dtype=np.float32)),
            plan=loaded,
            chunk=_chunk(plan, 0, 0, 2),
        )
    resumed = sink.recover_chunked("run1", plan=loaded, action="resume")
    assert resumed.status == "writing"


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


def test_hdf_chunked_sink_rejects_unknown_components(tmp_path: Path):
    signal = ChunkArrayLayout(component="signal", final_shape=(2,), dtype="float64")
    unknown = ChunkArrayLayout(component="metadata/value", final_shape=(2,), dtype="float64")
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
                arrays=(signal, unknown),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported chunk array component"):
        HDFChunkedProcessingSink(resource_location=tmp_path / "out.h5").initialize_chunked("run", plan)


def _complete_plan() -> ChunkPlan:
    return ChunkPlan(
        schema_version="1.0",
        plan_id="complete-basedata",
        total_chunks=3,
        expected_chunk_ids=("c0", "c1", "c2"),
        outputs=(
            ChunkOutputLayout(
                output_id="corrected_signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(
                    ChunkArrayLayout(component="signal", final_shape=(5, 3), dtype="float32"),
                    ChunkArrayLayout(
                        component="weights",
                        final_shape=(5, 3),
                        dtype="float32",
                        placement_binding=PlacementBinding(kind="broadcast"),
                    ),
                    ChunkArrayLayout(
                        component="uncertainties/poisson",
                        final_shape=(5, 3),
                        dtype="float32",
                    ),
                    ChunkArrayLayout(
                        component="uncertainties/calibration",
                        final_shape=(5, 3),
                        dtype="float32",
                    ),
                    ChunkArrayLayout(
                        component="axes/frame",
                        final_shape=(5,),
                        dtype="float64",
                        placement_binding=PlacementBinding(kind="axis_map", axis_map=(0,)),
                        units=str(ureg.Unit("second")),
                        rank_of_data=1,
                    ),
                    ChunkArrayLayout(
                        component="axes/Q",
                        final_shape=(3,),
                        dtype="float64",
                        placement_binding=PlacementBinding(kind="static"),
                        units=str(ureg.Unit("1/nm")),
                        rank_of_data=1,
                    ),
                ),
                axis_names=("frame", "Q"),
            ),
        ),
        driver={"full_shape": [5, 3]},
        batch_axes=(0,),
        data_axes=(1,),
    )


def _complete_processing_data(
    signal: np.ndarray,
    uncertainty: np.ndarray,
    frames: np.ndarray,
    q_values: np.ndarray,
    *,
    weights: np.ndarray,
) -> ProcessingData:
    frame_axis = BaseData(signal=frames, units=ureg.Unit("second"), rank_of_data=1)
    q_axis = BaseData(signal=q_values, units=ureg.Unit("1/nm"), rank_of_data=1)
    signal_data = BaseData(
        signal=signal,
        units=ureg.Unit("count"),
        uncertainties={"poisson": uncertainty, "calibration": uncertainty * np.float32(2.0)},
        weights=weights,
        axes=[frame_axis, q_axis],
        rank_of_data=1,
    )
    bundle = DataBundle({"signal": signal_data, "frame": frame_axis, "Q": q_axis})
    bundle.default_plot = "signal"
    return ProcessingData({"sample": bundle})


def _assert_hdf_nodes_equal(left: h5py.Group | h5py.Dataset, right: h5py.Group | h5py.Dataset) -> None:
    assert type(left) is type(right)
    assert set(left.attrs) == set(right.attrs)
    for name in left.attrs:
        left_value = np.asarray(left.attrs[name])
        right_value = np.asarray(right.attrs[name])
        np.testing.assert_array_equal(left_value, right_value)
    if isinstance(left, h5py.Dataset):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left[()], right[()])
        return
    assert set(left) == set(right)
    for name in left:
        _assert_hdf_nodes_equal(left[name], right[name])


def test_hdf_chunked_sink_matches_complete_basedata_tree(tmp_path: Path):
    chunked_file = tmp_path / "complete-chunked.h5"
    ordinary_file = tmp_path / "complete-ordinary.h5"
    plan = _complete_plan()
    signal = np.arange(15, dtype=np.float32).reshape(5, 3)
    uncertainty = np.linspace(0.1, 1.5, 15, dtype=np.float32).reshape(5, 3)
    frames = np.linspace(0.0, 0.4, 5, dtype=np.float64)
    q_values = np.linspace(0.01, 0.03, 3, dtype=np.float64)
    weight_row = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    ranges = ((0, 2), (2, 4), (4, 5))

    sink = HDFChunkedProcessingSink(resource_location=chunked_file)
    sink.initialize_chunked("run", plan)
    chunk_payloads = []
    for ordinal, (start, stop) in enumerate(ranges):
        selectors = (AxisSelector.sliced(start, stop), AxisSelector.all())
        spec = ChunkSpec(
            schema_version=plan.schema_version,
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            chunk_id=plan.expected_chunk_ids[ordinal],
            ordinal=ordinal,
            grid_index=(ordinal,),
            source_selection=selectors,
            expected_input_shape=(stop - start, 3),
            placements=(ChunkPlacement("corrected_signal", selectors, (stop - start, 3)),),
        )
        payload = _complete_processing_data(
            signal[start:stop],
            uncertainty[start:stop],
            frames[start:stop],
            q_values,
            weights=weight_row,
        )
        chunk_payloads.append((spec, payload))

    first_spec, first_payload = chunk_payloads[-1]
    sink.write_chunk("run", first_payload, plan=plan, chunk=first_spec)
    sink = HDFChunkedProcessingSink(resource_location=chunked_file)
    resumed = sink.initialize_chunked("run", plan, collision="resume")
    assert resumed.completed_chunks == 1

    for spec, payload in reversed(chunk_payloads[:-1]):
        sink.write_chunk("run", payload, plan=plan, chunk=spec)
    sink.finalize_chunked("run", plan=plan)

    complete = _complete_processing_data(
        signal,
        uncertainty,
        frames,
        q_values,
        weights=np.broadcast_to(weight_row, signal.shape).copy(),
    )
    HDFProcessingSink(resource_location=ordinary_file).write("run", complete, data_paths=["/sample/signal"])

    with h5py.File(chunked_file, "r") as chunked, h5py.File(ordinary_file, "r") as ordinary:
        _assert_hdf_nodes_equal(
            chunked["processing/result/run"],
            ordinary["processing/result/run"],
        )
        assert (
            _read_text(chunked["processing/chunk_plans/complete-basedata/components/" "corrected_signal/axes/Q/status"])
            == "complete"
        )


def test_hdf_chunked_sink_rejects_changed_static_axis(tmp_path: Path):
    plan = _complete_plan()
    sink = HDFChunkedProcessingSink(resource_location=tmp_path / "static-axis.h5")
    sink.initialize_chunked("run", plan)
    first = ChunkSpec(
        schema_version="1.0",
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        chunk_id="c0",
        ordinal=0,
        grid_index=(0,),
        source_selection=(AxisSelector.sliced(0, 2), AxisSelector.all()),
        expected_input_shape=(2, 3),
        placements=(
            ChunkPlacement(
                "corrected_signal",
                (AxisSelector.sliced(0, 2), AxisSelector.all()),
                (2, 3),
            ),
        ),
    )
    second = ChunkSpec(
        schema_version="1.0",
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        chunk_id="c1",
        ordinal=1,
        grid_index=(1,),
        source_selection=(AxisSelector.sliced(2, 4), AxisSelector.all()),
        expected_input_shape=(2, 3),
        placements=(
            ChunkPlacement(
                "corrected_signal",
                (AxisSelector.sliced(2, 4), AxisSelector.all()),
                (2, 3),
            ),
        ),
    )
    signal = np.ones((2, 3), dtype=np.float32)
    uncertainty = np.ones((2, 3), dtype=np.float32)
    weights = np.ones((1, 3), dtype=np.float32)
    sink.write_chunk(
        "run",
        _complete_processing_data(
            signal,
            uncertainty,
            np.array([0.0, 0.1]),
            np.array([0.01, 0.02, 0.03]),
            weights=weights,
        ),
        plan=plan,
        chunk=first,
    )

    with pytest.raises(ValueError, match="Static component.*changed"):
        sink.write_chunk(
            "run",
            _complete_processing_data(
                signal,
                uncertainty,
                np.array([0.2, 0.3]),
                np.array([0.01, 0.02, 0.04]),
                weights=weights,
            ),
            plan=plan,
            chunk=second,
        )


def test_hdf_chunked_sink_preserves_invariant_scalar_weight(tmp_path: Path):
    chunked_file = tmp_path / "scalar-weight-chunked.h5"
    ordinary_file = tmp_path / "scalar-weight-ordinary.h5"
    plan = ChunkPlan(
        schema_version="1.0",
        plan_id="scalar-weight",
        total_chunks=2,
        expected_chunk_ids=("c0", "c1"),
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(
                    ChunkArrayLayout(component="signal", final_shape=(4,), dtype="float32"),
                    ChunkArrayLayout(
                        component="weights",
                        final_shape=(),
                        dtype="float32",
                        placement_binding=PlacementBinding(kind="static"),
                    ),
                ),
            ),
        ),
        driver={"full_shape": [4]},
        batch_axes=(0,),
    )
    sink = HDFChunkedProcessingSink(resource_location=chunked_file)
    sink.initialize_chunked("run", plan)
    full_signal = np.arange(4, dtype=np.float32)
    for ordinal, (start, stop) in enumerate(((0, 2), (2, 4))):
        selector = (AxisSelector.sliced(start, stop),)
        spec = ChunkSpec(
            schema_version="1.0",
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            chunk_id=plan.expected_chunk_ids[ordinal],
            ordinal=ordinal,
            grid_index=(ordinal,),
            source_selection=selector,
            expected_input_shape=(2,),
            placements=(ChunkPlacement("signal", selector, (2,)),),
        )
        data = _processing_data(full_signal[start:stop])
        data["sample"]["signal"].weights = np.asarray(2.0, dtype=np.float32)
        sink.write_chunk("run", data, plan=plan, chunk=spec)
    sink.finalize_chunked("run", plan=plan)

    complete = _processing_data(full_signal)
    complete["sample"]["signal"].weights = np.asarray(2.0, dtype=np.float32)
    HDFProcessingSink(resource_location=ordinary_file).write("run", complete, data_paths=["/sample/signal"])

    with h5py.File(chunked_file, "r") as chunked, h5py.File(ordinary_file, "r") as ordinary:
        _assert_hdf_nodes_equal(
            chunked["processing/result/run/sample/signal"],
            ordinary["processing/result/run/sample/signal"],
        )


def test_axis_map_supports_indexed_signal_dimension(tmp_path: Path):
    out_file = tmp_path / "indexed-axis.h5"
    plan = ChunkPlan(
        schema_version="1.0",
        plan_id="indexed-axis",
        total_chunks=2,
        expected_chunk_ids=("c0", "c1"),
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(
                    ChunkArrayLayout(component="signal", final_shape=(2, 3), dtype="float32"),
                    ChunkArrayLayout(
                        component="axes/frame",
                        final_shape=(2,),
                        dtype="float64",
                        placement_binding=PlacementBinding(kind="axis_map", axis_map=(0,)),
                        units=str(ureg.Unit("second")),
                        rank_of_data=0,
                    ),
                    ChunkArrayLayout(
                        component="axes/Q",
                        final_shape=(3,),
                        dtype="float64",
                        placement_binding=PlacementBinding(kind="static"),
                        units=str(ureg.Unit("1/nm")),
                        rank_of_data=1,
                    ),
                ),
                axis_names=("frame", "Q"),
            ),
        ),
        driver={"full_shape": [2, 3]},
        batch_axes=(0,),
        data_axes=(1,),
    )
    full_signal = np.arange(6, dtype=np.float32).reshape(2, 3)
    frame_values = np.array([0.0, 0.1], dtype=np.float64)
    q_values = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_chunked("run", plan)

    for ordinal in range(2):
        selection = (AxisSelector.index(ordinal), AxisSelector.all())
        spec = ChunkSpec(
            schema_version="1.0",
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            chunk_id=plan.expected_chunk_ids[ordinal],
            ordinal=ordinal,
            grid_index=(ordinal,),
            source_selection=selection,
            expected_input_shape=(3,),
            placements=(ChunkPlacement("signal", selection, (3,)),),
        )
        frame = BaseData(
            signal=np.asarray(frame_values[ordinal]),
            units=ureg.Unit("second"),
            rank_of_data=0,
        )
        q_axis = BaseData(signal=q_values, units=ureg.Unit("1/nm"), rank_of_data=1)
        signal = BaseData(
            signal=full_signal[ordinal],
            units=ureg.Unit("count"),
            axes=[q_axis],
            rank_of_data=1,
        )
        bundle = DataBundle({"signal": signal, "frame": frame, "Q": q_axis})
        bundle.default_plot = "signal"
        sink.write_chunk(
            "run",
            ProcessingData({"sample": bundle}),
            plan=plan,
            chunk=spec,
        )

    sink.finalize_chunked("run", plan=plan)
    with h5py.File(out_file, "r") as h5:
        group = h5["processing/result/run/sample/signal"]
        np.testing.assert_array_equal(group["signal"], full_signal)
        np.testing.assert_array_equal(group["frame"], frame_values)
        np.testing.assert_array_equal(group["Q"], q_values)


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


def test_signal_only_plan_rejects_undeclared_basedata_components(tmp_path: Path):
    values = np.ones((2, 2), dtype=np.float32)
    axis = BaseData(signal=np.arange(2, dtype=np.float32), units=ureg.dimensionless, rank_of_data=1)
    payloads = (
        (
            ProcessingData(
                {
                    "sample": DataBundle(
                        {
                            "signal": BaseData(
                                signal=values,
                                units=ureg.Unit("count"),
                                uncertainties={"poisson": values},
                                rank_of_data=1,
                            )
                        }
                    )
                }
            ),
            "uncertainty keys",
        ),
        (
            ProcessingData(
                {
                    "sample": DataBundle(
                        {
                            "signal": BaseData(
                                signal=values,
                                units=ureg.Unit("count"),
                                weights=np.full_like(values, 2.0),
                                rank_of_data=1,
                            )
                        }
                    )
                }
            ),
            "no weights layout",
        ),
        (
            ProcessingData(
                {
                    "sample": DataBundle(
                        {
                            "signal": BaseData(
                                signal=values,
                                units=ureg.Unit("count"),
                                axes=[axis, None],
                                rank_of_data=1,
                            )
                        }
                    )
                }
            ),
            "axes not declared",
        ),
    )

    for index, (payload, message) in enumerate(payloads):
        plan = _plan()
        sink = HDFChunkedProcessingSink(resource_location=tmp_path / f"undeclared-{index}.h5")
        sink.initialize_chunked("run", plan)
        with pytest.raises(ValueError, match=message):
            sink.write_chunk("run", payload, plan=plan, chunk=_chunk(plan, 0, 0, 2))


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


def test_hdf_provisional_pilot_resolves_schema_and_writes_without_second_run(tmp_path: Path):
    out_file = tmp_path / "provisional.h5"
    provisional = _provisional_plan()
    input_plan = resolve_chunk_input_plan(provisional, (5, 2), np.float32)
    pilot_work = input_plan.chunks[0]
    pilot_data = _processing_data(np.arange(4, dtype=np.float32).reshape(2, 2))
    plan, specs = resolve_provisional_chunk_plan(
        provisional,
        input_plan,
        pilot_data,
        pilot_work.chunk_id,
    )
    sink = HDFChunkedProcessingSink(resource_location=out_file)

    initialized = sink.initialize_provisional_chunked("run", provisional, input_plan=input_plan)
    assert initialized.status == "awaiting_schema"
    assert sink.inspect_provisional_chunked("run", plan=provisional, input_plan=input_plan).missing_chunks == 3
    assert sink.load_provisional_chunked(provisional.plan_id) == ("run", provisional, input_plan)

    pilot_result = sink.resolve_provisional_chunked(
        "run",
        pilot_data,
        provisional_plan=provisional,
        input_plan=input_plan,
        plan=plan,
        chunk=specs[0],
        execution_metadata={"pilot": True},
    )
    assert pilot_result.status == "writing"
    assert pilot_result.completed_chunks == 1

    sink.write_chunk(
        "run",
        _processing_data(np.arange(4, 8, dtype=np.float32).reshape(2, 2)),
        plan=plan,
        chunk=specs[1],
    )
    sink.write_chunk(
        "run",
        _processing_data(np.arange(8, 10, dtype=np.float32).reshape(1, 2)),
        plan=plan,
        chunk=specs[2],
    )
    assert sink.finalize_chunked("run", plan=plan).status == "complete"

    with h5py.File(out_file, "r") as h5:
        plan_group = h5["processing/chunk_plans/provisional-assembly"]
        assert _read_text(plan_group["status"]) == "complete"
        assert "provisional-assembly.__awaiting_schema__" not in h5["processing/chunk_plans"]
        assert _read_text(plan_group["provisional_plan_json"]) == provisional.to_json()
        assert _read_text(plan_group["input_plan_json"]) == input_plan.to_json()
        np.testing.assert_array_equal(
            h5["processing/result/run/sample/signal/signal"],
            np.arange(10, dtype=np.float32).reshape(5, 2),
        )


def test_hdf_provisional_transition_retries_after_interrupted_pilot_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    out_file = tmp_path / "provisional-retry.h5"
    provisional = _provisional_plan()
    input_plan = resolve_chunk_input_plan(provisional, (5, 2), np.float32)
    pilot_data = _processing_data(np.arange(4, dtype=np.float32).reshape(2, 2))
    plan, specs = resolve_provisional_chunk_plan(
        provisional,
        input_plan,
        pilot_data,
        input_plan.chunks[0].chunk_id,
    )
    sink = HDFChunkedProcessingSink(resource_location=out_file)
    sink.initialize_provisional_chunked("run", provisional, input_plan=input_plan)
    original_write = HDFChunkedProcessingSink.write_chunk

    def interrupted_write(*args, **kwargs):
        raise OSError("synthetic pilot interruption")

    monkeypatch.setattr(HDFChunkedProcessingSink, "write_chunk", interrupted_write)
    with pytest.raises(OSError, match="synthetic pilot interruption"):
        sink.resolve_provisional_chunked(
            "run",
            pilot_data,
            provisional_plan=provisional,
            input_plan=input_plan,
            plan=plan,
            chunk=specs[0],
        )

    monkeypatch.setattr(HDFChunkedProcessingSink, "write_chunk", original_write)
    retried = sink.resolve_provisional_chunked(
        "run",
        pilot_data,
        provisional_plan=provisional,
        input_plan=input_plan,
        plan=plan,
        chunk=specs[0],
    )
    assert retried.completed_chunks == 1
    with h5py.File(out_file, "r") as h5:
        assert "provisional-assembly.__awaiting_schema__" not in h5["processing/chunk_plans"]
