# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunking import AxisSelector, ChunkArrayLayout, ChunkOutputLayout, ChunkPlacement, ChunkPlan, ChunkSpec
from modacor.io.hdf import HDFChunkedProcessingSink
from modacor.server.chunked_outputs import ChunkedOutputManager
from modacor.server.runtime_policy import RuntimePolicy


def _plan() -> ChunkPlan:
    return ChunkPlan(
        schema_version="1.0",
        plan_id="concurrent-plan",
        total_chunks=2,
        expected_chunk_ids=("c0", "c1"),
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units="count",
                rank_of_data=1,
                arrays=(ChunkArrayLayout(component="signal", final_shape=(4, 2), dtype="float32"),),
            ),
        ),
        driver={"full_shape": [4, 2]},
    )


def _chunk(plan: ChunkPlan, ordinal: int) -> ChunkSpec:
    start = ordinal * 2
    return ChunkSpec(
        schema_version=plan.schema_version,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        chunk_id=plan.expected_chunk_ids[ordinal],
        ordinal=ordinal,
        grid_index=(ordinal,),
        source_selection=(AxisSelector.sliced(start, start + 2), AxisSelector.all()),
        expected_input_shape=(2, 2),
        placements=(
            ChunkPlacement(
                output_id="signal",
                destination_selection=(AxisSelector.sliced(start, start + 2), AxisSelector.all()),
                expected_shape=(2, 2),
            ),
        ),
    )


def _data(value: float) -> ProcessingData:
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(
        signal=np.full((2, 2), value, dtype=np.float32),
        units=ureg.Unit("count"),
        rank_of_data=1,
    )
    processing_data["sample"] = bundle
    return processing_data


def test_manager_serializes_two_output_ids_for_the_same_hdf_file(monkeypatch, tmp_path: Path):
    manager = ChunkedOutputManager(policy=RuntimePolicy.trusted())
    plan = _plan()
    sink_spec = {"ref": "out", "type": "hdf_chunked", "location": str(tmp_path / "concurrent.h5")}
    first, _result = manager.initialize(sink_spec=sink_spec, subpath="run", plan=plan)
    second, _status = manager.reopen(sink_spec=sink_spec, plan_id=plan.plan_id)

    original_write = HDFChunkedProcessingSink.write_chunk
    counter_lock = Lock()
    active = 0
    max_active = 0

    def observed_write(self, *args, **kwargs):  # noqa: ANN001
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return original_write(self, *args, **kwargs)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(HDFChunkedProcessingSink, "write_chunk", observed_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(manager.write_chunk, first.output_id, _data(1.0), chunk=_chunk(plan, 0)),
            pool.submit(manager.write_chunk, second.output_id, _data(2.0), chunk=_chunk(plan, 1)),
        ]
        for future in futures:
            future.result()

    assert max_active == 1
    status = manager.inspect(first.output_id)
    assert status.completed_chunks == 2
