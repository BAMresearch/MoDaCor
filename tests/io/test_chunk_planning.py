# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunk_planning import resolve_chunk_input_plan, resolve_provisional_chunk_plan
from modacor.io.chunking import (
    AxisSelector,
    ChunkAxisRule,
    ChunkInputPlan,
    ChunkSourceBinding,
    PlacementBinding,
    ProvisionalChunkOutput,
    ProvisionalChunkPlan,
)


def _provisional(*, stride: int = 1) -> ProvisionalChunkPlan:
    return ProvisionalChunkPlan(
        schema_version="1.0",
        plan_id="pilot-plan",
        driver={"source": "raw::/entry/data", "rank_of_data": 2},
        axis_rules=(
            ChunkAxisRule(axis=0, chunk_size=1),
            ChunkAxisRule(axis=1, start=1, stop=10, stride=stride, chunk_size=3),
        ),
        outputs=(
            ProvisionalChunkOutput(
                output_id="reduced",
                processing_path="/sample/I",
                destination_path="sample/I",
            ),
        ),
        source_bindings=(ChunkSourceBinding("raw", "/entry/data", "aligned"),),
    )


def _pilot_data(batch_shape: tuple[int, int]) -> ProcessingData:
    q = BaseData(signal=np.linspace(0.1, 1.0, 5), units="1/nm", rank_of_data=1)
    signal = BaseData(
        signal=np.ones(batch_shape + (5,), dtype=np.float32),
        units="count",
        uncertainties={"poisson": np.full(batch_shape + (5,), 0.5, dtype=np.float32)},
        axes=[q],
        rank_of_data=1,
    )
    bundle = DataBundle(I=signal, Q=q)
    bundle.default_plot = "I"
    result = ProcessingData()
    result["sample"] = bundle
    return result


def test_provisional_contract_round_trips_and_hashes_canonically():
    provisional = _provisional()

    assert ProvisionalChunkPlan.from_dict(provisional.to_dict()) == provisional
    assert provisional.provisional_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="batch-axis"):
        ProvisionalChunkPlan(
            schema_version="1.0",
            plan_id="empty-rules",
            driver={"source": "raw::/data", "rank_of_data": 2},
            axis_rules=(),
            outputs=(ProvisionalChunkOutput("out", "/sample/I", "sample/I"),),
            source_bindings=(ChunkSourceBinding("raw", "/data", "aligned"),),
        )


def test_input_resolution_discovers_shape_and_packs_strided_selection():
    provisional = _provisional(stride=2)
    resolved = resolve_chunk_input_plan(provisional, (2, 10, 8, 6), np.uint16)

    assert ChunkInputPlan.from_dict(resolved.to_dict()) == resolved
    assert resolved.batch_axes == (0, 1)
    assert resolved.data_axes == (2, 3)
    assert resolved.final_batch_shape == (2, 5)
    assert len(resolved.chunks) == 4
    first = resolved.chunks[0]
    assert first.source_selection == (
        AxisSelector.sliced(0, 1),
        AxisSelector.sliced(1, 6, 2),
        AxisSelector.all(),
        AxisSelector.all(),
    )
    assert first.destination_batch_selection == (
        AxisSelector.sliced(0, 1),
        AxisSelector.sliced(0, 3),
    )
    assert first.expected_input_shape == (1, 3, 8, 6)
    assert resolved.chunks[1].expected_batch_shape == (1, 2)


def test_pilot_resolves_complete_outputs_and_regular_chunk_specs():
    provisional = _provisional()
    input_plan = resolve_chunk_input_plan(provisional, (2, 10, 8, 6), np.uint16)
    pilot = input_plan.chunks[0]

    plan, specs = resolve_provisional_chunk_plan(
        provisional,
        input_plan,
        _pilot_data(pilot.expected_batch_shape),
        pilot.chunk_id,
    )

    assert plan.driver["full_shape"] == (2, 10, 8, 6)
    assert plan.outputs[0].signal.final_shape == (2, 9, 5)
    assert plan.outputs[0].rank_of_data == 1
    assert plan.outputs[0].axis_names == (".", ".", "Q")
    assert [layout.component for layout in plan.outputs[0].arrays] == [
        "signal",
        "uncertainties/poisson",
        "axes/Q",
    ]
    assert plan.outputs[0].arrays[-1].placement_binding == PlacementBinding("static")
    assert len(specs) == 6
    assert specs[0].placements[0].expected_shape == (1, 3, 5)
    assert specs[-1].placements[0].expected_shape == (1, 3, 5)
    for spec in specs:
        spec.validate_for_plan(plan)


def test_coordinate_output_does_not_treat_itself_as_an_axis():
    provisional = ProvisionalChunkPlan(
        schema_version="1.0",
        plan_id="coordinate-pilot-plan",
        driver={"source": "raw::/entry/data", "rank_of_data": 2},
        axis_rules=(ChunkAxisRule(axis=0, chunk_size=1),),
        outputs=(ProvisionalChunkOutput("Q", "/sample/Q", "sample/Q"),),
    )
    input_plan = resolve_chunk_input_plan(provisional, (2, 8, 6), np.uint16)
    pilot = input_plan.chunks[0]
    processing_data = ProcessingData()
    processing_data["sample"] = DataBundle(
        Q=BaseData(
            signal=np.linspace(0.1, 1.0, 5)[None, :],
            units="1/nm",
            rank_of_data=1,
        )
    )

    plan, _specs = resolve_provisional_chunk_plan(
        provisional,
        input_plan,
        processing_data,
        pilot.chunk_id,
    )

    assert plan.outputs[0].axis_names == (".", ".")
    assert [layout.component for layout in plan.outputs[0].arrays] == ["signal"]


def test_pilot_rejects_pipeline_that_changes_batch_dimensions():
    provisional = _provisional()
    input_plan = resolve_chunk_input_plan(provisional, (2, 10, 8, 6), np.uint16)
    pilot = input_plan.chunks[0]
    invalid = ProcessingData()
    invalid["sample"] = DataBundle(
        I=BaseData(signal=np.ones((15,), dtype=np.float32), units=ureg.count, rank_of_data=1)
    )

    with pytest.raises(ValueError, match="batch dimensions"):
        resolve_provisional_chunk_plan(provisional, input_plan, invalid, pilot.chunk_id)
