# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import pytest

from modacor.io.chunking import (
    AxisSelector,
    ChunkArrayLayout,
    ChunkOutputLayout,
    ChunkPlacement,
    ChunkPlan,
    ChunkSourceBinding,
    ChunkSpec,
    PlacementBinding,
    selection_shape,
)


def _output_layout() -> ChunkOutputLayout:
    return ChunkOutputLayout(
        output_id="corrected_signal",
        processing_path="/sample/signal",
        destination_path="sample/signal",
        units="count",
        rank_of_data=1,
        arrays=(ChunkArrayLayout(component="signal", final_shape=(5, 2), dtype="float32"),),
    )


def _plan() -> ChunkPlan:
    return ChunkPlan(
        schema_version="1.0",
        plan_id="test-plan",
        total_chunks=3,
        expected_chunk_ids=("c000000", "c000001", "c000002"),
        outputs=(_output_layout(),),
        driver={"source": "sample::/data", "full_shape": [5, 2], "dtype": "float32"},
        batch_axes=(0,),
        data_axes=(1,),
        axis_rules=({"axis": 0, "chunk_size": 2},),
        bindings=({"source": "sample", "role": "aligned"},),
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


def test_axis_selector_shape_supports_indices_slices_and_edge_chunks():
    assert selection_shape(
        (2, 5, 7),
        (AxisSelector.index(0), AxisSelector.sliced(1, 5, 2), AxisSelector.all()),
    ) == (2, 7)

    with pytest.raises(ValueError, match="positive"):
        AxisSelector.sliced(0, 4, 0)
    with pytest.raises(ValueError, match="exceeds axis"):
        selection_shape((3,), (AxisSelector.sliced(0, 4),))


def test_chunk_plan_hash_is_canonical_and_round_trips():
    first = _plan()
    second = ChunkPlan(
        schema_version="1.0",
        plan_id="test-plan",
        total_chunks=3,
        expected_chunk_ids=("c000000", "c000001", "c000002"),
        outputs=(_output_layout(),),
        driver={"dtype": "float32", "full_shape": [5, 2], "source": "sample::/data"},
        batch_axes=(0,),
        data_axes=(1,),
        axis_rules=({"chunk_size": 2, "axis": 0},),
        bindings=({"role": "aligned", "source": "sample"},),
    )

    assert first.plan_hash == second.plan_hash
    assert ChunkPlan.from_dict(json.loads(first.to_json())) == first

    with pytest.raises(TypeError):
        first.driver["source"] = "changed"  # type: ignore[index]


def test_chunk_source_bindings_project_driver_selection_and_round_trip():
    driver_selection = (
        AxisSelector.index(0),
        AxisSelector.sliced(1, 6, 2),
        AxisSelector.all(),
        AxisSelector.all(),
    )
    aligned = ChunkSourceBinding("sample", "entry/data", "aligned")
    explicit = ChunkSourceBinding("sample", "/normalization", "explicit", axis_map=(0, 1))
    static = ChunkSourceBinding("mask", "/mask", "static")

    assert aligned.resolve_selection(driver_selection, (1, 10, 5, 4)) == driver_selection
    assert explicit.resolve_selection(driver_selection, (1, 10)) == driver_selection[:2]
    assert static.resolve_selection(driver_selection, (5, 4)) is None
    assert ChunkSourceBinding.from_dict(explicit.to_dict()) == explicit
    assert aligned.data_reference == "sample::/entry/data"


def test_chunk_plan_preserves_typed_source_bindings_without_changing_legacy_shape():
    legacy = _plan()
    assert "source_bindings" not in legacy.to_dict()

    plan = ChunkPlan(
        schema_version="1.0",
        plan_id="direct-source-plan",
        total_chunks=1,
        expected_chunk_ids=("c0",),
        outputs=(_output_layout(),),
        driver={"source": "sample::/data", "full_shape": [5, 2], "dtype": "float32"},
        batch_axes=(0,),
        data_axes=(1,),
        source_bindings=(
            ChunkSourceBinding("sample", "/data", "aligned"),
            ChunkSourceBinding("calibration", "/factor", "static"),
        ),
    )

    assert ChunkPlan.from_dict(plan.to_dict()) == plan
    assert plan.to_dict()["source_bindings"][0]["role"] == "aligned"

    with pytest.raises(ValueError, match="driver.source"):
        ChunkPlan(
            schema_version="1.0",
            plan_id="missing-driver",
            total_chunks=1,
            expected_chunk_ids=("c0",),
            outputs=(_output_layout(),),
            driver={"full_shape": [5, 2]},
            source_bindings=(ChunkSourceBinding("sample", "/data", "aligned"),),
        )


def test_chunk_plan_rejects_tampered_serialized_hash():
    payload = _plan().to_dict()
    payload["driver"]["source"] = "other::/data"
    with pytest.raises(ValueError, match="plan_hash"):
        ChunkPlan.from_dict(payload)


def test_chunk_output_layout_preserves_explicit_axis_metadata():
    output = ChunkOutputLayout(
        output_id="signal",
        processing_path="/sample/signal",
        destination_path="sample/signal",
        units="count",
        rank_of_data=1,
        arrays=(
            ChunkArrayLayout(component="signal", final_shape=(5,), dtype="float32"),
            ChunkArrayLayout(
                component="axes/Q",
                final_shape=(5,),
                dtype="float64",
                placement_binding=PlacementBinding(kind="axis_map", axis_map=(0,)),
                units="1/nm",
                rank_of_data=1,
            ),
        ),
        axis_names=("Q",),
    )

    assert ChunkOutputLayout.from_dict(output.to_dict()) == output
    assert output.to_dict()["arrays"][1]["units"] == "1/nm"


def test_chunk_output_layout_rejects_ambiguous_axis_schema():
    axis = ChunkArrayLayout(
        component="axes/Q",
        final_shape=(5,),
        dtype="float64",
        placement_binding=PlacementBinding(kind="static"),
        units="1/nm",
        rank_of_data=1,
    )
    with pytest.raises(ValueError, match="axis components"):
        ChunkOutputLayout(
            output_id="signal",
            processing_path="/sample/signal",
            destination_path="sample/signal",
            units="count",
            rank_of_data=1,
            arrays=(ChunkArrayLayout(component="signal", final_shape=(5,), dtype="float32"), axis),
        )

    with pytest.raises(ValueError, match="require component-level"):
        ChunkOutputLayout(
            output_id="signal",
            processing_path="/sample/signal",
            destination_path="sample/signal",
            units="count",
            rank_of_data=1,
            arrays=(
                ChunkArrayLayout(component="signal", final_shape=(5,), dtype="float32"),
                ChunkArrayLayout(component="axes/Q", final_shape=(5,), dtype="float64"),
            ),
            axis_names=("Q",),
        )


def test_chunk_spec_validates_plan_identity_shapes_and_ordinal():
    plan = _plan()
    edge = _chunk(plan, 2, 4, 5)
    edge.validate_for_plan(plan)

    wrong_ordinal = ChunkSpec(
        schema_version=edge.schema_version,
        plan_id=edge.plan_id,
        plan_hash=edge.plan_hash,
        chunk_id=edge.chunk_id,
        ordinal=1,
        grid_index=edge.grid_index,
        source_selection=edge.source_selection,
        expected_input_shape=edge.expected_input_shape,
        placements=edge.placements,
    )
    with pytest.raises(ValueError, match="ordinal"):
        wrong_ordinal.validate_for_plan(plan)

    wrong_shape = ChunkSpec(
        schema_version=edge.schema_version,
        plan_id=edge.plan_id,
        plan_hash=edge.plan_hash,
        chunk_id=edge.chunk_id,
        ordinal=edge.ordinal,
        grid_index=edge.grid_index,
        source_selection=edge.source_selection,
        expected_input_shape=(2, 2),
        placements=edge.placements,
    )
    with pytest.raises(ValueError, match="source selection resolves"):
        wrong_shape.validate_for_plan(plan)
