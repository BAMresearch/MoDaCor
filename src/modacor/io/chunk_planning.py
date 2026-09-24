# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunking import (
    AxisSelector,
    ChunkArrayLayout,
    ChunkAxisRule,
    ChunkInputPlan,
    ChunkOutputLayout,
    ChunkPlacement,
    ChunkPlan,
    ChunkSpec,
    PlacementBinding,
    ProvisionalChunkOutput,
    ProvisionalChunkPlan,
    ProvisionalChunkSpec,
)

__all__ = ["materialize_chunk_specs", "resolve_chunk_input_plan", "resolve_provisional_chunk_plan"]


def _rule_segments(
    rule: ChunkAxisRule,
    axis_size: int,
) -> tuple[list[tuple[AxisSelector, AxisSelector, int]], int]:
    start = 0 if rule.start is None else rule.start
    stop = axis_size if rule.stop is None else rule.stop
    if start >= axis_size or stop > axis_size:
        raise ValueError(f"Chunk rule for axis {rule.axis} selects [{start}:{stop}] outside axis length {axis_size}.")
    selected_count = len(range(start, stop, rule.stride))
    if selected_count < 1:
        raise ValueError(f"Chunk rule for axis {rule.axis} selects no elements.")

    segments: list[tuple[AxisSelector, AxisSelector, int]] = []
    for destination_start in range(0, selected_count, rule.chunk_size):
        count = min(rule.chunk_size, selected_count - destination_start)
        source_start = start + destination_start * rule.stride
        source_stop = source_start + (count - 1) * rule.stride + 1
        segments.append(
            (
                AxisSelector.sliced(source_start, source_stop, rule.stride),
                AxisSelector.sliced(destination_start, destination_start + count),
                count,
            )
        )
    return segments, selected_count


def resolve_chunk_input_plan(
    provisional: ProvisionalChunkPlan,
    source_shape: tuple[int, ...],
    source_dtype: Any | None = None,
) -> ChunkInputPlan:
    """Resolve source extents and deterministic work items without reading data."""

    full_shape = tuple(int(size) for size in source_shape)
    if not full_shape or any(size <= 0 for size in full_shape):
        raise ValueError("The chunk driver must have a non-empty, positive source shape.")
    declared_shape = provisional.driver.get("full_shape")
    if declared_shape is not None and tuple(int(size) for size in declared_shape) != full_shape:
        raise ValueError(
            f"Discovered driver shape {full_shape} does not match declared full_shape {tuple(declared_shape)}."
        )
    declared_dtype = provisional.driver.get("dtype")
    if declared_dtype is not None and source_dtype is not None:
        if np.dtype(declared_dtype) != np.dtype(source_dtype):
            raise ValueError(
                f"Discovered driver dtype {np.dtype(source_dtype)} does not match declared dtype "
                f"{np.dtype(declared_dtype)}."
            )

    rank_of_data = int(provisional.driver["rank_of_data"])
    if rank_of_data > len(full_shape):
        raise ValueError("driver.rank_of_data cannot exceed the discovered source rank.")
    batch_rank = len(full_shape) - rank_of_data
    batch_axes = tuple(range(batch_rank))
    data_axes = tuple(range(batch_rank, len(full_shape)))

    rules_by_axis = {rule.axis: rule for rule in provisional.axis_rules}
    invalid_axes = sorted(axis for axis in rules_by_axis if axis not in batch_axes)
    if invalid_axes:
        raise ValueError(f"Provisional chunking may only partition batch axes {batch_axes}; got {invalid_axes}.")

    axis_choices: list[list[tuple[AxisSelector, AxisSelector, int, int | None]]] = []
    final_batch_shape: list[int] = []
    for axis in batch_axes:
        rule = rules_by_axis.get(axis)
        if rule is None:
            axis_choices.append([(AxisSelector.all(), AxisSelector.all(), full_shape[axis], None)])
            final_batch_shape.append(full_shape[axis])
            continue
        segments, selected_count = _rule_segments(rule, full_shape[axis])
        axis_choices.append(
            [
                (source_selector, destination_selector, count, segment_index)
                for segment_index, (source_selector, destination_selector, count) in enumerate(segments)
            ]
        )
        final_batch_shape.append(selected_count)

    combinations = list(product(*axis_choices))
    width = max(6, len(str(max(0, len(combinations) - 1))))
    chunks: list[ProvisionalChunkSpec] = []
    for ordinal, combination in enumerate(combinations):
        source_batch = tuple(item[0] for item in combination)
        destination_batch = tuple(item[1] for item in combination)
        expected_batch_shape = tuple(item[2] for item in combination)
        grid_index = tuple(item[3] for item in combination if item[3] is not None)
        source_selection = source_batch + tuple(AxisSelector.all() for _ in data_axes)
        chunks.append(
            ProvisionalChunkSpec(
                schema_version=provisional.schema_version,
                plan_id=provisional.plan_id,
                provisional_hash=provisional.provisional_hash,
                chunk_id=f"c{ordinal:0{width}d}",
                ordinal=ordinal,
                grid_index=grid_index,
                source_selection=source_selection,
                expected_input_shape=expected_batch_shape + tuple(full_shape[axis] for axis in data_axes),
                destination_batch_selection=destination_batch,
                expected_batch_shape=expected_batch_shape,
            )
        )

    return ChunkInputPlan(
        schema_version=provisional.schema_version,
        plan_id=provisional.plan_id,
        provisional_hash=provisional.provisional_hash,
        full_shape=full_shape,
        dtype=None if source_dtype is None else np.dtype(source_dtype).str,
        batch_axes=batch_axes,
        data_axes=data_axes,
        final_batch_shape=tuple(final_batch_shape),
        chunks=tuple(chunks),
    )


def _resolve_basedata(processing_data: ProcessingData, output: ProvisionalChunkOutput) -> tuple[Any, BaseData]:
    bundle_name, basedata_name = tuple(part for part in output.processing_path.strip("/").split("/") if part)
    try:
        bundle = processing_data[bundle_name]
        basedata = bundle[basedata_name]
    except KeyError as exc:
        raise KeyError(f"Pilot result is missing output {output.processing_path!r}.") from exc
    if not isinstance(basedata, BaseData):
        raise TypeError(f"Pilot output {output.processing_path!r} is not BaseData.")
    return bundle, basedata


def _basedata_name(bundle: Any, candidate: BaseData | None) -> str | None:
    if candidate is None:
        return None
    for name, value in bundle.items():
        if value is candidate:
            return str(name)
    return None


def _pilot_axis_names(bundle: Any, basedata: BaseData, batch_rank: int) -> tuple[str, ...]:
    signal_rank = basedata.signal.ndim
    names = ["."] * signal_rank
    if basedata.axes:
        if len(basedata.axes) == signal_rank:
            offset = 0
        elif len(basedata.axes) == basedata.rank_of_data:
            offset = batch_rank
        else:
            raise ValueError(
                f"Pilot output has {len(basedata.axes)} axes for signal rank {signal_rank}; "
                "expected either the complete rank or rank_of_data."
            )
        for index, axis in enumerate(basedata.axes):
            if isinstance(axis, BaseData):
                names[offset + index] = _basedata_name(bundle, axis) or f"axis_{offset + index}"
    elif basedata.rank_of_data and isinstance(bundle.get("Q"), BaseData) and bundle.get("Q") is not basedata:
        for index in range(signal_rank - basedata.rank_of_data, signal_rank):
            names[index] = "Q"
    return tuple(names)


def _axis_by_name(bundle: Any, basedata: BaseData, axis_name: str, positions: list[int]) -> BaseData:
    candidates: list[BaseData] = []
    if len(basedata.axes) == basedata.signal.ndim:
        candidates.extend(
            axis for index, axis in enumerate(basedata.axes) if index in positions and isinstance(axis, BaseData)
        )
    elif len(basedata.axes) == basedata.rank_of_data:
        offset = basedata.signal.ndim - basedata.rank_of_data
        candidates.extend(
            axis
            for index, axis in enumerate(basedata.axes)
            if index + offset in positions and isinstance(axis, BaseData)
        )
    named = bundle.get(axis_name)
    if isinstance(named, BaseData):
        candidates.append(named)
    unique = {id(axis): axis for axis in candidates}
    if not unique:
        raise ValueError(f"Pilot output does not provide values for axis {axis_name!r}.")
    if len(unique) != 1:
        raise ValueError(f"Pilot output provides ambiguous values for axis {axis_name!r}.")
    return next(iter(unique.values()))


def _broadcast_layout(
    component: str,
    values: np.ndarray,
    pilot_signal_shape: tuple[int, ...],
    final_signal_shape: tuple[int, ...],
) -> ChunkArrayLayout:
    try:
        np.broadcast_shapes(values.shape, pilot_signal_shape)
    except ValueError as exc:
        raise ValueError(
            f"Pilot component {component!r} with shape {values.shape} cannot broadcast to signal "
            f"{pilot_signal_shape}."
        ) from exc
    binding = PlacementBinding("direct" if values.shape == pilot_signal_shape else "broadcast")
    return ChunkArrayLayout(
        component=component,
        final_shape=final_signal_shape,
        dtype=values.dtype.str,
        placement_binding=binding,
    )


def _resolve_output_layout(
    output: ProvisionalChunkOutput,
    processing_data: ProcessingData,
    input_plan: ChunkInputPlan,
    pilot: ProvisionalChunkSpec,
) -> ChunkOutputLayout:
    bundle, basedata = _resolve_basedata(processing_data, output)
    signal = np.asarray(basedata.signal)
    batch_rank = len(input_plan.batch_axes)
    if signal.ndim - basedata.rank_of_data != batch_rank:
        raise ValueError(
            f"Pilot output {output.output_id!r} has {signal.ndim - basedata.rank_of_data} batch dimensions; "
            f"the driver has {batch_rank}. The provisional workflow requires preserved batch dimensions."
        )
    if signal.shape[:batch_rank] != pilot.expected_batch_shape:
        raise ValueError(
            f"Pilot output {output.output_id!r} has batch shape {signal.shape[:batch_rank]}; "
            f"expected {pilot.expected_batch_shape}."
        )

    final_signal_shape = input_plan.final_batch_shape + signal.shape[batch_rank:]
    arrays: list[ChunkArrayLayout] = [
        ChunkArrayLayout(component="signal", final_shape=final_signal_shape, dtype=signal.dtype.str)
    ]

    weights = np.asarray(basedata.weights)
    if weights.size != 1 or float(weights.reshape(-1)[0]) != 1.0:
        if weights.size == 1:
            arrays.append(
                ChunkArrayLayout(
                    component="weights",
                    final_shape=(),
                    dtype=weights.dtype.str,
                    placement_binding=PlacementBinding("static"),
                )
            )
        else:
            arrays.append(_broadcast_layout("weights", weights, signal.shape, final_signal_shape))

    for name in sorted(basedata.uncertainties):
        arrays.append(
            _broadcast_layout(
                f"uncertainties/{name}",
                np.asarray(basedata.uncertainties[name]),
                signal.shape,
                final_signal_shape,
            )
        )

    axis_names = _pilot_axis_names(bundle, basedata, batch_rank)
    for axis_name in dict.fromkeys(name for name in axis_names if name != "."):
        positions = [index for index, name in enumerate(axis_names) if name == axis_name]
        axis = _axis_by_name(bundle, basedata, axis_name, positions)
        values = np.asarray(axis.signal)
        if all(position >= batch_rank for position in positions):
            expected_shape = tuple(signal.shape[position] for position in positions)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Static pilot axis {axis_name!r} has shape {values.shape}; expected {expected_shape}."
                )
            binding = PlacementBinding("static")
            final_shape = values.shape
        elif values.shape == signal.shape:
            binding = PlacementBinding("direct")
            final_shape = final_signal_shape
        else:
            expected_shape = tuple(signal.shape[position] for position in positions)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Pilot axis {axis_name!r} has shape {values.shape}; expected {expected_shape} "
                    "from its signal-axis positions."
                )
            binding = PlacementBinding("axis_map", tuple(positions))
            final_shape = tuple(final_signal_shape[position] for position in positions)
        arrays.append(
            ChunkArrayLayout(
                component=f"axes/{axis_name}",
                final_shape=final_shape,
                dtype=values.dtype.str,
                placement_binding=binding,
                units=str(axis.units),
                rank_of_data=axis.rank_of_data,
            )
        )

    return ChunkOutputLayout(
        output_id=output.output_id,
        processing_path=output.processing_path,
        destination_path=output.destination_path,
        units=str(basedata.units),
        rank_of_data=basedata.rank_of_data,
        arrays=tuple(arrays),
        axis_names=axis_names,
    )


def resolve_provisional_chunk_plan(
    provisional: ProvisionalChunkPlan,
    input_plan: ChunkInputPlan,
    processing_data: ProcessingData,
    pilot_chunk_id: str,
) -> tuple[ChunkPlan, tuple[ChunkSpec, ...]]:
    """Resolve output layouts from a pilot result and materialize every spec."""

    if input_plan.provisional_hash != provisional.provisional_hash:
        raise ValueError("ChunkInputPlan does not belong to the provisional plan.")
    pilot = input_plan.chunk(pilot_chunk_id)
    outputs = tuple(
        _resolve_output_layout(output, processing_data, input_plan, pilot) for output in provisional.outputs
    )
    driver = dict(provisional.driver)
    driver["full_shape"] = list(input_plan.full_shape)
    if input_plan.dtype is not None:
        driver["dtype"] = input_plan.dtype
    plan = ChunkPlan(
        schema_version=provisional.schema_version,
        plan_id=provisional.plan_id,
        total_chunks=len(input_plan.chunks),
        expected_chunk_ids=tuple(chunk.chunk_id for chunk in input_plan.chunks),
        outputs=outputs,
        driver=driver,
        batch_axes=input_plan.batch_axes,
        data_axes=input_plan.data_axes,
        axis_rules=tuple(rule.to_dict() for rule in provisional.axis_rules),
        bindings=provisional.bindings,
        source_bindings=provisional.source_bindings,
    )

    return plan, materialize_chunk_specs(plan, input_plan)


def materialize_chunk_specs(plan: ChunkPlan, input_plan: ChunkInputPlan) -> tuple[ChunkSpec, ...]:
    """Build final output placements from a resolved plan and stored source work."""

    if plan.plan_id != input_plan.plan_id or len(plan.expected_chunk_ids) != len(input_plan.chunks):
        raise ValueError("Resolved ChunkPlan and ChunkInputPlan identities do not match.")
    if plan.batch_axes != input_plan.batch_axes or plan.data_axes != input_plan.data_axes:
        raise ValueError("Resolved ChunkPlan and ChunkInputPlan axis roles do not match.")
    batch_rank = len(input_plan.batch_axes)
    specs: list[ChunkSpec] = []
    for provisional_chunk in input_plan.chunks:
        placements = tuple(
            ChunkPlacement(
                output_id=output.output_id,
                destination_selection=provisional_chunk.destination_batch_selection
                + tuple(AxisSelector.all() for _ in range(output.rank_of_data)),
                expected_shape=provisional_chunk.expected_batch_shape + output.signal.final_shape[batch_rank:],
            )
            for output in plan.outputs
        )
        specs.append(
            ChunkSpec(
                schema_version=plan.schema_version,
                plan_id=plan.plan_id,
                plan_hash=plan.plan_hash,
                chunk_id=provisional_chunk.chunk_id,
                ordinal=provisional_chunk.ordinal,
                grid_index=provisional_chunk.grid_index,
                source_selection=provisional_chunk.source_selection,
                expected_input_shape=provisional_chunk.expected_input_shape,
                placements=placements,
            )
        )
    return tuple(specs)
