# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import h5py
import numpy as np
from attrs import define, field, validators

from modacor import __version__
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunking import (
    AxisSelector,
    ChunkArrayLayout,
    ChunkOutputLayout,
    ChunkPlan,
    ChunkSpec,
    ChunkWriteResult,
    selection_shape,
)
from modacor.io.hdf.hdf_processing_sink import (
    _as_hdf_str_list,
    _json_dumps_bytes,
    _normalise_subpath,
    _recreate_group,
    _set_array_metadata,
    _set_nexus_default_chain,
    _write_text_dataset,
    _write_text_field,
)
from modacor.io.io_sink import IoSink

__all__ = ["HDFChunkedProcessingSink"]

CollisionPolicy = Literal["error", "resume", "replace"]


def _read_text(value: h5py.Dataset) -> str:
    payload = value[()]
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


def _status(group: h5py.Group) -> str:
    if "status" not in group:
        return "unknown"
    return _read_text(group["status"])


def _set_status(group: h5py.Group, status: str) -> None:
    _write_text_field(group, "status", status)


def _chunk_group(plan_group: h5py.Group, chunk_id: str) -> h5py.Group:
    chunks_group = plan_group.get("chunks")
    if not isinstance(chunks_group, h5py.Group) or chunk_id not in chunks_group:
        raise ValueError(f"Chunk {chunk_id!r} is not present in the stored manifest.")
    chunk_group = chunks_group[chunk_id]
    if not isinstance(chunk_group, h5py.Group):
        raise ValueError(f"Manifest entry for chunk {chunk_id!r} is not a group.")
    return chunk_group


def _completed_chunk_count(plan_group: h5py.Group) -> int:
    chunks_group = plan_group.get("chunks")
    if not isinstance(chunks_group, h5py.Group):
        return 0
    return sum(_status(chunks_group[name]) == "complete" for name in chunks_group)


def _fill_value(dtype: np.dtype) -> Any:
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.complexfloating):
        return complex(np.nan, np.nan)
    if np.issubdtype(dtype, np.bool_):
        return False
    return 0


def _destination_parts(destination_path: str) -> tuple[str, str]:
    parts = tuple(part for part in destination_path.strip("/").split("/") if part)
    if len(parts) != 2:  # guarded by ChunkOutputLayout; defensive at the I/O boundary
        raise ValueError("Chunk destination path must contain '<bundle>/<basedata>'.")
    return parts[0], parts[1]


def _component_parts(component: str) -> tuple[str, str | None]:
    if component in {"signal", "weights"}:
        return component, None
    prefix, separator, name = component.partition("/")
    if separator and prefix in {"uncertainties", "axes"} and name and "/" not in name:
        return prefix, name
    raise ValueError(
        f"Unsupported chunk array component {component!r}; expected signal, weights, "
        "uncertainties/<name>, or axes/<name>."
    )


def _component_dataset_path(basedata_path: str, component: str) -> str | None:
    kind, name = _component_parts(component)
    if kind == "weights" and name is None:
        return f"{basedata_path}/weights"
    if kind == "uncertainties":
        return f"{basedata_path}/uncertainties/{name}"
    if kind == "axes":
        return f"{basedata_path}/{name}"
    return f"{basedata_path}/signal"


def _component_selection(
    layout: ChunkArrayLayout,
    signal_selection: tuple[AxisSelector, ...],
) -> tuple[AxisSelector, ...]:
    binding = layout.placement_binding
    if binding.kind == "static":
        return tuple(AxisSelector.all() for _ in layout.final_shape)
    if binding.kind in {"direct", "broadcast"}:
        return signal_selection
    return tuple(signal_selection[axis] for axis in binding.axis_map)


def _component_state(plan_group: h5py.Group, output_id: str, component: str) -> h5py.Group:
    state_root = plan_group["components"]
    output_group = state_root[output_id]
    state = output_group
    for part in component.split("/"):
        state = state[part]
    if not isinstance(state, h5py.Group):  # pragma: no cover - defensive
        raise ValueError(f"Stored state for component {component!r} is not a group.")
    return state


def _values_equal(left: Any, right: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(left), right, equal_nan=True))


def _string_list(value: Any) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in list(value)]


@dataclass(slots=True)
class _PendingWrite:
    dataset: h5py.Dataset | None
    basedata_group: h5py.Group
    selection: tuple[slice | int, ...]
    values: np.ndarray
    static_state: h5py.Group | None = None
    scalar_weight: bool = False

    def write(self) -> None:
        if self.scalar_weight:
            self.basedata_group.attrs["weight_scalar"] = float(self.values.reshape(-1)[0])
            return
        assert self.dataset is not None
        if self.dataset.shape:
            self.dataset[self.selection] = self.values
        else:
            self.dataset[()] = self.values


def _create_component_dataset(
    basedata_group: h5py.Group,
    output: ChunkOutputLayout,
    layout: ChunkArrayLayout,
    *,
    compression: str | None,
    configured_chunks: Any,
) -> h5py.Dataset | None:
    kind, name = _component_parts(layout.component)
    if kind == "weights" and not layout.final_shape:
        return None
    if kind == "uncertainties":
        assert name is not None
        parent = basedata_group.require_group("uncertainties")
        dataset_name = name
    elif kind == "axes":
        assert name is not None
        parent = basedata_group
        dataset_name = name
    else:
        parent = basedata_group
        dataset_name = kind

    scalar = not layout.final_shape
    chunks = None if scalar else configured_chunks
    if isinstance(chunks, tuple) and len(chunks) != len(layout.final_shape):
        chunks = True
    dataset = parent.create_dataset(
        dataset_name,
        shape=layout.final_shape,
        dtype=np.dtype(layout.dtype),
        chunks=chunks,
        compression=None if scalar else compression,
        fillvalue=None if scalar else _fill_value(np.dtype(layout.dtype)),
    )
    if kind == "signal":
        _set_array_metadata(dataset, units=output.units, rank_of_data=output.rank_of_data)
    elif kind == "uncertainties":
        _set_array_metadata(dataset, units=output.units)
    elif kind == "axes":
        assert layout.units is not None and layout.rank_of_data is not None
        _set_array_metadata(dataset, units=layout.units, rank_of_data=layout.rank_of_data)
    return dataset


def _selection_bounds(
    full_shape: tuple[int, ...],
    selectors: tuple[AxisSelector, ...],
) -> tuple[tuple[int, int], ...]:
    if len(full_shape) != len(selectors):
        raise ValueError("Destination selector rank does not match the final signal rank.")
    bounds: list[tuple[int, int]] = []
    for size, selector in zip(full_shape, selectors, strict=True):
        if selector.kind == "all":
            bounds.append((0, size))
        elif selector.kind == "index":
            assert selector.value is not None
            bounds.append((selector.value, selector.value + 1))
        else:
            assert selector.start is not None and selector.stop is not None and selector.stride is not None
            if selector.stride != 1:
                raise NotImplementedError("The initial HDF chunk writer requires contiguous destination slices.")
            bounds.append((selector.start, selector.stop))
    return tuple(bounds)


def _bounds_overlap(left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]) -> bool:
    return all(
        max(left_start, right_start) < min(left_stop, right_stop)
        for (left_start, left_stop), (right_start, right_stop) in zip(left, right, strict=True)
    )


def _selection_volume(bounds: tuple[tuple[int, int], ...]) -> int:
    return math.prod(stop - start for start, stop in bounds)


def _resolve_output_data(
    processing_data: ProcessingData,
    output: ChunkOutputLayout,
) -> tuple[Any, BaseData]:
    bundle_key, basedata_name = tuple(part for part in output.processing_path.strip("/").split("/") if part)
    try:
        databundle = processing_data[bundle_key]
    except KeyError as exc:
        raise KeyError(f"ProcessingData is missing bundle {bundle_key!r} for output {output.output_id!r}.") from exc
    try:
        basedata = databundle[basedata_name]
    except KeyError as exc:
        raise KeyError(f"DataBundle {bundle_key!r} is missing BaseData {basedata_name!r}.") from exc
    if not isinstance(basedata, BaseData):
        raise TypeError(f"Chunk output {output.processing_path!r} did not resolve to BaseData.")
    return databundle, basedata


def _resolve_axis_basedata(
    databundle: Any,
    basedata: BaseData,
    output: ChunkOutputLayout,
    axis_name: str,
    signal_selection: tuple[AxisSelector, ...],
) -> BaseData:
    final_axis_indices = [index for index, name in enumerate(output.axis_names) if name == axis_name]
    retained_axes = [index for index, selector in enumerate(signal_selection) if selector.kind != "index"]

    candidates: list[BaseData] = []
    if len(basedata.axes) == len(output.axis_names):
        candidates.extend(
            axis
            for index, axis in enumerate(basedata.axes)
            if index in final_axis_indices and isinstance(axis, BaseData)
        )
    elif len(basedata.axes) == len(retained_axes):
        candidates.extend(
            axis
            for chunk_index, axis in enumerate(basedata.axes)
            if retained_axes[chunk_index] in final_axis_indices and isinstance(axis, BaseData)
        )

    named_candidate = databundle.get(axis_name) if hasattr(databundle, "get") else None
    if isinstance(named_candidate, BaseData):
        candidates.append(named_candidate)
    candidates = list({id(candidate): candidate for candidate in candidates}.values())
    if not candidates:
        raise ValueError(f"Chunk output {output.output_id!r} does not provide declared axis {axis_name!r}.")

    first = candidates[0]
    for candidate in candidates[1:]:
        if (
            str(candidate.units) != str(first.units)
            or candidate.rank_of_data != first.rank_of_data
            or not _values_equal(candidate.signal, np.asarray(first.signal))
        ):
            raise ValueError(f"Chunk output {output.output_id!r} provides conflicting values for axis {axis_name!r}.")
    return first


def _component_values(
    databundle: Any,
    basedata: BaseData,
    output: ChunkOutputLayout,
    layout: ChunkArrayLayout,
    signal_selection: tuple[AxisSelector, ...],
) -> tuple[np.ndarray, str | None, int | None, BaseData | None]:
    kind, name = _component_parts(layout.component)
    if kind == "signal":
        return np.asarray(basedata.signal), str(basedata.units), basedata.rank_of_data, None
    if kind == "weights":
        return np.asarray(basedata.weights), None, None, None
    if kind == "uncertainties":
        assert name is not None
        try:
            values = basedata.uncertainties[name]
        except KeyError as exc:
            raise ValueError(f"Chunk output {output.output_id!r} is missing uncertainty {name!r}.") from exc
        return np.asarray(values), str(basedata.units), None, None

    assert name is not None
    axis = _resolve_axis_basedata(databundle, basedata, output, name, signal_selection)
    return np.asarray(axis.signal), str(axis.units), axis.rank_of_data, axis


def _validate_basedata_schema(output: ChunkOutputLayout, basedata: BaseData) -> None:
    if str(basedata.units) != output.units:
        raise ValueError(f"Chunk output {output.output_id!r} has units {basedata.units}; expected {output.units}.")
    if basedata.rank_of_data != output.rank_of_data:
        raise ValueError(
            f"Chunk output {output.output_id!r} has rank_of_data {basedata.rank_of_data}; "
            f"expected {output.rank_of_data}."
        )

    declared_uncertainties = {
        layout.component.removeprefix("uncertainties/")
        for layout in output.arrays
        if layout.component.startswith("uncertainties/")
    }
    if set(basedata.uncertainties) != declared_uncertainties:
        raise ValueError(
            f"Chunk output {output.output_id!r} uncertainty keys {sorted(basedata.uncertainties)} "
            f"do not match the declared keys {sorted(declared_uncertainties)}."
        )
    has_weights_layout = any(layout.component == "weights" for layout in output.arrays)
    if not has_weights_layout and (basedata.weights.size != 1 or float(basedata.weights.reshape(-1)[0]) != 1.0):
        raise ValueError(f"Chunk output {output.output_id!r} has non-default weights but no weights layout.")


def _validate_component_metadata(
    output: ChunkOutputLayout,
    layout: ChunkArrayLayout,
    values: np.ndarray,
    component_units: str | None,
    component_rank: int | None,
) -> tuple[str, str | None]:
    expected_dtype = np.dtype(layout.dtype)
    if values.dtype != expected_dtype:
        raise ValueError(
            f"Chunk output {output.output_id!r} component {layout.component!r} has dtype "
            f"{values.dtype}; expected {expected_dtype}."
        )
    kind, name = _component_parts(layout.component)
    expected_units = output.units if kind in {"signal", "uncertainties"} else layout.units
    expected_rank = output.rank_of_data if kind == "signal" else layout.rank_of_data
    if expected_units is not None and component_units != expected_units:
        raise ValueError(
            f"Chunk output {output.output_id!r} component {layout.component!r} has units "
            f"{component_units}; expected {expected_units}."
        )
    if expected_rank is not None and component_rank != expected_rank:
        raise ValueError(
            f"Chunk output {output.output_id!r} component {layout.component!r} has "
            f"rank_of_data {component_rank}; expected {expected_rank}."
        )
    return kind, name


def _prepare_component_write(
    h5: h5py.File,
    plan_group: h5py.Group,
    basedata_group: h5py.Group,
    basedata_path: str,
    output: ChunkOutputLayout,
    layout: ChunkArrayLayout,
    databundle: Any,
    basedata: BaseData,
    signal_selection: tuple[AxisSelector, ...],
) -> tuple[_PendingWrite | None, BaseData | None]:
    values, component_units, component_rank, axis = _component_values(
        databundle,
        basedata,
        output,
        layout,
        signal_selection,
    )
    kind, axis_name = _validate_component_metadata(
        output,
        layout,
        values,
        component_units,
        component_rank,
    )
    if kind == "axes":
        assert axis_name is not None and axis is not None

    selectors = _component_selection(layout, signal_selection)
    expected_shape = selection_shape(layout.final_shape, selectors)
    if layout.placement_binding.kind == "broadcast":
        try:
            values = np.broadcast_to(values, expected_shape)
        except ValueError as exc:
            raise ValueError(
                f"Chunk output {output.output_id!r} component {layout.component!r} with shape "
                f"{values.shape} cannot broadcast to {expected_shape}."
            ) from exc
    elif values.shape != expected_shape:
        raise ValueError(
            f"Chunk output {output.output_id!r} component {layout.component!r} has shape "
            f"{values.shape}; expected {expected_shape}."
        )

    scalar_weight = kind == "weights" and not layout.final_shape
    dataset_path = _component_dataset_path(basedata_path, layout.component)
    assert dataset_path is not None
    dataset = None if scalar_weight else h5[dataset_path]
    static_state = None
    if layout.placement_binding.kind == "static":
        static_state = _component_state(plan_group, output.output_id, layout.component)
        if _status(static_state) == "complete":
            stored_values = (
                basedata_group.attrs.get("weight_scalar")
                if scalar_weight
                else dataset[()] if not layout.final_shape else dataset[...]
            )
            if not _values_equal(stored_values, values):
                raise ValueError(
                    f"Static component {layout.component!r} for output {output.output_id!r} changed between chunks."
                )
            return None, axis

    return (
        _PendingWrite(
            dataset=dataset,
            basedata_group=basedata_group,
            selection=tuple(selector.to_index() for selector in selectors),
            values=values,
            static_state=static_state,
            scalar_weight=scalar_weight,
        ),
        axis,
    )


def _prepare_output_writes(
    h5: h5py.File,
    plan_group: h5py.Group,
    run_name: str,
    processing_data: ProcessingData,
    output: ChunkOutputLayout,
    signal_selection: tuple[AxisSelector, ...],
) -> list[_PendingWrite]:
    databundle, basedata = _resolve_output_data(processing_data, output)
    _validate_basedata_schema(output, basedata)
    bundle_key, basedata_name = _destination_parts(output.destination_path)
    basedata_path = f"processing/result/{run_name}/{bundle_key}/{basedata_name}"
    basedata_group = h5[basedata_path]
    pending_writes: list[_PendingWrite] = []
    declared_axis_objects: set[int] = set()

    for layout in output.arrays:
        pending, axis = _prepare_component_write(
            h5,
            plan_group,
            basedata_group,
            basedata_path,
            output,
            layout,
            databundle,
            basedata,
            signal_selection,
        )
        if pending is not None:
            pending_writes.append(pending)
        if axis is not None:
            declared_axis_objects.add(id(axis))

    undeclared_axes = [
        axis for axis in basedata.axes if isinstance(axis, BaseData) and id(axis) not in declared_axis_objects
    ]
    if undeclared_axes:
        raise ValueError(f"Chunk output {output.output_id!r} contains axes not declared by the plan.")
    return pending_writes


@define(kw_only=True)
class HDFChunkedProcessingSink(IoSink):
    """Assemble signal arrays into fixed-shape HDF5 destinations."""

    supports_chunked_writes: ClassVar[bool] = True

    resource_location: Path = field(converter=Path, validator=validators.instance_of(Path))
    iosink_method_kwargs: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))
    logger: MessageHandler = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.logger = MessageHandler(level=self.logging_level, name="HDFChunkedProcessingSink")

    def _path(self, override_resource_location: Path | None = None) -> Path:
        return (override_resource_location or self.resource_location).expanduser()

    @staticmethod
    def _validate_supported_plan(plan: ChunkPlan) -> None:
        destination_paths = [output.destination_path for output in plan.outputs]
        if len(set(destination_paths)) != len(destination_paths):
            raise ValueError("ChunkPlan output destination paths must be unique.")
        for output in plan.outputs:
            if not output.signal.final_shape:
                raise ValueError("Chunked signal outputs must not be scalar.")
            for layout in output.arrays:
                _component_parts(layout.component)
                binding = layout.placement_binding
                if binding.kind in {"direct", "broadcast"} and layout.final_shape != output.signal.final_shape:
                    raise ValueError(
                        f"Component {layout.component!r} in output {output.output_id!r} must match "
                        "the signal final shape for direct or broadcast placement."
                    )
                if binding.kind == "axis_map" and len(binding.axis_map) != len(layout.final_shape):
                    raise ValueError(
                        f"Component {layout.component!r} in output {output.output_id!r} has an invalid axis map."
                    )

    @staticmethod
    def _plan_group(h5: h5py.File, plan: ChunkPlan) -> h5py.Group:
        path = f"processing/chunk_plans/{plan.plan_id}"
        if path not in h5:
            raise ValueError(f"Chunk plan {plan.plan_id!r} has not been initialized in this file.")
        plan_group = h5[path]
        if not isinstance(plan_group, h5py.Group):
            raise ValueError(f"Stored chunk plan path {path!r} is not a group.")
        if str(plan_group.attrs.get("plan_hash", "")) != plan.plan_hash:
            raise ValueError("Stored chunk plan hash does not match the requested plan.")
        stored_plan = ChunkPlan.from_dict(json.loads(_read_text(plan_group["plan_json"])))
        if stored_plan.plan_hash != plan.plan_hash:
            raise ValueError("Stored chunk plan content does not match the requested plan.")
        return plan_group

    @staticmethod
    def _validate_stored_layout(h5: h5py.File, run_name: str, plan: ChunkPlan) -> None:
        plan_group = h5[f"processing/chunk_plans/{plan.plan_id}"]
        if not isinstance(plan_group, h5py.Group):  # pragma: no cover - guarded by _plan_group
            raise ValueError(f"Stored chunk plan {plan.plan_id!r} is not a group.")
        for output in plan.outputs:
            bundle_key, basedata_name = _destination_parts(output.destination_path)
            basedata_path = f"processing/result/{run_name}/{bundle_key}/{basedata_name}"
            if basedata_path not in h5 or not isinstance(h5[basedata_path], h5py.Group):
                raise ValueError(f"Initialized chunk output is missing BaseData group {basedata_path!r}.")
            basedata_group = h5[basedata_path]
            expected_axis_names = list(output.axis_names or (".",) * len(output.signal.final_shape))
            if _string_list(basedata_group.attrs.get("axes", [])) != expected_axis_names:
                raise ValueError(f"Initialized BaseData group {basedata_path!r} has different axis metadata.")

            for layout in output.arrays:
                if layout.component == "weights" and not layout.final_shape:
                    state = _component_state(plan_group, output.output_id, "weights")
                    if _status(state) == "complete" and "weight_scalar" not in basedata_group.attrs:
                        raise ValueError(f"Initialized BaseData group {basedata_path!r} is missing weight_scalar.")
                    continue
                dataset_path = _component_dataset_path(basedata_path, layout.component)
                assert dataset_path is not None
                if dataset_path not in h5:
                    raise ValueError(f"Initialized chunk output is missing dataset {dataset_path!r}.")
                dataset = h5[dataset_path]
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"Initialized chunk output path {dataset_path!r} is not a dataset.")
                if tuple(dataset.shape) != layout.final_shape or dataset.dtype != np.dtype(layout.dtype):
                    raise ValueError(f"Initialized dataset {dataset_path!r} does not match the ChunkPlan layout.")
                kind, _ = _component_parts(layout.component)
                expected_units = output.units if kind in {"signal", "uncertainties"} else layout.units
                expected_rank = output.rank_of_data if kind == "signal" else layout.rank_of_data
                if expected_units is not None and str(dataset.attrs.get("units", "")) != expected_units:
                    raise ValueError(f"Initialized dataset {dataset_path!r} has different units metadata.")
                if expected_rank is not None and int(dataset.attrs.get("rank_of_data", -1)) != expected_rank:
                    raise ValueError(f"Initialized dataset {dataset_path!r} has different rank metadata.")

    @staticmethod
    def _stored_specs(plan_group: h5py.Group, *, exclude_chunk_id: str | None = None) -> list[ChunkSpec]:
        chunks_group = plan_group["chunks"]
        specs: list[ChunkSpec] = []
        for chunk_id in chunks_group:
            if chunk_id == exclude_chunk_id:
                continue
            entry = chunks_group[chunk_id]
            if _status(entry) not in {"writing", "complete"} or "spec_json" not in entry:
                continue
            specs.append(ChunkSpec.from_dict(json.loads(_read_text(entry["spec_json"]))))
        return specs

    @staticmethod
    def _validate_disjoint_placements(plan: ChunkPlan, chunk: ChunkSpec, stored_specs: list[ChunkSpec]) -> None:
        for placement in chunk.placements:
            output = plan.output(placement.output_id)
            for layout in output.arrays:
                if layout.placement_binding.kind == "static":
                    continue
                candidate_selection = _component_selection(layout, placement.destination_selection)
                candidate_bounds = _selection_bounds(layout.final_shape, candidate_selection)
                for stored_spec in stored_specs:
                    stored_placement = next(
                        item for item in stored_spec.placements if item.output_id == placement.output_id
                    )
                    stored_selection = _component_selection(layout, stored_placement.destination_selection)
                    stored_bounds = _selection_bounds(layout.final_shape, stored_selection)
                    if _bounds_overlap(candidate_bounds, stored_bounds):
                        raise ValueError(
                            f"Chunk {chunk.chunk_id!r} output {placement.output_id!r}, component "
                            f"{layout.component!r} overlaps chunk {stored_spec.chunk_id!r}."
                        )

    @staticmethod
    def _validate_complete_coverage(plan_group: h5py.Group, plan: ChunkPlan, specs: list[ChunkSpec]) -> None:
        grid_indices = [spec.grid_index for spec in specs]
        if len(set(grid_indices)) != len(grid_indices):
            raise ValueError("Completed chunk manifest contains duplicate grid_index values.")

        for output in plan.outputs:
            for layout in output.arrays:
                if layout.placement_binding.kind == "static":
                    state = _component_state(plan_group, output.output_id, layout.component)
                    if _status(state) != "complete":
                        raise ValueError(
                            f"Static component {layout.component!r} for output {output.output_id!r} is incomplete."
                        )
                    continue

                bounds = []
                for spec in specs:
                    placement = next(item for item in spec.placements if item.output_id == output.output_id)
                    component_selection = _component_selection(layout, placement.destination_selection)
                    bounds.append(_selection_bounds(layout.final_shape, component_selection))
                for index, left in enumerate(bounds):
                    for right in bounds[index + 1 :]:
                        if _bounds_overlap(left, right):
                            raise ValueError(
                                f"Completed chunks overlap for output {output.output_id!r}, "
                                f"component {layout.component!r}."
                            )
                selected_volume = sum(_selection_volume(item) for item in bounds)
                expected_volume = math.prod(layout.final_shape)
                if selected_volume != expected_volume:
                    raise ValueError(
                        f"Completed chunks cover {selected_volume} elements for output {output.output_id!r}, "
                        f"component {layout.component!r}; expected {expected_volume}."
                    )

    def _result(
        self,
        plan_group: h5py.Group,
        plan: ChunkPlan,
        *,
        chunk_id: str | None = None,
        resource_location: Path | None = None,
    ) -> ChunkWriteResult:
        return ChunkWriteResult(
            status=_status(plan_group),
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            expected_chunks=plan.total_chunks,
            completed_chunks=_completed_chunk_count(plan_group),
            chunk_id=chunk_id,
            resource_location=str(resource_location or self.resource_location),
        )

    def initialize_chunked(
        self,
        subpath: str,
        plan: ChunkPlan,
        *,
        collision: CollisionPolicy = "error",
        override_resource_location: Path | None = None,
    ) -> ChunkWriteResult:
        self._validate_supported_plan(plan)
        if collision not in {"error", "resume", "replace"}:
            raise ValueError("collision must be 'error', 'resume', or 'replace'.")

        out_path = self._path(override_resource_location)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_name = _normalise_subpath(subpath)

        with h5py.File(out_path, "a") as h5:
            processing_group = h5.require_group("processing")
            result_root = processing_group.require_group("result")
            plans_root = processing_group.require_group("chunk_plans")
            has_run = run_name in result_root
            has_plan = plan.plan_id in plans_root

            if has_run or has_plan:
                if collision == "error":
                    raise FileExistsError(f"Chunked output run {run_name!r} or plan {plan.plan_id!r} already exists.")
                if collision == "resume":
                    if not has_run or not has_plan:
                        raise ValueError("Cannot resume a partially initialized chunked output; use replace.")
                    plan_group = self._plan_group(h5, plan)
                    self._validate_stored_layout(h5, run_name, plan)
                    return self._result(plan_group, plan, resource_location=out_path)
                if has_run:
                    del result_root[run_name]
                if has_plan:
                    del plans_root[plan.plan_id]

            plan_group = plans_root.create_group(plan.plan_id)
            plan_group.attrs["plan_id"] = plan.plan_id
            plan_group.attrs["plan_hash"] = plan.plan_hash
            plan_group.attrs["run_name"] = run_name
            plan_group.attrs["schema_version"] = plan.schema_version
            plan_group.attrs["modacor_version"] = __version__
            plan_group.attrs["expected_chunks"] = plan.total_chunks
            plan_group.attrs["completed_chunks"] = 0
            _write_text_field(plan_group, "plan_json", plan.to_json())
            _set_status(plan_group, "initializing")
            chunks_group = plan_group.create_group("chunks")
            for ordinal, chunk_id in enumerate(plan.expected_chunk_ids):
                entry = chunks_group.create_group(chunk_id)
                entry.attrs["ordinal"] = ordinal
                _set_status(entry, "pending")
            components_group = plan_group.create_group("components")

            run_group = result_root.create_group(run_name)
            run_group.attrs["NX_class"] = "NXcollection"
            run_group.attrs["modacor_version"] = __version__
            compression = self.iosink_method_kwargs.get("compression")
            hdf_chunks = self.iosink_method_kwargs.get("chunks", True)
            for output in plan.outputs:
                bundle_key, basedata_name = _destination_parts(output.destination_path)
                bundle_group = run_group.require_group(bundle_key)
                basedata_group = bundle_group.create_group(basedata_name)
                basedata_group.attrs["NX_class"] = "NXdata"
                basedata_group.attrs["default"] = "signal"
                basedata_group.attrs["signal"] = "signal"
                axis_names = output.axis_names or (".",) * len(output.signal.final_shape)
                basedata_group.attrs["axes"] = _as_hdf_str_list(axis_names)
                output_states = components_group.create_group(output.output_id)
                for layout in output.arrays:
                    _create_component_dataset(
                        basedata_group,
                        output,
                        layout,
                        compression=compression,
                        configured_chunks=hdf_chunks,
                    )
                    if layout.placement_binding.kind == "static":
                        state = output_states.require_group(layout.component)
                        _set_status(state, "pending")

            _write_text_field(processing_group, "program_name", "MoDaCor")
            _write_text_field(processing_group, "program_version", __version__)
            _set_status(plan_group, "writing")
            h5.flush()
            return self._result(plan_group, plan, resource_location=out_path)

    def write_chunk(
        self,
        subpath: str,
        processing_data: ProcessingData,
        *,
        plan: ChunkPlan,
        chunk: ChunkSpec,
        execution_metadata: dict[str, Any] | None = None,
        override_resource_location: Path | None = None,
    ) -> ChunkWriteResult:
        self._validate_supported_plan(plan)
        chunk.validate_for_plan(plan)
        out_path = self._path(override_resource_location)
        run_name = _normalise_subpath(subpath)

        pending_writes: list[_PendingWrite] = []
        with h5py.File(out_path, "r+") as h5:
            plan_group = self._plan_group(h5, plan)
            if _status(plan_group) != "writing":
                raise RuntimeError(f"Chunk plan {plan.plan_id!r} is not writable (status={_status(plan_group)!r}).")
            self._validate_stored_layout(h5, run_name, plan)
            manifest_entry = _chunk_group(plan_group, chunk.chunk_id)
            entry_status = _status(manifest_entry)
            if entry_status == "complete":
                stored_spec = _read_text(manifest_entry["spec_json"])
                if stored_spec != chunk.to_json():
                    raise ValueError(f"Completed chunk {chunk.chunk_id!r} has a conflicting ChunkSpec.")
                return self._result(
                    plan_group,
                    plan,
                    chunk_id=chunk.chunk_id,
                    resource_location=out_path,
                )
            if entry_status not in {"pending", "writing"}:
                raise RuntimeError(f"Chunk {chunk.chunk_id!r} cannot be written from status {entry_status!r}.")
            if entry_status == "writing" and "spec_json" in manifest_entry:
                if _read_text(manifest_entry["spec_json"]) != chunk.to_json():
                    raise ValueError(f"Interrupted chunk {chunk.chunk_id!r} has a conflicting ChunkSpec.")

            self._validate_disjoint_placements(
                plan,
                chunk,
                self._stored_specs(plan_group, exclude_chunk_id=chunk.chunk_id),
            )

            for placement in chunk.placements:
                output = plan.output(placement.output_id)
                pending_writes.extend(
                    _prepare_output_writes(
                        h5,
                        plan_group,
                        run_name,
                        processing_data,
                        output,
                        placement.destination_selection,
                    )
                )

            _write_text_field(manifest_entry, "spec_json", chunk.to_json())
            if execution_metadata is not None:
                _write_text_field(
                    manifest_entry,
                    "execution_json",
                    json.dumps(execution_metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                )
            _set_status(manifest_entry, "writing")
            for pending in pending_writes:
                if pending.static_state is not None:
                    _set_status(pending.static_state, "writing")
            h5.flush()
            for pending in pending_writes:
                pending.write()
            h5.flush()
            for pending in pending_writes:
                if pending.static_state is not None:
                    _set_status(pending.static_state, "complete")
            _set_status(manifest_entry, "complete")
            plan_group.attrs["completed_chunks"] = _completed_chunk_count(plan_group)
            h5.flush()
            return self._result(
                plan_group,
                plan,
                chunk_id=chunk.chunk_id,
                resource_location=out_path,
            )

    def finalize_chunked(
        self,
        subpath: str,
        *,
        plan: ChunkPlan,
        pipeline_spec: dict[str, Any] | None = None,
        pipeline_yaml: str | None = None,
        override_resource_location: Path | None = None,
    ) -> ChunkWriteResult:
        self._validate_supported_plan(plan)
        out_path = self._path(override_resource_location)
        run_name = _normalise_subpath(subpath)

        with h5py.File(out_path, "r+") as h5:
            plan_group = self._plan_group(h5, plan)
            current_status = _status(plan_group)
            if current_status == "complete":
                return self._result(plan_group, plan, resource_location=out_path)
            if current_status not in {"writing", "finalizing"}:
                raise RuntimeError(f"Chunk plan {plan.plan_id!r} cannot finalize from status {current_status!r}.")

            incomplete = [
                chunk_id
                for chunk_id in plan.expected_chunk_ids
                if _status(_chunk_group(plan_group, chunk_id)) != "complete"
            ]
            if incomplete:
                preview = ", ".join(incomplete[:10])
                suffix = "" if len(incomplete) <= 10 else f" (+{len(incomplete) - 10} more)"
                raise RuntimeError(f"Cannot finalize chunk plan; incomplete chunks: {preview}{suffix}.")

            completed_specs = self._stored_specs(plan_group)
            if len(completed_specs) != plan.total_chunks:
                raise ValueError("Completed chunk manifest is missing one or more stored ChunkSpec records.")
            self._validate_complete_coverage(plan_group, plan, completed_specs)

            _set_status(plan_group, "finalizing")
            h5.flush()
            self._validate_stored_layout(h5, run_name, plan)

            processing_group = h5["processing"]
            pipeline_root = processing_group.require_group("pipeline")
            pipeline_group = _recreate_group(pipeline_root, run_name)
            if pipeline_spec is None and pipeline_yaml is None:
                pipeline_group.attrs["empty"] = True
            else:
                if pipeline_spec is not None:
                    pipeline_group.create_dataset("spec", data=_json_dumps_bytes(pipeline_spec))
                if pipeline_yaml is not None:
                    _write_text_dataset(pipeline_group, "yaml", pipeline_yaml)

            tracer_root = processing_group.require_group("tracer")
            tracer_group = _recreate_group(tracer_root, run_name)
            tracer_group.attrs["empty"] = True

            first_output = plan.outputs[0]
            default_path = _destination_parts(first_output.destination_path)
            _set_nexus_default_chain(h5, run_name=run_name, default_path=default_path)
            plan_group.attrs["completed_chunks"] = plan.total_chunks
            _set_status(plan_group, "complete")
            h5.flush()
            result = self._result(plan_group, plan, resource_location=out_path)

        self.logger.info(f"Finalized chunked processing results in {out_path} (run={run_name}).")
        return result
