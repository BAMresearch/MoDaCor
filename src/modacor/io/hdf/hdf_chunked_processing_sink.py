# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import h5py
import numpy as np
from attrs import define, field, validators

from modacor import __version__
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunking import AxisSelector, ChunkPlan, ChunkSpec, ChunkWriteResult
from modacor.io.hdf.hdf_processing_sink import (
    _as_hdf_str_list,
    _json_dumps_bytes,
    _normalise_subpath,
    _recreate_group,
    _set_nexus_default_chain,
    _write_text_dataset,
    _write_text_field,
)
from modacor.io.io_sink import IoSink
from modacor.io.processing_path import resolve_processing_path

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
            components = {array.component for array in output.arrays}
            if components != {"signal"}:
                raise NotImplementedError(
                    "The initial HDF chunk writer supports signal arrays only; "
                    f"output {output.output_id!r} declares {sorted(components)}."
                )
            if not output.signal.final_shape:
                raise ValueError("Chunked signal outputs must not be scalar.")

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
        for output in plan.outputs:
            bundle_key, basedata_name = _destination_parts(output.destination_path)
            dataset_path = f"processing/result/{run_name}/{bundle_key}/{basedata_name}/signal"
            if dataset_path not in h5:
                raise ValueError(f"Initialized chunk output is missing dataset {dataset_path!r}.")
            dataset = h5[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"Initialized chunk output path {dataset_path!r} is not a dataset.")
            if tuple(dataset.shape) != output.signal.final_shape or dataset.dtype != np.dtype(output.signal.dtype):
                raise ValueError(f"Initialized dataset {dataset_path!r} does not match the ChunkPlan layout.")

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
            candidate_bounds = _selection_bounds(output.signal.final_shape, placement.destination_selection)
            for stored_spec in stored_specs:
                stored_placement = next(
                    item for item in stored_spec.placements if item.output_id == placement.output_id
                )
                stored_bounds = _selection_bounds(
                    output.signal.final_shape,
                    stored_placement.destination_selection,
                )
                if _bounds_overlap(candidate_bounds, stored_bounds):
                    raise ValueError(
                        f"Chunk {chunk.chunk_id!r} output {placement.output_id!r} overlaps "
                        f"chunk {stored_spec.chunk_id!r}."
                    )

    @staticmethod
    def _validate_complete_coverage(plan: ChunkPlan, specs: list[ChunkSpec]) -> None:
        grid_indices = [spec.grid_index for spec in specs]
        if len(set(grid_indices)) != len(grid_indices):
            raise ValueError("Completed chunk manifest contains duplicate grid_index values.")

        for output in plan.outputs:
            bounds = []
            for spec in specs:
                placement = next(item for item in spec.placements if item.output_id == output.output_id)
                bounds.append(_selection_bounds(output.signal.final_shape, placement.destination_selection))
            for index, left in enumerate(bounds):
                for right in bounds[index + 1 :]:
                    if _bounds_overlap(left, right):
                        raise ValueError(f"Completed chunks overlap for output {output.output_id!r}.")
            selected_volume = sum(_selection_volume(item) for item in bounds)
            expected_volume = math.prod(output.signal.final_shape)
            if selected_volume != expected_volume:
                raise ValueError(
                    f"Completed chunks cover {selected_volume} elements for output {output.output_id!r}; "
                    f"expected {expected_volume}."
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
                basedata_group.attrs["axes"] = _as_hdf_str_list(["."] * len(output.signal.final_shape))
                basedata_group.attrs["output_id"] = output.output_id
                dataset = basedata_group.create_dataset(
                    "signal",
                    shape=output.signal.final_shape,
                    dtype=np.dtype(output.signal.dtype),
                    chunks=hdf_chunks,
                    compression=compression,
                    fillvalue=_fill_value(np.dtype(output.signal.dtype)),
                )
                dataset.attrs["units"] = output.units
                dataset.attrs["rank_of_data"] = output.rank_of_data

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

        resolved: list[tuple[h5py.Dataset, tuple[slice | int, ...], np.ndarray]] = []
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
                basedata = resolve_processing_path(processing_data, output.processing_path)
                if not isinstance(basedata, BaseData):
                    raise TypeError(f"Chunk output {output.processing_path!r} did not resolve to BaseData.")
                array = np.asarray(basedata.signal)
                if array.shape != placement.expected_shape:
                    raise ValueError(
                        f"Chunk output {output.output_id!r} has shape {array.shape}; "
                        f"expected {placement.expected_shape}."
                    )
                if array.dtype != np.dtype(output.signal.dtype):
                    raise ValueError(
                        f"Chunk output {output.output_id!r} has dtype {array.dtype}; "
                        f"expected {np.dtype(output.signal.dtype)}."
                    )
                if str(basedata.units) != output.units:
                    raise ValueError(
                        f"Chunk output {output.output_id!r} has units {basedata.units}; expected {output.units}."
                    )
                if basedata.rank_of_data != output.rank_of_data:
                    raise ValueError(
                        f"Chunk output {output.output_id!r} has rank_of_data {basedata.rank_of_data}; "
                        f"expected {output.rank_of_data}."
                    )
                bundle_key, basedata_name = _destination_parts(output.destination_path)
                dataset = h5[f"processing/result/{run_name}/{bundle_key}/{basedata_name}/signal"]
                selection = tuple(selector.to_index() for selector in placement.destination_selection)
                resolved.append((dataset, selection, array))

            _write_text_field(manifest_entry, "spec_json", chunk.to_json())
            if execution_metadata is not None:
                _write_text_field(
                    manifest_entry,
                    "execution_json",
                    json.dumps(execution_metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                )
            _set_status(manifest_entry, "writing")
            h5.flush()
            for dataset, selection, array in resolved:
                dataset[selection] = array
            h5.flush()
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
            self._validate_complete_coverage(plan, completed_specs)

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
