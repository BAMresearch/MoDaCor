# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

__all__ = [
    "AxisSelector",
    "ChunkArrayLayout",
    "ChunkOutputLayout",
    "ChunkPlacement",
    "ChunkPlan",
    "ChunkSpec",
    "ChunkWriteResult",
    "PlacementBinding",
    "UnsupportedSinkCapability",
    "selection_shape",
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
SelectorKind = Literal["all", "index", "slice"]
PlacementKind = Literal["direct", "static", "broadcast", "axis_map"]


class UnsupportedSinkCapability(RuntimeError):
    """Raised when an optional operation is requested from an incapable sink."""

    def __init__(self, sink_type: type[Any], capability: str) -> None:
        super().__init__(f"Sink {sink_type.__name__} does not support {capability.replace('_', ' ')}.")
        self.sink_type = sink_type
        self.capability = capability


def _require_identifier(value: str, field_name: str) -> str:
    identifier = str(value).strip()
    if not identifier or _IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise ValueError(f"{field_name} must contain only letters, digits, '.', '_', or '-'.")
    return identifier


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return result


def _freeze_json(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinite values.")
        return value
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze_json(item, field_name) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain JSON-compatible values, got {type(value).__name__}.")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class AxisSelector:
    """Normalized, JSON-safe representation of one array-axis selector."""

    kind: SelectorKind
    value: int | None = None
    start: int | None = None
    stop: int | None = None
    stride: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"all", "index", "slice"}:
            raise ValueError("AxisSelector.kind must be 'all', 'index', or 'slice'.")

        if self.kind == "all":
            if any(item is not None for item in (self.value, self.start, self.stop, self.stride)):
                raise ValueError("An 'all' selector cannot define value, start, stop, or stride.")
            return

        if self.kind == "index":
            if self.value is None:
                raise ValueError("An 'index' selector requires value.")
            value = _require_non_negative_int(self.value, "AxisSelector.value")
            if any(item is not None for item in (self.start, self.stop, self.stride)):
                raise ValueError("An 'index' selector cannot define start, stop, or stride.")
            object.__setattr__(self, "value", value)
            return

        if self.value is not None:
            raise ValueError("A 'slice' selector cannot define value.")
        if self.start is None or self.stop is None:
            raise ValueError("A normalized 'slice' selector requires start and stop.")
        start = _require_non_negative_int(self.start, "AxisSelector.start")
        stop = _require_non_negative_int(self.stop, "AxisSelector.stop")
        stride = 1 if self.stride is None else self.stride
        if isinstance(stride, bool) or not isinstance(stride, Integral):
            raise TypeError("AxisSelector.stride must be an integer.")
        stride = int(stride)
        if stride <= 0:
            raise ValueError("AxisSelector.stride must be positive.")
        if stop <= start:
            raise ValueError("A normalized slice must select at least one element.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "stride", stride)

    @classmethod
    def all(cls) -> AxisSelector:
        return cls(kind="all")

    @classmethod
    def index(cls, value: int) -> AxisSelector:
        return cls(kind="index", value=value)

    @classmethod
    def sliced(cls, start: int, stop: int, stride: int = 1) -> AxisSelector:
        return cls(kind="slice", start=start, stop=stop, stride=stride)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AxisSelector:
        return cls(
            kind=str(payload.get("kind", "")),  # type: ignore[arg-type]
            value=payload.get("value"),
            start=payload.get("start"),
            stop=payload.get("stop"),
            stride=payload.get("stride"),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "all":
            return {"kind": "all"}
        if self.kind == "index":
            return {"kind": "index", "value": self.value}
        return {"kind": "slice", "start": self.start, "stop": self.stop, "stride": self.stride}

    def to_index(self) -> slice | int:
        if self.kind == "all":
            return slice(None)
        if self.kind == "index":
            assert self.value is not None
            return self.value
        return slice(self.start, self.stop, self.stride)


def selection_shape(full_shape: Sequence[int], selectors: Sequence[AxisSelector]) -> tuple[int, ...]:
    """Return the NumPy result shape for normalized basic indexing."""

    shape = tuple(int(size) for size in full_shape)
    selection = tuple(selectors)
    if len(shape) != len(selection):
        raise ValueError(f"Selection has {len(selection)} axes, but the destination has {len(shape)} dimensions.")

    selected_shape: list[int] = []
    for axis, (size, selector) in enumerate(zip(shape, selection, strict=True)):
        if size < 0:
            raise ValueError("Full shapes must contain only non-negative dimensions.")
        if selector.kind == "all":
            selected_shape.append(size)
            continue
        if selector.kind == "index":
            assert selector.value is not None
            if selector.value >= size:
                raise ValueError(f"Index {selector.value} exceeds axis {axis} with length {size}.")
            continue

        assert selector.start is not None and selector.stop is not None and selector.stride is not None
        if selector.stop > size:
            raise ValueError(f"Slice stop {selector.stop} exceeds axis {axis} with length {size}.")
        extent = len(range(selector.start, selector.stop, selector.stride))
        if extent < 1:
            raise ValueError(f"Slice on axis {axis} selects no elements.")
        selected_shape.append(extent)
    return tuple(selected_shape)


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    """Describe how a BaseData array component follows signal placement."""

    kind: PlacementKind = "direct"
    axis_map: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"direct", "static", "broadcast", "axis_map"}:
            raise ValueError("Unsupported placement binding kind.")
        axis_map = tuple(_require_non_negative_int(axis, "PlacementBinding.axis_map") for axis in self.axis_map)
        if self.kind == "axis_map" and not axis_map:
            raise ValueError("An axis_map placement binding requires at least one mapped axis.")
        if self.kind != "axis_map" and axis_map:
            raise ValueError(f"A {self.kind!r} placement binding cannot define axis_map.")
        if len(set(axis_map)) != len(axis_map):
            raise ValueError("PlacementBinding.axis_map must not contain duplicate axes.")
        object.__setattr__(self, "axis_map", axis_map)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlacementBinding:
        return cls(kind=str(payload.get("kind", "direct")), axis_map=tuple(payload.get("axis_map", ())))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.axis_map:
            payload["axis_map"] = list(self.axis_map)
        return payload


@dataclass(frozen=True, slots=True)
class ChunkArrayLayout:
    """Final storage contract for one array component of a chunked output."""

    component: str
    final_shape: tuple[int, ...]
    dtype: str
    placement_binding: PlacementBinding = field(default_factory=PlacementBinding)
    units: str | None = None
    rank_of_data: int | None = None

    def __post_init__(self) -> None:
        component = str(self.component).strip().strip("/")
        if not component or component.startswith("../") or "/../" in component:
            raise ValueError("ChunkArrayLayout.component must be a non-empty relative path.")
        shape = tuple(_require_non_negative_int(size, "ChunkArrayLayout.final_shape") for size in self.final_shape)
        if any(size == 0 for size in shape):
            raise ValueError("ChunkArrayLayout.final_shape dimensions must be positive.")
        try:
            dtype = np.dtype(self.dtype).str
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ChunkArrayLayout dtype {self.dtype!r}.") from exc
        units = None if self.units is None else str(self.units)
        rank = self.rank_of_data
        if rank is not None:
            rank = _require_non_negative_int(rank, "ChunkArrayLayout.rank_of_data")
            if rank > len(shape):
                raise ValueError("ChunkArrayLayout.rank_of_data cannot exceed the component array rank.")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "final_shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "rank_of_data", rank)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkArrayLayout:
        return cls(
            component=str(payload["component"]),
            final_shape=tuple(payload["final_shape"]),
            dtype=str(payload["dtype"]),
            placement_binding=PlacementBinding.from_dict(payload.get("placement_binding", {})),
            units=payload.get("units"),
            rank_of_data=payload.get("rank_of_data"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "component": self.component,
            "final_shape": list(self.final_shape),
            "dtype": self.dtype,
            "placement_binding": self.placement_binding.to_dict(),
        }
        if self.units is not None:
            payload["units"] = self.units
        if self.rank_of_data is not None:
            payload["rank_of_data"] = self.rank_of_data
        return payload


def _normalise_output_axis_names(
    axis_names: Sequence[str],
    signal: ChunkArrayLayout,
    arrays: tuple[ChunkArrayLayout, ...],
) -> tuple[str, ...]:
    normalized = tuple(str(name).strip() for name in axis_names)
    if normalized and len(normalized) != len(signal.final_shape):
        raise ValueError("ChunkOutputLayout.axis_names must match the signal array rank.")
    if any(not name or "/" in name for name in normalized):
        raise ValueError("ChunkOutputLayout.axis_names must contain non-empty HDF field names or '.'.")
    declared = {array.component.removeprefix("axes/") for array in arrays if array.component.startswith("axes/")}
    referenced = {name for name in normalized if name != "."}
    reserved = referenced & {"signal", "weights", "uncertainties"}
    if reserved:
        raise ValueError(f"ChunkOutputLayout.axis_names contain reserved fields: {sorted(reserved)}.")
    if declared != referenced:
        raise ValueError("ChunkOutputLayout axis components and non-'.' axis_names must identify the same fields.")
    return normalized


def _validate_component_layout(
    array: ChunkArrayLayout,
    signal: ChunkArrayLayout,
    axis_names: tuple[str, ...],
) -> None:
    is_axis = array.component.startswith("axes/")
    if is_axis and (array.units is None or array.rank_of_data is None):
        raise ValueError("Axis array layouts require component-level units and rank_of_data.")
    if not is_axis and (array.units is not None or array.rank_of_data is not None):
        raise ValueError(
            "Component-level units and rank_of_data are reserved for axis array layouts; "
            "signal and uncertainty units/rank come from ChunkOutputLayout."
        )

    binding = array.placement_binding
    if binding.kind in {"direct", "broadcast"} and array.final_shape != signal.final_shape:
        raise ValueError(f"A {binding.kind!r} component must have the same final shape as the signal array.")
    if binding.kind == "axis_map":
        if len(binding.axis_map) != len(array.final_shape):
            raise ValueError("An axis_map must contain one signal axis for each component dimension.")
        if any(axis >= len(signal.final_shape) for axis in binding.axis_map):
            raise ValueError("A component axis_map references an axis outside the signal array rank.")
    if array.component == "weights" and not array.final_shape and binding.kind != "static":
        raise ValueError("Scalar weights must use a static placement binding.")
    if is_axis and binding.kind == "axis_map":
        axis_name = array.component.removeprefix("axes/")
        mapped_axes = {index for index, name in enumerate(axis_names) if name == axis_name}
        if not set(binding.axis_map).issubset(mapped_axes):
            raise ValueError("An axis component axis_map must refer to signal dimensions bearing its name.")


@dataclass(frozen=True, slots=True)
class ChunkOutputLayout:
    """Final storage contract for one BaseData output."""

    output_id: str
    processing_path: str
    destination_path: str
    units: str
    rank_of_data: int
    arrays: tuple[ChunkArrayLayout, ...]
    axis_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        output_id = _require_identifier(self.output_id, "ChunkOutputLayout.output_id")
        processing_path = "/" + str(self.processing_path).strip().strip("/")
        destination_path = str(self.destination_path).strip().strip("/")
        if len([part for part in processing_path.split("/") if part]) != 2:
            raise ValueError("ChunkOutputLayout.processing_path must identify one BaseData root.")
        if len([part for part in destination_path.split("/") if part]) != 2:
            raise ValueError("ChunkOutputLayout.destination_path must contain '<bundle>/<basedata>'.")
        rank = _require_non_negative_int(self.rank_of_data, "ChunkOutputLayout.rank_of_data")
        arrays = tuple(self.arrays)
        components = [array.component for array in arrays]
        if len(set(components)) != len(components):
            raise ValueError("ChunkOutputLayout array components must be unique.")
        signal_arrays = [array for array in arrays if array.component == "signal"]
        if len(signal_arrays) != 1:
            raise ValueError("ChunkOutputLayout requires exactly one signal array.")
        if signal_arrays[0].placement_binding.kind != "direct":
            raise ValueError("The signal array must use direct placement.")
        if rank > len(signal_arrays[0].final_shape):
            raise ValueError("rank_of_data cannot exceed the signal array rank.")
        axis_names = _normalise_output_axis_names(self.axis_names, signal_arrays[0], arrays)
        for array in arrays:
            _validate_component_layout(array, signal_arrays[0], axis_names)
        object.__setattr__(self, "output_id", output_id)
        object.__setattr__(self, "processing_path", processing_path)
        object.__setattr__(self, "destination_path", destination_path)
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "rank_of_data", rank)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "axis_names", axis_names)

    @property
    def signal(self) -> ChunkArrayLayout:
        return next(array for array in self.arrays if array.component == "signal")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkOutputLayout:
        return cls(
            output_id=str(payload["output_id"]),
            processing_path=str(payload["processing_path"]),
            destination_path=str(payload["destination_path"]),
            units=str(payload["units"]),
            rank_of_data=payload["rank_of_data"],
            arrays=tuple(ChunkArrayLayout.from_dict(item) for item in payload["arrays"]),
            axis_names=tuple(payload.get("axis_names", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "output_id": self.output_id,
            "processing_path": self.processing_path,
            "destination_path": self.destination_path,
            "units": self.units,
            "rank_of_data": self.rank_of_data,
            "arrays": [array.to_dict() for array in self.arrays],
        }
        if self.axis_names:
            payload["axis_names"] = list(self.axis_names)
        return payload


@dataclass(frozen=True, slots=True)
class ChunkPlacement:
    """Destination selection and expected signal shape for one output."""

    output_id: str
    destination_selection: tuple[AxisSelector, ...]
    expected_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_id", _require_identifier(self.output_id, "ChunkPlacement.output_id"))
        object.__setattr__(self, "destination_selection", tuple(self.destination_selection))
        shape = tuple(_require_non_negative_int(size, "ChunkPlacement.expected_shape") for size in self.expected_shape)
        if any(size == 0 for size in shape):
            raise ValueError("ChunkPlacement.expected_shape dimensions must be positive.")
        object.__setattr__(self, "expected_shape", shape)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkPlacement:
        return cls(
            output_id=str(payload["output_id"]),
            destination_selection=tuple(AxisSelector.from_dict(item) for item in payload["destination_selection"]),
            expected_shape=tuple(payload["expected_shape"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "destination_selection": [selector.to_dict() for selector in self.destination_selection],
            "expected_shape": list(self.expected_shape),
        }


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Immutable plan-wide contract shared by every chunk."""

    schema_version: str
    plan_id: str
    total_chunks: int
    expected_chunk_ids: tuple[str, ...]
    outputs: tuple[ChunkOutputLayout, ...]
    driver: Mapping[str, Any] = field(default_factory=dict)
    batch_axes: tuple[int, ...] = ()
    data_axes: tuple[int, ...] = ()
    axis_rules: tuple[Mapping[str, Any], ...] = ()
    bindings: tuple[Mapping[str, Any], ...] = ()
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("ChunkPlan.schema_version must be non-empty.")
        plan_id = _require_identifier(self.plan_id, "ChunkPlan.plan_id")
        total_chunks = _require_non_negative_int(self.total_chunks, "ChunkPlan.total_chunks")
        if total_chunks < 1:
            raise ValueError("ChunkPlan.total_chunks must be positive.")
        chunk_ids = tuple(_require_identifier(item, "ChunkPlan.expected_chunk_ids") for item in self.expected_chunk_ids)
        if len(chunk_ids) != total_chunks:
            raise ValueError("ChunkPlan.expected_chunk_ids length must equal total_chunks.")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("ChunkPlan.expected_chunk_ids must be unique.")
        outputs = tuple(self.outputs)
        if not outputs:
            raise ValueError("ChunkPlan.outputs must not be empty.")
        output_ids = [output.output_id for output in outputs]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("ChunkPlan output ids must be unique.")
        batch_axes = tuple(_require_non_negative_int(axis, "ChunkPlan.batch_axes") for axis in self.batch_axes)
        data_axes = tuple(_require_non_negative_int(axis, "ChunkPlan.data_axes") for axis in self.data_axes)
        if set(batch_axes) & set(data_axes):
            raise ValueError("ChunkPlan batch_axes and data_axes must not overlap.")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "total_chunks", total_chunks)
        object.__setattr__(self, "expected_chunk_ids", chunk_ids)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "driver", _freeze_json(self.driver, "ChunkPlan.driver"))
        object.__setattr__(self, "batch_axes", batch_axes)
        object.__setattr__(self, "data_axes", data_axes)
        object.__setattr__(self, "axis_rules", _freeze_json(self.axis_rules, "ChunkPlan.axis_rules"))
        object.__setattr__(self, "bindings", _freeze_json(self.bindings, "ChunkPlan.bindings"))
        digest = sha256(_canonical_json(self.to_dict(include_hash=False)).encode("utf-8")).hexdigest()
        object.__setattr__(self, "plan_hash", f"sha256:{digest}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkPlan:
        plan = cls(
            schema_version=str(payload["schema_version"]),
            plan_id=str(payload["plan_id"]),
            total_chunks=payload["total_chunks"],
            expected_chunk_ids=tuple(payload["expected_chunk_ids"]),
            outputs=tuple(ChunkOutputLayout.from_dict(item) for item in payload["outputs"]),
            driver=payload.get("driver", {}),
            batch_axes=tuple(payload.get("batch_axes", ())),
            data_axes=tuple(payload.get("data_axes", ())),
            axis_rules=tuple(payload.get("axis_rules", ())),
            bindings=tuple(payload.get("bindings", ())),
        )
        expected_hash = payload.get("plan_hash")
        if expected_hash is not None and str(expected_hash) != plan.plan_hash:
            raise ValueError("Serialized ChunkPlan plan_hash does not match its canonical content.")
        return plan

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "driver": _json_ready(self.driver),
            "batch_axes": list(self.batch_axes),
            "data_axes": list(self.data_axes),
            "axis_rules": _json_ready(self.axis_rules),
            "bindings": _json_ready(self.bindings),
            "total_chunks": self.total_chunks,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "outputs": [output.to_dict() for output in self.outputs],
        }
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def output(self, output_id: str) -> ChunkOutputLayout:
        for output in self.outputs:
            if output.output_id == output_id:
                return output
        raise KeyError(f"ChunkPlan has no output {output_id!r}.")


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """Immutable description of one resolved unit of chunked work."""

    schema_version: str
    plan_id: str
    plan_hash: str
    chunk_id: str
    ordinal: int
    grid_index: tuple[int, ...]
    source_selection: tuple[AxisSelector, ...]
    expected_input_shape: tuple[int, ...]
    placements: tuple[ChunkPlacement, ...]

    def __post_init__(self) -> None:
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("ChunkSpec.schema_version must be non-empty.")
        plan_hash = str(self.plan_hash).strip()
        if not plan_hash.startswith("sha256:"):
            raise ValueError("ChunkSpec.plan_hash must use the 'sha256:' prefix.")
        ordinal = _require_non_negative_int(self.ordinal, "ChunkSpec.ordinal")
        grid_index = tuple(_require_non_negative_int(index, "ChunkSpec.grid_index") for index in self.grid_index)
        input_shape = tuple(
            _require_non_negative_int(size, "ChunkSpec.expected_input_shape") for size in self.expected_input_shape
        )
        if any(size == 0 for size in input_shape):
            raise ValueError("ChunkSpec.expected_input_shape dimensions must be positive.")
        placements = tuple(self.placements)
        if not placements:
            raise ValueError("ChunkSpec.placements must not be empty.")
        output_ids = [placement.output_id for placement in placements]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("ChunkSpec placement output ids must be unique.")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "plan_id", _require_identifier(self.plan_id, "ChunkSpec.plan_id"))
        object.__setattr__(self, "plan_hash", plan_hash)
        object.__setattr__(self, "chunk_id", _require_identifier(self.chunk_id, "ChunkSpec.chunk_id"))
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "grid_index", grid_index)
        object.__setattr__(self, "source_selection", tuple(self.source_selection))
        object.__setattr__(self, "expected_input_shape", input_shape)
        object.__setattr__(self, "placements", placements)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkSpec:
        return cls(
            schema_version=str(payload["schema_version"]),
            plan_id=str(payload["plan_id"]),
            plan_hash=str(payload["plan_hash"]),
            chunk_id=str(payload["chunk_id"]),
            ordinal=payload["ordinal"],
            grid_index=tuple(payload.get("grid_index", ())),
            source_selection=tuple(AxisSelector.from_dict(item) for item in payload["source_selection"]),
            expected_input_shape=tuple(payload["expected_input_shape"]),
            placements=tuple(ChunkPlacement.from_dict(item) for item in payload["placements"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "chunk_id": self.chunk_id,
            "ordinal": self.ordinal,
            "grid_index": list(self.grid_index),
            "source_selection": [selector.to_dict() for selector in self.source_selection],
            "expected_input_shape": list(self.expected_input_shape),
            "placements": [placement.to_dict() for placement in self.placements],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def validate_for_plan(self, plan: ChunkPlan) -> None:
        if self.schema_version != plan.schema_version:
            raise ValueError("ChunkSpec schema_version does not match ChunkPlan.")
        if self.plan_id != plan.plan_id or self.plan_hash != plan.plan_hash:
            raise ValueError("ChunkSpec plan identity does not match ChunkPlan.")
        if self.chunk_id not in plan.expected_chunk_ids:
            raise ValueError(f"ChunkSpec chunk_id {self.chunk_id!r} is not expected by the plan.")
        if self.ordinal >= plan.total_chunks:
            raise ValueError("ChunkSpec ordinal exceeds the plan's chunk count.")
        if plan.expected_chunk_ids[self.ordinal] != self.chunk_id:
            raise ValueError("ChunkSpec ordinal does not match its position in ChunkPlan.expected_chunk_ids.")
        driver_shape = plan.driver.get("full_shape")
        if driver_shape is not None:
            derived_input_shape = selection_shape(tuple(driver_shape), self.source_selection)
            if derived_input_shape != self.expected_input_shape:
                raise ValueError(
                    f"ChunkSpec expected input shape {self.expected_input_shape}, "
                    f"but its source selection resolves to {derived_input_shape}."
                )
        expected_outputs = {output.output_id for output in plan.outputs}
        actual_outputs = {placement.output_id for placement in self.placements}
        if actual_outputs != expected_outputs:
            raise ValueError("ChunkSpec placements must cover every ChunkPlan output exactly once.")
        for placement in self.placements:
            output = plan.output(placement.output_id)
            derived_shape = selection_shape(output.signal.final_shape, placement.destination_selection)
            if derived_shape != placement.expected_shape:
                raise ValueError(
                    f"Chunk placement {placement.output_id!r} expected shape {placement.expected_shape}, "
                    f"but its destination selection resolves to {derived_shape}."
                )


@dataclass(frozen=True, slots=True)
class ChunkWriteResult:
    """Small serializable summary returned by chunk sink lifecycle operations."""

    status: str
    plan_id: str
    plan_hash: str
    expected_chunks: int
    completed_chunks: int
    chunk_id: str | None = None
    resource_location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "expected_chunks": self.expected_chunks,
            "completed_chunks": self.completed_chunks,
            "chunk_id": self.chunk_id,
            "resource_location": self.resource_location,
        }
