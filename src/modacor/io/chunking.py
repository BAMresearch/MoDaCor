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
    "ChunkAxisRule",
    "ChunkArrayLayout",
    "ChunkInputPlan",
    "ChunkOutputLayout",
    "ChunkPlacement",
    "ChunkPlan",
    "ChunkSourceBinding",
    "ChunkSpec",
    "ProvisionalChunkOutput",
    "ProvisionalChunkPlan",
    "ProvisionalChunkSpec",
    "ChunkOutputStatus",
    "ChunkWriteResult",
    "PlacementBinding",
    "UnsupportedSinkCapability",
    "selection_shape",
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
SelectorKind = Literal["all", "index", "slice"]
PlacementKind = Literal["direct", "static", "broadcast", "axis_map"]
SourceBindingRole = Literal["aligned", "static", "explicit"]


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
class ChunkSourceBinding:
    """Project one chunk driver selection onto one registered source dataset.

    ``aligned`` applies the driver selection unchanged and therefore requires
    the source dataset to have the same rank as the driver. ``explicit`` maps
    every source axis to a driver axis, with ``None`` selecting the complete
    source axis. ``static`` records provenance but applies no slice.
    """

    source_ref: str
    data_key: str
    role: SourceBindingRole
    axis_map: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        source_ref = _require_identifier(self.source_ref, "ChunkSourceBinding.source_ref")
        data_key = "/" + str(self.data_key).strip().strip("/")
        if data_key == "/" or "/../" in data_key or data_key.endswith("/.."):
            raise ValueError("ChunkSourceBinding.data_key must identify a source dataset.")
        if self.role not in {"aligned", "static", "explicit"}:
            raise ValueError("ChunkSourceBinding.role must be 'aligned', 'static', or 'explicit'.")

        axis_map: list[int | None] = []
        for axis in self.axis_map:
            if axis is None:
                axis_map.append(None)
            else:
                axis_map.append(_require_non_negative_int(axis, "ChunkSourceBinding.axis_map"))
        mapped_axes = [axis for axis in axis_map if axis is not None]
        if len(set(mapped_axes)) != len(mapped_axes):
            raise ValueError("ChunkSourceBinding.axis_map must not map multiple source axes to one driver axis.")
        if self.role == "explicit" and not axis_map:
            raise ValueError("An explicit ChunkSourceBinding requires axis_map.")
        if self.role != "explicit" and axis_map:
            raise ValueError(f"A {self.role!r} ChunkSourceBinding cannot define axis_map.")

        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "data_key", data_key)
        object.__setattr__(self, "axis_map", tuple(axis_map))

    @property
    def data_reference(self) -> str:
        return f"{self.source_ref}::{self.data_key}"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkSourceBinding:
        return cls(
            source_ref=str(payload["source_ref"]),
            data_key=str(payload["data_key"]),
            role=str(payload["role"]),  # type: ignore[arg-type]
            axis_map=tuple(payload.get("axis_map", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_ref": self.source_ref,
            "data_key": self.data_key,
            "role": self.role,
        }
        if self.axis_map:
            payload["axis_map"] = list(self.axis_map)
        return payload

    def resolve_selection(
        self,
        driver_selection: Sequence[AxisSelector],
        source_shape: Sequence[int],
    ) -> tuple[AxisSelector, ...] | None:
        """Return and validate the effective selector for this source dataset."""

        if self.role == "static":
            return None

        driver_selection = tuple(driver_selection)
        source_shape = tuple(int(size) for size in source_shape)
        if self.role == "aligned":
            if len(source_shape) != len(driver_selection):
                raise ValueError(
                    f"Aligned source {self.data_reference!r} has rank {len(source_shape)}, "
                    f"but the chunk driver has rank {len(driver_selection)}."
                )
            selection = driver_selection
        else:
            if len(self.axis_map) != len(source_shape):
                raise ValueError(
                    f"Explicit source {self.data_reference!r} axis_map has {len(self.axis_map)} entries, "
                    f"but the source dataset has rank {len(source_shape)}."
                )
            if any(axis is not None and axis >= len(driver_selection) for axis in self.axis_map):
                raise ValueError(f"Explicit source {self.data_reference!r} maps outside the chunk driver rank.")
            selection = tuple(
                AxisSelector.all() if driver_axis is None else driver_selection[driver_axis]
                for driver_axis in self.axis_map
            )

        selection_shape(source_shape, selection)
        return selection


@dataclass(frozen=True, slots=True)
class ChunkAxisRule:
    """Partition one source axis into chunks of selected elements."""

    axis: int
    chunk_size: int
    start: int | None = None
    stop: int | None = None
    stride: int = 1

    def __post_init__(self) -> None:
        axis = _require_non_negative_int(self.axis, "ChunkAxisRule.axis")
        chunk_size = _require_non_negative_int(self.chunk_size, "ChunkAxisRule.chunk_size")
        if chunk_size < 1:
            raise ValueError("ChunkAxisRule.chunk_size must be positive.")
        start = None if self.start is None else _require_non_negative_int(self.start, "ChunkAxisRule.start")
        stop = None if self.stop is None else _require_non_negative_int(self.stop, "ChunkAxisRule.stop")
        if isinstance(self.stride, bool) or not isinstance(self.stride, Integral):
            raise TypeError("ChunkAxisRule.stride must be an integer.")
        stride = int(self.stride)
        if stride < 1:
            raise ValueError("ChunkAxisRule.stride must be positive.")
        if start is not None and stop is not None and stop <= start:
            raise ValueError("ChunkAxisRule.stop must be greater than start.")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "stride", stride)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkAxisRule:
        return cls(
            axis=payload["axis"],
            chunk_size=payload["chunk_size"],
            start=payload.get("start"),
            stop=payload.get("stop"),
            stride=payload.get("stride", 1),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "axis": self.axis,
            "chunk_size": self.chunk_size,
            "stride": self.stride,
        }
        if self.start is not None:
            payload["start"] = self.start
        if self.stop is not None:
            payload["stop"] = self.stop
        return payload


@dataclass(frozen=True, slots=True)
class ProvisionalChunkOutput:
    """Output identity whose numerical and metadata schema comes from a pilot."""

    output_id: str
    processing_path: str
    destination_path: str

    def __post_init__(self) -> None:
        output_id = _require_identifier(self.output_id, "ProvisionalChunkOutput.output_id")
        processing_path = "/" + str(self.processing_path).strip().strip("/")
        destination_path = str(self.destination_path).strip().strip("/")
        if len([part for part in processing_path.split("/") if part]) != 2:
            raise ValueError("ProvisionalChunkOutput.processing_path must identify one BaseData root.")
        if len([part for part in destination_path.split("/") if part]) != 2:
            raise ValueError("ProvisionalChunkOutput.destination_path must contain '<bundle>/<basedata>'.")
        object.__setattr__(self, "output_id", output_id)
        object.__setattr__(self, "processing_path", processing_path)
        object.__setattr__(self, "destination_path", destination_path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvisionalChunkOutput:
        return cls(
            output_id=str(payload["output_id"]),
            processing_path=str(payload["processing_path"]),
            destination_path=str(payload["destination_path"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "processing_path": self.processing_path,
            "destination_path": self.destination_path,
        }


@dataclass(frozen=True, slots=True)
class ProvisionalChunkPlan:
    """Immutable request for server-resolved input extents and output schema.

    The constrained provisional workflow chunks only batch dimensions. Data
    dimensions are the trailing ``driver.rank_of_data`` axes and are always
    read in full.
    """

    schema_version: str
    plan_id: str
    driver: Mapping[str, Any]
    axis_rules: tuple[ChunkAxisRule, ...]
    outputs: tuple[ProvisionalChunkOutput, ...]
    source_bindings: tuple[ChunkSourceBinding, ...] = ()
    bindings: tuple[Mapping[str, Any], ...] = ()
    provisional_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("ProvisionalChunkPlan.schema_version must be non-empty.")
        plan_id = _require_identifier(self.plan_id, "ProvisionalChunkPlan.plan_id")
        driver = _freeze_json(self.driver, "ProvisionalChunkPlan.driver")
        source = str(driver.get("source", "")).strip()
        source_ref, separator, data_key = source.partition("::")
        if not separator or not source_ref.strip() or not data_key.strip():
            raise ValueError("ProvisionalChunkPlan.driver.source must use '<source_ref>::<data_key>'.")
        rank_of_data = _require_non_negative_int(
            driver.get("rank_of_data"),
            "ProvisionalChunkPlan.driver.rank_of_data",
        )
        if rank_of_data > 3:
            raise ValueError("ProvisionalChunkPlan.driver.rank_of_data cannot exceed 3.")
        if "full_shape" in driver:
            full_shape = tuple(
                _require_non_negative_int(size, "ProvisionalChunkPlan.driver.full_shape")
                for size in driver["full_shape"]
            )
            if not full_shape or any(size == 0 for size in full_shape):
                raise ValueError("ProvisionalChunkPlan.driver.full_shape must contain positive dimensions.")
            if rank_of_data > len(full_shape):
                raise ValueError("driver.rank_of_data cannot exceed driver.full_shape rank.")

        axis_rules = tuple(
            rule if isinstance(rule, ChunkAxisRule) else ChunkAxisRule.from_dict(rule) for rule in self.axis_rules
        )
        if not axis_rules:
            raise ValueError("ProvisionalChunkPlan.axis_rules must contain at least one batch-axis rule.")
        axes = [rule.axis for rule in axis_rules]
        if len(set(axes)) != len(axes):
            raise ValueError("ProvisionalChunkPlan.axis_rules must identify unique axes.")

        outputs = tuple(
            output if isinstance(output, ProvisionalChunkOutput) else ProvisionalChunkOutput.from_dict(output)
            for output in self.outputs
        )
        if not outputs:
            raise ValueError("ProvisionalChunkPlan.outputs must not be empty.")
        output_ids = [output.output_id for output in outputs]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("ProvisionalChunkPlan output ids must be unique.")

        source_bindings = tuple(
            binding if isinstance(binding, ChunkSourceBinding) else ChunkSourceBinding.from_dict(binding)
            for binding in self.source_bindings
        )
        normalized_driver = f"{source_ref.strip()}::/{data_key.strip().strip('/')}"
        matches = [binding for binding in source_bindings if binding.data_reference == normalized_driver]
        if source_bindings and (len(matches) != 1 or matches[0].role != "aligned"):
            raise ValueError("ProvisionalChunkPlan driver.source requires exactly one aligned source binding.")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "driver", driver)
        object.__setattr__(self, "axis_rules", axis_rules)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "source_bindings", source_bindings)
        object.__setattr__(self, "bindings", _freeze_json(self.bindings, "ProvisionalChunkPlan.bindings"))
        digest = sha256(_canonical_json(self.to_dict(include_hash=False)).encode("utf-8")).hexdigest()
        object.__setattr__(self, "provisional_hash", f"sha256:{digest}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvisionalChunkPlan:
        plan = cls(
            schema_version=str(payload["schema_version"]),
            plan_id=str(payload["plan_id"]),
            driver=payload["driver"],
            axis_rules=tuple(ChunkAxisRule.from_dict(item) for item in payload["axis_rules"]),
            outputs=tuple(ProvisionalChunkOutput.from_dict(item) for item in payload["outputs"]),
            source_bindings=tuple(ChunkSourceBinding.from_dict(item) for item in payload.get("source_bindings", ())),
            bindings=tuple(payload.get("bindings", ())),
        )
        expected_hash = payload.get("provisional_hash")
        if expected_hash is not None and str(expected_hash) != plan.provisional_hash:
            raise ValueError("Serialized provisional_hash does not match canonical content.")
        return plan

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "driver": _json_ready(self.driver),
            "axis_rules": [rule.to_dict() for rule in self.axis_rules],
            "outputs": [output.to_dict() for output in self.outputs],
            "source_bindings": [binding.to_dict() for binding in self.source_bindings],
            "bindings": _json_ready(self.bindings),
        }
        if include_hash:
            payload["provisional_hash"] = self.provisional_hash
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProvisionalChunkSpec:
    """One source-side work item before processed output layouts are known."""

    schema_version: str
    plan_id: str
    provisional_hash: str
    chunk_id: str
    ordinal: int
    grid_index: tuple[int, ...]
    source_selection: tuple[AxisSelector, ...]
    expected_input_shape: tuple[int, ...]
    destination_batch_selection: tuple[AxisSelector, ...]
    expected_batch_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("ProvisionalChunkSpec.schema_version must be non-empty.")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "plan_id", _require_identifier(self.plan_id, "ProvisionalChunkSpec.plan_id"))
        provisional_hash = str(self.provisional_hash).strip()
        if not provisional_hash.startswith("sha256:"):
            raise ValueError("ProvisionalChunkSpec.provisional_hash must use the 'sha256:' prefix.")
        object.__setattr__(self, "provisional_hash", provisional_hash)
        object.__setattr__(self, "chunk_id", _require_identifier(self.chunk_id, "ProvisionalChunkSpec.chunk_id"))
        object.__setattr__(self, "ordinal", _require_non_negative_int(self.ordinal, "ProvisionalChunkSpec.ordinal"))
        object.__setattr__(
            self,
            "grid_index",
            tuple(_require_non_negative_int(value, "ProvisionalChunkSpec.grid_index") for value in self.grid_index),
        )
        object.__setattr__(self, "source_selection", tuple(self.source_selection))
        input_shape = tuple(
            _require_non_negative_int(value, "ProvisionalChunkSpec.expected_input_shape")
            for value in self.expected_input_shape
        )
        batch_shape = tuple(
            _require_non_negative_int(value, "ProvisionalChunkSpec.expected_batch_shape")
            for value in self.expected_batch_shape
        )
        if any(value == 0 for value in (*input_shape, *batch_shape)):
            raise ValueError("Provisional chunk shapes must contain positive dimensions.")
        object.__setattr__(self, "expected_input_shape", input_shape)
        object.__setattr__(self, "destination_batch_selection", tuple(self.destination_batch_selection))
        object.__setattr__(self, "expected_batch_shape", batch_shape)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvisionalChunkSpec:
        return cls(
            schema_version=str(payload["schema_version"]),
            plan_id=str(payload["plan_id"]),
            provisional_hash=str(payload["provisional_hash"]),
            chunk_id=str(payload["chunk_id"]),
            ordinal=payload["ordinal"],
            grid_index=tuple(payload.get("grid_index", ())),
            source_selection=tuple(AxisSelector.from_dict(item) for item in payload["source_selection"]),
            expected_input_shape=tuple(payload["expected_input_shape"]),
            destination_batch_selection=tuple(
                AxisSelector.from_dict(item) for item in payload["destination_batch_selection"]
            ),
            expected_batch_shape=tuple(payload["expected_batch_shape"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "provisional_hash": self.provisional_hash,
            "chunk_id": self.chunk_id,
            "ordinal": self.ordinal,
            "grid_index": list(self.grid_index),
            "source_selection": [selector.to_dict() for selector in self.source_selection],
            "expected_input_shape": list(self.expected_input_shape),
            "destination_batch_selection": [selector.to_dict() for selector in self.destination_batch_selection],
            "expected_batch_shape": list(self.expected_batch_shape),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def identity_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "provisional_hash": self.provisional_hash,
            "chunk_id": self.chunk_id,
            "ordinal": self.ordinal,
            "provisional": True,
        }


@dataclass(frozen=True, slots=True)
class ChunkInputPlan:
    """Resolved driver extent and source-side work for a provisional plan."""

    schema_version: str
    plan_id: str
    provisional_hash: str
    full_shape: tuple[int, ...]
    dtype: str | None
    batch_axes: tuple[int, ...]
    data_axes: tuple[int, ...]
    final_batch_shape: tuple[int, ...]
    chunks: tuple[ProvisionalChunkSpec, ...]
    resolution_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("ChunkInputPlan.schema_version must be non-empty.")
        plan_id = _require_identifier(self.plan_id, "ChunkInputPlan.plan_id")
        provisional_hash = str(self.provisional_hash).strip()
        if not provisional_hash.startswith("sha256:"):
            raise ValueError("ChunkInputPlan.provisional_hash must use the 'sha256:' prefix.")
        full_shape = tuple(_require_non_negative_int(value, "ChunkInputPlan.full_shape") for value in self.full_shape)
        if not full_shape or any(value == 0 for value in full_shape):
            raise ValueError("ChunkInputPlan.full_shape must contain positive dimensions.")
        batch_axes = tuple(_require_non_negative_int(value, "ChunkInputPlan.batch_axes") for value in self.batch_axes)
        data_axes = tuple(_require_non_negative_int(value, "ChunkInputPlan.data_axes") for value in self.data_axes)
        if batch_axes + data_axes != tuple(range(len(full_shape))):
            raise ValueError("ChunkInputPlan supports leading batch axes followed by trailing data axes.")
        final_batch_shape = tuple(
            _require_non_negative_int(value, "ChunkInputPlan.final_batch_shape") for value in self.final_batch_shape
        )
        chunks = tuple(self.chunks)
        if not chunks:
            raise ValueError("ChunkInputPlan.chunks must not be empty.")
        for ordinal, chunk in enumerate(chunks):
            if (
                chunk.schema_version != schema_version
                or chunk.plan_id != plan_id
                or chunk.provisional_hash != provisional_hash
                or chunk.ordinal != ordinal
            ):
                raise ValueError("ChunkInputPlan contains a chunk with a conflicting provisional identity.")
            if selection_shape(full_shape, chunk.source_selection) != chunk.expected_input_shape:
                raise ValueError("ChunkInputPlan contains a chunk with an invalid source selection.")
            if selection_shape(final_batch_shape, chunk.destination_batch_selection) != chunk.expected_batch_shape:
                raise ValueError("ChunkInputPlan contains a chunk with an invalid batch destination selection.")
        dtype = None if self.dtype is None else np.dtype(self.dtype).str
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "provisional_hash", provisional_hash)
        object.__setattr__(self, "full_shape", full_shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "batch_axes", batch_axes)
        object.__setattr__(self, "data_axes", data_axes)
        object.__setattr__(self, "final_batch_shape", final_batch_shape)
        object.__setattr__(self, "chunks", chunks)
        digest = sha256(_canonical_json(self.to_dict(include_hash=False)).encode("utf-8")).hexdigest()
        object.__setattr__(self, "resolution_hash", f"sha256:{digest}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkInputPlan:
        result = cls(
            schema_version=str(payload["schema_version"]),
            plan_id=str(payload["plan_id"]),
            provisional_hash=str(payload["provisional_hash"]),
            full_shape=tuple(payload["full_shape"]),
            dtype=payload.get("dtype"),
            batch_axes=tuple(payload["batch_axes"]),
            data_axes=tuple(payload["data_axes"]),
            final_batch_shape=tuple(payload["final_batch_shape"]),
            chunks=tuple(ProvisionalChunkSpec.from_dict(item) for item in payload["chunks"]),
        )
        expected_hash = payload.get("resolution_hash")
        if expected_hash is not None and str(expected_hash) != result.resolution_hash:
            raise ValueError("Serialized resolution_hash does not match canonical content.")
        return result

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "provisional_hash": self.provisional_hash,
            "full_shape": list(self.full_shape),
            "dtype": self.dtype,
            "batch_axes": list(self.batch_axes),
            "data_axes": list(self.data_axes),
            "final_batch_shape": list(self.final_batch_shape),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        if include_hash:
            payload["resolution_hash"] = self.resolution_hash
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def chunk(self, chunk_id: str) -> ProvisionalChunkSpec:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"ChunkInputPlan has no chunk {chunk_id!r}.")


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
    source_bindings: tuple[ChunkSourceBinding, ...] = ()
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
        source_bindings = tuple(
            binding if isinstance(binding, ChunkSourceBinding) else ChunkSourceBinding.from_dict(binding)
            for binding in self.source_bindings
        )
        data_references = [binding.data_reference for binding in source_bindings]
        if len(set(data_references)) != len(data_references):
            raise ValueError("ChunkPlan source_bindings must identify unique source datasets.")
        if source_bindings:
            driver_reference = str(self.driver.get("source", "")).strip()
            driver_ref, separator, driver_key = driver_reference.partition("::")
            if not separator or not driver_ref.strip() or not driver_key.strip():
                raise ValueError(
                    "A ChunkPlan with source_bindings requires driver.source in '<source_ref>::<data_key>' form."
                )
            normalized_driver = f"{driver_ref.strip()}::/{driver_key.strip().strip('/')}"
            matching_driver_bindings = [
                binding for binding in source_bindings if binding.data_reference == normalized_driver
            ]
            if len(matching_driver_bindings) != 1 or matching_driver_bindings[0].role != "aligned":
                raise ValueError("ChunkPlan driver.source requires exactly one aligned source binding.")
            if "full_shape" not in self.driver:
                raise ValueError("A ChunkPlan with source_bindings requires driver.full_shape.")

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
        object.__setattr__(self, "source_bindings", source_bindings)
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
            source_bindings=tuple(ChunkSourceBinding.from_dict(item) for item in payload.get("source_bindings", ())),
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
        if self.source_bindings:
            payload["source_bindings"] = [binding.to_dict() for binding in self.source_bindings]
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

    def identity_dict(self) -> dict[str, Any]:
        """Return the compact execution identity suitable for logs and traces."""

        return {
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "chunk_id": self.chunk_id,
            "ordinal": self.ordinal,
        }

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


@dataclass(frozen=True, slots=True)
class ChunkOutputStatus:
    """Serializable authoritative status of a chunked output manifest."""

    status: str
    plan_id: str
    plan_hash: str
    expected_chunks: int
    completed_chunks: int
    writing_chunks: int
    failed_chunks: int
    missing_chunks: int
    chunks: tuple[Mapping[str, Any], ...] = ()
    resource_location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "expected_chunks": self.expected_chunks,
            "completed_chunks": self.completed_chunks,
            "writing_chunks": self.writing_chunks,
            "failed_chunks": self.failed_chunks,
            "missing_chunks": self.missing_chunks,
            "chunks": [_json_ready(item) for item in self.chunks],
            "resource_location": self.resource_location,
        }
