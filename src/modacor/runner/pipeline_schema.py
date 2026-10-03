# SPDX-License-Identifier: BSD-3-Clause
"""Compile compact pipeline documents into the ordinary step schema."""

from __future__ import annotations

__all__ = [
    "ExpandedPipelineDocument",
    "PipelineSchemaError",
    "StepOrigin",
    "expand_pipeline_document",
    "expand_pipeline_yaml",
]

import re
from copy import deepcopy
from typing import Any, Mapping

import yaml
from attrs import define, field

_ID_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_PARAMETER_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class PipelineSchemaError(ValueError):
    """Raised when a compact pipeline document cannot be expanded safely."""


@define(frozen=True, slots=True)
class StepOrigin:
    """Authored block location for one expanded execution step."""

    block: str
    item: str
    local_step: str
    block_index: int
    item_index: int
    local_step_index: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "block": self.block,
            "item": self.item,
            "local_step": self.local_step,
            "block_index": self.block_index,
            "item_index": self.item_index,
            "local_step_index": self.local_step_index,
        }


@define(frozen=True, slots=True)
class ExpandedPipelineDocument:
    """Normalized authored document and its executable expansion."""

    authored_spec: dict[str, Any] = field(converter=deepcopy)
    expanded_spec: dict[str, Any] = field(converter=deepcopy)
    origins: dict[str, StepOrigin] = field(converter=dict)

    def expanded_yaml(self) -> str:
        return yaml.safe_dump(self.expanded_spec, sort_keys=False)


def _as_mapping(value: Any, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise PipelineSchemaError(f"{path} must be a mapping, got {type(value).__name__}.")
    return value


def _normalize_id(value: Any, path: str, *, segment: bool = False) -> str:
    identifier = str(value)
    if not identifier:
        raise PipelineSchemaError(f"{path} must be a non-empty identifier.")
    if segment and not _ID_SEGMENT_PATTERN.fullmatch(identifier):
        raise PipelineSchemaError(
            f"{path} must contain only letters, digits, underscores, or hyphens; got {identifier!r}."
        )
    return identifier


def _normalize_requires(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PipelineSchemaError(f"{path} must be a list of step ids.")
    return [str(item) for item in value]


def _normalize_steps(
    value: Any,
    path: str,
    *,
    segment_ids: bool,
    normalize_requires: bool = True,
) -> dict[str, dict[str, Any]]:
    steps = _as_mapping(value, path)
    normalized: dict[str, dict[str, Any]] = {}
    for raw_step_id, raw_step in steps.items():
        step_id = _normalize_id(raw_step_id, f"{path} key", segment=segment_ids)
        if step_id in normalized:
            raise PipelineSchemaError(f"{path} contains duplicate normalized step id {step_id!r}.")
        step = dict(_as_mapping(raw_step, f"{path}.{step_id}"))
        if normalize_requires and "requires_steps" in step:
            step["requires_steps"] = _normalize_requires(
                step["requires_steps"],
                f"{path}.{step_id}.requires_steps",
            )
        normalized[step_id] = deepcopy(step)
    return normalized


def _substitute(value: Any, parameters: Mapping[str, Any], path: str) -> Any:
    if isinstance(value, str):
        match = _PARAMETER_PATTERN.fullmatch(value)
        if match is not None:
            parameter = match.group(1)
            if parameter not in parameters:
                raise PipelineSchemaError(f"{path} refers to missing parameter {parameter!r}.")
            return deepcopy(parameters[parameter])
        if "${" in value:
            raise PipelineSchemaError(
                f"{path} uses partial parameter interpolation; version 1 supports whole values only."
            )
        return value
    if isinstance(value, list):
        return [_substitute(item, parameters, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_substitute(item, parameters, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, Mapping):
        return {key: _substitute(item, parameters, f"{path}.{key}") for key, item in value.items()}
    return deepcopy(value)


def _normalize_block(
    raw_block_id: Any,
    raw_block: Any,
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    block_id = _normalize_id(raw_block_id, "step_blocks key", segment=True)
    block = dict(_as_mapping(raw_block, f"step_blocks.{block_id}"))
    unknown = set(block) - {"for_each", "steps"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise PipelineSchemaError(f"step_blocks.{block_id} contains unsupported field(s): {names}.")
    if "for_each" not in block or "steps" not in block:
        raise PipelineSchemaError(f"step_blocks.{block_id} requires both 'for_each' and 'steps'.")

    raw_items = _as_mapping(block["for_each"], f"step_blocks.{block_id}.for_each")
    if not raw_items:
        raise PipelineSchemaError(f"step_blocks.{block_id}.for_each must not be empty.")
    items: dict[str, dict[str, Any]] = {}
    for raw_item_id, raw_parameters in raw_items.items():
        item_id = _normalize_id(raw_item_id, f"step_blocks.{block_id}.for_each key", segment=True)
        if item_id in items:
            raise PipelineSchemaError(f"step_blocks.{block_id}.for_each contains duplicate item {item_id!r}.")
        parameters = _as_mapping(raw_parameters, f"step_blocks.{block_id}.for_each.{item_id}")
        if not all(isinstance(name, str) for name in parameters):
            raise PipelineSchemaError(f"step_blocks.{block_id}.for_each.{item_id} parameter names must be strings.")
        items[item_id] = deepcopy(dict(parameters))

    templates = _normalize_steps(
        block["steps"],
        f"step_blocks.{block_id}.steps",
        segment_ids=True,
        normalize_requires=False,
    )
    if not templates:
        raise PipelineSchemaError(f"step_blocks.{block_id}.steps must not be empty.")
    for local_step_id, template in templates.items():
        if "step_id" in template:
            raise PipelineSchemaError(
                f"step_blocks.{block_id}.steps.{local_step_id} must not declare step_id; "
                "the expanded id is generated from the block, item, and local step ids."
            )

    normalized_block = {"for_each": deepcopy(items), "steps": deepcopy(templates)}
    return block_id, normalized_block, items, templates


def expand_pipeline_document(
    document: Mapping[str, Any],
    *,
    max_expanded_steps: int | None = None,
) -> ExpandedPipelineDocument:
    """Expand declarative ``step_blocks`` into an ordinary ``steps`` mapping."""

    if max_expanded_steps is not None and max_expanded_steps < 1:
        raise ValueError("max_expanded_steps must be a positive integer or None.")
    source = dict(_as_mapping(document, "pipeline document"))
    ordinary_steps = _normalize_steps(source.get("steps", {}) or {}, "steps", segment_ids=False)
    raw_blocks = _as_mapping(source.get("step_blocks", {}) or {}, "step_blocks")

    authored = deepcopy(source)
    authored["steps"] = deepcopy(ordinary_steps)
    normalized_blocks: dict[str, dict[str, Any]] = {}
    expanded_steps = deepcopy(ordinary_steps)
    origins: dict[str, StepOrigin] = {}

    for block_index, (raw_block_id, raw_block) in enumerate(raw_blocks.items()):
        block_id, normalized_block, items, templates = _normalize_block(raw_block_id, raw_block)
        if block_id in normalized_blocks:
            raise PipelineSchemaError(f"step_blocks contains duplicate normalized block id {block_id!r}.")
        normalized_blocks[block_id] = normalized_block

        local_ids = set(templates)
        projected_steps = len(expanded_steps) + len(items) * len(templates)
        if max_expanded_steps is not None and projected_steps > max_expanded_steps:
            raise PipelineSchemaError(
                f"Pipeline expands to at least {projected_steps} steps, "
                f"exceeding the limit of {max_expanded_steps}."
            )
        for item_index, (item_id, parameters) in enumerate(items.items()):
            for local_step_index, (local_step_id, template) in enumerate(templates.items()):
                expanded_id = f"{block_id}.{item_id}.{local_step_id}"
                if expanded_id in expanded_steps:
                    raise PipelineSchemaError(f"Expanded step id {expanded_id!r} collides with another step.")
                path = f"step_blocks.{block_id}.for_each.{item_id}.steps.{local_step_id}"
                expanded_step = _substitute(template, parameters, path)
                dependencies = _normalize_requires(expanded_step.get("requires_steps"), f"{path}.requires_steps")
                resolved_dependencies: list[str] = []
                for dependency in dependencies:
                    if dependency.startswith("."):
                        local_dependency = dependency[1:]
                        if local_dependency not in local_ids:
                            raise PipelineSchemaError(
                                f"{path}.requires_steps refers to unknown local step {dependency!r}."
                            )
                        resolved_dependencies.append(f"{block_id}.{item_id}.{local_dependency}")
                    else:
                        resolved_dependencies.append(dependency)
                if resolved_dependencies:
                    expanded_step["requires_steps"] = resolved_dependencies
                else:
                    expanded_step.pop("requires_steps", None)
                expanded_steps[expanded_id] = expanded_step
                origins[expanded_id] = StepOrigin(
                    block=block_id,
                    item=item_id,
                    local_step=local_step_id,
                    block_index=block_index,
                    item_index=item_index,
                    local_step_index=local_step_index,
                )

    if normalized_blocks:
        authored["step_blocks"] = normalized_blocks
    else:
        authored.pop("step_blocks", None)
    if max_expanded_steps is not None and len(expanded_steps) > max_expanded_steps:
        raise PipelineSchemaError(
            f"Pipeline expands to {len(expanded_steps)} steps, exceeding the limit of {max_expanded_steps}."
        )

    expanded = deepcopy(source)
    expanded.pop("step_blocks", None)
    expanded["steps"] = expanded_steps
    return ExpandedPipelineDocument(authored_spec=authored, expanded_spec=expanded, origins=origins)


def expand_pipeline_yaml(
    yaml_string: str,
    *,
    max_expanded_steps: int | None = None,
) -> ExpandedPipelineDocument:
    """Parse and expand one pipeline YAML document."""

    parsed = yaml.safe_load(yaml_string) or {}
    if not isinstance(parsed, Mapping):
        raise PipelineSchemaError(f"pipeline document must be a mapping, got {type(parsed).__name__}.")
    return expand_pipeline_document(parsed, max_expanded_steps=max_expanded_steps)
