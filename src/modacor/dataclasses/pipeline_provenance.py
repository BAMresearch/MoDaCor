# SPDX-License-Identifier: BSD-3-Clause
"""Authored and expanded representations of one executable pipeline."""

from __future__ import annotations

__all__ = ["PipelineProvenance"]

from copy import deepcopy
from typing import Any

from attrs import define, field, validators


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(value)


@define(frozen=True, slots=True)
class PipelineProvenance:
    """Complete reproducibility record for one pipeline execution."""

    authored_yaml: str = field(converter=str, validator=validators.instance_of(str))
    authored_spec: dict[str, Any] = field(converter=_copy_mapping, validator=validators.instance_of(dict))
    expanded_yaml: str = field(converter=str, validator=validators.instance_of(str))
    expanded_spec: dict[str, Any] = field(converter=_copy_mapping, validator=validators.instance_of(dict))
    schema_version: str = field(default="1.0", converter=str, validator=validators.instance_of(str))
