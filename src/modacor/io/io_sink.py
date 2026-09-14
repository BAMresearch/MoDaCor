# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from logging import WARNING
from typing import TYPE_CHECKING, Any, ClassVar

import attrs
from attrs import define, field

from modacor.io.chunking import UnsupportedSinkCapability

if TYPE_CHECKING:
    from modacor.dataclasses.processing_data import ProcessingData
    from modacor.io.chunking import (
        ChunkInputPlan,
        ChunkOutputStatus,
        ChunkPlan,
        ChunkSpec,
        ChunkWriteResult,
        ProvisionalChunkPlan,
    )


def default_config() -> dict[str, Any]:
    return {}


@define
class IoSink:
    """
    Base class for IO sinks. Mirrors IoSource.

    Sinks are registered with a resource_location (file/socket/etc.).
    The routed write call passes an optional 'subpath' after '::', which may be empty.
    """

    supports_chunked_writes: ClassVar[bool] = False

    configuration: dict[str, Any] = field(factory=default_config)
    sink_reference: str = field(default="", converter=str, validator=attrs.validators.instance_of(str))
    type_reference: str = "IoSink"
    iosink_method_kwargs: dict[str, Any] = field(factory=dict, validator=attrs.validators.instance_of(dict))
    logging_level: int = field(default=WARNING, validator=attrs.validators.instance_of(int))

    def write(self, subpath: str, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def initialize_chunked(self, subpath: str, plan: ChunkPlan, **kwargs: Any) -> ChunkWriteResult:
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def initialize_provisional_chunked(
        self,
        subpath: str,
        plan: ProvisionalChunkPlan,
        *,
        input_plan: ChunkInputPlan,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        raise UnsupportedSinkCapability(type(self), "provisional_chunked_writes")

    def resolve_provisional_chunked(
        self,
        subpath: str,
        processing_data: ProcessingData,
        *,
        provisional_plan: ProvisionalChunkPlan,
        input_plan: ChunkInputPlan,
        plan: ChunkPlan,
        chunk: ChunkSpec,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        raise UnsupportedSinkCapability(type(self), "provisional_chunked_writes")

    def inspect_provisional_chunked(
        self,
        subpath: str,
        *,
        plan: ProvisionalChunkPlan,
        input_plan: ChunkInputPlan,
        **kwargs: Any,
    ) -> ChunkOutputStatus:
        raise UnsupportedSinkCapability(type(self), "provisional_chunked_writes")

    def load_provisional_chunked(
        self,
        plan_id: str,
        **kwargs: Any,
    ) -> tuple[str, ProvisionalChunkPlan, ChunkInputPlan]:
        raise UnsupportedSinkCapability(type(self), "provisional_chunked_writes")

    def load_chunked_resolution(
        self,
        plan_id: str,
        **kwargs: Any,
    ) -> tuple[ProvisionalChunkPlan, ChunkInputPlan] | None:
        raise UnsupportedSinkCapability(type(self), "provisional_chunked_writes")

    def write_chunk(
        self,
        subpath: str,
        processing_data: ProcessingData,
        *,
        plan: ChunkPlan,
        chunk: ChunkSpec,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def inspect_chunked(self, subpath: str, *, plan: ChunkPlan, **kwargs: Any) -> ChunkOutputStatus:
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def load_chunked_plan(self, plan_id: str, **kwargs: Any) -> tuple[str, ChunkPlan]:
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def recover_chunked(
        self,
        subpath: str,
        *,
        plan: ChunkPlan,
        action: str,
        **kwargs: Any,
    ) -> ChunkOutputStatus:
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def finalize_chunked(self, subpath: str, *, plan: ChunkPlan, **kwargs: Any) -> ChunkWriteResult:
        raise UnsupportedSinkCapability(type(self), "chunked_writes")
