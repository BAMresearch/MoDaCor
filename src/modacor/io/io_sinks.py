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

__all__ = ["IoSinks"]

from typing import Any

from attrs import define, field

from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunking import (
    ChunkInputPlan,
    ChunkOutputStatus,
    ChunkPlan,
    ChunkSpec,
    ChunkWriteResult,
    ProvisionalChunkPlan,
    UnsupportedSinkCapability,
)
from modacor.io.io_sink import IoSink


@define
class IoSinks:
    """
    Registry for IoSink instances. Mirrors IoSources.
    """

    defined_sinks: dict[str, IoSink] = field(factory=dict)

    def register_sink(self, sink: IoSink, sink_reference: str | None = None) -> None:
        if not isinstance(sink, IoSink):
            raise TypeError("sink must be an instance of IoSink")
        if sink_reference is None:
            sink_reference = sink.sink_reference
        if not isinstance(sink_reference, str):
            raise TypeError("sink_reference must be a string")
        if sink_reference in self.defined_sinks:
            raise ValueError(f"Sink {sink_reference} already registered.")
        self.defined_sinks[sink_reference] = sink

    def get_sink(self, sink_reference: str) -> IoSink:
        if sink_reference not in self.defined_sinks:
            raise KeyError(f"Sink {sink_reference} not registered.")
        return self.defined_sinks[sink_reference]

    def split_target_reference(self, target_reference: str) -> tuple[str, str]:
        """
        Split 'sink_ref::subpath'. Subpath may be empty (e.g. 'export_csv::').
        """
        _split = target_reference.split("::", 1)
        if len(_split) != 2:
            raise ValueError(
                "target_reference must be in the format 'sink_ref::subpath' with a double colon separator."
            )
        return _split[0], _split[1]

    def write_data(self, target_reference: str, *args, **kwargs) -> Any:
        sink_ref, subpath = self.split_target_reference(target_reference)
        sink = self.get_sink(sink_ref)
        return sink.write(subpath, *args, **kwargs)

    def _get_chunked_sink(self, target_reference: str) -> tuple[IoSink, str]:
        sink_ref, subpath = self.split_target_reference(target_reference)
        sink = self.get_sink(sink_ref)
        if not sink.supports_chunked_writes:
            raise UnsupportedSinkCapability(type(sink), "chunked_writes")
        return sink, subpath

    def initialize_chunked(
        self,
        target_reference: str,
        plan: ChunkPlan,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.initialize_chunked(subpath, plan, **kwargs)

    def initialize_provisional_chunked(
        self,
        target_reference: str,
        plan: ProvisionalChunkPlan,
        *,
        input_plan: ChunkInputPlan,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.initialize_provisional_chunked(subpath, plan, input_plan=input_plan, **kwargs)

    def inspect_provisional_chunked(
        self,
        target_reference: str,
        *,
        plan: ProvisionalChunkPlan,
        input_plan: ChunkInputPlan,
        **kwargs: Any,
    ) -> ChunkOutputStatus:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.inspect_provisional_chunked(subpath, plan=plan, input_plan=input_plan, **kwargs)

    def resolve_provisional_chunked(
        self,
        target_reference: str,
        processing_data: ProcessingData,
        *,
        provisional_plan: ProvisionalChunkPlan,
        input_plan: ChunkInputPlan,
        plan: ChunkPlan,
        chunk: ChunkSpec,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.resolve_provisional_chunked(
            subpath,
            processing_data,
            provisional_plan=provisional_plan,
            input_plan=input_plan,
            plan=plan,
            chunk=chunk,
            **kwargs,
        )

    def load_provisional_chunked(
        self,
        sink_reference: str,
        plan_id: str,
        **kwargs: Any,
    ) -> tuple[str, ProvisionalChunkPlan, ChunkInputPlan]:
        sink = self.get_sink(sink_reference)
        if not sink.supports_chunked_writes:
            raise UnsupportedSinkCapability(type(sink), "provisional_chunked_writes")
        return sink.load_provisional_chunked(plan_id, **kwargs)

    def write_chunk(
        self,
        target_reference: str,
        processing_data: ProcessingData,
        *,
        plan: ChunkPlan,
        chunk: ChunkSpec,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.write_chunk(subpath, processing_data, plan=plan, chunk=chunk, **kwargs)

    def inspect_chunked(
        self,
        target_reference: str,
        *,
        plan: ChunkPlan,
        **kwargs: Any,
    ) -> ChunkOutputStatus:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.inspect_chunked(subpath, plan=plan, **kwargs)

    def load_chunked_plan(self, sink_reference: str, plan_id: str, **kwargs: Any) -> tuple[str, ChunkPlan]:
        sink = self.get_sink(sink_reference)
        if not sink.supports_chunked_writes:
            raise UnsupportedSinkCapability(type(sink), "chunked_writes")
        return sink.load_chunked_plan(plan_id, **kwargs)

    def recover_chunked(
        self,
        target_reference: str,
        *,
        plan: ChunkPlan,
        action: str,
        **kwargs: Any,
    ) -> ChunkOutputStatus:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.recover_chunked(subpath, plan=plan, action=action, **kwargs)

    def finalize_chunked(
        self,
        target_reference: str,
        *,
        plan: ChunkPlan,
        **kwargs: Any,
    ) -> ChunkWriteResult:
        sink, subpath = self._get_chunked_sink(target_reference)
        return sink.finalize_chunked(subpath, plan=plan, **kwargs)
