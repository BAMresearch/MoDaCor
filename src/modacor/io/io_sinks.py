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
