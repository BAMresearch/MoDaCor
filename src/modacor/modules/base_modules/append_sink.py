# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["AppendSink"]
__version__ = "20260109.1"

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.io.io_sinks import IoSinks

# Module-level handler; facilities can swap MessageHandler implementation as needed
logger = MessageHandler(name=__name__)


class AppendSink(ProcessStep):
    """
    Appends an IoSink to self.io_sinks.

    This mirrors AppendSource: it augments the set of available I/O sinks but does
    not touch the actual data bundles.
    """

    documentation = ProcessStepDescriber(
        calling_name="Append Sink",
        calling_id="AppendSink",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},  # sinks only; no data modified
        arguments={
            "sink_identifier": {
                "type": (str, list),
                "required": True,
                "default": "",
                "doc": "Identifier(s) to register the ioSink(s) under.",
            },
            "sink_location": {
                "type": (str, list),
                "required": True,
                "default": "",
                "doc": "Resource location(s) understood by the sink.",
            },
            "iosink_module": {
                "type": str,
                "required": True,
                "default": "",
                "doc": "Fully qualified module path to the ioSink class.",
            },
            "iosink_method_kwargs": {
                "type": dict,
                "default": {},
                "doc": "Keyword arguments forwarded to the ioSink constructor.",
            },
        },
        step_keywords=["append", "sink"],
        step_doc="Append an IoSink to the available data sinks",
        step_reference="",
        step_note="This adds an IoSink to the data sinks registry.",
    )

    # -------------------------------------------------------------------------
    # Public API used by the pipeline
    # -------------------------------------------------------------------------
    def calculate(self) -> dict[str, DataBundle]:
        output: dict[str, DataBundle] = {}

        sink_ids: str | list[str] = self.configuration["sink_identifier"]
        sink_locations: str | list[str] = self.configuration["sink_location"]
        iosink_module: str = self.configuration["iosink_module"]

        # Normalise to lists
        if isinstance(sink_ids, str):
            sink_ids = [sink_ids]
        if isinstance(sink_locations, str):
            sink_locations = [sink_locations]

        if len(sink_ids) != len(sink_locations):
            raise ValueError("If multiple sink_identifiers and sink_locations are provided, their counts must match.")

        for sink_id, sink_location in zip(sink_ids, sink_locations):
            if sink_id not in self.io_sinks.defined_sinks:
                self._append_sink_by_name(
                    sink_name=iosink_module,
                    sink_location=sink_location,
                    sink_identifier=sink_id,
                    iosink_method_kwargs=self.configuration.get("iosink_method_kwargs", {}),
                )

        return output

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _append_sink_by_name(
        self,
        sink_name: str,
        sink_location: str,
        sink_identifier: str,
        iosink_method_kwargs: dict[str, Any] = {},
    ) -> None:
        sink_callable = self._resolve_iosink_callable(sink_name)

        # Ensure io_sinks exists or initialize it
        if not hasattr(self, "io_sinks") or self.io_sinks is None:
            self.io_sinks = IoSinks()
            logger.info("Initialized self.io_sinks in AppendSink step.")

        self.io_sinks.register_sink(
            sink_callable(
                sink_reference=sink_identifier,
                resource_location=sink_location,
                iosink_method_kwargs=iosink_method_kwargs,
            )
        )

    def _resolve_iosink_callable(self, sink_name: str) -> Callable[..., Any]:
        module_path, attr_name = sink_name.rsplit(".", 1)
        module = import_module(module_path)
        try:
            sink_obj = getattr(module, attr_name)
        except AttributeError as exc:
            raise ImportError(
                f"Could not find '{attr_name}' in module '{module_path}' for iosink_module='{sink_name}'."
            ) from exc
        return sink_obj
