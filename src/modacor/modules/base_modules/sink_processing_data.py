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

__all__ = ["SinkProcessingData"]
__version__ = "20260901.1"

from pathlib import Path, PurePosixPath

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.io.processing_path import parse_processing_path

# Module-level handler; facilities can swap MessageHandler implementation as needed
logger = MessageHandler(name=__name__)


class SinkProcessingData(ProcessStep):
    """
    Export ProcessingData to an IoSink.

    - target: 'sink_id::subpath' (for CSV usually 'export_csv::')
    - data_paths: ProcessingData paths without '::', e.g. '/sample/Q/signal'
      HDF sinks also accept bundle roots, e.g. '/sample'.
    """

    documentation = ProcessStepDescriber(
        calling_name="Sink Processing Data",
        calling_id="SinkProcessingData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],  # no new databundle produced
        modifies={},  # side-effect only (writing)
        arguments={
            "target": {
                "type": str,
                "required": True,
                "default": "",
                "doc": "Sink target in the form 'sink_id::subpath'.",
            },
            "data_paths": {
                "type": (str, list),
                "required": True,
                "default": [],
                "doc": "ProcessingData paths to write (string or list of strings).",
            },
        },
        step_keywords=["sink", "export", "write"],
        step_doc="Write selected ProcessingData leaves to an IoSink.",
        step_reference="",
        step_note="This step performs an export side-effect and returns an empty output dict.",
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        data_paths = self.configuration.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        read_patterns: set[str] = set()
        for data_path in data_paths:
            parts = tuple(part for part in PurePosixPath(str(data_path).strip()).parts if part != "/")
            if len(parts) == 1:
                read_patterns.add(f"{parts[0]}.*")
                continue

            try:
                parsed = parse_processing_path(str(data_path))
            except (TypeError, ValueError):
                read_patterns.add("*")
            else:
                read_patterns.add(f"{parsed.databundle_key}.{parsed.basedata_name}")

        return ProcessStepDependencies(
            processing_reads=read_patterns,
            processing_writes=(),
        )

    def calculate(self) -> dict[str, DataBundle]:
        output: dict[str, DataBundle] = {}

        target: str = self.configuration["target"]
        data_paths: str | list[str] = self.configuration["data_paths"]

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        # Delegate determinism + validation to sink implementation
        self.io_sinks.write_data(
            target,
            self.processing_data,
            data_paths=data_paths,
        )

        logger.debug(f"SinkProcessingData wrote {len(data_paths)} paths to target '{target}'.")
        return output
