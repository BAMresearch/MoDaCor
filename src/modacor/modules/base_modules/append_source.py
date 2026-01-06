# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "30/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["AppendSource"]
__version__ = "20251030.1"

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.io.io_sources import IoSources

# Module-level handler; facilities can swap MessageHandler implementation as needed
logger = MessageHandler(name=__name__)


class AppendSource(ProcessStep):
    """
    Appends an ioSource to self.io_sources.

    This step is intended for pipeline-graph / provenance operations: it augments
    the set of available I/O sources but does not touch the actual data bundles.
    """

    documentation = ProcessStepDescriber(
        calling_name="Append Source",
        calling_id="AppendSource",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},  # sources only; no data modified
        required_arguments=[
            "source_identifier",  # The identifier to use for the appended ioSource in the data sources. Can be a string or list of strings for multiple sources
            "source_location",  # The ioSource path or other location identifier of the object to append to the IoSources. Can be a string or list of strings for multiple sources
            "iosource_module",  # The fully qualified import path to the module to load the source_location into an ioSource object, e.g. 'modacor.io.yaml.yaml_source.YAMLSource' or 'modacor.io.hdf.hdf_source.HDFSource'. Choose only one.
        ],
        default_configuration={
            "source_identifier": "",
            "source_location": "",
            "iosource_module": "",
            "iosource_method_kwargs": {},
        },
        step_keywords=["append", "source"],
        step_doc="Append an ioSource to the available data sources",
        step_reference="",
        step_note="This adds an ioSource to the data sources of the databundle.",
    )

    # -------------------------------------------------------------------------
    # Public API used by the pipeline
    # -------------------------------------------------------------------------
    def calculate(self) -> dict[str, DataBundle]:
        """
        Append one or more sources to ``self.io_sources``.

        Notes
        -----
        - No ``DataBundle`` objects are modified or created.
        - The pipeline can treat an empty output dict as "no-op on data",
          while the side-effect on ``self.io_sources`` persists.
        """
        output: dict[str, DataBundle] = {}

        source_ids: str | list[str] = self.configuration["source_identifier"]
        source_locations: str | list[str] = self.configuration["source_location"]
        iosource_module: str = self.configuration["iosource_module"]

        # Normalise to lists
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        if isinstance(source_locations, str):
            source_locations = [source_locations]

        if len(source_ids) != len(source_locations):
            raise ValueError(
                "If multiple source_identifiers and source_locations are provided, their counts must match."
            )

        for source_id, source_location in zip(source_ids, source_locations):
            # Only append if not already present
            if source_id not in self.io_sources.defined_sources:
                self._append_loader_by_name(
                    loader_name=iosource_module,
                    source_location=source_location,
                    source_identifier=source_id,
                    iosource_method_kwargs=self.configuration.get("iosource_method_kwargs", {}),
                )
        # No data modified – only side-effect is on self.io_sources
        return output

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _append_loader_by_name(
        self,
        loader_name: str,
        source_location: str,
        source_identifier: str,
        iosource_method_kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Resolve the requested loader and append the resulting ioSource to
        ``self.io_sources``.

        Parameters
        ----------
        loader_name:
            Either a fully qualified import path
            (e.g. ``"modacor.io.hdf.hdf_source.HDFSource"``).
        source_location:
            Path / URI / identifier understood by the loader.
        source_identifier:
            Key under which the resulting ioSource will be stored in
            ``self.io_sources``.
        iosource_method_kwargs:
            Additional keyword arguments to pass to the loader callable.
        """
        source_callable = self._resolve_iosource_callable(loader_name)

        # Ensure io_sources exists or initialize it
        if not hasattr(self, "io_sources") or self.io_sources is None:
            # ProcessStep normally sets this up, but be defensive.
            self.io_sources = IoSources()
            logger.info("Initialized self.io_sources in AppendSource step.")

        self.io_sources.register_source(
            source_callable(
                source_reference=source_identifier,
                resource_location=source_location,
                iosource_method_kwargs=iosource_method_kwargs,
            )
        )

    def _resolve_iosource_callable(self, loader_name: str) -> Callable[..., Any]:
        """
        Resolve the configured loader into a callable.

        Strategy
        --------
        1. If ``loader_name`` contains a dot, treat it as a fully qualified
           import path like ``package.module.ClassOrFunc``.
        """

        module_path, attr_name = loader_name.rsplit(".", 1)
        module = import_module(module_path)
        try:
            loader_obj = getattr(module, attr_name)
        except AttributeError as exc:
            raise ImportError(
                f"Could not find '{attr_name}' in module '{module_path}' for iosource_module='{loader_name}'."
            ) from exc

        return loader_obj
