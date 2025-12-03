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

__all__ = ["AppendProcessingData"]
__version__ = "20251030.3"

from pathlib import Path
from typing import Any

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.io.io_sources import IoSources

# Module-level handler; facilities can swap MessageHandler implementation as needed
logger = MessageHandler(name=__name__)


class AppendProcessingData(ProcessStep):
    """
    Load signal data from ``self.io_sources`` into a processing :class:`DataBundle`
    in ``self.processing_data``.

    This step creates or updates a single :class:`DataBundle` from existing
    :class:`IoSources` entries in ``self.io_sources``. It:

    1. Loads the signal array from ``signal_location`` (string reference).
    2. Loads units either from:
       - ``units_location`` via :meth:`IoSources.get_static_metadata`, or
       - ``units_override`` as a direct units string, or
       - defaults to dimensionless if neither is provided.
    3. Optionally loads uncertainty arrays from ``uncertainties_sources``.
    4. Wraps everything in a :class:`BaseData` instance.
    5. Sets ``BaseData.rank_of_data`` based on the configured ``rank_of_data``:
       - If it is an ``int``, it is used directly.
       - If it is a ``str``, it is interpreted as an IoSources metadata reference
         (``'<io_source_id>::<dataset_path>'``) and read via
         :meth:`IoSources.get_static_metadata`, then converted to ``int``.
       - Validation and bounds checking are handled by :func:`validate_rank_of_data`
         inside :class:`BaseData`.
    6. Stores the resulting :class:`BaseData` under the configured
       ``databundle_output_key`` (default: ``"signal"``) in a
       :class:`DataBundle` at ``self.processing_data[processing_key]``. If that
       DataBundle already exists, it is updated: existing entries are preserved
       and the ``databundle_output_key`` entry is overwritten or added.

    The resulting mapping ``{processing_key: DataBundle}`` is returned.
    """

    documentation = ProcessStepDescriber(
        calling_name="Append Processing Data",
        calling_id="AppendProcessingData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],  # this step creates/updates a DataBundle
        modifies={},  # processing_key: databundle: databundle_output_key
        required_arguments=[
            "processing_key",  # Processing data key to create/update. Must be a string.
            "signal_location",  # Data identifier to read from, '<io_source_id>::<dataset_path>'. Must be a string.
            "rank_of_data",  # Rank of the created BaseData array. int or str (io_source location).
        ],
        calling_arguments={
            "processing_key": "",  # must be set by the user
            "signal_location": "",  # must be set by the user
            "rank_of_data": 2,
            # key under which the BaseData will be stored in the DataBundle
            "databundle_output_key": "signal",
            # optional location for units definition, in the form
            # '<io_source_id>::<dataset_path>' or with attribute suffix
            # '<io_source_id>::<dataset_path>@units'. If None, units will be
            # set to ureg.dimensionless unless units_override is given.
            "units_location": None,
            # optional direct units string to override any loaded units
            "units_override": None,
            # optional sources for uncertainties data, in the form:
            # {'<uncertainty_name>': '<io_source_id>::<dataset_path>'}
            "uncertainties_sources": {},
        },
        step_keywords=["append", "processing", "data", "signal"],
        step_doc="Append signal data from IoSources into a processing DataBundle.",
        step_reference="",
        step_note=(
            "This step reads from existing IoSources and creates or updates a named DataBundle "
            "with a BaseData entry (default 'signal') for use in the processing pipeline."
        ),
    )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _resolve_rank_of_data(self, rank_cfg: Any, io_sources: IoSources) -> int:
        """
        Resolve the configured rank_of_data to an integer.

        Parameters
        ----------
        rank_cfg :
            Either an integer directly, or a string reference of the form
            '<io_source_id>::<dataset_path>' pointing to metadata that contains
            the rank as an integer-compatible value.

        io_sources :
            The IoSources object used to resolve metadata references.

        Returns
        -------
        int
            The resolved rank_of_data. Actual bounds checking is performed by
            BaseData's internal validation.
        """
        # Direct int → use as-is (with int() for safety)
        if isinstance(rank_cfg, int):
            return int(rank_cfg)

        # If it *looks* like an io_source reference, treat it as such
        if isinstance(rank_cfg, str):
            logger.debug(
                f"AppendProcessingData: resolving rank_of_data from IoSources metadata reference '{rank_cfg}'."
            )
            meta_value = io_sources.get_static_metadata(rank_cfg)
            try:
                return int(meta_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not convert rank_of_data metadata from '{rank_cfg}' to int (value={meta_value!r})."
                ) from exc

        # Fallback: try to cast whatever it is to int
        try:
            return int(rank_cfg)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rank_of_data must be an int or an IoSources metadata reference string, "
                f"got {rank_cfg!r} ({type(rank_cfg).__name__})."
            ) from exc

    def _load_and_validate_configuration(self) -> dict[str, Any]:
        """
        Load and validate configuration values from ``self.configuration``.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the resolved configuration:
            - processing_key (str)
            - signal_location (str)
            - rank_of_data (int)
            - databundle_output_key (str)
            - units_location (str | None)
            - units_override (str | None)
            - uncertainties_sources (dict[str, str])
        """
        cfg = self.configuration

        processing_key = cfg.get("processing_key")
        if not isinstance(processing_key, str) or not processing_key:
            raise ValueError("AppendProcessingData requires 'processing_key' to be a non-empty string.")

        signal_location = cfg.get("signal_location")
        if not isinstance(signal_location, str) or not signal_location:
            raise ValueError("AppendProcessingData requires 'signal_location' to be a non-empty string.")

        if "rank_of_data" not in cfg:
            raise ValueError("AppendProcessingData requires 'rank_of_data' in the configuration.")
        rank_cfg = cfg["rank_of_data"]
        resolved_rank = self._resolve_rank_of_data(rank_cfg, self.io_sources)

        databundle_output_key = cfg.get("databundle_output_key", "signal")
        if not isinstance(databundle_output_key, str) or not databundle_output_key:
            raise ValueError("AppendProcessingData requires 'databundle_output_key' to be a non-empty string.")

        units_location = cfg.get("units_location")
        if units_location is not None and not isinstance(units_location, str):
            raise TypeError("'units_location' must be a string '<source_ref>::<dataset_path>' or None.")

        units_override = cfg.get("units_override")
        if units_override is not None and not isinstance(units_override, str):
            raise TypeError("'units_override' must be a units string if provided.")

        uncertainties_sources: dict[str, str] = cfg.get("uncertainties_sources", {}) or {}
        if not isinstance(uncertainties_sources, dict):
            raise TypeError(
                f"'uncertainties_sources' must be a dict[str, str], got {type(uncertainties_sources).__name__}."
            )

        return {
            "processing_key": processing_key,
            "signal_location": signal_location,
            "rank_of_data": resolved_rank,
            "databundle_output_key": databundle_output_key,
            "units_location": units_location,
            "units_override": units_override,
            "uncertainties_sources": uncertainties_sources,
        }

    # -------------------------------------------------------------------------
    # Public API used by the pipeline
    # -------------------------------------------------------------------------
    def calculate(self) -> dict[str, DataBundle]:
        """
        Create or update a DataBundle from ``self.io_sources`` and return it.

        Configuration fields:

        - ``processing_key`` (str):
            Name under which the DataBundle will be stored in
            ``self.processing_data``.

        - ``signal_location`` (str):
            Data reference in the form ``'<source_ref>::<dataset_path>'``.

        - ``rank_of_data`` (int or str):
            Desired rank for the created :class:`BaseData` object.
            If a string, it is treated as an IoSources metadata reference and
            resolved via :meth:`IoSources.get_static_metadata`. Validation and
            bounds checking are handled by :class:`BaseData`.

        - ``databundle_output_key`` (str, default: ``"signal"``):
            Key under which the new :class:`BaseData` will be stored inside the
            :class:`DataBundle`. If the DataBundle already contains an entry
            under this key, it will be overwritten.

        - ``units_location`` (str | None):
            Data reference pointing to a static metadata entry that defines the
            units. If provided, the value from
            :meth:`IoSources.get_static_metadata` is passed to :func:`ureg.Unit`.
            If omitted and ``units_override`` is None, units default to
            dimensionless.

        - ``units_override`` (str | None):
            Direct units string (e.g. ``"counts"`` or ``"1/m"``) that overrides
            any value loaded via ``units_location``.

        - ``uncertainties_sources`` (dict[str, str]):
            Mapping from uncertainty name (e.g. ``"poisson"``) to data reference
            (``'<source_ref>::<dataset_path>'``).
        """
        cfg = self._load_and_validate_configuration()

        processing_key: str = cfg["processing_key"]
        signal_location: str = cfg["signal_location"]
        rank_of_data: int = cfg["rank_of_data"]
        databundle_output_key: str = cfg["databundle_output_key"]
        units_location = cfg["units_location"]
        units_override = cfg["units_override"]
        uncertainties_sources: dict[str, str] = cfg["uncertainties_sources"]

        io_sources: IoSources = self.io_sources

        logger.info(
            (
                f"AppendProcessingData: creating/updating DataBundle '{processing_key}' "
                f"from signal_location='{signal_location}' into key '{databundle_output_key}'."
            ),
        )

        # Load BaseData via helper: signal + units + uncertainties
        bd: BaseData = basedata_from_sources(
            io_sources=io_sources,
            signal_source=signal_location,
            units_source=units_location,
            uncertainty_sources=uncertainties_sources,
        )

        # Override units if requested
        if units_override is not None:
            logger.debug(
                f"AppendProcessingData: overriding units for '{processing_key}' to '{units_override}'.",
            )
            bd.units = ureg.Unit(units_override)

        # Set rank_of_data; BaseData's own validation handles bounds
        bd.rank_of_data = rank_of_data

        # Create or update the DataBundle in processing_data
        existing_bundle = self.processing_data.get(processing_key)
        if existing_bundle is None:
            databundle = DataBundle()
        else:
            databundle = existing_bundle

        # Update/insert the BaseData at the requested key
        databundle[databundle_output_key] = bd

        # If no default_plot is set yet, use this key as a sensible default
        if getattr(databundle, "default_plot", None) is None:
            databundle.default_plot = databundle_output_key

        # Store back into processing_data and build output
        self.processing_data[processing_key] = databundle
        output: dict[str, DataBundle] = {processing_key: databundle}

        logger.info(
            f"AppendProcessingData: DataBundle '{processing_key}' now contains datasets {list(databundle.keys())}.",
        )

        return output
