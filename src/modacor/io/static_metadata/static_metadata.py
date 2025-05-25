# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from modacor.dataclasses.source_data import SourceData

__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__license__ = "BSD3"
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "24/05/2025"
__version__ = "20250524.1"
__status__ = "Development"  # "Development", "Production"
from logging import WARNING

import h5py
import numpy as np

from modacor.administration.licenses import BSD3Clause as __license__  # noqa: F401
from modacor.dataclasses.messagehandler import MessageHandler

from ..io_source import IoSource

# end of header and standard imports


class StaticMetadata(IoSource):
    """
    This IoSource is used to load and make experiment metadata available to
    the processing pipeline modules.
    It can be filled in with information such as wavelength,
    geometry and other relevant information which is needed in multiple
    processing steps.
    The metadata can be loaded from a yaml file with mappings. this is set in the configuraiton
    The entries are returned as BaseData elements, with units and uncertainties.
    """

    _data_cache: dict[str, SourceData] = None
    _static_metadata_cache: dict[str, Any] = None

    def __init__(self, source_reference: str, logging_level=WARNING):
        super().__init__(source_reference)
        self.logger = MessageHandler(level=logging_level, name="StaticMetadata")
        self._data_cache = {}  # for values with units and uncertainties
        self._static_metadata_cache = {}  # for other elements such as strings and tags

    def _load_from_yaml(self, file_path: Path) -> None:
        """
        Load static metadata from a YAML file.
        This method should be implemented to parse the YAML file and populate
        the _data_cache with SourceData objects.
        """
        assert file_path.exists(), f"Static metadataa file {file_path} does not exist."
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        for key, entry in data.items():
            if isinstance(entry, dict):
                if all(k in entry for k in ("value", "units", "variance")):
                    self._data_cache[key] = SourceData(
                        value=np.array(entry.pop("value", [])),
                        units=entry.pop("units", "rankine"),
                        variance=np.array(entry.pop("variance", [])),
                        attributes=entry if entry else {},
                    )
                else:
                    # invalid entry, raise an error or log it
                    self.logger.error(
                        f"Invalid entry for key '{key}': {entry}. Expected 'value', 'units', and"
                        " 'variance'."
                    )
            else:
                # Store other metadata as static metadata
                self._static_metadata_cache[key] = entry

    def get_static_metadata(self, data_key: str):
        if data_key not in self._static_metadata_cache:
            self.logger.error(f"Static metadata key '{data_key}' not in cache.")
            return None

        return self._static_metadata_cache.get(data_key)

    def get_data(self, data_key: str) -> SourceData:
        """
        Get the data from the HDF5 file.
        """
        if data_key not in self._data_cache:
            self.logger.error(f"Data key '{data_key}' not in static metadata cache.")
            return None

        return self._data_cache.get(data_key)
