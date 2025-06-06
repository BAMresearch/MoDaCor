# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__license__ = "BSD3"
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "06/06/2025"
__version__ = "20250505.1"
__status__ = "Development"  # "Development", "Production"

from logging import WARNING
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from modacor.administration.licenses import BSD3Clause as __license__  # noqa: F401
from modacor.dataclasses.messagehandler import MessageHandler

from ..io_source import IoSource


def get_from_nested_dict_by_path(data, path):
    """
    Get a value from a nested dictionary using a slash-separated path.
    """
    # remove leading and trailing slashes
    path = path.strip("/")
    for key in path.split("/"):
        data = data[key]
    return data


class YamlLoader(IoSource):
    """
    This IoSource is used to load and make experiment metadata available to
    the processing pipeline modules.
    It can be filled in with information such as wavelength,
    geometry and other relevant information which is needed in multiple
    processing steps.
    The metadata can be loaded from a yaml file with mappings. this is set in the configuraiton
    The entries are returned as BaseData elements, with units and uncertainties.
    """

    _yaml_data: dict[str, Any] = dict()
    _data_cache: dict[str, np.ndarray] = None

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
            self._yaml_data.update(yaml.safe_load(f))

    def get_static_metadata(self, data_key: str) -> Any:
        """Returns static metadata, which can be anything"""
        try:
            return get_from_nested_dict_by_path(self._yaml_data, data_key)
        except KeyError as e:
            self.logger.error(f"Static metadata key '{data_key}' not in YAML data: {e}")
            return None

    def get_data(self, data_key: str) -> np.ndarray:
        """
        Get the data from the static metadata.
        """
        if data_key not in self._data_cache:
            self.logger.info(f"Data key '{data_key}' not in static metadata cache yet.")
            # try to convert from the yaml data into an np.asarray
            self._data_cache.update({data_key: self.get_static_metadata(data_key)})

        return np.asarray(self._data_cache.get(data_key), dtype=float)
