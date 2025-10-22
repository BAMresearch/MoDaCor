# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

__coding__ = "utf-8"
__author__ = "Tim Snow, Brian R. Pauw"
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["HDFLoader"]

from logging import WARNING
from pathlib import Path

import h5py
import numpy as np

from modacor.dataclasses.messagehandler import MessageHandler

# from modacor.dataclasses.basedata import BaseData
from modacor.io.io_source import ArraySlice

from ..io_source import IoSource


class HDFLoader(IoSource):
    _data_cache: dict[str, np.ndarray] = None
    _file_path: Path | None = None
    _static_metadata_cache: dict[str, Any] = None

    def __init__(self, source_reference: str, logging_level=WARNING, resource_location: Path | str | None = None):
        super().__init__(source_reference=source_reference)
        self.logger = MessageHandler(level=logging_level, name="HDFLoader")
        self._file_path = Path(resource_location) if resource_location is not None else None
        # self._file_reference = None  # let's not leave open file references lying around if we can help it.
        self._file_datasets = []
        self._file_datasets_shapes = {}
        self._file_datasets_dtypes = {}
        self._data_cache = {}
        self._static_metadata_cache = {}

    def _preload(self):
        assert self._file_path.is_file(), self.logger.error(f"HDF5 file {self._file_path} does not exist.")
        try:
            with h5py.File(self._file_path, "r") as f:
                f.visititems(self._find_datasets)
        except OSError as error:
            self.logger.log.error(error)
            raise OSError(error)

    def _find_datasets(self, path_name, path_object):
        """
        An internal function to be used to walk the tree of an HDF5 file and return a list of
        the datasets within
        """
        if isinstance(path_object, h5py._hl.dataset.Dataset):
            self._file_datasets.append(path_name)
            self._file_datasets_shapes[path_name] = path_object.shape
            self._file_datasets_dtypes[path_name] = path_object.dtype

    def get_static_metadata(self, data_key):
        if data_key not in self._static_metadata_cache:
            with h5py.File(self._file_path, "r") as f:
                value = f[data_key][()]
                # decode bytes to string if necessary
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                self._static_metadata_cache[data_key] = value
        return self._static_metadata_cache[data_key]

    def get_data(self, data_key: str, load_slice: ArraySlice = ...) -> np.ndarray:
        if data_key not in self._data_cache:
            with h5py.File(self._file_path, "r") as f:
                data_array = f[data_key][load_slice]  # if load_slice is not None else f[data_key][()]
                self._data_cache[data_key] = np.array(data_array)
        return self._data_cache[data_key]

    def get_data_shape(self, data_key: str) -> tuple[int, ...]:
        if data_key in self._file_datasets_shapes:
            return self._file_datasets_shapes[data_key]
        return ()

    def get_data_dtype(self, data_key: str) -> np.dtype | None:
        if data_key in self._file_datasets_dtypes:
            return self._file_datasets_dtypes[data_key]
        return None

    def get_data_attributes(self, data_key: str) -> dict[str, Any]:
        attributes = {}
        with h5py.File(self._file_path, "r") as f:
            if data_key in f:
                dataset = f[data_key]
                for attr_key in dataset.attrs:
                    attributes[attr_key] = dataset.attrs[attr_key]
        return attributes
