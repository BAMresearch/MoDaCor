# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

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

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler

from ..io_source import IoSource


class HDFLoader(IoSource):
    _data_cache: dict[str, np.ndarray] = None
    _file_path: Path | None = None

    def __init__(self, source_reference: str, logging_level=WARNING, resource_location: Path | str | None = None):
        super().__init__(source_reference)
        self.logger = MessageHandler(level=logging_level, name="HDFLoader")
        self._file_path = Path(resource_location) if resource_location is not None else None
        # self._file_reference = None  # let's not leave open file references lying around if we can help it.
        self._file_datasets = []
        self._file_datasets_shapes = {}
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

    def get_data(self, data_key: str) -> BaseData:
        raise (NotImplementedError("get_data method not yet implemented in HDFLoader class."))

    def get_static_metadata(self, data_key):
        raise (NotImplementedError("get_static_metadata method not yet implemented in HDFLoader class."))
