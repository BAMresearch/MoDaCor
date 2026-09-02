# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

__coding__ = "utf-8"
__authors__ = ["Tim Snow", "Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["HDFSource"]

from pathlib import Path

import h5py
import numpy as np
from attrs import define, field, validators

from modacor.dataclasses.messagehandler import MessageHandler

# from modacor.dataclasses.basedata import BaseData
from modacor.io.io_source import ArraySlice

from ..io_source import IoSource

try:
    import hdf5plugin  # noqa: F401
except ImportError as error:
    _HDF5PLUGIN_IMPORT_ERROR: ImportError | None = error
else:
    _HDF5PLUGIN_IMPORT_ERROR = None


def _slice_cache_key(load_slice: ArraySlice) -> Any:
    if load_slice is Ellipsis:
        return ("ellipsis",)
    if load_slice is None:
        return ("none",)
    if isinstance(load_slice, slice):
        return ("slice", load_slice.start, load_slice.stop, load_slice.step)
    if isinstance(load_slice, tuple):
        return tuple(_slice_cache_key(item) for item in load_slice)
    try:
        hash(load_slice)
    except TypeError:
        return repr(load_slice)
    return load_slice


def _raise_hdf5_read_error(error: OSError) -> None:
    message = str(error)
    lower_message = message.lower()
    plugin_related = any(token in lower_message for token in ("plugin", "filter", "blosc", "bshuf", "lz4"))
    if plugin_related:
        if _HDF5PLUGIN_IMPORT_ERROR is None:
            hint = "hdf5plugin is installed, but HDF5 could not load the required filter plugin."
        else:
            hint = (
                "This HDF5 dataset appears to require an external compression filter. "
                "Install hdf5plugin in the active MoDaCor environment."
            )
        raise OSError(f"{message}\n{hint}") from error
    raise OSError(error) from error


@define(kw_only=True)
class HDFSource(IoSource):
    resource_location: Path | str | None = field(
        init=True, default=None, validator=validators.optional(validators.instance_of((Path, str)))
    )
    _data_cache: dict[Any, np.ndarray] = field(init=False, factory=dict, validator=validators.instance_of(dict))
    _file_path: Path | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(Path))
    )
    _file_datasets_shapes: dict[str, tuple[int, ...]] = field(
        init=False, factory=dict, validator=validators.instance_of(dict)
    )
    _file_datasets_dtypes: dict[str, np.dtype] = field(init=False, factory=dict, validator=validators.instance_of(dict))
    _static_metadata_cache: dict[str, Any] = field(init=False, factory=dict, validator=validators.instance_of(dict))
    logger: MessageHandler = field(init=False)

    # source_reference comes from IoSource
    # iosource_method_kwargs comes from IoSource

    def __attrs_post_init__(self):
        # super().__init__(source_reference=source_reference)
        self.logger = MessageHandler(level=self.logging_level, name="HDFSource")
        self._file_path = Path(self.resource_location) if self.resource_location is not None else None
        # self._file_datasets = []
        self._file_datasets_shapes = {}
        self._file_datasets_dtypes = {}
        self._data_cache = {}
        self._static_metadata_cache = {}
        self._preload()  # load the HDF5 file structure immediately so we have some information, but not the data

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
            # self._file_datasets.append(path_name)
            self._file_datasets_shapes[path_name] = path_object.shape
            self._file_datasets_dtypes[path_name] = path_object.dtype

    def get_static_metadata(self, data_key):
        if data_key not in self._static_metadata_cache:
            # if there's an "@" in the key, it's an attribute, we need to split it
            if "@" in data_key:
                dkey, akey = data_key.rsplit("@", 1)
                self._static_metadata_cache[data_key] = self.get_data_attributes(dkey).get(akey, None)
            else:
                try:
                    with h5py.File(self._file_path, "r") as f:
                        value = f[data_key][()]
                        # decode bytes to string if necessary
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        self._static_metadata_cache[data_key] = value
                except OSError as error:
                    _raise_hdf5_read_error(error)
        return self._static_metadata_cache[data_key]

    def get_data(self, data_key: str, load_slice: ArraySlice = ...) -> np.ndarray:
        cache_key = (data_key, _slice_cache_key(load_slice))
        if cache_key not in self._data_cache:
            try:
                with h5py.File(self._file_path, "r") as f:
                    data_array = f[data_key][load_slice]  # if load_slice is not None else f[data_key][()]
                    self._data_cache[cache_key] = np.array(data_array)
            except OSError as error:
                _raise_hdf5_read_error(error)
        return np.array(self._data_cache[cache_key], copy=True)

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
