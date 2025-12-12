# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["CSVSource"]

from collections.abc import Callable
from logging import WARNING
from pathlib import Path
from typing import Any

import numpy as np
from attrs import define, field, validators

from modacor.dataclasses.messagehandler import MessageHandler
from modacor.io.io_source import ArraySlice

from ..io_source import IoSource


def _is_callable(_, __, value):
    if not callable(value):
        raise TypeError("method must be callable")


@define(kw_only=True)
class CSVSource(IoSource):
    """
    IoSource for loading columnar data from CSV-like text files using NumPy's
    loadtxt or genfromtxt.

    Expected usage
    --------------
    - Data is 1D per column (no multi-dimensional fields).
    - Columns are returned as 1D arrays; each column corresponds to one data_key.
    - for np.loadtxt, column names must be provided via dtype with field names, e.g.:
        dtype=[("q", float), ("I", float), ("I_sigma", float)]
    - for np.genfromtxt, column names come from the first row or are specified explicitly via the `names` parameter. Typical patterns:
        * np.genfromtxt(..., names=True, delimiter=..., ...)  # use first row as names
        * np.genfromtxt(..., names=["q", "I", "I_sigma"], ...)  # specify names explicitly
      so that they can be clearly identified later.

    Configuration
    -------------
    `iosource_method_kwargs` is passed directly to the NumPy function `method`.
    This allows you to use all standard NumPy options, e.g.:

    For np.genfromtxt:
        delimiter=","
        skip_header=3
        max_rows=1000
        usecols=(0, 1, 2)
        names=True or names=["q", "I", "sigma"]
        dtype=None or dtype=float
        encoding="utf-8"
        comments="#"
        ...

    For np.loadtxt:
        delimiter=","
        skiprows=3
        max_rows=1000
        usecols=(0, 1, 2)
        dtype=float
        encoding="utf-8"
        comments="#"
        ...

    Notes
    -----
    - 2D arrays (no field names) are not supported in this implementation.
      If the resulting array does not have `dtype.names`, a ValueError is raised.
    """

    # external API:
    resource_location: Path = field(converter=Path, validator=validators.instance_of((Path)))
    method: Callable[..., np.ndarray] = field(
        default=np.genfromtxt, validator=_is_callable
    )  # default to genfromtxt, better for names
    # internal use (type hints; real values set per-instance)
    _data_cache: np.ndarray | None = field(init=False, default=None)
    _data_dict_cache: dict[str, np.ndarray] = field(factory=dict)
    _file_datasets_dtypes: dict[str, np.dtype] = field(init=False)
    _file_datasets_shapes: dict[str, tuple[int, ...]] = field(init=False)
    logger: MessageHandler = field(init=False)

    def __attrs_post_init__(self) -> None:
        # super().__init__(source_reference=self.source_reference, iosource_method_kwargs=self.iosource_method_kwargs)
        self.logger = MessageHandler(level=WARNING, name="CSVSource")
        # Set file path
        if not self.resource_location.is_file():
            self.logger.error(f"CSVSource: file {self.resource_location} does not exist.")

        # Bookkeeping structures for IoSource API
        self._file_datasets_shapes: dict[str, tuple[int, ...]] = {}
        self._file_datasets_dtypes: dict[str, np.dtype] = {}

        # Load and preprocess data immediately
        self._load_data()
        self._preload()

    # ------------------------------------------------------------------ #
    # Internal loading / preprocessing                                    #
    # ------------------------------------------------------------------ #

    def _load_data(self) -> None:
        """
        Load the CSV data into a structured NumPy array using the configured
        method (np.genfromtxt or np.loadtxt).

        iosource_method_kwargs are passed directly to that method.
        """
        self.logger.warning(
            f"CSVSource loading data from {self.resource_location} "
            f"using {self.method.__name__} with options: {self.iosource_method_kwargs}"
        )

        try:
            self._data_cache = self.method(self.resource_location, **self.iosource_method_kwargs)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Error while loading CSV data from {self.resource_location}: {exc}")
            raise

        if self._data_cache is None:
            raise ValueError(f"CSVSource: no data loaded from file {self.resource_location}.")
        # Ensure we have a structured array with named fields
        if self._data_cache.dtype.names is None:
            raise ValueError(
                "CSVSource expected a structured array with named fields, "
                "but dtype.names is None.\n"
                "Hint: use np.genfromtxt with 'names=True' or 'names=[...]', "
                "or provide an appropriate 'dtype' with field names."
            )

    def _preload(self) -> None:
        """
        Populate dataset lists, shapes, and dtypes from the structured array.
        """
        assert self._data_cache is not None  # for type checkers

        self._data_dict_cache = {}
        self._file_datasets_shapes.clear()
        self._file_datasets_dtypes.clear()

        for name in self._data_cache.dtype.names:
            column = self._data_cache[name]
            self._data_dict_cache[name] = self._data_cache[name]
            self._file_datasets_shapes[name] = column.shape
            self._file_datasets_dtypes[name] = column.dtype

        self.logger.info(f"CSVSource loaded datasets: {self._file_datasets_shapes.keys()}")

    # ------------------------------------------------------------------ #
    # IoSource API                                                       #
    # ------------------------------------------------------------------ #

    def get_static_metadata(self, data_key: str) -> None:
        """
        CSVSource does not support static metadata; always returns None.
        """
        self.logger.warning(
            f"You asked for static metadata '{data_key}', but CSVSource does not support static metadata."
        )
        return None

    def get_data(self, data_key: str, load_slice: ArraySlice = ...) -> np.ndarray:
        """
        Return the data column corresponding to `data_key`, cast to float, apply `load_slice`.

        - data_key must match one of the field names in the structured array.
        - `load_slice` is applied to that 1D column (e.g. ellipsis, slice, array of indices).
        """
        if self._data_cache is None:
            raise RuntimeError("CSVSource data cache is empty; loading may have failed.")

        try:
            column = self._data_dict_cache[data_key]
        except KeyError:
            raise KeyError(
                f"Data key '{data_key}' not found in CSV data. Available keys: {list(self._data_dict_cache.keys())}"  # noqa: E713
            ) from None

        return np.asarray(column[load_slice]).astype(float)

    def get_data_shape(self, data_key: str) -> tuple[int, ...]:
        if data_key in self._file_datasets_shapes:
            return self._file_datasets_shapes[data_key]
        return ()

    def get_data_dtype(self, data_key: str) -> np.dtype | None:
        if data_key in self._file_datasets_dtypes:
            return self._file_datasets_dtypes[data_key]
        return None

    def get_data_attributes(self, data_key: str) -> dict[str, Any]:
        """
        CSV has no per-dataset attributes; return a dict with None.
        """
        self.logger.warning(
            f"You asked for attributes of '{data_key}', but CSVSource does not support data attributes."
        )
        return {data_key: None}
