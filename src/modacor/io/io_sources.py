# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "06/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["IoSources"]


from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from attrs import define, field

from modacor.io.io_source import ArraySlice, IoSource


@define
class IoSources:
    """
    IoSources is a collection of al the defined IoSource instances to load data.

    It provides the interface to register new sources and access the different
    data sources through a single interface.
    """

    defined_sources: dict[str, IoSource] = field(factory=dict)
    data_slice_bindings: dict[str, ArraySlice] = field(factory=dict, repr=False)

    @staticmethod
    def normalize_data_reference(data_reference: str) -> str:
        source_ref, separator, data_key = str(data_reference).partition("::")
        if not separator or not source_ref.strip() or not data_key.strip():
            raise ValueError(
                "data_reference must be in the format 'source_ref::data_key' with a "
                "double colon separating both entries."
            )
        return f"{source_ref.strip()}::/{data_key.strip().strip('/')}"

    def set_data_slice_bindings(self, bindings: Mapping[str, ArraySlice]) -> None:
        """Replace request-scoped source slices applied by :meth:`get_data`."""

        normalized: dict[str, ArraySlice] = {}
        for data_reference, load_slice in bindings.items():
            reference = self.normalize_data_reference(data_reference)
            source_ref, _data_key = self.split_data_reference(reference)
            self.get_source(source_ref)
            if load_slice is None or load_slice is Ellipsis:
                raise ValueError(f"Bound data slice for {reference!r} must be explicit.")
            normalized[reference] = load_slice
        self.data_slice_bindings = normalized

    def register_source(self, source: IoSource, source_reference: str | None = None) -> None:
        """
        Register a new source class with the given name. If no source_reference is provided, the
        source's own source_reference attribute will be used.

        Parameters
        ----------
        source : IoSource
            The class of the source to register.
        source_reference : str
            The reference name of the source to register.
        """
        if not isinstance(source, IoSource):
            raise TypeError("source_class must be a subclass of IoSource")
        if source_reference is None:
            source_reference = source.source_reference
        if not isinstance(source_reference, str):
            raise TypeError("source_name must be a string")
        if source_reference in self.defined_sources:
            raise ValueError(f"Source {source_reference} already registered.")
        self.defined_sources[source_reference] = source

    def get_source(self, source_reference: str) -> IoSource:
        """
        Get the source class associated with the given name.

        Parameters
        ----------
        source_reference : str
            The reference name of the source to access.

        Returns
        -------
        IoSource :
            The source class associated with the provided name.
        """
        if source_reference not in self.defined_sources:
            raise KeyError(f"Source {source_reference} not registered.")
        return self.defined_sources[source_reference]

    def split_data_reference(self, data_reference: str) -> tuple[str, str]:
        """
        Split the data reference into source reference and data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.

        Returns
        -------
        tuple[str, str] :
            A tuple containing the source reference and the data key.
        """
        _split = data_reference.split("::", 1)
        if len(_split) != 2:
            raise ValueError(
                "data_reference must be in the format 'source_ref::data_key' with a "
                "double colon separating both entries."
            )
        return _split[0], _split[1]

    def get_data(self, data_reference: str, load_slice: Optional[ArraySlice] = ...) -> np.ndarray:
        """
        Get data from the specified source using the provided data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.
        load_slice : Optional[ArraySlice]
            A slice or tuple of slices to apply to the data. If None or ellipsis, the entire data is returned.

        Returns
        -------
        Any :
            The data associated with the provided key.
        """
        _source_ref, _data_key = self.split_data_reference(data_reference)
        _source = self.get_source(_source_ref)
        bound_slice = self.data_slice_bindings.get(self.normalize_data_reference(data_reference))
        if bound_slice is not None:
            if load_slice is not None and load_slice is not Ellipsis:
                raise ValueError(f"Data reference {data_reference!r} has both a bound and an explicit slice.")
            load_slice = bound_slice
        return _source.get_data(_data_key, load_slice=load_slice)

    def get_data_shape(self, data_reference: str) -> np.ndarray:
        """
        Get data from the specified source using the provided data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.

        Returns
        -------
        Any :
            The data associated with the provided key.
        """
        _source_ref, _data_key = self.split_data_reference(data_reference)
        _source = self.get_source(_source_ref)
        return _source.get_data_shape(_data_key)

    def get_data_dtype(self, data_reference: str) -> np.ndarray:
        """
        Get data from the specified source using the provided data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.

        Returns
        -------
        Any :
            The data associated with the provided key.
        """
        _source_ref, _data_key = self.split_data_reference(data_reference)
        _source = self.get_source(_source_ref)
        return _source.get_data_dtype(_data_key)

    def get_data_attributes(self, data_reference: str) -> np.ndarray:
        """
        Get data from the specified source using the provided data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.
        index : int
            The index to access the data.

        Returns
        -------
        Any :
            The data associated with the provided key.
        """
        _source_ref, _data_key = self.split_data_reference(data_reference)
        _source = self.get_source(_source_ref)
        return _source.get_data_attributes(_data_key)

    def get_static_metadata(self, data_reference: str) -> Any:
        """
        Get static metadata from the specified source using the provided data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.

        Returns
        -------
        Any :
            The static metadata associated with the provided key.
        """
        _source_ref, _data_key = data_reference.split("::", 1)
        _source = self.get_source(_source_ref)
        return _source.get_static_metadata(_data_key)
