# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "06/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["IoSources"]


from typing import Any

import numpy as np
from attrs import define, field

from modacor.io.io_source import IoSource


@define
class IoSources:
    """
    IoSources is a collection of al the defined IoSource instances to load data.

    It provides the interface to register new sources and access the different
    data sources through a single interface.
    """

    defined_sources: dict[str, IoSource] = field(factory=dict)

    def register_source(self, source: IoSource, source_reference: str | None = None):
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

    def get_data(self, data_reference: str, index: int) -> np.ndarray:
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
        return _source.get_data(index, _data_key)

    def get_data_shape(self, data_reference: str, index: int) -> np.ndarray:
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
        return _source.get_data_shape(index, _data_key)

    def get_data_dtype(self, data_reference: str, index: int) -> np.ndarray:
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
        return _source.get_data_dtype(index, _data_key)

    def get_data_attributes(self, data_reference: str, index: int) -> np.ndarray:
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
        return _source.get_data_attributes(index, _data_key)

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
