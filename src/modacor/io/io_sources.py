# SPDX-License-Identifier: BSD-3-Clause


__license__ = "BSD-3-Clause"
__all__ = ["IoSources"]


from typing import Any, Type

import numpy as np
from attrs import define, field

from modacor.io.io_source import IoSource


@define
class IoSources:

    source_registry: dict[str, IoSource] = field(factory=dict)

    def register_source(self, source_reference: str, source: IoSource):
        """
        Register a new source class with the given name.

        Parameters
        ----------
        source_reference : str
            The reference name of the source to register.
        source : Type
            The class of the source to register.
        """
        if not isinstance(source_reference, str):
            raise TypeError("source_name must be a string")
        if not isinstance(source, IoSource):
            raise TypeError("source_class must be a subclass of IoSource")
        self.source_registry[source_reference] = source

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
        if source_reference not in self.source_registry:
            raise ValueError(f"Source {source_reference} not registered.")
        return self.source_registry[source_reference]

    def get_data(self, data_reference: str, index: int | tuple[int]) -> np.ndarray:
        """
        Get data from the specified source using the provided data key.

        The data_reference is composed of the source reference and the internal
        data reference, separated by "::".

        Parameters
        ----------
        data_reference : str
            The reference name of the source to access.
        index : int | tuple[int]
            The index or indices to access the data.

        Returns
        -------
        Any :
            The data associated with the provided key.
        """
        _source_ref, _data_key = data_reference.split("::", 1)
        _source = self.get_source(_source_ref)
        return _source.get_data(index, _data_key)

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
