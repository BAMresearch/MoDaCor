# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "14/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from typing import Any, Optional, Tuple, Union

import numpy as np
from attrs import define, field

# for type hinting of slicing:
Index = Union[int, slice, type(Ellipsis)]
ArraySlice = Union[Index, Tuple[Index, ...]]


def default_config() -> dict[str, Any]:
    """
    Default configuration for the IoSource class.

    Returns
    -------
    dict[str, Any] :
        A dictionary containing the default configuration.
    """
    return {
        "data_rank": 1,
        "data_key": None,
        "data_rank_dims": (0,),
        "metadata_key": None,
        "non_data_slicing": "",
    }


@define
class IoSource:
    """
    IoSource is a base class for all IO sources in the MoDaCor framework.

    It provides access to a specific IO source and its associated methods.

    Required configuration keys are:

        data_rank : int
            The rank of the data.
        data_key : str
            The key to access the data.
        data_rank_dims : tuple[int]
            The dimensions of the data rank.
        non_data_slicing : str
            Slicing information for non-data dimensions. This must be a
            string that can be evaluated to a slice object. Multiple data
            slices can be separated by double semicolon ';;'.
    """

    type_reference = "IoSource"

    configuration: dict[str, Any] = field(factory=default_config)

    def get_data(self, data_key: str, load_slice: Optional[ArraySlice] = None) -> np.ndarray:
        """
        Get data from the IO source using the provided data key.

        Parameters
        ----------
        data_key : str
            The key to access the data, e.g. '/entry1/instrument/detector00/data'.
        load_slice : Optional[ArraySlice]
            A slice or tuple of slices to apply to the data. If None, the entire data is returned.
            Slicing is not yet implemented, so this will raise NotImplementedError if used.
            Consider using the numpy.s_ or numpy.index_exp for simplifying the slicing syntax.

        Returns
        -------
        np.ndarray :
            The data array associated with the provided key. For scalars, this is a 0-d array.
        """
        if load_slice is not None:
            raise NotImplementedError("Slicing is not yet implemented.")
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_data_shape(self, data_key: str) -> Tuple[int, ...]:
        """
        Get the shape of the data from the IO source if the format supports it else empty tuple.

        Parameters
        ----------
        data_key : str
            The key to the data for which the shape is requested.

        Returns
        -------
        Tuple[int, ...] :
            The shape of the data associated with the provided key.
            Returns an empty tuple if nothing available or unsupported.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_data_dtype(self, data_key: str) -> Optional[np.dtype]:
        """
        Get the data type of the data from the IO source if the format supports it else None.

        Parameters
        ----------
        data_key : str
            The key to the data for which the dtype is requested.

        Returns
        -------
        Optional[np.dtype] :
            The data type of the data associated with the provided key.
            Returns None if nothing available or unsupported.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_data_attributes(self, data_key: str) -> dict[str, Any]:
        """
        Get data attributes from the IO source if the format supports it else empty dict.

        Parameters
        ----------
        data_key : str
            The key to the data for which attributes are requested.

        Returns
        -------
        dict[str, Any] :
            The attributes associated with the data.
            Returns an empty dictionary if nothing available or unsupported.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_static_metadata(self, data_key: str) -> Any:
        """
        Get static metadata from the IO source using the provided data key.

        Parameters
        ----------
        data_key : str
            The key to access the metadata.

        Returns
        -------
        Any :
            The static metadata associated with the provided key.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
