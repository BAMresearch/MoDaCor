# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2025 MoDaCor Authors
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__license__ = "BSD-3-Clause"
__copyright__ = "Copyright 2025 MoDaCor Authors"
__status__ = "Alpha"
__all__ = ["IoSource"]

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
