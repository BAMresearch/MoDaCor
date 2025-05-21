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

    def register_source(self, source_reference: str, source: IoSource):
        """
        Register a new source class with the given name.

        Parameters
        ----------
        source_reference : str
            The reference name of the source to register.
        source : IoSource
            The class of the source to register.
        """
        if not isinstance(source_reference, str):
            raise TypeError("source_name must be a string")
        if not isinstance(source, IoSource):
            raise TypeError("source_class must be a subclass of IoSource")
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
