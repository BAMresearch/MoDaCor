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


__all__ = ["ProcessingData"]
__license__ = "BSD-3-Clause"
__version__ = "0.0.1"

from typing import Any

from modacor.dataclasses.databundle import DataBundle


class ProcessingData(dict):
    """
    The ProcessingData class is a dictionary-like object that stores reference
    to DataBundles.
    """

    def __setitem__(self, key: str, item: DataBundle | Any):
        """
        Assign a value to a dictionary key.

        Parameters
        ----------
        key : str
            The dictionary key.
        item : DataBundle | Any
            The value / object to be added to the dictionary.

        Raises
        ------
        TypeError
            If the item is not an instance of DataBundle.
        TypeError
            If the key is not a string.
        """
        if not isinstance(item, DataBundle):
            raise TypeError(f"Expected a DataBundle instance, got {type(item).__name__}.")
        if not isinstance(key, str):
            raise TypeError(f"Expected a string key, got {type(key).__name__}.")
        super().__setitem__(key, item)

    def __str__(self):
        """
        Print the information of all DataBundles stored in the ProcessingData.
        """
        for key in self.keys():
            print(f"DataBundle '{key}': contains datasets {list(self[key].keys())}")
            for dkey in self[key].keys():
                print(f" Dataset '{dkey}': shape {self[key][dkey].signal.shape}, units {self[key][dkey].units}")
                print(f"         available uncertainties: {list(self[key][dkey].uncertainties.keys())}")
