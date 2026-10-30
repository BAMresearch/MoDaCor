# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__version__ = "20261030.1"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from collections.abc import Mapping
from typing import Any

from modacor.dataclasses.databundle import DataBundle


class ProcessingData(dict):
    """
    The ProcessingData class is a dictionary-like object that stores reference
    to DataBundles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    @staticmethod
    def _validate_item(key, item):
        if not isinstance(item, DataBundle):
            raise TypeError(f"Expected a DataBundle instance, got {type(item).__name__}.")
        if not isinstance(key, str):
            raise TypeError(f"Expected a string key, got {type(key).__name__}.")
        return key, item

    def setdefault(self, key, default=None):
        key, default = self._validate_item(key, default)
        return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 positional argument, got {len(args)}.")

        items = []
        if args:
            other = args[0]
            if isinstance(other, Mapping):
                items.extend(other.items())
            else:
                items.extend(other)
        items.extend(kwargs.items())
        for key, item in items:
            self[key] = item

    def __ior__(self, other):
        self.update(other)
        return self

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
        key, item = self._validate_item(key, item)
        super().__setitem__(key, item)

    def __repr__(self):
        """
        Print the information of all DataBundles stored in the ProcessingData.
        """
        result = []
        for key in self.keys():
            result.append(f"DataBundle '{key}': contains datasets {list(self[key].keys())}")
            for dkey in self[key].keys():
                result.append(f" Dataset '{dkey}': shape {self[key][dkey].signal.shape}, units {self[key][dkey].units}")
                result.append(f"         available uncertainties: {list(self[key][dkey].uncertainties.keys())}")
        return "\n".join(result)
