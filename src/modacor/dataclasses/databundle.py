# src/modacor/dataclasses/databundle.py
# -*- coding: utf-8 -*-
__author__ = "Jerome Kieffer"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "21/05/2025"
__version__ = "20250521.1"
__status__ = "Production"  # "Development", "Production"
# end of header and standard imports

from collections.abc import Mapping


class DataBundle(dict):
    """
    DataBundle is a specialized data class for storing related data.
    It contains a dictionary of BaseData data elements, for example Signal,
    a wavelength and flux spectrum, Qx, Qy, Qz, Psi, etc. Process steps can
    add further BaseData objects to this bundle.

    """

    description: str | None = None
    # as per NXcanSAS, tells which basedata to plot
    default_plot: str | None = None

    def __init__(self, *args, **kwargs):
        """
        Create a bundle from BaseData entries.

        Supports normal dict construction styles such as
        ``DataBundle({"signal": bd})`` and ``DataBundle(signal=bd)``.
        """
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    @staticmethod
    def _validate_item(key, value) -> tuple[str, object]:
        from modacor.dataclasses.basedata import BaseData

        if not isinstance(key, str):
            raise TypeError(f"DataBundle keys must be strings, got {type(key).__name__}.")
        if not key:
            raise ValueError("DataBundle keys must not be empty.")
        if not isinstance(value, BaseData):
            raise TypeError(f"DataBundle values must be BaseData instances, got {type(value).__name__}.")
        return key, value

    def __setitem__(self, key, value):
        key, value = self._validate_item(key, value)
        super().__setitem__(key, value)

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

        for key, value in items:
            self[key] = value
