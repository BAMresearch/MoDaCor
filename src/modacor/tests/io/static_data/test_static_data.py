# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__license__ = "BSD3"
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "05/06/2025"
__version__ = "20250505.1"
__status__ = "Development"  # "Development", "Production"

from pathlib import Path

import numpy as np
import pytest

from ....io.static_data.static_data import StaticData

filepath = Path(__file__).parent / "static_data_example.yaml"


def test_static_data_initialization():
    """
    Test the initialization of the StaticData class.
    """
    source = StaticData("defaults")
    source._load_from_yaml(filepath)
    assert isinstance(source._yaml_data, dict)
    assert isinstance(source._data_cache, dict)


def test_static_data_get_value():
    source = StaticData("defaults")
    source._load_from_yaml(filepath)
    # at this point, data_cache should be empty:
    assert source._data_cache == {}
    v = source.get_data("probe_properties/wavelength/value")
    assert isinstance(v, np.ndarray)
    # and now we should have the value in the cache:
    assert "probe_properties/wavelength/value" in source._data_cache
    # this should raise a valueerror as the string cannot be converted to a float array:
    with pytest.raises(ValueError):
        source.get_data("probe_properties/wavelength/unit")
    # but this works:
    assert source.get_static_metadata("probe_properties/wavelength/unit") == "nm"
