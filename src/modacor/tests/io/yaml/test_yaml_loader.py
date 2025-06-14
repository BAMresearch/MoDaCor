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

from pathlib import Path

import numpy as np
import pytest

from ....io.yaml.yaml_loader import YamlLoader

filepath = Path(__file__).parent / "static_data_example.yaml"


def test_yaml_loader_initialization():
    """
    Test the initialization of the YamlLoader class.
    """
    source = YamlLoader("defaults")
    source._load_from_yaml(filepath)
    assert isinstance(source._yaml_data, dict)
    assert isinstance(source._data_cache, dict)


def test_yaml_loader_get_value():
    source = YamlLoader("defaults")
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
