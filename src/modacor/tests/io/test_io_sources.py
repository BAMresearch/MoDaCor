# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest

from modacor.io import IoSources
from modacor.io.io_source import IoSource

data_shape = (20, 20)


class OnesSource(IoSource):
    type_reference = "ones_source"

    def __attrs_post_init__(self):
        self.configuration["shape"] = data_shape

    def get_data(self, index, key):
        return np.ones(self.configuration["shape"])

    def get_metadata(self, key):
        return key


class IndexSource(IoSource):
    type_reference = "index_source"

    def __attrs_post_init__(self):
        self.configuration["shape"] = data_shape

    def get_data(self, index, key):
        return index * np.ones(self.configuration["shape"])

    def get_metadata(self, key):
        return key


@pytest.fixture
def io_sources():
    return IoSources()


def configure_test_sources(io_sources):
    _source_30x10 = OnesSource()
    _source_30x10.configuration["shape"] = (30, 10)
    io_sources.register_source(source_reference="ones_30x10", source=_source_30x10)

    _source_20x20 = OnesSource()
    _source_20x20.configuration["shape"] = (20, 20)
    io_sources.register_source(source_reference="ones_20x20", source=_source_20x20)

    _index_source_25x25 = IndexSource()
    _index_source_25x25.configuration["shape"] = (25, 25)
    io_sources.register_source(source_reference="index_25x25", source=_index_source_25x25)
    return io_sources


@pytest.mark.parametrize("ref", [42, ["a", "ref2"]])
def test_register_source__wrong_ref_type(io_sources, ref):
    with pytest.raises(TypeError):
        io_sources.register_source(source_reference=ref, source=OnesSource())


@pytest.mark.parametrize("source", [IoSource, object()])
def test_register_source__wrong_source_type(io_sources, source):
    with pytest.raises(TypeError):
        io_sources.register_source(source_reference="source", source=source)


def test_register_source__duplicate_ref(io_sources):
    io_sources.register_source(source_reference="source", source=OnesSource())
    with pytest.raises(ValueError):
        io_sources.register_source(source_reference="source", source=OnesSource())


def test_register_source__valid(io_sources):
    _source = OnesSource()
    io_sources.register_source(source_reference="ones", source=_source)
    assert "ones" in io_sources.defined_sources
    assert io_sources.defined_sources["ones"] == _source


def test_get_source__invalid(io_sources):
    with pytest.raises(KeyError):
        io_sources.get_source("invalid_source")


def test_get_source__valid(io_sources):
    io_sources = configure_test_sources(io_sources)
    for _key in io_sources.defined_sources:
        source = io_sources.get_source(_key)
        assert isinstance(source, IoSource)


def test_split_data_reference__invalid(io_sources):
    with pytest.raises(ValueError):
        io_sources.split_data_reference("invalid_reference")


@pytest.mark.parametrize("ref", ["test::ref", "test::ref::extra", "test::/entry/data", "test::/entry/data::extra"])
def test_split_data_reference__valid(io_sources, ref):
    source_ref, data_key = io_sources.split_data_reference(ref)
    assert source_ref == "test"
    assert data_key == ref.lstrip("test::")


if __name__ == "__main__":
    pytest.main()
