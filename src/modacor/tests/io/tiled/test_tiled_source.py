# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest

from modacor.io.tiled.tiled_source import TiledSource


class _DummyStructure:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _DummyLeaf:
    def __init__(self, data: np.ndarray, metadata: dict[str, object] | None = None):
        self._data = np.asarray(data)
        self.metadata = metadata or {}

    def read(self, slice=None):
        if slice is None:
            return self._data
        return self._data[slice]

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def structure(self):
        return _DummyStructure(self._data.shape, self._data.dtype)

    @property
    def attrs(self):
        return self.metadata.get("attrs", {})


class _DummyNode:
    def __init__(self, children: dict[str, object], metadata: dict[str, object] | None = None):
        self._children = children
        self.metadata = metadata or {}

    def __getitem__(self, item: str):
        return self._children[item]

    @property
    def attrs(self):
        return self.metadata.get("attrs", {})


@pytest.fixture
def dummy_source() -> TiledSource:
    root = _DummyNode(
        {
            "entry": _DummyNode(
                {
                    "data": _DummyLeaf(
                        data=np.arange(6).reshape(2, 3),
                        metadata={"attrs": {"units": "counts"}},
                    ),
                    "scalar": _DummyLeaf(np.array(42)),
                },
                metadata={"attrs": {"title": "Example"}},
            )
        }
    )

    return TiledSource(
        source_reference="dummy",
        root_node=root,
        iosource_method_kwargs={"base_item_path": "entry"},
    )


def test_tiled_source_reads_array(dummy_source: TiledSource):
    data = dummy_source.get_data("data")
    np.testing.assert_array_equal(data, np.arange(6).reshape(2, 3))

    cached = dummy_source.get_data("data")
    np.testing.assert_array_equal(cached, data)


def test_tiled_source_slicing(dummy_source: TiledSource):
    sliced = dummy_source.get_data("data", load_slice=np.s_[1, :])
    np.testing.assert_array_equal(sliced, np.array([3, 4, 5]))


def test_tiled_source_shape_dtype(dummy_source: TiledSource):
    assert dummy_source.get_data_shape("data") == (2, 3)
    assert dummy_source.get_data_dtype("data") == np.dtype(int)


def test_tiled_source_attributes(dummy_source: TiledSource):
    attrs = dummy_source.get_data_attributes("data")
    assert attrs == {"units": "counts"}

    assert dummy_source.get_static_metadata("data@units") == "counts"
    # For metadata without explicit attribute name, the full metadata mapping is returned
    metadata = dummy_source.get_static_metadata("data")
    assert metadata == {"attrs": {"units": "counts"}}


def test_tiled_source_resolves_base_metadata(dummy_source: TiledSource):
    assert dummy_source.get_static_metadata("") == {}
    assert dummy_source.get_static_metadata("scalar") == {}
    np.testing.assert_array_equal(dummy_source.get_data("scalar"), np.array(42))
