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


def test_tiled_source_returns_copies_of_cached_data_and_attributes(dummy_source: TiledSource):
    data = dummy_source.get_data("data")
    data[:] = -1
    np.testing.assert_array_equal(dummy_source.get_data("data"), np.arange(6).reshape(2, 3))

    attributes = dummy_source.get_data_attributes("data")
    attributes["units"] = "changed"
    assert dummy_source.get_data_attributes("data") == {"units": "counts"}


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
    assert dummy_source.get_static_metadata("") == {"attrs": {"title": "Example"}}
    assert dummy_source.get_static_metadata("@title") == "Example"
    assert dummy_source.get_static_metadata("scalar") == {}
    np.testing.assert_array_equal(dummy_source.get_data("scalar"), np.array(42))


def test_slice_fallback_applies_slice_locally():
    class UnslicedLeaf:
        def read(self):
            return np.arange(6).reshape(2, 3)

    source = TiledSource(root_node={"data": UnslicedLeaf()})
    np.testing.assert_array_equal(source.get_data("data", np.s_[1, :]), [3, 4, 5])


def test_metadata_accepts_read_only_mapping():
    from types import MappingProxyType

    leaf = _DummyLeaf(np.array(1))
    leaf.metadata = MappingProxyType({"attrs": MappingProxyType({"units": "count"})})
    assert TiledSource(root_node={"data": leaf}).get_data_attributes("data") == {"units": "count"}


def test_structure_dtype_without_array_download():
    class DataType:
        def to_numpy_dtype(self):
            return np.dtype("uint16")

    class Structure:
        data_type = DataType()
        shape = (2, 3)

    class Leaf:
        def structure(self):
            return Structure()

    source = TiledSource(root_node={"data": Leaf()})
    assert source.get_data_dtype("data") == np.dtype("uint16")
    assert source.get_data_shape("data") == (2, 3)


def test_client_mapping_does_not_import_tiled(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tiled.client", None)
    root = {}
    assert TiledSource(resource_location={"client": root})._root_node is root
    with pytest.raises(ImportError, match="modacor\\[tiled\\]"):
        TiledSource(resource_location="http://localhost:8000")


def test_clear_cache_refreshes_data():
    leaf = _DummyLeaf(np.array([1]))
    source = TiledSource(root_node={"data": leaf})
    np.testing.assert_array_equal(source.get_data("data"), [1])
    leaf._data = np.array([2])
    source.clear_cache()
    np.testing.assert_array_equal(source.get_data("data"), [2])


@pytest.mark.parametrize(
    "location, constructor, value",
    [
        ("https://example.test/api/v1", "from_uri", "https://example.test/api/v1"),
        ("profile:beamline", "from_profile", "beamline"),
        ("profile://beamline", "from_profile", "beamline"),
        ({"profile": "beamline"}, "from_profile", "beamline"),
        ({"uri": "https://example.test/api/v1"}, "from_uri", "https://example.test/api/v1"),
    ],
)
def test_connection_descriptors(monkeypatch, location, constructor, value):
    import sys
    from types import SimpleNamespace

    calls = []
    root = {}

    def connect(descriptor, **kwargs):
        calls.append((descriptor, kwargs))
        return root

    monkeypatch.setitem(
        sys.modules,
        "tiled.client",
        SimpleNamespace(
            **{
                "from_uri": connect if constructor == "from_uri" else None,
                "from_profile": connect if constructor == "from_profile" else None,
            }
        ),
    )
    source = TiledSource(
        resource_location=location,
        iosource_method_kwargs={"base_path": "entry", "connection_kwargs": {"api_key": "test-key"}},
    )
    assert source._root_node is root
    assert calls == [(value, {"api_key": "test-key"})]
