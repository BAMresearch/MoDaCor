# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sinks import IoSinks
from modacor.io.tiled import TiledSink, TiledSource


@pytest.fixture
def processing_data():
    bundle = DataBundle()
    bundle["signal"] = BaseData(
        signal=np.arange(6, dtype=np.float32).reshape(2, 3),
        units=ureg.count,
        uncertainties={"poisson": np.ones((2, 3), dtype=np.float32)},
        weights=np.full((2, 3), 0.5, dtype=np.float32),
        rank_of_data=2,
    )
    data = ProcessingData()
    data["sample"] = bundle
    return data


class ArrayNode:
    def __init__(self, array, metadata):
        self.array = np.asarray(array)
        self.metadata = metadata

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def read(self, slice=None):
        return self.array if slice is None else self.array[slice]

    def write(self, array):
        self.array = np.asarray(array)

    def update_metadata(self, metadata):
        self.metadata = metadata


class Container(dict):
    def __init__(self, metadata=None):
        super().__init__()
        self.metadata = metadata or {}

    def create_container(self, key, metadata=None):
        assert key not in self
        self[key] = Container(metadata)
        return self[key]

    def write_array(self, array, key, metadata=None):
        assert key not in self
        self[key] = ArrayNode(array, metadata or {})
        return self[key]

    def update_metadata(self, metadata):
        self.metadata = metadata


def test_sink_registry_round_trip(processing_data):
    root = Container()
    sink = TiledSink(
        sink_reference="out", resource_location={"client": root}, iosink_method_kwargs={"base_path": "results"}
    )
    sinks = IoSinks()
    sinks.register_sink(sink)
    result = sinks.write_data("out::run", processing_data, data_paths=["/sample/signal", "/sample/signal/units"])
    assert len(result["arrays"]) == 3
    assert result["metadata"] == ["run/sample/signal/units"]
    source = TiledSource(root_node=root, iosource_method_kwargs={"base_path": "results/run"})
    np.testing.assert_array_equal(source.get_data("sample/signal/signal"), processing_data["sample"]["signal"].signal)
    np.testing.assert_array_equal(source.get_data("sample/signal/weights"), 0.5)
    np.testing.assert_array_equal(source.get_data("sample/signal/uncertainties/poisson"), 1)
    assert source.get_static_metadata("sample/signal/signal@units") == "count"
    assert source.get_static_metadata("sample/signal/units@value") == "count"


def test_sink_overwrite_requires_opt_in_and_matching_shape(processing_data):
    root = Container()
    sink = TiledSink(root_node=root)
    path = "/sample/signal/signal"
    sink.write("", processing_data, path)
    with pytest.raises(FileExistsError):
        sink.write("", processing_data, path)
    overwrite = TiledSink(root_node=root, iosink_method_kwargs={"overwrite": True})
    processing_data["sample"]["signal"].signal[:] = 10
    overwrite.write("", processing_data, path)
    np.testing.assert_array_equal(root["sample"]["signal"]["signal"].array, 10)
    root["sample"]["signal"]["signal"].array = np.array([1])
    with pytest.raises(ValueError, match="shape, or dtype"):
        overwrite.write("", processing_data, path)
    np.testing.assert_array_equal(root["sample"]["signal"]["signal"].array, [1])


def test_sink_rejects_invalid_paths_before_writing(processing_data):
    root = Container()
    sink = TiledSink(root_node=root)
    with pytest.raises(ValueError, match="data_paths"):
        sink.write("", processing_data, [])
    with pytest.raises(KeyError):
        sink.write("", processing_data, ["/sample/signal", "/missing/signal"])
    assert not root


def test_sink_round_trip_through_tiled_catalog(tmp_path, processing_data):
    pytest.importorskip("tiled")
    pytest.importorskip("sqlalchemy", reason="Install modacor[tiled-tests] for server integration tests")
    from tiled.catalog import in_memory
    from tiled.client import Context, from_context
    from tiled.server.app import build_app

    catalog = in_memory(writable_storage=str(tmp_path / "arrays"))
    with Context.from_app(build_app(catalog)) as context:
        client = from_context(context)
        sink = TiledSink(root_node=client, iosink_method_kwargs={"base_path": "results"})
        sink.write("run", processing_data, ["/sample/signal", "/sample/signal/units"])
        source = TiledSource(root_node=client, iosource_method_kwargs={"base_path": "results/run"})
        assert source.get_data_shape("sample/signal/signal") == (2, 3)
        assert source.get_data_dtype("sample/signal/signal") == np.dtype("float32")
        assert source.get_static_metadata("sample/signal/signal@units") == "count"
        assert source.get_static_metadata("sample/signal/units@value") == "count"
        for leaf in ("signal", "weights", "uncertainties/poisson"):
            expected = (
                processing_data["sample"]["signal"].uncertainties["poisson"]
                if leaf.startswith("uncertainties")
                else getattr(processing_data["sample"]["signal"], leaf)
            )
            np.testing.assert_array_equal(source.get_data(f"sample/signal/{leaf}"), expected)
        np.testing.assert_array_equal(source.get_data("sample/signal/signal", np.s_[1, :]), [3, 4, 5])
        with pytest.raises(FileExistsError):
            sink.write("run", processing_data, "/sample/signal/signal")
        processing_data["sample"]["signal"].signal[:] = 12
        TiledSink(root_node=client, iosink_method_kwargs={"base_path": "results", "overwrite": True}).write(
            "run", processing_data, "/sample/signal/signal"
        )
        source.clear_cache()
        np.testing.assert_array_equal(source.get_data("sample/signal/signal"), 12)
