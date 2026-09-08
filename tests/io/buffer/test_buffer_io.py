from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.buffer import BufferSink, BufferSource, RuntimeBufferStore, decode_npy, encode_npy


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
    ],
)
def test_npy_codec_preserves_shape_and_dtype(dtype):
    source = np.arange(12, dtype=dtype).reshape(1, 3, 4)

    decoded = decode_npy(encode_npy(source))

    assert decoded.shape == source.shape
    assert decoded.dtype == source.dtype
    np.testing.assert_array_equal(decoded, source)


def test_runtime_buffer_store_arrays_attrs_metadata_manifest_and_clear():
    store = RuntimeBufferStore()
    store.put_array("s1", "source", "chunk", "sample/signal/signal", np.arange(6, dtype=np.float32).reshape(2, 3))
    store.put_array("s1", "source", "chunk", "sample/mask/signal", np.ones((2, 3), dtype=np.uint32))
    store.put_attrs("s1", "source", "chunk", "sample/signal/signal", {"units": "counts"})
    store.put_metadata("s1", "source", "chunk", "sample/rank", 2)

    manifest = store.manifest("s1", "sources", "chunk")

    assert manifest["arrays"] == ["sample/mask/signal", "sample/signal/signal"]
    assert manifest["attrs"] == ["sample/signal/signal"]
    assert manifest["metadata"] == ["sample/rank"]
    assert manifest["array_details"]["sample/mask/signal"]["dtype"] == "uint32"
    assert store.get_metadata("s1", "source", "chunk", "sample/rank") == 2

    store.put_array("s1", "source", "chunk", "sample/mask/signal", np.zeros((2, 3), dtype=np.uint8))
    assert store.get_array_dtype("s1", "source", "chunk", "sample/mask/signal") == np.dtype("uint8")

    removed = store.clear("s1", kind="source", ref="chunk", data_key="sample/mask/signal")
    assert removed == 1
    assert not store.has_array("s1", "source", "chunk", "sample/mask/signal")

    removed = store.clear("s1", kind="source", ref="chunk")
    assert removed == 3
    assert store.manifest("s1", "source", "chunk")["arrays"] == []


def test_buffer_source_reads_arrays_metadata_and_attrs():
    store = RuntimeBufferStore()
    signal = np.arange(6, dtype=np.float32).reshape(2, 3)
    store.put_array("s1", "source", "chunk", "sample/signal/signal", signal)
    store.put_attrs("s1", "source", "chunk", "sample/signal/signal", {"units": "counts"})
    store.put_metadata("s1", "source", "chunk", "sample/rank", 2)

    source = BufferSource(
        source_reference="chunk",
        resource_location="buffer://session",
        session_id="s1",
        buffer_store=store,
    )

    np.testing.assert_array_equal(source.get_data("sample/signal/signal"), signal)
    np.testing.assert_array_equal(source.get_data("sample/signal/signal", load_slice=np.s_[0]), signal[0])
    assert source.get_data_shape("sample/signal/signal") == (2, 3)
    assert source.get_data_dtype("sample/signal/signal") == np.dtype("float32")
    assert source.get_static_metadata("sample/signal/signal@units") == "counts"
    assert source.get_static_metadata("sample/rank") == 2
    with pytest.raises(KeyError):
        source.get_data("missing")


def test_buffer_sink_writes_basedata_roots_and_leaf_paths():
    store = RuntimeBufferStore()
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(
        signal=np.arange(6, dtype=np.float32).reshape(2, 3),
        units=ureg.Unit("count"),
        uncertainties={"poisson": np.ones((2, 3), dtype=np.float32)},
        weights=np.full((2, 3), 0.5, dtype=np.float32),
        rank_of_data=2,
    )
    bundle["mask"] = BaseData(signal=np.ones((2, 3), dtype=np.int16), units=ureg.dimensionless, rank_of_data=2)
    processing_data["sample"] = bundle

    sink = BufferSink(
        sink_reference="out",
        resource_location="buffer://session",
        session_id="s1",
        buffer_store=store,
    )

    result = sink.write("current", processing_data, data_paths=["/sample/signal", "/sample/mask/signal"])

    assert "current/sample/signal/signal" in result["arrays"]
    np.testing.assert_array_equal(
        store.get_array("s1", "sink", "out", "current/sample/signal/uncertainties/poisson"),
        np.ones((2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        store.get_array("s1", "sink", "out", "current/sample/mask/signal"),
        np.ones((2, 3), dtype=np.int16),
    )
    attrs = store.get_attrs("s1", "sink", "out", "current/sample/signal/signal")
    assert attrs["units"] == "count"
    assert attrs["rank_of_data"] == 2
