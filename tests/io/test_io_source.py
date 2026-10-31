import pytest

from modacor.io.io_source import IoSource


@pytest.mark.parametrize(
    "method,kwargs,message",
    [
        ("get_data", {}, "implemented in subclasses"),
        ("get_data", {"load_slice": slice(None)}, "Slicing is not yet implemented"),
        ("get_data_shape", {}, "implemented in subclasses"),
        ("get_data_dtype", {}, "implemented in subclasses"),
        ("get_data_attributes", {}, "implemented in subclasses"),
        ("get_static_metadata", {}, "implemented in subclasses"),
    ],
)
def test_unsupported_source_methods(method, kwargs, message):
    with pytest.raises(NotImplementedError, match=message):
        getattr(IoSource(), method)("/data", **kwargs)
