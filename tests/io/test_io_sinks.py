import pytest

from modacor.io.io_sink import IoSink
from modacor.io.io_sinks import IoSinks


class RecordingSink(IoSink):
    def write(self, subpath, *args, **kwargs):
        return subpath, args, kwargs


def test_registration_and_routed_write():
    sink = RecordingSink(sink_reference="export")
    sinks = IoSinks()
    sinks.register_sink(sink)
    assert sinks.get_sink("export") is sink
    assert sinks.write_data("export::/out", 42, option=True) == ("/out", (42,), {"option": True})
    sinks.register_sink(sink, sink_reference="alias")
    assert sinks.get_sink("alias") is sink


def test_duplicate_registration_preserves_original_sink():
    sinks = IoSinks()
    original = IoSink(sink_reference="export")
    sinks.register_sink(original)
    with pytest.raises(ValueError, match="Sink export already registered"):
        sinks.register_sink(IoSink(sink_reference="export"))
    assert sinks.get_sink("export") is original


@pytest.mark.parametrize("operation", ["lookup", "write"])
def test_unknown_reference(operation):
    sinks = IoSinks()
    with pytest.raises(KeyError, match="Sink missing not registered"):
        if operation == "lookup":
            sinks.get_sink("missing")
        else:
            sinks.write_data("missing::/out")


@pytest.mark.parametrize("target", ["export", "export:/out", ""])
def test_missing_double_colon_is_rejected(target):
    sinks = IoSinks()
    with pytest.raises(ValueError, match="double colon separator"):
        sinks.split_target_reference(target)
    with pytest.raises(ValueError, match="double colon separator"):
        sinks.write_data(target)


@pytest.mark.parametrize(
    "target,expected",
    [
        ("export::/out", ("export", "/out")),
        ("export::", ("export", "")),
        ("export::/out::leaf", ("export", "/out::leaf")),
    ],
)
def test_split_target_reference(target, expected):
    assert IoSinks().split_target_reference(target) == expected


def test_invalid_registration_types():
    sinks = IoSinks()
    with pytest.raises(TypeError, match="instance of IoSink"):
        sinks.register_sink(object())
    with pytest.raises(TypeError, match="sink_reference must be a string"):
        sinks.register_sink(IoSink(), sink_reference=42)
    assert not sinks.defined_sinks
