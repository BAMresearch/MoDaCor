import pytest

from modacor.io.chunking import UnsupportedSinkCapability
from modacor.io.io_sink import IoSink


# R05: guide requires NotImplementedError for unsupported interface subsets;
# UnsupportedSinkCapability currently derives from RuntimeError instead.
# Keep the guide expectation visible until the maintainer resolves the conflict.
@pytest.mark.xfail(
    strict=True, raises=UnsupportedSinkCapability, reason="R05: guide/runtime unsupported-capability conflict"
)
@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("initialize_chunked", ("/out", None), {}),
        ("initialize_provisional_chunked", ("/out", None), {"input_plan": None}),
        (
            "resolve_provisional_chunked",
            ("/out", None),
            {"provisional_plan": None, "input_plan": None, "plan": None, "chunk": None},
        ),
        ("inspect_provisional_chunked", ("/out",), {"plan": None, "input_plan": None}),
        ("load_provisional_chunked", ("plan-id",), {}),
        ("load_chunked_resolution", ("plan-id",), {}),
        ("write_chunk", ("/out", None), {"plan": None, "chunk": None}),
        ("inspect_chunked", ("/out",), {"plan": None}),
        ("load_chunked_plan", ("plan-id",), {}),
        ("recover_chunked", ("/out",), {"plan": None, "action": "resume"}),
        ("finalize_chunked", ("/out",), {"plan": None}),
    ],
)
def test_optional_sink_methods_follow_guide(method, args, kwargs):
    with pytest.raises(NotImplementedError, match="does not support"):
        getattr(IoSink(), method)(*args, **kwargs)


def test_base_write_requires_implementation():
    with pytest.raises(NotImplementedError, match="implemented in subclasses"):
        IoSink().write("/out")
