# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from modacor.io.hdf.hdf_processing_sink import _json_ready
from modacor.io.tiled.tiled_source import TiledSource


def _raise(error):
    raise error("injected failure")


@pytest.mark.parametrize("method", ["to_dict", "item"])
@pytest.mark.parametrize("error", [RuntimeError, TypeError, ValueError])
def test_hdf_metadata_conversion_only_falls_back_for_expected_errors(method, error):
    class Metadata:
        def __str__(self):
            return "metadata"

    setattr(Metadata, method, lambda self: _raise(error))
    if error is RuntimeError:
        with pytest.raises(RuntimeError, match="injected failure"):
            _json_ready(Metadata())
    else:
        assert _json_ready(Metadata()) == "metadata"


@pytest.mark.parametrize("conversion", ["records", "values", "structure"])
@pytest.mark.parametrize("unexpected", [False, True])
def test_tiled_optional_conversion_errors_propagate(conversion, unexpected):
    source = TiledSource(source_reference="test", root_node={})
    expected_error = NotImplementedError if conversion == "structure" else ValueError
    error = RuntimeError if unexpected else expected_error

    class Data:
        def __array__(self, dtype=None, copy=None):
            return np.array([1.0], dtype=dtype)

    if conversion == "records":
        Data.to_records = lambda self, **kwargs: _raise(error)
    elif conversion == "values":

        class InvalidArray:
            def __array__(self, dtype=None, copy=None):
                _raise(error)

        Data.values = InvalidArray()
    else:
        Data.structure = lambda self: _raise(error)

    def call():
        return source._call_structure(Data()) if conversion == "structure" else source._to_numpy(Data())

    if unexpected:
        with pytest.raises(RuntimeError, match="injected failure"):
            call()
    elif conversion == "structure":
        assert call() is None
    else:
        np.testing.assert_allclose(call(), [1.0], rtol=1e-12, atol=0)
