# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.trace_event import _to_jsonable
from modacor.io.io_sources import IoSources


@pytest.mark.parametrize("source", [None, "", "  "])
def test_basedata_from_sources_requires_signal_before_io(source):
    with pytest.raises(ValueError, match="^signal_source is required$"):
        basedata_from_sources(IoSources(), source)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError])
def test_trace_scalar_conversion_only_falls_back_for_expected_errors(error):
    class Metadata:
        def item(self):
            raise error("injected failure")

        def __str__(self):
            return "metadata"

    if error is RuntimeError:
        with pytest.raises(RuntimeError, match="injected failure"):
            _to_jsonable(Metadata())
    else:
        assert _to_jsonable(Metadata()) == "metadata"


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, IndexError])
def test_indexed_weights_only_fall_back_for_expected_errors(monkeypatch, error):
    data = BaseData(np.array([1.0, 2.0]), ureg.count, weights=np.array([0.5, 1.0]))
    weights = data.weights
    asarray = np.asarray

    def convert(value, *args, **kwargs):
        if value is weights:
            raise error("injected failure")
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", convert)
    if error is RuntimeError:
        with pytest.raises(RuntimeError, match="injected failure"):
            data.indexed(slice(None))
    else:
        result = data.indexed(slice(None))
        np.testing.assert_allclose(result.signal, data.signal, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.weights, 1.0, rtol=1e-12, atol=0)
