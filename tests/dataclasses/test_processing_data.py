from __future__ import annotations

import operator

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData

INSERTIONS = [
    lambda data, key, value: data.__init__({key: value}),
    lambda data, key, value: data.__init__([(key, value)]),
    lambda data, key, value: data.update({key: value}),
    lambda data, key, value: data.update([(key, value)]),
    lambda data, key, value: data.setdefault(key, value),
    operator.setitem,
]


@pytest.mark.parametrize("route", range(8))
@pytest.mark.parametrize("key,value,message", [(1, DataBundle(), "string key"), ("sample", 1, "DataBundle instance")])
def test_all_insertion_routes_validate(route, key, value, message):
    data = ProcessingData()
    with pytest.raises(TypeError, match=message):
        if route == 6:
            data |= {key: value}
        elif route == 7:
            ProcessingData.fromkeys([key], value)
        else:
            INSERTIONS[route](data, key, value)
    assert not data


@pytest.mark.parametrize("route", ["constructor", "update"])
def test_keyword_insertions_validate(route):
    with pytest.raises(TypeError, match="DataBundle instance"):
        if route == "constructor":
            ProcessingData(sample=1)
        else:
            ProcessingData().update(sample=1)


def test_valid_construction_and_mutation_preserve_bundles():
    bundle = DataBundle()
    data = ProcessingData({"sample": bundle}, background=bundle)
    data.update([("extra", bundle)], other=bundle)
    data.update({"mapped": bundle})
    data.update()
    assert data.setdefault("sample", DataBundle()) is bundle
    assert data.setdefault("new", bundle) is bundle
    assert operator.ior(data, {"union": bundle}) is data
    assert all(value is bundle for value in data.values())
    assert len(data) == 7
    # Preserve the existing string-key contract, including an empty key.
    data[""] = bundle


def test_setdefault_validates_even_when_key_exists():
    data = ProcessingData(sample=DataBundle())
    with pytest.raises(TypeError, match="DataBundle instance"):
        data.setdefault("sample", 1)
    with pytest.raises(TypeError, match="DataBundle instance"):
        data.setdefault("new")


@pytest.mark.parametrize("method", ["constructor", "update"])
def test_too_many_positional_arguments(method):
    with pytest.raises(TypeError, match="at most 1"):
        if method == "constructor":
            ProcessingData({}, {})
        else:
            ProcessingData().update({}, {})


def test_repr_describes_datasets_and_uncertainties():
    data = ProcessingData(
        sample=DataBundle(
            signal=BaseData(
                signal=np.ones((2, 3)),
                units=ureg.count,
                uncertainties={"poisson": np.ones((2, 3))},
                rank_of_data=2,
            )
        )
    )
    assert repr(ProcessingData()) == ""
    assert repr(data) == (
        "DataBundle 'sample': contains datasets ['signal']\n"
        " Dataset 'signal': shape (2, 3), units count\n"
        "         available uncertainties: ['poisson']"
    )
