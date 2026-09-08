from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle


def _basedata() -> BaseData:
    return BaseData(signal=np.array([1.0, 2.0]), units=ureg.dimensionless)


def test_databundle_accepts_mapping_and_keyword_entries():
    signal = _basedata()
    q = _basedata()

    bundle = DataBundle({"signal": signal}, Q=q)

    assert bundle["signal"] is signal
    assert bundle["Q"] is q


def test_databundle_rejects_non_string_keys():
    with pytest.raises(TypeError, match="keys must be strings"):
        DataBundle({1: _basedata()})


def test_databundle_rejects_empty_keys():
    with pytest.raises(ValueError, match="must not be empty"):
        DataBundle({"": _basedata()})


def test_databundle_rejects_non_basedata_values():
    with pytest.raises(TypeError, match="values must be BaseData"):
        DataBundle(signal=np.array([1.0, 2.0]))
