"""Stage 0 acceptance tests; requirement ids link to the normative specification."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from modacor import ureg
from modacor.dataclasses.basedata import BaseData, dict_signal_converter


def test_t0_06_constructor_and_converter_ownership():
    source = np.full(4, 3.0)
    source.flags.writeable = False
    p = BaseData(source, ureg.m, {"Poisson": source}, weights=source)
    q = BaseData(source, ureg.m, {"Poisson": source})
    converted = dict_signal_converter({"u": source})
    for value in (p.signal, p.weights, p.uncertainties["Poisson"], converted["u"]):
        assert not np.shares_memory(value, source)
        assert value.flags.writeable
    p /= 3
    assert_array_equal(q.signal, source)
    assert_array_equal(q.uncertainties["Poisson"], source)


def test_s0_02_copy_and_conversion_detach_external_arrays():
    p = BaseData(np.ones(3), ureg.m, {"u": np.ones(3)})
    q = p.copy()
    assert not np.shares_memory(q.uncertainties["u"], p.uncertainties["u"])
    external = np.ones(3)
    p.uncertainties["u"] = external
    p.to_units(ureg.cm)
    assert_array_equal(external, np.ones(3))
    assert_allclose(p.uncertainties["u"], 100)
    assert_allclose(q.uncertainties["u"], 1)
