# SPDX-License-Identifier: BSD-3-Clause

import importlib

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.modules import IndexByCoordinate, PolarizationCorrection

capillary = importlib.import_module("modacor.modules.technique_modules.scattering.capillary_self_absorption_correction")
indexing = importlib.import_module("modacor.modules.base_modules.index_by_coordinate")


@pytest.mark.parametrize("handler", ["length", "rotation", "polarization", "indexing"])
@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_scientific_conversion_handlers_preserve_unexpected_errors(handler, error, monkeypatch):
    def fail(*args, **kwargs):
        raise error("injected failure")

    angle = BaseData(np.array([0.5]), ureg.radian)
    if handler == "length":
        monkeypatch.setattr(BaseData, "magnitudes", fail)

        def call():
            return capillary._length_array([1.0], "m", name="position")

    elif handler == "rotation":
        monkeypatch.setattr(capillary, "scalar_in_units", fail)

        def call():
            return capillary._rotation_radians({})

    elif handler == "polarization":
        monkeypatch.setattr(BaseData, "to_units", fail)

        def call():
            return PolarizationCorrection._angle_signal_radian(angle, name="angle")

    else:
        monkeypatch.setattr(indexing, "scalar_in_units", fail)

        def call():
            return IndexByCoordinate()._edges_for(angle)

    with pytest.raises(error) as caught:
        call()
    if error is RuntimeError:
        assert str(caught.value) == "injected failure"
        assert caught.value.__cause__ is None
    else:
        assert isinstance(caught.value.__cause__, ValueError)
        assert str(caught.value.__cause__) == "injected failure"
