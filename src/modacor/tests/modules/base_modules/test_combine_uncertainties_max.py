# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.modules.base_modules.combine_uncertainties_max import CombineUncertaintiesMax


def _make_processing_data() -> ProcessingData:
    signal = np.array([[10.0, 12.0], [8.0, 9.0]], dtype=float)
    poisson = 0.1 * np.ones_like(signal)
    readout = np.array([[0.2, 0.05], [0.15, 0.25]])
    background = 0.05 * np.ones_like(signal)

    basedata = BaseData(
        signal=signal,
        units=ureg.Unit("count"),
        uncertainties={
            "poisson": poisson,
            "readout": readout,
            "background": background,
        },
    )

    databundle = DataBundle(signal=basedata)
    processing_data = ProcessingData()
    processing_data["sample"] = databundle
    return processing_data


def test_combine_uncertainties_maximum():
    processing_data = _make_processing_data()
    step = CombineUncertaintiesMax()
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        combinations={"sigma_max": ["poisson", "readout", "background"]},
    )

    step(processing_data)

    bd = processing_data["sample"]["signal"]
    expected = np.maximum.reduce(
        [
            np.broadcast_to(bd.uncertainties["poisson"], bd.signal.shape),
            np.broadcast_to(bd.uncertainties["readout"], bd.signal.shape),
            np.broadcast_to(bd.uncertainties["background"], bd.signal.shape),
        ]
    )
    np.testing.assert_allclose(bd.uncertainties["sigma_max"], expected)


def test_combine_uncertainties_max_drop_sources():
    processing_data = _make_processing_data()
    step = CombineUncertaintiesMax()
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        combinations={"sigma_max": ["poisson", "readout"]},
        drop_source_keys=True,
    )

    step(processing_data)

    bd = processing_data["sample"]["signal"]
    assert "sigma_max" in bd.uncertainties
    assert "poisson" not in bd.uncertainties
    assert "readout" not in bd.uncertainties
    assert "background" in bd.uncertainties


def test_combine_uncertainties_max_missing_sources_error():
    processing_data = _make_processing_data()
    step = CombineUncertaintiesMax()
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        combinations={"sigma_max": ["poisson", "absent_key"]},
    )

    with pytest.raises(KeyError):
        step(processing_data)


def test_combine_uncertainties_max_ignore_missing():
    processing_data = _make_processing_data()
    step = CombineUncertaintiesMax()
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        combinations={"sigma_max": ["poisson", "absent_key"]},
        ignore_missing=True,
    )

    step(processing_data)

    bd = processing_data["sample"]["signal"]
    np.testing.assert_allclose(bd.uncertainties["sigma_max"], bd.uncertainties["poisson"])
