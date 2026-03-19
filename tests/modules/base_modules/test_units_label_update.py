# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources

# adjust import if your file lives elsewhere
from modacor.modules.base_modules.units_label_update import UnitsLabelUpdate


def _make_pd() -> ProcessingData:
    pd = ProcessingData()
    db = DataBundle()

    db["intensity_scale_factor"] = BaseData(
        signal=np.array(2.0),
        units="dimensionless",
        uncertainties={"propagate_to_all": np.array(0.05)},
    )
    db["other_factor"] = BaseData(
        signal=np.array([1.0, 2.0, 3.0]),
        units="dimensionless",
        uncertainties={"propagate_to_all": np.array([0.1, 0.1, 0.1])},
    )

    pd["intensity_calibration"] = db
    return pd


def test_units_update_sets_units_for_multiple_keys_without_touching_values():
    pd = _make_pd()
    before_sf = pd["intensity_calibration"]["intensity_scale_factor"].signal.copy()
    before_sf_u = dict(pd["intensity_calibration"]["intensity_scale_factor"].uncertainties)

    before_other = pd["intensity_calibration"]["other_factor"].signal.copy()
    before_other_u = dict(pd["intensity_calibration"]["other_factor"].uncertainties)

    step = UnitsLabelUpdate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["intensity_calibration"],
            "update_pairs": {
                "intensity_scale_factor": {"units": "meter"},
                "other_factor": {"units": "1/second"},
            },
        }
    )
    step.execute(pd)

    sf = pd["intensity_calibration"]["intensity_scale_factor"]
    other = pd["intensity_calibration"]["other_factor"]

    assert sf.units == ureg.Unit("meter")
    assert other.units == ureg.Unit("1/second")

    np.testing.assert_allclose(sf.signal, before_sf)
    np.testing.assert_allclose(other.signal, before_other)
    np.testing.assert_allclose(sf.uncertainties["propagate_to_all"], before_sf_u["propagate_to_all"])
    np.testing.assert_allclose(other.uncertainties["propagate_to_all"], before_other_u["propagate_to_all"])


def test_units_update_accepts_shorthand_string_form():
    pd = _make_pd()

    step = UnitsLabelUpdate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["intensity_calibration"],
            "update_pairs": {
                "intensity_scale_factor": "second",
            },
        }
    )
    step.execute(pd)

    assert pd["intensity_calibration"]["intensity_scale_factor"].units == ureg.Unit("second")
