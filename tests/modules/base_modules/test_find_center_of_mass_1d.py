from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules import FindCenterOfMass1D


def test_find_center_of_mass_uses_contiguous_window_and_propagates_uncertainties():
    yaw = BaseData(
        np.array([-20.0, -1.0, 0.0, 1.0, 20.0]),
        ureg.microradian,
        uncertainties={"encoder": np.full(5, 0.1)},
        rank_of_data=1,
    )
    signal = BaseData(
        np.array([3.0, 4.0, 10.0, 6.0, 3.0]),
        ureg.count,
        uncertainties={"SEM": np.ones(5)},
        rank_of_data=1,
    )
    processing_data = ProcessingData()
    processing_data["rear"] = DataBundle(signal=signal, yaw=yaw)
    step = FindCenterOfMass1D(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["rear"],
            "half_width": 2.0,
            "width_units": "microradian",
            "maximum_iterations": 4,
        }
    )

    output = step.calculate()["rear"]

    assert_allclose(output["beam_center"].signal, 0.1)
    assert_allclose(output["centroid_count"].signal, 3)
    assert bool(output["centroid_converged"].signal)
    assert_allclose(output["centroid_window_min"].signal, -1.0)
    assert_allclose(output["centroid_window_max"].signal, 1.0)
    assert set(output["beam_center"].uncertainties) == {"signal:SEM", "axis:encoder"}
    assert output["beam_center"].units == ureg.microradian


def test_find_center_of_mass_dependency_contract_is_exact():
    step = FindCenterOfMass1D(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["low_rear", "high_rear"],
            "axis_key": "yaw",
            "mask_key": "Mask",
            "output_key": "yaw_zero",
            "diagnostic_prefix": "centre",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={
            "low_rear.signal",
            "low_rear.yaw",
            "low_rear.Mask",
            "high_rear.signal",
            "high_rear.yaw",
            "high_rear.Mask",
        },
        processing_writes={
            "low_rear.yaw_zero",
            "low_rear.centre_count",
            "low_rear.centre_converged",
            "low_rear.centre_window_min",
            "low_rear.centre_window_max",
            "high_rear.yaw_zero",
            "high_rear.centre_count",
            "high_rear.centre_converged",
            "high_rear.centre_window_min",
            "high_rear.centre_window_max",
        },
    )
