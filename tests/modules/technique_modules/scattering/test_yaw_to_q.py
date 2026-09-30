from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules import YawToQ


def test_yaw_to_q_uses_measured_energy_center_and_retains_sign():
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(
        yaw=BaseData(
            np.array([5.0, 10.0, 15.0]),
            ureg.microradian,
            uncertainties={"encoder": np.full(3, 0.1)},
            rank_of_data=1,
        ),
        energy=BaseData(
            np.asarray(14.0),
            ureg.keV,
            uncertainties={"energy": np.asarray(0.01)},
        ),
        beam_center=BaseData(
            np.asarray(10.0),
            ureg.microradian,
            uncertainties={"centre": np.asarray(0.2)},
        ),
    )
    step = YawToQ(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_kwargs(with_processing_keys=["scan"], output_units="1/nm")

    q = step.calculate()["scan"]["Q"]

    wavelength_nm = (ureg.planck_constant * ureg.speed_of_light / (14.0 * ureg.keV)).to("nm").magnitude
    expected = 4.0 * np.pi / wavelength_nm * np.sin(np.array([-5.0, 0.0, 5.0]) * 1.0e-6 / 2.0)
    assert_allclose(q.signal, expected)
    assert q.units == ureg.Unit("1/nm")
    assert set(q.uncertainties) == {"encoder", "centre", "energy"}
    assert q.signal[0] < 0.0 < q.signal[2]


def test_yaw_to_q_configured_zero_omits_center_dependency():
    step = YawToQ(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["scan"],
            "yaw_zero": 0.0,
            "yaw_zero_units": "microradian",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"scan.yaw", "scan.energy"},
        processing_writes={"scan.Q"},
    )


def test_yaw_to_q_center_dependency_is_exact():
    step = YawToQ(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["scan"])

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"scan.yaw", "scan.energy", "scan.beam_center"},
        processing_writes={"scan.Q"},
    )
