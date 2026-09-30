from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.geometry.scattering_angle import signed_q_from_angle
from modacor.io.io_sources import IoSources
from modacor.modules import AngleToQ


def _processing_data(*, incident: BaseData, incident_key: str = "energy") -> ProcessingData:
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(
        angle=BaseData(
            np.array([5.0, 10.0, 15.0]),
            ureg.microradian,
            uncertainties={"encoder": np.full(3, 0.1)},
            rank_of_data=1,
        ),
        beam_center=BaseData(
            np.asarray(10.0),
            ureg.microradian,
            uncertainties={"centre": np.asarray(0.2)},
        ),
        **{incident_key: incident},
    )
    return processing_data


def _run_step(processing_data: ProcessingData, **configuration) -> BaseData:
    step = AngleToQ(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_kwargs(with_processing_keys=["scan"], output_units="1/nm", **configuration)
    return step.calculate()["scan"]["Q"]


def test_angle_to_q_uses_energy_center_and_retains_sign() -> None:
    energy = BaseData(
        np.asarray(14.0),
        ureg.keV,
        uncertainties={"energy": np.asarray(0.01)},
    )

    q = _run_step(_processing_data(incident=energy))

    wavelength_nm = (ureg.planck_constant * ureg.speed_of_light / (14.0 * ureg.keV)).to("nm").magnitude
    expected = signed_q_from_angle(np.array([-5.0, 0.0, 5.0]) * 1.0e-6, wavelength_nm)
    assert_allclose(q.signal, expected)
    assert q.units == ureg.Unit("1/nm")
    assert set(q.uncertainties) == {"encoder", "centre", "energy"}
    assert q.signal[0] < 0.0 < q.signal[2]


def test_angle_to_q_accepts_wavelength_and_bragg_angle() -> None:
    wavelength = BaseData(
        np.asarray(0.1),
        ureg.nm,
        uncertainties={"wavelength": np.asarray(1.0e-4)},
    )

    q = _run_step(
        _processing_data(incident=wavelength, incident_key="wavelength"),
        incident_key="wavelength",
        incident_quantity="wavelength",
        angle_convention="bragg_angle",
    )

    expected = signed_q_from_angle(
        np.array([-5.0, 0.0, 5.0]) * 1.0e-6,
        np.asarray(0.1),
        convention="bragg_angle",
    )
    assert_allclose(q.signal, expected)
    assert set(q.uncertainties) == {"encoder", "centre", "wavelength"}


def test_angle_to_q_configured_zero_omits_center_dependency() -> None:
    step = AngleToQ(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["scan"],
            "angle_zero": 0.0,
            "angle_zero_units": "microradian",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"scan.angle", "scan.energy"},
        processing_writes={"scan.Q"},
    )


def test_angle_to_q_center_dependency_is_exact() -> None:
    step = AngleToQ(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["scan"])

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"scan.angle", "scan.energy", "scan.beam_center"},
        processing_writes={"scan.Q"},
    )


def test_angle_to_q_wavelength_dependency_is_exact() -> None:
    step = AngleToQ(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["scan"],
        incident_key="wavelength",
        incident_quantity="wavelength",
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"scan.angle", "scan.wavelength", "scan.beam_center"},
        processing_writes={"scan.Q"},
    )


def test_angle_to_q_rejects_unknown_incident_quantity() -> None:
    step = AngleToQ(io_sources=IoSources())
    step.processing_data = _processing_data(incident=BaseData(np.asarray(14.0), ureg.keV))
    step.modify_config_by_kwargs(with_processing_keys=["scan"], incident_quantity="frequency")

    with pytest.raises(ValueError, match="incident_quantity"):
        step.calculate()
