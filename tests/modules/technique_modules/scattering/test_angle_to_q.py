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
from modacor.modules import AngleToQ, ConvertUnits


def _processing_data() -> ProcessingData:
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
    )
    return processing_data


class DummyAngleToQ(AngleToQ):
    def __init__(self, *, photon: BaseData, **kwargs):
        super().__init__(**kwargs)
        self._photon = photon

    def _load_photon(self) -> BaseData:
        return self._photon


def _run_step(processing_data: ProcessingData, *, photon: BaseData, **configuration) -> BaseData:
    step = DummyAngleToQ(photon=photon, io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_kwargs(with_processing_keys=["scan"], **configuration)
    return step.calculate()["scan"]["Q"]


def test_angle_to_q_uses_energy_center_and_retains_sign() -> None:
    energy = BaseData(
        np.asarray(14.0),
        ureg.keV,
        uncertainties={"energy": np.asarray(0.01)},
    )

    q = _run_step(_processing_data(), photon=energy)

    wavelength_m = (ureg.planck_constant * ureg.speed_of_light / (14.0 * ureg.keV)).to("m").magnitude
    expected = signed_q_from_angle(np.array([-5.0, 0.0, 5.0]) * 1.0e-6, wavelength_m)
    assert_allclose(q.signal, expected)
    assert q.units == ureg.Unit("1/m")
    assert set(q.uncertainties) == {"encoder", "centre", "energy"}
    assert q.signal[0] < 0.0 < q.signal[2]


def test_angle_to_q_accepts_wavelength_and_bragg_angle() -> None:
    wavelength = BaseData(
        np.asarray(0.1),
        ureg.nm,
        uncertainties={"wavelength": np.asarray(1.0e-4)},
    )

    q = _run_step(
        _processing_data(),
        photon=wavelength,
        angle_convention="bragg_angle",
    )

    expected = signed_q_from_angle(
        np.array([-5.0, 0.0, 5.0]) * 1.0e-6,
        np.asarray(1e-10),
        convention="bragg_angle",
    )
    assert_allclose(q.signal, expected)
    assert set(q.uncertainties) == {"encoder", "centre", "wavelength"}


def test_angle_to_q_configured_zero_omits_center_dependency() -> None:
    step = AngleToQ(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["scan"],
            "photon_source": "measurement::/beam/energy",
            "photon_units_source": "measurement::/beam/energy@units",
            "angle_zero": 0.0,
            "angle_zero_units": "microradian",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        source_refs={"measurement"},
        processing_reads={"scan.angle"},
        processing_writes={"scan.Q"},
    )


def test_angle_to_q_center_dependency_is_exact() -> None:
    step = AngleToQ(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["scan"],
        photon_source="measurement::/beam/wavelength",
        photon_units_source="measurement::/beam/wavelength@units",
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        source_refs={"measurement"},
        processing_reads={"scan.angle", "scan.beam_center"},
        processing_writes={"scan.Q"},
    )


def test_angle_to_q_rejects_non_photon_metadata_units() -> None:
    with pytest.raises(ValueError, match="energy or wavelength units"):
        _run_step(_processing_data(), photon=BaseData(np.asarray(1.0), ureg.second))


def test_output_units_is_removed():
    with pytest.raises(KeyError, match="output_units"):
        AngleToQ(configuration={"output_units": "1/nm"})


def test_convert_units_selects_q_output_units():
    processing = _processing_data()
    q = _run_step(processing, photon=BaseData(0.1, "nm", uncertainties={"wavelength": 0.0001}))
    signal = q.signal.copy()
    uncertainties = {key: value.copy() for key, value in q.uncertainties.items()}
    ConvertUnits(configuration={"with_processing_keys": ["scan"], "units": {"Q": "1/nm"}}).execute(processing)
    assert processing["scan"]["Q"] is q
    assert q.units == ureg.Unit("1/nm")
    assert_allclose(q.signal, signal * 1e-9, rtol=1e-12, atol=0)
    for key, values in uncertainties.items():
        assert_allclose(q.uncertainties[key], values * 1e-9, rtol=1e-12, atol=0)
