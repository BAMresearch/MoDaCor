# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.attenuator_plate_correction import AttenuatorPlateCorrection
from modacor.modules.technique_modules.scattering.detector_efficiency_correction import DetectorEfficiencyCorrection
from modacor.modules.technique_modules.scattering.material_attenuation import (
    HC_KEV_ANGSTROM,
    energy_kev_from_config_or_wavelength,
)


def _processing_data() -> ProcessingData:
    pd = ProcessingData()
    pd["sample"] = DataBundle(
        signal=BaseData(
            signal=np.array([10.0, 10.0]),
            units=ureg.count,
            uncertainties={"Poisson": np.array([1.0, 1.0])},
            rank_of_data=1,
        ),
        CosAlpha=BaseData(signal=np.array([1.0, 0.5]), units=ureg.dimensionless, rank_of_data=1),
    )
    return pd


def _run_step(step, pd: ProcessingData, **configuration):
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "linear_attenuation_coefficient": 100.0,
            "linear_attenuation_coefficient_units": "1/m",
            "thickness": 0.01,
            "thickness_units": "m",
            **configuration,
        }
    )
    step.processing_data = pd
    step.calculate()


def test_detector_efficiency_correction_defaults_to_normal_incidence_normalization():
    pd = _processing_data()

    _run_step(DetectorEfficiencyCorrection(io_sources=IoSources()), pd)

    efficiency = 1.0 - np.exp(-1.0 / np.array([1.0, 0.5]))
    expected_divisor = efficiency / efficiency[0]
    np.testing.assert_allclose(pd["sample"]["detector_efficiency"].signal, expected_divisor)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, np.array([10.0, 10.0]) / expected_divisor)
    np.testing.assert_allclose(pd["sample"]["signal"].uncertainties["Poisson"], np.array([1.0, 1.0]) / expected_divisor)


def test_detector_efficiency_correction_can_apply_absolute_efficiency():
    pd = _processing_data()

    _run_step(
        DetectorEfficiencyCorrection(io_sources=IoSources()),
        pd,
        normalize_to_normal_incidence=False,
    )

    expected_divisor = 1.0 - np.exp(-1.0 / np.array([1.0, 0.5]))
    np.testing.assert_allclose(pd["sample"]["detector_efficiency"].signal, expected_divisor)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, np.array([10.0, 10.0]) / expected_divisor)


def test_attenuator_plate_correction_defaults_to_absolute_transmission():
    pd = _processing_data()

    _run_step(AttenuatorPlateCorrection(io_sources=IoSources()), pd)

    expected_divisor = np.exp(-1.0 / np.array([1.0, 0.5]))
    np.testing.assert_allclose(pd["sample"]["attenuator_transmission"].signal, expected_divisor)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, np.array([10.0, 10.0]) / expected_divisor)


def test_attenuator_plate_correction_can_normalize_to_normal_incidence():
    pd = _processing_data()

    _run_step(
        AttenuatorPlateCorrection(io_sources=IoSources()),
        pd,
        normalize_to_normal_incidence=True,
    )

    transmission = np.exp(-1.0 / np.array([1.0, 0.5]))
    expected_divisor = transmission / transmission[0]
    np.testing.assert_allclose(pd["sample"]["attenuator_transmission"].signal, expected_divisor)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, np.array([10.0, 10.0]) / expected_divisor)


def test_attenuator_plate_correction_can_apply_transmission_for_dawn_crosscheck():
    pd = _processing_data()

    _run_step(
        AttenuatorPlateCorrection(io_sources=IoSources()),
        pd,
        apply_as="multiply",
    )

    transmission = np.exp(-1.0 / np.array([1.0, 0.5]))
    np.testing.assert_allclose(pd["sample"]["attenuator_transmission"].signal, transmission)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, np.array([10.0, 10.0]) * transmission)


def test_attenuator_plate_correction_rejects_unknown_apply_as():
    pd = _processing_data()

    with pytest.raises(ValueError, match="apply_as"):
        _run_step(
            AttenuatorPlateCorrection(io_sources=IoSources()),
            pd,
            apply_as="subtract",
        )


def test_material_attenuation_can_derive_energy_from_wavelength():
    energy = energy_kev_from_config_or_wavelength(
        IoSources(),
        {
            "wavelength": 1.0,
            "wavelength_units": "angstrom",
        },
    )

    assert energy == pytest.approx(HC_KEV_ANGSTROM)


def test_detector_efficiency_dependency_contract_tracks_sources_and_cos_alpha():
    step = DetectorEfficiencyCorrection(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "cos_alpha_key": "CosAlpha",
            "material_source": "calibration::/entry/detector/sensor_material",
            "density_source": "calibration::/entry/detector/sensor_density",
            "density_units_source": "calibration::/entry/detector/sensor_density@units",
            "thickness_source": "calibration::/entry/detector/sensor_thickness",
        }
    )

    contract = step.dependency_contract()

    assert isinstance(contract, ProcessStepDependencies)
    assert "calibration" in contract.source_refs
    assert "sample.CosAlpha" in contract.processing_reads
    assert "sample.*" in contract.processing_writes
