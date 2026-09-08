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
from modacor.modules.technique_modules.scattering.polarization_correction import PolarizationCorrection


def _processing_data(*, psi_units=ureg.radian) -> ProcessingData:
    psi = np.array([0.0, np.pi / 2.0])
    if psi_units == ureg.degree:
        psi = np.rad2deg(psi)
    pd = ProcessingData()
    pd["sample"] = DataBundle(
        signal=BaseData(
            signal=np.array([10.0, 10.0]),
            units=ureg.count,
            uncertainties={"Poisson": np.array([1.0, 1.0])},
            rank_of_data=1,
        ),
        TwoTheta=BaseData(signal=np.array([np.pi / 4.0, np.pi / 4.0]), units=ureg.radian, rank_of_data=1),
        Psi=BaseData(signal=psi, units=psi_units, rank_of_data=1),
    )
    return pd


def _run_step(pd: ProcessingData, **configuration) -> None:
    step = PolarizationCorrection(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "polarization_factor": 0.9,
            "polarization_angular_offset": 0.0,
            "polarization_angular_offset_units": "radian",
            **configuration,
        }
    )
    step.processing_data = pd
    step.calculate()


def test_linear_fraction_polarization_correction_matches_formula():
    pd = _processing_data()

    _run_step(pd)

    two_theta = np.array([np.pi / 4.0, np.pi / 4.0])
    psi = np.array([0.0, np.pi / 2.0])
    expected = 0.9 * (1.0 - np.sin(two_theta) ** 2 * np.cos(psi) ** 2) + 0.1 * (
        1.0 - np.sin(two_theta) ** 2 * np.sin(psi) ** 2
    )
    np.testing.assert_allclose(pd["sample"]["polarization_factor_map"].signal, expected)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, np.array([10.0, 10.0]) / expected)
    np.testing.assert_allclose(pd["sample"]["signal"].uncertainties["Poisson"], np.array([1.0, 1.0]) / expected)


def test_linear_fraction_unpolarized_case_reduces_to_standard_factor():
    pd = _processing_data()

    _run_step(pd, polarization_factor=0.5)

    expected = (1.0 + np.cos(np.array([np.pi / 4.0, np.pi / 4.0])) ** 2) / 2.0
    np.testing.assert_allclose(pd["sample"]["polarization_factor_map"].signal, expected)


def test_polarisation_aliases_and_degree_units_are_supported():
    pd = _processing_data(psi_units=ureg.degree)

    _run_step(
        pd,
        polarization_factor=None,
        polarisation_factor=0.9,
        polarisation_angular_offset=90.0,
        polarisation_angular_offset_units="degree",
    )

    two_theta = np.array([np.pi / 4.0, np.pi / 4.0])
    psi = np.array([0.0, np.pi / 2.0]) - np.pi / 2.0
    expected = 0.9 * (1.0 - np.sin(two_theta) ** 2 * np.cos(psi) ** 2) + 0.1 * (
        1.0 - np.sin(two_theta) ** 2 * np.sin(psi) ** 2
    )
    np.testing.assert_allclose(pd["sample"]["polarization_factor_map"].signal, expected)


def test_stokes_mode_is_reserved_but_not_implemented():
    pd = _processing_data()

    with pytest.raises(NotImplementedError):
        _run_step(pd, mode="stokes")


def test_rejects_invalid_factor_and_too_small_correction():
    pd = _processing_data()
    with pytest.raises(ValueError):
        _run_step(pd, polarization_factor=1.5)

    pd = _processing_data()
    pd["sample"]["TwoTheta"] = BaseData(signal=np.array([np.pi / 2.0]), units=ureg.radian, rank_of_data=1)
    pd["sample"]["Psi"] = BaseData(signal=np.array([0.0]), units=ureg.radian, rank_of_data=1)
    pd["sample"]["signal"] = BaseData(signal=np.array([1.0]), units=ureg.count, rank_of_data=1)
    with pytest.raises(ValueError):
        _run_step(pd, polarization_factor=1.0)


def test_dependency_contract_is_exact():
    step = PolarizationCorrection(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "two_theta_key": "TwoTheta",
            "psi_key": "Psi",
            "correction_key": "P",
        }
    )

    contract = step.dependency_contract()

    assert isinstance(contract, ProcessStepDependencies)
    assert contract.source_refs == frozenset()
    assert contract.processing_reads == frozenset({"sample.signal", "sample.TwoTheta", "sample.Psi"})
    assert contract.processing_writes == frozenset({"sample.signal", "sample.P"})
