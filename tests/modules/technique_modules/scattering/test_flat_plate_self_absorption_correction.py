# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_source import IoSource
from modacor.io.io_sources import IoSources
from modacor.models.attenuation import flat_plate_relative_attenuation
from modacor.modules.technique_modules.scattering.flat_plate_self_absorption_correction import (
    FlatPlateSelfAbsorptionCorrection,
)


class TransmissionSource(IoSource):
    def get_data(self, data_key, load_slice=...):
        if data_key == "/transmission_sem":
            return np.asarray(0.01)
        return np.asarray(0.5)

    def get_data_attributes(self, data_key):
        return {"units": self.get_static_metadata(f"{data_key}@units")}

    def get_static_metadata(self, data_key):
        return "dimensionless"


def _sources() -> IoSources:
    sources = IoSources()
    sources.register_source(TransmissionSource(source_reference="measurement"))
    return sources


def _processing_data(cos_alpha) -> ProcessingData:
    cos_alpha = np.asarray(cos_alpha, dtype=float)
    pd = ProcessingData()
    pd["sample"] = DataBundle(
        signal=BaseData(
            signal=np.full(cos_alpha.shape, 10.0),
            units=ureg.count,
            uncertainties={"Poisson": np.ones(cos_alpha.shape)},
            rank_of_data=cos_alpha.ndim,
        ),
        CosAlpha=BaseData(signal=cos_alpha, units=ureg.dimensionless, rank_of_data=cos_alpha.ndim),
    )
    return pd


def _run(pd: ProcessingData):
    step = FlatPlateSelfAbsorptionCorrection(io_sources=_sources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "transmission_source": "measurement::/transmission",
            "transmission_units_source": "measurement::/transmission@units",
            "transmission_uncertainties_sources": {"transmission_SEM": "measurement::/transmission_sem"},
        }
    )
    step.processing_data = pd
    step.calculate()
    return step


def test_flat_plate_factor_matches_depth_integral():
    cos_alpha = np.array([1.0, 0.8, 0.5])
    pd = _processing_data(cos_alpha)

    _run(pd)

    x = (1.0 / cos_alpha - 1.0) * np.log(0.5)
    expected = np.ones_like(x)
    expected[1:] = np.expm1(x[1:]) / x[1:]
    np.testing.assert_allclose(pd["sample"]["flat_plate_self_absorption"].signal, expected)
    np.testing.assert_allclose(pd["sample"]["signal"].signal, 10.0 / expected)
    np.testing.assert_allclose(pd["sample"]["signal"].uncertainties["Poisson"], 1.0 / expected)
    derivative_x = np.full_like(x, 0.5)
    derivative_x[1:] = (np.exp(x[1:]) * x[1:] - np.expm1(x[1:])) / x[1:] ** 2
    derivative = derivative_x * (1.0 / cos_alpha - 1.0) / 0.5
    expected_sem = np.abs(derivative) * 0.01
    np.testing.assert_allclose(
        pd["sample"]["flat_plate_self_absorption"].uncertainties["transmission_SEM"], expected_sem
    )
    np.testing.assert_allclose(
        pd["sample"]["signal"].uncertainties["transmission_SEM"],
        10.0 * expected_sem / expected**2,
    )


def test_forward_angle_and_unit_transmission_limits_are_one():
    assert flat_plate_relative_attenuation(0.2, np.array([1.0]))[0] == 1.0
    np.testing.assert_array_equal(
        flat_plate_relative_attenuation(1.0, np.array([1.0, 0.5])),
        np.ones(2),
    )


@pytest.mark.parametrize("transmission", [0.0, -0.1, 1.1, np.nan])
def test_invalid_transmission_is_rejected(transmission):
    source = TransmissionSource(source_reference="measurement")
    source.get_data = lambda data_key, load_slice=...: np.asarray(transmission)
    sources = IoSources()
    sources.register_source(source)
    step = FlatPlateSelfAbsorptionCorrection(io_sources=sources)
    step.modify_config_by_dict(
        {"with_processing_keys": ["sample"], "transmission_source": "measurement::/transmission"}
    )
    step.processing_data = _processing_data([1.0])
    with pytest.raises(ValueError, match="transmission"):
        step.calculate()


def test_dependency_contract_tracks_transmission_and_geometry():
    step = FlatPlateSelfAbsorptionCorrection(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "transmission_source": "measurement::/transmission",
            "transmission_units_source": "measurement::/transmission@units",
            "transmission_uncertainties_sources": {"SEM": "statistics::/transmission_sem"},
        }
    )

    contract = step.dependency_contract()

    assert isinstance(contract, ProcessStepDependencies)
    assert contract.source_refs == frozenset({"measurement", "statistics"})
    assert contract.processing_reads == frozenset({"sample.signal", "sample.CosAlpha"})
    assert contract.processing_writes == frozenset({"sample.signal", "sample.flat_plate_self_absorption"})


@pytest.mark.parametrize("shape", [(), (2, 1, 1)])
@pytest.mark.parametrize("uncertainty_key", ["transmission_SEM", "propagate_to_all"])
def test_percent_transmission_matches_dimensionless(monkeypatch, shape, uncertainty_key):
    original_data = TransmissionSource.get_data
    original_metadata = TransmissionSource.get_static_metadata
    results = []
    for units, factor in [("dimensionless", 1.0), ("percent", 100.0)]:

        def get_data(self, key, load_slice=...):
            if key == "/transmission":
                return np.full(shape, 0.5 * factor)
            if key == "/transmission_sem":
                return np.full(shape, 0.01 * factor)
            return original_data(self, key, load_slice)

        def get_metadata(self, key):
            return units if "transmission" in key else original_metadata(self, key)

        monkeypatch.setattr(TransmissionSource, "get_data", get_data)
        monkeypatch.setattr(TransmissionSource, "get_static_metadata", get_metadata)
        data = _processing_data([1.0, 0.8, 0.5])
        step = FlatPlateSelfAbsorptionCorrection(io_sources=_sources(), processing_data=data)
        step.modify_config_by_dict(
            {
                "with_processing_keys": ["sample"],
                "transmission_source": "measurement::/transmission",
                "transmission_units_source": "measurement::/transmission@units",
                "transmission_uncertainties_sources": {uncertainty_key: "measurement::/transmission_sem"},
            }
        )
        step.calculate()
        results.append(data["sample"])
    for key, expected in results[0].items():
        actual = results[1][key]
        np.testing.assert_allclose(actual.signal, expected.signal, rtol=1e-12, atol=0)
        assert actual.uncertainties.keys() == expected.uncertainties.keys()
        for component in expected.uncertainties:
            np.testing.assert_allclose(
                actual.uncertainties[component], expected.uncertainties[component], rtol=1e-12, atol=0
            )
