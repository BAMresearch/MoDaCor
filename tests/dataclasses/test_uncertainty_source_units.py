# SPDX-License-Identifier: BSD-3-Clause
"""Uncertainty-source units, override labels, and shared material loading."""

import logging

import h5py
import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sources import IoSources
from modacor.io.yaml.yaml_source import YAMLSource
from modacor.modules import AppendProcessingData
from modacor.modules.helpers.scattering.material_attenuation import scalar_quantity_from_config_or_source


def _sources(tmp_path, *, signal_units="nm", uncertainty_units="angstrom", scalar=False):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as h5:
        signal = h5.create_dataset("signal", data=1.0 if scalar else [1.0, 2.0])
        signal.attrs["units"] = signal_units
        sigma = h5.create_dataset("sigma", data=0.005 if scalar else [0.005, 0.01])
        if uncertainty_units is not None:
            sigma.attrs["units"] = (
                np.bytes_(uncertainty_units) if isinstance(uncertainty_units, bytes) else uncertainty_units
            )
    sources = IoSources()
    sources.register_source(HDFSource(source_reference="measurement", resource_location=path))
    return sources


def _load(sources, **kwargs):
    return basedata_from_sources(
        sources,
        "measurement::/signal",
        units_source="measurement::/signal@units",
        uncertainty_sources={"measurement_error": "measurement::/sigma"},
        **kwargs,
    )


@pytest.mark.parametrize("source_units", ["angstrom", b"angstrom"])
def test_uncertainty_units_convert_through_basedata_and_log_at_debug(tmp_path, source_units, caplog):
    sources = _sources(tmp_path, uncertainty_units=source_units)
    with caplog.at_level(logging.DEBUG, logger="modacor.dataclasses.helpers"):
        data = _load(sources)
    np.testing.assert_allclose(data.signal, [1.0, 2.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(data.uncertainties["measurement_error"], [0.0005, 0.001], rtol=1e-12, atol=0)
    assert data.units == ureg.nm
    records = [r for r in caplog.records if r.name == "modacor.dataclasses.helpers"]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG
    assert "measurement_error" in records[0].message and "measurement::/sigma" in records[0].message


@pytest.mark.parametrize("source_units", [None, "", b"", "  ", "nm"])
def test_missing_or_matching_uncertainty_units_keep_values(tmp_path, source_units, caplog):
    sources = _sources(tmp_path, uncertainty_units=source_units)
    with caplog.at_level(logging.DEBUG, logger="modacor.dataclasses.helpers"):
        data = _load(sources)
    np.testing.assert_allclose(data.uncertainties["measurement_error"], [0.005, 0.01], rtol=1e-12, atol=0)
    assert not [r for r in caplog.records if r.name == "modacor.dataclasses.helpers"]


@pytest.mark.parametrize(
    "signal_units, uncertainty_units",
    [
        ("nm", "s"),
        ("nm", "not_a_defined_unit"),
        ("nm", "m/("),
        ("nm", "1/"),
        ("nm", "m s-1"),
        ("K", "degC"),
        ("K", "degF"),
        ("degC", "degC"),
        ("degC", "K"),
    ],
)
def test_invalid_or_offset_uncertainty_units_name_both_units_and_source(tmp_path, signal_units, uncertainty_units):
    sources = _sources(tmp_path, signal_units=signal_units, uncertainty_units=uncertainty_units)
    with pytest.raises(ValueError) as caught:
        _load(sources)
    message = str(caught.value)
    assert "measurement_error" in message and "measurement::/sigma" in message
    assert uncertainty_units in message and str(ureg.Unit(signal_units)) in message
    assert caught.value.__cause__ is not None


def test_unexpected_uncertainty_conversion_error_propagates(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    error = RuntimeError("unexpected conversion failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(BaseData, "magnitudes", fail)
    with pytest.raises(RuntimeError, match="unexpected conversion failure") as caught:
        _load(sources)
    assert caught.value is error


@pytest.mark.parametrize("route", ["helper", "step"])
@pytest.mark.parametrize("override", ["nm", "%", "count/%"])
@pytest.mark.parametrize("scalar", [False, True])
def test_units_override_preserves_values_and_never_reads_attributes(tmp_path, monkeypatch, route, override, scalar):
    sources = _sources(tmp_path, uncertainty_units="not_a_defined_unit", scalar=scalar)

    def fail(*args, **kwargs):
        raise AssertionError("Override must skip source unit metadata and uncertainty attributes")

    monkeypatch.setattr(IoSources, "get_data_attributes", fail)
    monkeypatch.setattr(IoSources, "get_static_metadata", fail)
    if route == "helper":
        data = _load(sources, units_override=override)
    else:
        processing = ProcessingData()
        step = AppendProcessingData(
            io_sources=sources,
            configuration={
                "processing_key": "sample",
                "signal_location": "measurement::/signal",
                "units_location": "measurement::/signal@units",
                "units_override": override,
                "uncertainties_sources": {"measurement_error": "measurement::/sigma"},
                "rank_of_data": 0 if scalar else 1,
            },
        )
        step.execute(processing)
        data = processing["sample"]["signal"]
    assert data.units == ureg.Unit(override)
    np.testing.assert_allclose(data.signal, 1.0 if scalar else [1.0, 2.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        data.uncertainties["measurement_error"], 0.005 if scalar else [0.005, 0.01], rtol=1e-12, atol=0
    )


def test_material_direct_value_branch_converts_uncertainty_units(tmp_path):
    sources = _sources(tmp_path, scalar=True)
    data = scalar_quantity_from_config_or_source(
        sources,
        {"thickness": 2.0, "thickness_units": "nm"},
        "thickness",
        default_units="m",
        uncertainty_sources={"material_error": "measurement::/sigma"},
    )
    np.testing.assert_allclose(data.signal, 2.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(data.uncertainties["material_error"], 0.0005, rtol=1e-12, atol=0)
    assert data.units == ureg.nm


def test_yaml_source_without_attributes_preserves_uncertainty_values(tmp_path):
    path = tmp_path / "source.yaml"
    path.write_text("signal: [1.0, 2.0]\nsigma: [0.005, 0.01]\nunits: nm\n")
    sources = IoSources()
    sources.register_source(YAMLSource(source_reference="measurement", resource_location=path))
    data = basedata_from_sources(
        sources,
        "measurement::signal",
        units_source="measurement::units",
        uncertainty_sources={"measurement_error": "measurement::sigma"},
    )
    np.testing.assert_allclose(data.uncertainties["measurement_error"], [0.005, 0.01], rtol=1e-12, atol=0)
