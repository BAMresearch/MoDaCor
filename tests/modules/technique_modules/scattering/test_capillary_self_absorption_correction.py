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
from modacor.modules.technique_modules.scattering.capillary_self_absorption_correction import (
    CapillarySelfAbsorptionCorrection,
)


class CapillarySource(IoSource):
    def get_data(self, data_key, load_slice=...):
        values = {
            "/transmission": np.asarray(0.5),
            "/transmission_sem": np.asarray(0.01),
            "/beam": np.asarray([[1.0, 2.0], [2.0, 1.0]]),
            "/sample_phase_absorption": np.asarray(1.0 - np.exp(-0.5)),
            "/sample_phase_transmission": np.asarray(np.exp(-0.5)),
            "/sample_phase_thickness": np.asarray(1.0e-3),
            "/sample_phase_factor_sem": np.asarray(2.0e-3),
            "/sample_phase_thickness_sem": np.asarray(1.0e-5),
        }
        return values[data_key]

    def get_static_metadata(self, data_key):
        return "m" if "thickness" in data_key else "dimensionless"


def _sources():
    sources = IoSources()
    sources.register_source(CapillarySource(source_reference="measurement"))
    return sources


def _processing_data(*, masked=True, nan_at_mask=False):
    slow = np.linspace(-0.20, 0.20, 7)
    fast = np.linspace(-0.25, 0.25, 9)
    coord_x, coord_y = np.meshgrid(fast, slow)
    coord_z = np.ones_like(coord_x)
    mask = np.zeros_like(coord_x, dtype=bool)
    if masked:
        mask[3, 4] = True
        if nan_at_mask:
            coord_x[3, 4] = np.nan

    bundle = DataBundle(
        signal=BaseData(
            signal=np.full(coord_x.shape, 10.0),
            units=ureg.count,
            uncertainties={"Poisson": np.ones(coord_x.shape)},
            rank_of_data=2,
        ),
        coord_x=BaseData(signal=coord_x, units=ureg.m, rank_of_data=2),
        coord_y=BaseData(signal=coord_y, units=ureg.m, rank_of_data=2),
        coord_z=BaseData(signal=coord_z, units=ureg.m, rank_of_data=2),
        Mask=BaseData(signal=mask, units=ureg.dimensionless, rank_of_data=2),
    )
    processing = ProcessingData()
    processing["sample"] = bundle
    return processing


def _configuration(**overrides):
    config = {
        "with_processing_keys": ["sample"],
        "sample_radius": 1.0,
        "sample_radius_units": "mm",
        "wall_thickness": 0.1,
        "wall_thickness_units": "mm",
        "sample_mu": 500.0,
        "sample_mu_units": "1/m",
        "wall_mu": 1000.0,
        "wall_mu_units": "1/m",
        "beam_profile": {
            "type": "trapezoid_2d",
            "plateau_width": [0.2, 0.3],
            "ramp_widths": [[0.1, 0.1], [0.1, 0.1]],
            "width_units": "mm",
            "quadrature_order_per_region": 3,
        },
        "chord_order": 6,
        "evaluation_mode": "exact",
        "input_state": "raw",
    }
    config.update(overrides)
    return config


def _run(processing, **overrides):
    step = CapillarySelfAbsorptionCorrection(io_sources=_sources())
    step.modify_config_by_dict(_configuration(**overrides))
    step.processing_data = processing
    step.calculate()
    return step


def test_raw_signal_is_corrected_by_absolute_sample_origin_factor():
    processing = _processing_data(masked=True, nan_at_mask=True)

    _run(processing)

    bundle = processing["sample"]
    attenuation = bundle["capillary_sample_attenuation"].signal
    correction = bundle["capillary_self_absorption"].signal
    active = ~bundle["Mask"].signal
    assert np.all((attenuation[active] > 0.0) & (attenuation[active] < 1.0))
    np.testing.assert_array_equal(attenuation[~active], 1.0)
    np.testing.assert_array_equal(correction, attenuation)
    np.testing.assert_allclose(bundle["signal"].signal[active], 10.0 / attenuation[active])
    np.testing.assert_array_equal(bundle["signal"].signal[~active], 10.0)
    assert 0.0 < float(bundle["capillary_calculated_transmission"].signal) < 1.0
    assert 0.0 < float(bundle["capillary_beam_profile_retained_fraction"].signal) <= 1.0
    assert not np.any(bundle["capillary_attenuation_evaluated"].signal[~active])


def test_transmission_normalized_signal_uses_residual_factor_and_propagates_uncertainty():
    processing = _processing_data(masked=False)

    _run(
        processing,
        input_state="transmission_normalized",
        transmission_source="measurement::/transmission",
        transmission_units_source="measurement::/transmission@units",
        transmission_uncertainties_sources={"transmission_SEM": "measurement::/transmission_sem"},
    )

    bundle = processing["sample"]
    attenuation = bundle["capillary_sample_attenuation"].signal
    correction = bundle["capillary_self_absorption"]
    np.testing.assert_allclose(correction.signal, attenuation / 0.5)
    np.testing.assert_allclose(bundle["signal"].signal, 5.0 / attenuation)
    np.testing.assert_allclose(
        correction.uncertainties["transmission_SEM"],
        attenuation * 0.01 / 0.5**2,
    )
    np.testing.assert_allclose(
        bundle["signal"].uncertainties["transmission_SEM"],
        10.0 * 0.01 / attenuation,
    )


def test_adaptive_mode_matches_exact_and_respects_mask_boundary():
    exact_processing = _processing_data(masked=True)
    adaptive_processing = _processing_data(masked=True)
    _run(exact_processing)

    _run(adaptive_processing, evaluation_mode="adaptive", relative_tolerance=2e-4)

    exact = exact_processing["sample"]["capillary_sample_attenuation"].signal
    adaptive_bundle = adaptive_processing["sample"]
    adaptive = adaptive_bundle["capillary_sample_attenuation"].signal
    active = ~adaptive_bundle["Mask"].signal
    np.testing.assert_allclose(adaptive[active], exact[active], rtol=6e-4, atol=1e-12)
    assert not np.any(adaptive_bundle["capillary_attenuation_evaluated"].signal[~active])


def test_scan_shaped_mask_evaluates_pixels_active_in_any_frame():
    processing = _processing_data(masked=False)
    static_mask = processing["sample"]["Mask"].signal
    scan_mask = np.zeros((2, *static_mask.shape), dtype=bool)
    scan_mask[:, 0, 0] = True
    scan_mask[0, 1, 1] = True
    processing["sample"]["Mask"] = BaseData(
        signal=scan_mask,
        units=ureg.dimensionless,
        rank_of_data=2,
    )

    _run(processing, evaluation_mode="adaptive")

    bundle = processing["sample"]
    attenuation = bundle["capillary_sample_attenuation"].signal
    evaluated = bundle["capillary_attenuation_evaluated"].signal
    assert attenuation[0, 0] == 1.0
    assert not evaluated[0, 0]
    assert attenuation[1, 1] < 1.0


def test_image_profile_can_be_loaded_from_io_source():
    processing = _processing_data(masked=False)

    _run(
        processing,
        beam_profile={
            "type": "image",
            "signal_source": "measurement::/beam",
            "pixel_pitch": [0.1, 0.1],
            "pixel_pitch_units": "mm",
        },
    )

    assert np.all(np.isfinite(processing["sample"]["capillary_sample_attenuation"].signal))


@pytest.mark.parametrize(
    "phase_input",
    [
        {"sample_phase_transmission_source": "measurement::/sample_phase_transmission"},
        {"sample_phase_absorption_source": "measurement::/sample_phase_absorption"},
    ],
)
def test_effective_sample_mu_can_be_derived_from_phase_factor_and_thickness_sources(phase_input):
    direct = _processing_data(masked=False)
    derived = _processing_data(masked=False)
    _run(direct)

    _run(
        derived,
        sample_mu=None,
        sample_phase_thickness_source="measurement::/sample_phase_thickness",
        sample_phase_thickness_units_source="measurement::/sample_phase_thickness@units",
        **phase_input,
    )
    np.testing.assert_allclose(
        derived["sample"]["capillary_sample_attenuation"].signal,
        direct["sample"]["capillary_sample_attenuation"].signal,
        rtol=1e-14,
        atol=1e-14,
    )


def test_derived_mu_uncertainties_propagate_to_factors_and_corrected_signal():
    processing = _processing_data(masked=False)
    _run(
        processing,
        sample_mu=None,
        sample_phase_transmission_source="measurement::/sample_phase_transmission",
        sample_phase_transmission_uncertainties_sources={
            "sample_transmission_SEM": "measurement::/sample_phase_factor_sem"
        },
        sample_phase_thickness_source="measurement::/sample_phase_thickness",
        sample_phase_thickness_units_source="measurement::/sample_phase_thickness@units",
        sample_phase_thickness_uncertainties_sources={
            "sample_thickness_SEM": "measurement::/sample_phase_thickness_sem"
        },
    )

    bundle = processing["sample"]
    resolved_mu = bundle["capillary_effective_sample_mu"]
    transmission = np.exp(-0.5)
    expected_mu_uncertainties = {
        "sample_transmission_SEM": 2.0e-3 / (1.0e-3 * transmission),
        "sample_thickness_SEM": 500.0 * 1.0e-5 / 1.0e-3,
    }
    assert float(resolved_mu.signal) == pytest.approx(500.0)
    for name, expected in expected_mu_uncertainties.items():
        assert float(resolved_mu.uncertainties[name]) == pytest.approx(expected)

    decreased = _processing_data(masked=False)
    increased = _processing_data(masked=False)
    _run(decreased, sample_mu=490.0)
    _run(increased, sample_mu=510.0)
    attenuation = bundle["capillary_sample_attenuation"]
    derivative = (
        increased["sample"]["capillary_sample_attenuation"].signal
        - decreased["sample"]["capillary_sample_attenuation"].signal
    ) / 20.0
    for name, mu_uncertainty in expected_mu_uncertainties.items():
        expected_factor_uncertainty = np.abs(derivative) * mu_uncertainty
        np.testing.assert_allclose(attenuation.uncertainties[name], expected_factor_uncertainty, rtol=1e-13, atol=1e-15)
        expected_signal_uncertainty = 10.0 * expected_factor_uncertainty / attenuation.signal**2
        np.testing.assert_allclose(
            bundle["signal"].uncertainties[name], expected_signal_uncertainty, rtol=1e-13, atol=1e-15
        )
        assert name in bundle["capillary_calculated_transmission"].uncertainties


def test_absorbed_fraction_uncertainty_propagates_to_effective_mu():
    processing = _processing_data(masked=False)
    _run(
        processing,
        sample_mu=None,
        sample_phase_absorption_source="measurement::/sample_phase_absorption",
        sample_phase_absorption_uncertainties_sources={
            "sample_absorption_SEM": "measurement::/sample_phase_factor_sem"
        },
        sample_phase_thickness_source="measurement::/sample_phase_thickness",
        sample_phase_thickness_units_source="measurement::/sample_phase_thickness@units",
    )

    resolved_mu = processing["sample"]["capillary_effective_sample_mu"]
    expected = 2.0e-3 / (1.0e-3 * np.exp(-0.5))
    assert float(resolved_mu.uncertainties["sample_absorption_SEM"]) == pytest.approx(expected)


def test_adaptive_derived_mu_uncertainty_matches_exact_evaluation():
    exact = _processing_data(masked=True)
    adaptive = _processing_data(masked=True)
    uncertainty_configuration = {
        "sample_mu": None,
        "sample_phase_transmission_source": "measurement::/sample_phase_transmission",
        "sample_phase_transmission_uncertainties_sources": {
            "sample_transmission_SEM": "measurement::/sample_phase_factor_sem"
        },
        "sample_phase_thickness_source": "measurement::/sample_phase_thickness",
        "sample_phase_thickness_units_source": "measurement::/sample_phase_thickness@units",
    }
    _run(exact, **uncertainty_configuration)
    _run(
        adaptive,
        evaluation_mode="adaptive",
        relative_tolerance=2e-4,
        **uncertainty_configuration,
    )

    exact_bundle = exact["sample"]
    adaptive_bundle = adaptive["sample"]
    active = ~exact_bundle["Mask"].signal
    for key in ("capillary_sample_attenuation", "signal"):
        np.testing.assert_allclose(
            adaptive_bundle[key].uncertainties["sample_transmission_SEM"][active],
            exact_bundle[key].uncertainties["sample_transmission_SEM"][active],
            rtol=2e-3,
            atol=1e-12,
        )


def test_default_horizontal_axis_gives_fast_axis_symmetric_factor_on_planar_detector():
    processing = _processing_data(masked=False)

    _run(processing)

    attenuation = processing["sample"]["capillary_sample_attenuation"].signal
    np.testing.assert_allclose(attenuation, attenuation[:, ::-1], rtol=1e-14, atol=1e-14)


def test_dependency_contract_tracks_geometry_mask_and_nested_profile_source():
    step = CapillarySelfAbsorptionCorrection(io_sources=IoSources())
    step.modify_config_by_dict(
        _configuration(
            beam_profile={
                "type": "image",
                "signal_source": "measurement::/beam",
                "pixel_pitch": [0.1, 0.1],
                "pixel_pitch_units": "mm",
            },
            transmission_source="measurement::/transmission",
        )
    )

    contract = step.dependency_contract()

    assert isinstance(contract, ProcessStepDependencies)
    assert contract.source_refs == frozenset({"measurement"})
    assert {
        "sample.signal",
        "sample.coord_x",
        "sample.coord_y",
        "sample.coord_z",
        "sample.mask",
        "sample.Mask",
    } <= contract.processing_reads
    assert {
        "sample.signal",
        "sample.capillary_sample_attenuation",
        "sample.capillary_self_absorption",
    } <= contract.processing_writes


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"sample_radius": 0.0}, "sample_radius"),
        ({"wall_thickness": -0.1}, "wall_thickness"),
        ({"sample_mu": -1.0}, "sample_mu"),
        (
            {"sample_phase_transmission": 0.5, "sample_phase_thickness": 1.0},
            "sample_mu or",
        ),
        (
            {"sample_mu": None, "sample_phase_transmission": 0.5},
            "sample_phase_thickness",
        ),
        (
            {
                "sample_mu": None,
                "sample_phase_absorption": 1.0,
                "sample_phase_thickness": 1.0,
            },
            "sample_phase_absorption",
        ),
        ({"beam_profile": {"type": "unknown"}}, "beam_profile"),
        ({"input_state": "transmission_normalized"}, "transmission_source"),
        ({"evaluation_mode": "central_ray"}, "evaluation_mode"),
    ],
)
def test_invalid_configuration_is_rejected(overrides, message):
    processing = _processing_data(masked=False)
    with pytest.raises((ValueError, TypeError), match=message):
        _run(processing, **overrides)
