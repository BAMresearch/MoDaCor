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
from modacor.modules.technique_modules.scattering.capillary_sample_container_correction import (
    CapillarySampleContainerCorrection,
)


class CompositeUncertaintySource(IoSource):
    def get_data(self, data_key, load_slice=...):
        values = {
            "/sample_transmission_sem": np.asarray(2.0e-3),
            "/sample_thickness_sem": np.asarray(1.0e-2),
        }
        return values[data_key]


def _processing(filled_signal, empty_signal, *, filled_mask=None, empty_mask=None):
    filled_signal = np.asarray(filled_signal, dtype=float)
    empty_signal = np.asarray(empty_signal, dtype=float)
    slow = np.linspace(-0.20, 0.20, filled_signal.shape[-2])
    fast = np.linspace(-0.25, 0.25, filled_signal.shape[-1])
    coord_x, coord_y = np.meshgrid(fast, slow)
    coord_z = np.ones_like(coord_x)
    if filled_mask is None:
        filled_mask = np.zeros(filled_signal.shape, dtype=bool)
    if empty_mask is None:
        empty_mask = np.zeros(empty_signal.shape, dtype=bool)

    processing = ProcessingData()
    processing["sample"] = DataBundle(
        signal=BaseData(
            signal=filled_signal,
            units=ureg.count / ureg.s,
            uncertainties={"filled_noise": np.full(filled_signal.shape, 0.1)},
            rank_of_data=2,
        ),
        coord_x=BaseData(signal=coord_x, units=ureg.m, rank_of_data=2),
        coord_y=BaseData(signal=coord_y, units=ureg.m, rank_of_data=2),
        coord_z=BaseData(signal=coord_z, units=ureg.m, rank_of_data=2),
        mask=BaseData(signal=filled_mask, units=ureg.dimensionless, rank_of_data=2),
    )
    processing["background"] = DataBundle(
        signal=BaseData(
            signal=empty_signal,
            units=ureg.count / ureg.s,
            uncertainties={"empty_noise": np.full(empty_signal.shape, 0.05)},
            rank_of_data=2,
        ),
        mask=BaseData(signal=empty_mask, units=ureg.dimensionless, rank_of_data=2),
    )
    return processing


def _configuration(**overrides):
    config = {
        "filled_processing_key": "sample",
        "empty_processing_key": "background",
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
    }
    config.update(overrides)
    return config


def _run(processing, **overrides):
    sources = IoSources()
    sources.register_source(CompositeUncertaintySource(source_reference="measurement"))
    step = CapillarySampleContainerCorrection(io_sources=sources)
    step.modify_config_by_dict(_configuration(**overrides))
    step.processing_data = processing
    step.calculate()
    return step


def _factor_maps(shape=(7, 9)):
    processing = _processing(np.zeros(shape), np.zeros(shape))
    _run(processing)
    bundle = processing["sample"]
    return (
        bundle["capillary_sample_attenuation"].signal,
        bundle["capillary_wall_attenuation_filled"].signal,
        bundle["capillary_wall_attenuation_empty"].signal,
    )


def test_raw_composite_correction_recovers_known_sample_signal():
    sample_factor, wall_filled_factor, wall_empty_factor = _factor_maps()
    row, column = np.indices(sample_factor.shape)
    true_sample = 5.0 + 0.2 * row + 0.1 * column
    true_wall = 2.0 + 0.05 * column
    filled_observed = sample_factor * true_sample + wall_filled_factor * true_wall
    empty_observed = wall_empty_factor * true_wall
    processing = _processing(filled_observed, empty_observed)

    _run(processing)

    bundle = processing["sample"]
    np.testing.assert_allclose(bundle["signal"].signal, true_sample, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        bundle["capillary_wall_subtraction_scale"].signal,
        wall_filled_factor / wall_empty_factor,
    )
    assert np.all(wall_filled_factor <= wall_empty_factor)


def test_common_incident_flux_normalization_is_preserved():
    sample_factor, wall_filled_factor, wall_empty_factor = _factor_maps()
    true_sample = np.full(sample_factor.shape, 7.0)
    true_wall = np.full(sample_factor.shape, 2.0)
    filled_raw = sample_factor * true_sample + wall_filled_factor * true_wall
    empty_raw = wall_empty_factor * true_wall
    incident_flux = 5.0
    processing = _processing(filled_raw / incident_flux, empty_raw / incident_flux)

    _run(processing)

    corrected = processing["sample"]["signal"]
    np.testing.assert_allclose(corrected.signal, true_sample / incident_flux, rtol=2e-14, atol=2e-14)


def test_composite_uses_effective_mu_derived_from_sample_phase_transmission():
    direct_processing = _processing(np.ones((5, 7)), np.ones((5, 7)))
    derived_processing = _processing(np.ones((5, 7)), np.ones((5, 7)))
    _run(direct_processing)
    _run(
        derived_processing,
        sample_mu=None,
        sample_phase_transmission=np.exp(-0.5),
        sample_phase_thickness=1.0,
        sample_phase_thickness_units="mm",
    )

    for key in (
        "capillary_sample_attenuation",
        "capillary_wall_attenuation_filled",
        "capillary_wall_subtraction_scale",
    ):
        np.testing.assert_allclose(
            derived_processing["sample"][key].signal,
            direct_processing["sample"][key].signal,
            rtol=1e-14,
            atol=1e-14,
        )


def test_composite_propagates_derived_mu_uncertainty_as_one_correlated_source():
    sample_factor, wall_filled_factor, wall_empty_factor = _factor_maps(shape=(5, 7))
    true_sample = np.full(sample_factor.shape, 7.0)
    true_wall = np.full(sample_factor.shape, 2.0)
    filled_observed = sample_factor * true_sample + wall_filled_factor * true_wall
    empty_observed = wall_empty_factor * true_wall
    processing = _processing(filled_observed, empty_observed)

    _run(
        processing,
        sample_mu=None,
        sample_phase_transmission=np.exp(-0.5),
        sample_phase_transmission_uncertainties_sources={
            "sample_transmission_SEM": "measurement::/sample_transmission_sem"
        },
        sample_phase_thickness=1.0,
        sample_phase_thickness_units="mm",
        sample_phase_thickness_uncertainties_sources={"sample_thickness_SEM": "measurement::/sample_thickness_sem"},
    )

    decreased = _processing(filled_observed, empty_observed)
    increased = _processing(filled_observed, empty_observed)
    _run(decreased, sample_mu=490.0)
    _run(increased, sample_mu=510.0)
    nominal = processing["sample"]
    attenuation_derivative = (
        increased["sample"]["capillary_sample_attenuation"].signal
        - decreased["sample"]["capillary_sample_attenuation"].signal
    ) / 20.0
    wall_derivative = (
        increased["sample"]["capillary_wall_attenuation_filled"].signal
        - decreased["sample"]["capillary_wall_attenuation_filled"].signal
    ) / 20.0
    scale_derivative = wall_derivative / wall_empty_factor
    numerator = filled_observed - nominal["capillary_wall_subtraction_scale"].signal * empty_observed
    corrected_derivative = (
        -scale_derivative * empty_observed / sample_factor - numerator * attenuation_derivative / sample_factor**2
    )
    expected_mu_uncertainties = {
        "sample_transmission_SEM": 2.0e-3 / (1.0e-3 * np.exp(-0.5)),
        "sample_thickness_SEM": 5.0,
    }
    for name, mu_uncertainty in expected_mu_uncertainties.items():
        np.testing.assert_allclose(
            nominal["signal"].uncertainties[name],
            np.abs(corrected_derivative) * mu_uncertainty,
            rtol=1e-13,
            atol=1e-15,
        )
        assert name in nominal["capillary_sample_attenuation"].uncertainties
        assert name in nominal["capillary_wall_attenuation_filled"].uncertainties
        assert name in nominal["capillary_wall_subtraction_scale"].uncertainties
        assert name in nominal["capillary_filled_calculated_transmission"].uncertainties


def test_union_mask_is_applied_and_masked_filled_values_are_preserved():
    shape = (7, 9)
    filled_mask = np.zeros(shape, dtype=bool)
    empty_mask = np.zeros(shape, dtype=bool)
    filled_mask[1, 2] = True
    empty_mask[4, 5] = True
    filled_signal = np.full(shape, 10.0)
    processing = _processing(
        filled_signal,
        np.full(shape, 2.0),
        filled_mask=filled_mask,
        empty_mask=empty_mask,
    )

    _run(processing, evaluation_mode="adaptive")

    bundle = processing["sample"]
    expected_mask = filled_mask | empty_mask
    np.testing.assert_array_equal(bundle["mask"].signal, expected_mask)
    np.testing.assert_array_equal(bundle["signal"].signal[expected_mask], filled_signal[expected_mask])
    for key in (
        "capillary_sample_attenuation_evaluated",
        "capillary_wall_filled_attenuation_evaluated",
        "capillary_wall_empty_attenuation_evaluated",
    ):
        assert not np.any(bundle[key].signal[expected_mask])


def test_scan_masks_evaluate_only_pixels_active_in_both_measurements_together():
    shape = (2, 5, 7)
    filled_mask = np.zeros(shape, dtype=bool)
    empty_mask = np.zeros(shape, dtype=bool)
    filled_mask[:, 0, 0] = True
    empty_mask[:, 0, 0] = True
    filled_mask[0, 1, 1] = True
    empty_mask[1, 1, 1] = True
    processing = _processing(
        np.full(shape, 10.0),
        np.full(shape, 2.0),
        filled_mask=filled_mask,
        empty_mask=empty_mask,
    )

    _run(processing, evaluation_mode="adaptive")

    bundle = processing["sample"]
    evaluated = bundle["capillary_sample_attenuation_evaluated"].signal
    assert not evaluated[0, 0]
    assert not evaluated[1, 1]


def test_calculated_transmission_diagnostics_distinguish_filled_and_empty():
    processing = _processing(np.ones((5, 7)), np.ones((5, 7)))

    _run(processing)

    bundle = processing["sample"]
    filled = float(bundle["capillary_filled_calculated_transmission"].signal)
    empty = float(bundle["capillary_empty_calculated_transmission"].signal)
    assert 0.0 < filled < empty < 1.0
    assert float(bundle["capillary_beam_profile_retained_fraction"].signal) == pytest.approx(1.0)


def test_dependency_contract_tracks_both_signals_geometry_masks_and_sources():
    step = CapillarySampleContainerCorrection(io_sources=IoSources())
    step.modify_config_by_dict(
        _configuration(
            empty_centre_mu_source="measurement::/entry/sample/empty_centre_mu",
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
        "background.signal",
        "background.mask",
    } <= contract.processing_reads
    assert {
        "sample.signal",
        "sample.capillary_sample_attenuation",
        "sample.capillary_wall_attenuation_filled",
        "sample.capillary_wall_attenuation_empty",
    } <= contract.processing_writes


def test_composite_module_does_not_accept_transmission_normalization_inputs():
    step = CapillarySampleContainerCorrection(io_sources=IoSources())

    with pytest.raises(KeyError, match="filled_input_state"):
        step.modify_config_by_dict({"filled_input_state": "transmission_normalized"})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"wall_thickness": 0.0}, "positive wall_thickness"),
        ({"empty_centre_mu": -1.0}, "empty_centre_mu"),
        ({"filled_processing_key": "sample", "empty_processing_key": "sample"}, "must be different"),
        ({"wall_chord_order": 0}, "chord_order"),
        ({"minimum_attenuation_factor": 0.0}, "minimum_attenuation_factor"),
    ],
)
def test_invalid_composite_configuration_is_rejected(overrides, message):
    processing = _processing(np.ones((5, 7)), np.ones((5, 7)))
    with pytest.raises((ValueError, KeyError), match=message):
        _run(processing, **overrides)
