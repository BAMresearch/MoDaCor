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
from modacor.modules.technique_modules.scattering.capillary_self_absorption_correction import (
    CapillarySelfAbsorptionCorrection,
)


class CapillarySource(IoSource):
    def get_data(self, data_key, load_slice=...):
        values = {
            "/sample_mu": np.asarray(500.0),
            "/sample_mu_sem": np.asarray(5.0),
            "/transmission": np.asarray(0.5),
            "/transmission_sem": np.asarray(0.01),
            "/phase_percent": np.asarray(50.0),
            "/beam": np.asarray([[1.0, 2.0], [2.0, 1.0]]),
            "/sample_phase_absorption": np.asarray(1.0 - np.exp(-0.5)),
            "/sample_phase_transmission": np.asarray(np.exp(-0.5)),
            "/sample_phase_thickness": np.asarray(1.0e-3),
            "/sample_phase_factor_sem": np.asarray(2.0e-3),
            "/sample_phase_thickness_sem": np.asarray(1.0e-5),
        }
        return values[data_key]

    def get_static_metadata(self, data_key):
        if "sample_mu" in data_key:
            return "1/m"
        if "phase_percent" in data_key:
            return "percent"
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
        mask=BaseData(signal=mask, units=ureg.dimensionless, rank_of_data=2),
    )
    processing = ProcessingData()
    processing["sample"] = bundle
    return processing


def _configuration(**overrides):
    config = {
        "with_processing_keys": ["sample"],
        "mask_key": "mask",
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


@pytest.mark.parametrize("phase_input", ["absorption", "transmission"])
@pytest.mark.parametrize("shared_name", [False, True])
def test_sample_mu_matches_analytic_uncertainties_and_basedata(phase_input, shared_name):
    factor_name = "shared" if shared_name else "factor"
    thickness_name = "shared" if shared_name else "thickness"
    transmission = np.exp(-0.5)
    factor = 1.0 - transmission if phase_input == "absorption" else transmission
    thickness = BaseData(
        signal=np.asarray(1.0),
        units=ureg.mm,
        uncertainties={thickness_name: np.asarray(0.01)},
    )
    phase = BaseData(
        signal=np.asarray(factor),
        units=ureg.dimensionless,
        uncertainties={factor_name: np.asarray(0.002)},
    )
    basedata_transmission = 1 - phase if phase_input == "absorption" else phase
    basedata_mu = -basedata_transmission.log() / thickness
    basedata_mu.to_units(ureg.Unit("1/m"))
    expected = {factor_name: 0.002 / (0.001 * transmission)}
    thickness_component = 500.0 * 0.00001 / 0.001
    if shared_name:
        expected[factor_name] = np.hypot(expected[factor_name], thickness_component)
    else:
        expected[thickness_name] = thickness_component

    step = CapillarySelfAbsorptionCorrection(io_sources=_sources())
    step.modify_config_by_dict(
        _configuration(
            sample_mu=None,
            **{
                f"sample_phase_{phase_input}_source": f"measurement::/sample_phase_{phase_input}",
                f"sample_phase_{phase_input}_uncertainties_sources": {
                    factor_name: "measurement::/sample_phase_factor_sem",
                },
                "sample_phase_thickness_source": "measurement::/sample_phase_thickness",
                "sample_phase_thickness_units_source": "measurement::/sample_phase_thickness@units",
                "sample_phase_thickness_uncertainties_sources": {
                    thickness_name: "measurement::/sample_phase_thickness_sem",
                },
            },
        )
    )
    resolved = step._sample_mu()
    assert resolved.value == pytest.approx(500.0, rel=1e-14)
    assert float(basedata_mu.signal) == pytest.approx(500.0, rel=1e-14)
    assert set(resolved.uncertainties) == set(basedata_mu.uncertainties) == set(expected)
    for name, uncertainty in expected.items():
        assert resolved.uncertainties[name] == pytest.approx(uncertainty, rel=1e-14)
        assert float(basedata_mu.uncertainties[name]) == pytest.approx(uncertainty, rel=1e-14)


def test_raw_signal_is_corrected_by_absolute_sample_origin_factor():
    processing = _processing_data(masked=True, nan_at_mask=True)

    _run(processing)

    bundle = processing["sample"]
    attenuation = bundle["capillary_sample_attenuation"].signal
    correction = bundle["capillary_self_absorption"].signal
    active = ~bundle["mask"].signal
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
    active = ~adaptive_bundle["mask"].signal
    np.testing.assert_allclose(adaptive[active], exact[active], rtol=6e-4, atol=1e-12)
    assert not np.any(adaptive_bundle["capillary_attenuation_evaluated"].signal[~active])


def test_scan_shaped_mask_evaluates_pixels_active_in_any_frame():
    processing = _processing_data(masked=False)
    static_mask = processing["sample"]["mask"].signal
    scan_mask = np.zeros((2, *static_mask.shape), dtype=bool)
    scan_mask[:, 0, 0] = True
    scan_mask[0, 1, 1] = True
    processing["sample"]["mask"] = BaseData(
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
    active = ~exact_bundle["mask"].signal
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
    assert "sample.Mask" not in contract.processing_reads
    assert {
        "sample.signal",
        "sample.coord_x",
        "sample.coord_y",
        "sample.coord_z",
        "sample.mask",
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


def test_configured_mask_does_not_fall_back_to_capitalized_key():
    processing = _processing_data(masked=True)
    bundle = processing["sample"]
    bundle["Mask"] = bundle.pop("mask")
    step = CapillarySelfAbsorptionCorrection(io_sources=_sources())
    step.modify_config_by_dict(_configuration(mask_key="mask"))
    with pytest.raises(KeyError, match="mask.*sample"):
        step._active_mask(bundle, bundle["Mask"].shape, bundle_key="sample")
    bundle["mask"] = BaseData(np.zeros(bundle["Mask"].shape, dtype=bool), ureg.dimensionless, rank_of_data=2)
    assert np.all(step._active_mask(bundle, bundle["Mask"].shape, bundle_key="sample"))


@pytest.mark.parametrize("present_mask", [False, True])
def test_no_mask_by_default_ignores_present_mask(present_mask):
    processing = _processing_data(masked=True)
    if not present_mask:
        processing["sample"].pop("mask")
    cfg = _configuration()
    cfg.pop("mask_key")
    step = CapillarySelfAbsorptionCorrection(io_sources=_sources(), configuration=cfg)
    assert step.configuration["mask_key"] is None
    step.processing_data = processing
    step.calculate()
    assert processing["sample"]["capillary_sample_attenuation"].signal[3, 4] < 1.0


def test_configured_mask_is_applied():
    processing = _processing_data(masked=True)
    _run(processing, mask_key="mask")
    assert processing["sample"]["capillary_sample_attenuation"].signal[3, 4] == 1.0


def test_missing_configured_mask_names_key_and_bundle():
    with pytest.raises(KeyError, match="excluded.*sample"):
        _run(_processing_data(), mask_key="excluded")


@pytest.mark.parametrize("mask_key", [None, "excluded"])
def test_optional_mask_dependency_contract(mask_key):
    step = CapillarySelfAbsorptionCorrection(io_sources=_sources(), configuration=_configuration(mask_key=mask_key))
    contract = step.dependency_contract()
    assert contract.source_refs == frozenset()
    assert contract.processing_reads == {
        "sample.signal",
        "sample.coord_x",
        "sample.coord_y",
        "sample.coord_z",
    } | ({f"sample.{mask_key}"} if mask_key is not None else set())
    assert contract.processing_writes == {
        "sample.signal",
        "sample.capillary_sample_attenuation",
        "sample.capillary_self_absorption",
        "sample.capillary_calculated_transmission",
        "sample.capillary_effective_sample_mu",
        "sample.capillary_beam_profile_retained_fraction",
        "sample.capillary_attenuation_evaluated",
    }


@pytest.mark.parametrize("step_type", [CapillarySelfAbsorptionCorrection, CapillarySampleContainerCorrection])
@pytest.mark.parametrize("factor", ["absorption", "transmission"])
@pytest.mark.parametrize("from_source", [False, True])
def test_phase_factor_percent_units_convert_to_dimensionless(step_type, factor, from_source):
    name = f"sample_phase_{factor}"
    config = {"sample_phase_thickness": 1.0e-3}
    if from_source:
        config.update(
            {
                f"{name}_source": "measurement::/phase_percent",
                f"{name}_units_source": "measurement::/phase_percent@units",
            }
        )
    else:
        config.update({name: 50.0, f"{name}_units": "percent"})
    step = step_type(io_sources=_sources(), configuration=config)

    resolved = step._sample_mu()

    assert resolved.value == pytest.approx(-np.log(0.5) / 1.0e-3)
    assert resolved.uncertainties == {}
    contract = step.dependency_contract()
    expected_sources = {"measurement"} if from_source else set()
    assert contract.source_refs == frozenset(expected_sources)


@pytest.mark.parametrize("from_source", [False, True])
def test_direct_mu_uncertainty_matches_derived_route(from_source):
    direct = _processing_data(masked=False)
    derived = _processing_data(masked=False)
    nominal = _processing_data(masked=False)
    config = {"sample_mu_uncertainties_sources": {"mu_SEM": "measurement::/sample_mu_sem"}}
    if from_source:
        config.update(
            sample_mu=None,
            sample_mu_source="measurement::/sample_mu",
            sample_mu_units_source="measurement::/sample_mu@units",
        )
    _run(direct, **config)
    _run(
        derived,
        sample_mu=None,
        sample_phase_transmission=float(np.exp(-0.5)),
        sample_phase_thickness_source="measurement::/sample_phase_thickness",
        sample_phase_thickness_units_source="measurement::/sample_phase_thickness@units",
        sample_phase_thickness_uncertainties_sources={"mu_SEM": "measurement::/sample_phase_thickness_sem"},
    )
    _run(nominal)
    for key in ("capillary_effective_sample_mu", "capillary_sample_attenuation", "signal"):
        actual = direct["sample"][key]
        expected = derived["sample"][key]
        # with uncertainty sources the kernel evaluates the nominal μ batched with
        # the perturbed coefficient sets (`np.stack`), which can change floating-point
        # summation order by one ulp
        np.testing.assert_allclose(actual.signal, nominal["sample"][key].signal, rtol=1e-14, atol=0)
        np.testing.assert_allclose(actual.signal, expected.signal, rtol=1e-13)
        np.testing.assert_allclose(
            actual.uncertainties["mu_SEM"], expected.uncertainties["mu_SEM"], rtol=1e-13, atol=1e-15
        )
        assert "mu_SEM" not in nominal["sample"][key].uncertainties
    assert np.any(direct["sample"]["signal"].uncertainties["mu_SEM"] > 0)


def test_dependency_contract_includes_direct_mu_uncertainty_sources():
    step = CapillarySelfAbsorptionCorrection(
        configuration={"sample_mu_uncertainties_sources": {"mu_SEM": "calibration::/mu_sem"}},
    )
    baseline = CapillarySelfAbsorptionCorrection().dependency_contract()
    contract = step.dependency_contract()
    assert contract == ProcessStepDependencies(
        source_refs=frozenset({"calibration"}),
        processing_reads=baseline.processing_reads,
        processing_writes=baseline.processing_writes,
    )


def test_stage0_capillary_residual_components_legacy_fold(caplog):
    processing = _processing_data()
    config = _configuration(
        input_state="transmission_normalized",
        sample_mu_units_source="measurement::/sample_mu@units",
        transmission_units_source="measurement::/transmission@units",
        sample_mu=None,
        sample_mu_source="measurement::/sample_mu",
        sample_mu_uncertainties_sources={"propagate_to_all": "measurement::/sample_mu_sem"},
        transmission_source="measurement::/transmission",
        transmission_uncertainties_sources={"measurement": "measurement::/transmission_sem"},
    )
    step = CapillarySelfAbsorptionCorrection(io_sources=_sources(), configuration=config)
    with caplog.at_level("INFO"):
        step.execute(processing)
    correction = processing["sample"]["capillary_self_absorption"]
    assert set(correction.uncertainties) == {"measurement"}
    assert np.any(correction.uncertainties["measurement"] > 0.0)
    records = [r for r in caplog.records if "legacy fold" in r.message]
    assert len(records) == 1


def test_stage0_correlated_mu_components_fold_together_and_mask_validation_is_atomic(caplog):
    cls = CapillarySampleContainerCorrection
    corrected = BaseData(np.array([4.0, 4.0]), ureg.count, {"propagate_to_all": 3.0})
    filled = BaseData(np.array([10.0, 10.0]), ureg.count)
    empty = BaseData(np.array([2.0, 2.0]), ureg.count)
    with caplog.at_level("INFO"):
        cls._add_correlated_sample_mu_uncertainties(
            corrected,
            filled,
            empty,
            np.full(2, 2.0),
            np.ones(2),
            np.ones(2),
            np.zeros(2),
            np.ones(2),
            {"u": 0.5, "v": 1.0},
        )
    assert set(corrected.uncertainties) == {"u", "v"}
    np.testing.assert_allclose(corrected.uncertainties["u"], np.hypot(1, 3))
    np.testing.assert_allclose(corrected.uncertainties["v"], np.hypot(2, 3))
    assert len([r for r in caplog.records if "legacy fold" in r.message]) == 1
    corrected.uncertainties["propagate_to_all"] = np.ones(2)
    original_signal = corrected.signal.copy()
    with pytest.raises(ValueError, match="propagate_to_all"):
        cls._restore_masked_values(corrected, filled, np.ones(2, dtype=bool))
    np.testing.assert_array_equal(corrected.signal, original_signal)
