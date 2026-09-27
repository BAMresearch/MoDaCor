"""Tests for analyser-angle USAXS geometry construction."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.usaxs.xs_geometry_from_analyser_angle import (
    XSGeometryFromAnalyserAngle,
)


def _run_step(processing_data: ProcessingData, **configuration) -> None:
    """Execute the geometry module using a referenced wavelength."""
    wavelength = processing_data["scan"].get(
        "wavelength",
        BaseData(
            signal=np.asarray(0.088560141738),
            units="nanometer",
            rank_of_data=0,
        ),
    )
    if "psi" not in processing_data["scan"]:
        processing_data["scan"]["psi"] = BaseData(signal=np.asarray(0.0), units="degree", rank_of_data=0)
    step = XSGeometryFromAnalyserAngle(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["scan"],
        angle_key="yaw",
        wavelength_source="calibration::/wavelength",
        psi_key="psi",
        q_units="1 / nanometer",
        **configuration,
    )
    with patch.object(step, "_load_from_sources", return_value=wavelength):
        step(processing_data)


def test_applies_analyser_zero_and_preserves_signed_wing_coordinate() -> None:
    """Produce standard geometry fields without replacing the raw angle."""
    yaw = np.asarray([-3.0, 1.0, 5.0])
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(
        yaw=BaseData(signal=yaw, units="microradian", rank_of_data=1),
        peak_center=BaseData(signal=np.asarray(1.0), units="microradian", rank_of_data=0),
        angular_calibration=BaseData(signal=np.asarray(2.0), units="dimensionless", rank_of_data=0),
        psi=BaseData(signal=np.asarray(90.0), units="degree", rank_of_data=0),
    )

    _run_step(
        processing_data,
        angular_zero_key="peak_center",
        angular_multiplier_key="angular_calibration",
        two_theta_units="microradian",
    )

    bundle = processing_data["scan"]
    np.testing.assert_array_equal(bundle["yaw"].signal, yaw)
    np.testing.assert_allclose(bundle["TwoTheta"].signal, [-8.0, 0.0, 8.0])
    np.testing.assert_allclose(bundle["Psi"].signal, np.pi / 2.0)

    wavelength_nm = 0.088560141738
    expected_signed_q = 4.0 * np.pi / wavelength_nm * np.sin(np.asarray([-8.0, 0.0, 8.0]) * 1.0e-6 / 2.0)
    np.testing.assert_allclose(bundle["signed_q"].signal, expected_signed_q)
    np.testing.assert_allclose(bundle["Q"].signal, np.abs(expected_signed_q))
    assert float(bundle["applied_angular_zero"].signal) == pytest.approx(1.0)
    assert float(bundle["applied_angular_multiplier"].signal) == pytest.approx(2.0)


def test_propagates_angle_zero_multiplier_and_wavelength_uncertainties_separately() -> None:
    """Retain each independent calibration contribution through the q calculation."""
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(
        yaw=BaseData(
            signal=np.asarray([3.0]),
            units="microradian",
            uncertainties={"encoder_SEM": np.asarray([0.1])},
            rank_of_data=1,
        ),
        zero=BaseData(
            signal=np.asarray(1.0),
            units="microradian",
            uncertainties={"zero_SEM": np.asarray(0.2)},
            rank_of_data=0,
        ),
        wavelength=BaseData(
            signal=np.asarray(0.1),
            units="nanometer",
            uncertainties={"wavelength_SEM": np.asarray(0.002)},
            rank_of_data=0,
        ),
        angular_calibration=BaseData(
            signal=np.asarray(1.5),
            units="dimensionless",
            uncertainties={"angular_multiplier_SEM": np.asarray(0.015)},
            rank_of_data=0,
        ),
    )

    _run_step(
        processing_data,
        angular_zero_key="zero",
        angular_multiplier_key="angular_calibration",
    )

    bundle = processing_data["scan"]
    assert set(bundle["TwoTheta"].uncertainties) == {
        "encoder_SEM",
        "zero_SEM",
        "angular_multiplier_SEM",
    }
    assert set(bundle["signed_q"].uncertainties) == {
        "encoder_SEM",
        "zero_SEM",
        "angular_multiplier_SEM",
        "wavelength_SEM",
    }
    assert set(bundle["Q"].uncertainties) == set(bundle["signed_q"].uncertainties)
    np.testing.assert_allclose(
        bundle["Q"].uncertainties["wavelength_SEM"],
        np.abs(bundle["Q"].signal) * 0.02,
        rtol=1.0e-10,
    )


def test_can_take_analyser_zero_from_a_reference_bundle() -> None:
    """Allow one separately fitted zero to calibrate a target rocking curve."""
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(
        yaw=BaseData(signal=np.asarray([1.0, 2.0]), units="microradian", rank_of_data=1)
    )
    processing_data["reference"] = DataBundle(
        fitted_zero=BaseData(signal=np.asarray(1.5), units="microradian", rank_of_data=0)
    )

    _run_step(
        processing_data,
        angular_zero_key="fitted_zero",
        angular_zero_processing_key="reference",
    )

    np.testing.assert_allclose(
        processing_data["scan"]["TwoTheta"].signal,
        np.asarray([-0.5, 0.5]) * 1.0e-6,
    )


def test_source_wavelength_and_axis_metadata_follow_xs_geometry_conventions() -> None:
    """Use the shared source/scalar lifecycle and retain the scan-axis metadata."""
    scan_axis = BaseData(signal=np.asarray([0.0, 1.0]), units="dimensionless", rank_of_data=1)
    processing_data = ProcessingData()
    processing_data["scan"] = DataBundle(
        yaw=BaseData(
            signal=np.asarray([-1.0, 1.0]),
            units="microradian",
            axes=[scan_axis],
            rank_of_data=1,
        ),
        psi=BaseData(signal=np.asarray(0.0), units="degree", rank_of_data=0),
    )
    wavelength_samples = BaseData(
        signal=np.asarray([0.1, 0.1]),
        units="nanometer",
        rank_of_data=0,
    )
    step = XSGeometryFromAnalyserAngle(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["scan"],
        angle_key="yaw",
        wavelength_source="calibration::/wavelength",
        psi_key="psi",
    )

    with patch.object(step, "_load_from_sources", return_value=wavelength_samples):
        step(processing_data)

    bundle = processing_data["scan"]
    for key in ("TwoTheta", "signed_q", "Q", "Psi"):
        assert bundle[key].rank_of_data == 1
        assert bundle[key].axes == [scan_axis]
    assert "wavelength_jitter" in bundle["Q"].uncertainties
    assert bundle["Q"].units.is_compatible_with("1 / meter")
    assert bundle["TwoTheta"].units.is_compatible_with("radian")


def test_numeric_radiation_and_calibration_literals_are_not_module_arguments() -> None:
    """Require physical metadata to arrive through BaseData or IoSources."""
    arguments = XSGeometryFromAnalyserAngle.documentation.arguments
    assert "wavelength" not in arguments
    assert "wavelength_key" not in arguments
    assert "beam_energy" not in arguments
    assert "angular_multiplier" not in arguments
    assert "psi" not in arguments
    assert arguments["wavelength_source"]["required"] is True


def test_dependency_contract_tracks_exact_inputs_outputs_and_sources() -> None:
    """Expose precise dependencies for partial graph reruns."""
    step = XSGeometryFromAnalyserAngle(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["sample"],
        angle_key="yaw",
        angular_zero_key="peak_center",
        angular_zero_processing_key="reference",
        angular_multiplier_source="instrument::/usaxs/angular_multiplier",
        wavelength_source="instrument::/usaxs/wavelength",
        psi_source="instrument::/usaxs/psi",
    )

    contract = step.dependency_contract()

    assert isinstance(contract, ProcessStepDependencies)
    assert contract.source_refs == frozenset({"instrument"})
    assert contract.processing_reads == frozenset(
        {
            "sample.yaw",
            "reference.peak_center",
        }
    )
    assert contract.processing_writes == frozenset(f"sample.{key}" for key in step.output_keys)
