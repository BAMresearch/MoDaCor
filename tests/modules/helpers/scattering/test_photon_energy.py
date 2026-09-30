from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.modules.helpers.scattering.photon_energy import (
    photon_energy_from_wavelength,
    photon_wavelength_from_energy,
)


def test_photon_energy_wavelength_roundtrip_preserves_uncertainty_and_metadata() -> None:
    axis = BaseData(np.array([0.0, 1.0]), ureg.second, rank_of_data=1)
    energy = BaseData(
        np.array([12.398419843320026, 6.199209921660013]),
        ureg.keV,
        uncertainties={"calibration": np.array([0.01, 0.02])},
        weights=np.array([0.8, 0.6]),
        axes=[axis],
        rank_of_data=1,
    )

    wavelength = photon_wavelength_from_energy(energy, output_units="angstrom")
    roundtrip = photon_energy_from_wavelength(wavelength, output_units="keV")

    assert_allclose(wavelength.signal, np.array([1.0, 2.0]), rtol=2.0e-12)
    assert_allclose(
        wavelength.uncertainties["calibration"],
        wavelength.signal * energy.uncertainties["calibration"] / energy.signal,
    )
    assert wavelength.axes == energy.axes
    assert wavelength.rank_of_data == energy.rank_of_data
    assert_allclose(wavelength.weights, energy.weights)
    assert_allclose(roundtrip.signal, energy.signal)
    assert_allclose(roundtrip.uncertainties["calibration"], energy.uncertainties["calibration"])


@pytest.mark.parametrize(
    ("value", "conversion", "message"),
    [
        (BaseData(np.asarray(0.0), ureg.keV), photon_wavelength_from_energy, "finite and positive"),
        (BaseData(np.asarray(np.nan), ureg.angstrom), photon_energy_from_wavelength, "finite and positive"),
        (BaseData(np.asarray(1.0), ureg.second), photon_wavelength_from_energy, "energy units"),
    ],
)
def test_photon_conversion_rejects_invalid_input(value: BaseData, conversion, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        conversion(value)


def test_photon_conversion_rejects_incompatible_output_units() -> None:
    with pytest.raises(ValueError, match="output units"):
        photon_wavelength_from_energy(BaseData(np.asarray(12.0), ureg.keV), output_units="second")
