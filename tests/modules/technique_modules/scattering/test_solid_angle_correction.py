from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.solid_angle_correction import SolidAngleCorrection


@pytest.mark.parametrize("keys", [["sample"], ["sample", "background"]])
def test_divides_signal_with_units_and_both_operand_uncertainties(keys):
    data = ProcessingData()
    for key in [*keys, "untouched"]:
        data[key] = DataBundle(
            signal=BaseData(
                signal=np.array([12.0, 24.0]),
                units=ureg.count,
                rank_of_data=1,
                uncertainties={"counting": np.array([3.0, 6.0])},
            ),
            Omega=BaseData(
                signal=np.array([2.0, 4.0]),
                units=ureg.steradian,
                rank_of_data=1,
                uncertainties={"geometry": np.array([0.2, 0.4])},
            ),
        )
    untouched = data["untouched"]["signal"]
    omegas = {key: data[key]["Omega"] for key in keys}
    step = SolidAngleCorrection(io_sources=IoSources(), configuration={"with_processing_keys": keys})
    step.processing_data = data
    output = step.calculate()

    assert set(output) == set(keys)
    for key in keys:
        assert output[key] is data[key]
        result = data[key]["signal"]
        np.testing.assert_allclose(result.signal, [6.0, 6.0])
        assert result.units == ureg.count / ureg.steradian
        np.testing.assert_allclose(result.uncertainties["counting"], [1.5, 1.5])
        np.testing.assert_allclose(result.uncertainties["geometry"], [0.6, 0.6])
        assert result.rank_of_data == 1
        assert data[key]["Omega"] is omegas[key]
        np.testing.assert_allclose(data[key]["Omega"].signal, [2.0, 4.0])
    assert data["untouched"]["signal"] is untouched


def test_missing_omega_raises_key_error():
    step = SolidAngleCorrection(io_sources=IoSources(), configuration={"with_processing_keys": ["sample"]})
    signal = BaseData(signal=np.array([12.0]), units=ureg.count)
    step.processing_data = ProcessingData(sample=DataBundle(signal=signal))
    with pytest.raises(KeyError, match="Omega"):
        step.calculate()
    assert step.processing_data["sample"]["signal"] is signal
