# SPDX-License-Identifier: BSD-3-Clause
import logging

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.modules import ConvertUnits


def data():
    result = ProcessingData()
    for name in ["sample", "background"]:
        q = BaseData(np.array([1, 2]), "1/m", uncertainties={"SEM": [0.1, 0.2], "STD": 0.3}, weights=[2, 3])
        result[name] = DataBundle(Q=q, signal=BaseData([3, 4], "count/sr", axes=[q]))
    return result


def test_several_keys_bundles_uncertainties_weights_and_axis_identity():
    processing = data()
    axes = [processing[name]["Q"] for name in processing]
    weights = [axis.weights for axis in axes]
    step = ConvertUnits(
        configuration={"with_processing_keys": ["sample", "background"], "units": {"Q": "1/nm", "signal": "count/sr/%"}}
    )
    step.execute(processing)
    for index, bundle in enumerate(processing.values()):
        q = bundle["Q"]
        assert bundle["signal"].axes[0] is q is axes[index]
        assert q.weights is weights[index]
        assert q.units == ureg.Unit("1/nm")
        np.testing.assert_allclose(q.signal, [1e-9, 2e-9], rtol=1e-12, atol=0)
        np.testing.assert_allclose(q.uncertainties["SEM"], [1e-10, 2e-10], rtol=1e-12, atol=0)
        np.testing.assert_allclose(q.uncertainties["STD"], 3e-10, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(q.weights, [2, 3])
        np.testing.assert_allclose(bundle["signal"].signal, [0.03, 0.04], rtol=1e-12, atol=0)


def test_wildcard_skips_missing_keys_and_logs(caplog):
    processing = data()
    del processing["background"]["Q"]
    processing["empty"] = DataBundle()
    step = ConvertUnits(configuration={"with_processing_keys": "*", "units": {"Q": "1/nm", "signal": "count/sr"}})
    with caplog.at_level(logging.DEBUG, logger=step.logger.name):
        step.execute(processing)
    assert "Q" not in processing["background"]
    assert processing["sample"]["Q"].units == ureg.Unit("1/nm")
    messages = [(record.levelno, record.getMessage()) for record in caplog.records if record.name == step.logger.name]
    summaries = [message for level, message in messages if level == logging.INFO]
    assert summaries == [
        "ConvertUnits: converted 'Q' in 1 of 3, 'signal' in 2 of 3 bundles; bundles without a key were skipped."
    ]
    details = [message for level, message in messages if level == logging.DEBUG and "skipped per key" in message]
    assert len(details) == 1 and "'background'" in details[0] and "'empty'" in details[0]


def test_explicit_missing_key_names_bundle_and_key():
    step = ConvertUnits(configuration={"with_processing_keys": ["sample"], "units": {"missing": "m"}})
    with pytest.raises(KeyError, match="sample.*missing"):
        step.execute(data())


@pytest.mark.parametrize("units", ["s", "not_a_real_unit"])
def test_incompatible_units_error_names_bundle_key_and_units(units):
    step = ConvertUnits(configuration={"with_processing_keys": ["sample"], "units": {"Q": units}})
    with pytest.raises(ValueError, match=f"sample.*Q.*{units}"):
        step.execute(data())


@pytest.mark.parametrize(
    "selection,paths",
    [
        (["sample", "background"], {"sample.Q", "background.Q", "sample.signal", "background.signal"}),
        ("*", {"*"}),
    ],
)
def test_mapping_dependency_contract(selection, paths):
    step = ConvertUnits(configuration={"with_processing_keys": selection, "units": {"Q": "1/nm", "signal": "count"}})
    assert step.dependency_contract() == ProcessStepDependencies(processing_reads=paths, processing_writes=paths)


def test_empty_mapping_has_empty_contract():
    assert ConvertUnits().dependency_contract() == ProcessStepDependencies()
