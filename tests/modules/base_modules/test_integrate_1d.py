from __future__ import annotations

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.integrate_1d import Integrate1D


def test_integrate_1d_uses_common_valid_nonuniform_domain() -> None:
    q = BaseData(
        signal=np.asarray([0.0, 0.5, 1.5, 3.0, 5.0]),
        units="1 / meter",
        rank_of_data=1,
    )
    sample = BaseData(
        signal=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        units="count",
        uncertainties={"SEM": np.ones(5)},
        weights=np.ones(5),
        rank_of_data=1,
    )
    blank = BaseData(
        signal=np.ones(5),
        units="count",
        uncertainties={"SEM": np.full(5, 0.5)},
        weights=np.asarray([1.0, 1.0, 1.0, 1.0, 0.0]),
        rank_of_data=1,
    )
    data = ProcessingData()
    data["sample"] = DataBundle(signal=sample, q=q)
    data["blank"] = DataBundle(signal=blank, q=q.copy())
    step = Integrate1D(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["sample", "blank"],
        axis_key="q",
        output_processing_keys=["sample_integral", "blank_integral"],
    )

    step(data)

    np.testing.assert_allclose(
        data["sample_integral"]["signal"].signal,
        np.trapezoid(sample.signal[:4], q.signal[:4]),
    )
    np.testing.assert_allclose(
        data["blank_integral"]["signal"].signal,
        np.trapezoid(blank.signal[:4], q.signal[:4]),
    )
    expected_sem = np.sqrt(np.sum(np.asarray([0.25, 0.75, 1.25, 0.75]) ** 2))
    np.testing.assert_allclose(data["sample_integral"]["signal"].uncertainties["SEM"], expected_sem)
    np.testing.assert_allclose(data["blank_integral"]["signal"].uncertainties["SEM"], 0.5 * expected_sem)
    assert data["sample_integral"]["signal"].units == ureg.count / ureg.meter


def test_integrate_1d_can_store_simpson_result_in_source_bundle() -> None:
    x = BaseData(signal=np.asarray([0.0, 1.0, 2.0]), units="second", rank_of_data=1)
    signal = BaseData(
        signal=x.signal**2,
        units="count",
        uncertainties={"SEM": np.ones(3)},
        rank_of_data=1,
    )
    data = ProcessingData()
    data["curve"] = DataBundle(signal=signal, x=x)
    step = Integrate1D(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["curve"],
        axis_key="x",
        method="simpson",
        output_key="area",
    )

    step(data)

    np.testing.assert_allclose(data["curve"]["area"].signal, 8.0 / 3.0)
    np.testing.assert_allclose(data["curve"]["area"].uncertainties["SEM"], np.sqrt(2.0))
