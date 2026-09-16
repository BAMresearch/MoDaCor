from __future__ import annotations

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.framewise_indexed_averager import (
    FramewiseIndexedAverager,
)


def test_integrates_each_frame_with_static_geometry_and_mask() -> None:
    processing = ProcessingData()
    processing["prefilter"] = DataBundle(
        signal=BaseData(
            signal=np.array(
                [
                    [[1.0, 100.0], [3.0, 5.0]],
                    [[2.0, 100.0], [4.0, 8.0]],
                ]
            ),
            units=ureg.count,
            rank_of_data=2,
        ),
        Q=BaseData(
            signal=np.array([[0.01, 0.01], [0.20, 0.20]]),
            units=ureg.angstrom**-1,
            rank_of_data=2,
        ),
        Psi=BaseData(
            signal=np.zeros((2, 2)),
            units=ureg.degree,
            rank_of_data=2,
        ),
        pixel_index=BaseData(
            signal=np.array([[0, 0], [1, 1]]),
            units=ureg.dimensionless,
            rank_of_data=2,
        ),
        mask=BaseData(
            signal=np.array([[0, 1], [0, 0]], dtype=np.uint32),
            units=ureg.dimensionless,
            rank_of_data=2,
        ),
    )

    step = FramewiseIndexedAverager(
        io_sources=IoSources(),
        configuration={
            "with_processing_keys": ["prefilter"],
            "averaging_direction": "azimuthal",
            "use_signal_weights": False,
            "stats_keys": ["signal"],
        },
    )
    step.execute(processing)

    result = processing["prefilter"]
    np.testing.assert_allclose(result["signal"].signal, [[1.0, 4.0], [2.0, 6.0]])
    np.testing.assert_allclose(result["Q"].signal, [[0.01, 0.20], [0.01, 0.20]])
    assert result["signal"].shape == (2, 2)
    assert result["signal"].rank_of_data == 1
    assert result["signal"].axes == [None]


def test_preserves_two_leading_batch_dimensions() -> None:
    processing = ProcessingData()
    processing["prefilter"] = DataBundle(
        signal=BaseData(
            signal=np.arange(16.0).reshape(1, 2, 2, 4),
            units=ureg.count,
            rank_of_data=2,
        ),
        Q=BaseData(
            signal=np.tile(np.array([0.01, 0.01, 0.2, 0.2]), (2, 1)),
            units=ureg.angstrom**-1,
            rank_of_data=2,
        ),
        Psi=BaseData(signal=np.zeros((2, 4)), units=ureg.degree, rank_of_data=2),
        pixel_index=BaseData(
            signal=np.tile(np.array([0, 0, 1, 1]), (2, 1)),
            units=ureg.dimensionless,
            rank_of_data=2,
        ),
    )
    step = FramewiseIndexedAverager(
        io_sources=IoSources(),
        configuration={"with_processing_keys": ["prefilter"], "use_signal_weights": False},
    )
    step.execute(processing)
    assert processing["prefilter"]["signal"].shape == (1, 2, 2)
    assert processing["prefilter"]["signal"].axes == [None]


def test_retains_batch_shape_when_one_frame_is_fully_masked() -> None:
    processing = ProcessingData()
    processing["prefilter"] = DataBundle(
        signal=BaseData(
            signal=np.ones((2, 2, 2)),
            units=ureg.count,
            uncertainties={"poisson": np.ones((2, 2, 2))},
            rank_of_data=2,
        ),
        Q=BaseData(signal=np.array([[0.01, 0.01], [0.2, 0.2]]), units=ureg.angstrom**-1, rank_of_data=2),
        Psi=BaseData(signal=np.zeros((2, 2)), units=ureg.degree, rank_of_data=2),
        pixel_index=BaseData(signal=np.array([[0, 0], [1, 1]]), units=ureg.dimensionless, rank_of_data=2),
        mask=BaseData(
            signal=np.array(
                [
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                ],
                dtype=np.uint32,
            ),
            units=ureg.dimensionless,
            rank_of_data=2,
        ),
    )
    step = FramewiseIndexedAverager(
        io_sources=IoSources(),
        configuration={"with_processing_keys": ["prefilter"], "use_signal_weights": False},
    )
    step.execute(processing)

    result = processing["prefilter"]["signal"]
    np.testing.assert_allclose(result.signal[0], [1.0, 1.0])
    assert np.isnan(result.signal[1]).all()
    assert np.isnan(result.uncertainties["poisson"][1]).all()
