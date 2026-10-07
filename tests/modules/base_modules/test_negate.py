from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules import Negate


def test_negate_selected_basedata_preserves_uncertainties():
    data = ProcessingData()
    data["curve"] = DataBundle(
        Q=BaseData(
            signal=np.array([-2.0, 1.0]),
            units="1/nm",
            uncertainties={"dq": np.array([0.1, 0.2])},
            rank_of_data=1,
        )
    )
    step = Negate(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["curve"], data_key="Q")

    step(data)

    assert_allclose(data["curve"]["Q"].signal, [2.0, -1.0])
    assert_allclose(data["curve"]["Q"].uncertainties["dq"], [0.1, 0.2])
    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"curve.Q"}, processing_writes={"curve.Q"}
    )
