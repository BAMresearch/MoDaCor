# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Anja Hörmann, Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "18/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest

from modacor import ureg

from ...dataclasses.basedata import BaseData
from ...dataclasses.databundle import DataBundle
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.processing_data import ProcessingData
from ...io.io_sources import IoSources
from ...modules.base_modules.poisson_uncertainties import PoissonUncertainties
from ...runner.pipeline import Pipeline

TEST_IO_SOURCES = IoSources()
TEST_DATA = ProcessingData()
TEST_DATA["data"] = DataBundle()


@pytest.fixture
def flat_data():
    data = ProcessingData()
    data["bundle1"] = DataBundle()
    data["bundle2"] = DataBundle()
    data["bundle1"]["signal"] = BaseData(signal=np.arange(50), units=ureg.Unit("count"))
    data["bundle2"]["signal"] = BaseData(signal=np.ones((10, 10)), units=ureg.Unit("count"))
    return data


class DummyProcessStep(ProcessStep):
    def calculate(self):
        return {"test": DataBundle()}


def test_processstep_pipeline(flat_data):
    "tests execution of a linear processstep pipeline (not actually doing anything)"
    steps = [DummyProcessStep(TEST_IO_SOURCES, step_id=i) for i in range(3)]
    graph = {steps[i]: {steps[i + 1]} for i in range(len(steps) - 1)}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    sequence = []
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.processing_data = flat_data
            sequence.append(node)
            node.execute(flat_data)
            pipeline.done(node)
    assert pipeline._nfinished == len(steps)


def test_actual_processstep(flat_data):
    "test running the PoissonUncertainties Process step"
    step = PoissonUncertainties(TEST_IO_SOURCES)
    # we need to supply a list of values here
    step.modify_config_by_kwargs(with_processing_keys=["bundle2"])
    graph = {step: {}}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.processing_data = flat_data
            node.execute(flat_data)
            pipeline.done(node)
    assert node.produced_outputs["bundle2"]["signal"].variances["Poisson"].mean().astype(int) == 1
