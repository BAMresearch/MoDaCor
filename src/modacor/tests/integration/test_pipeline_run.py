import pytest
from pathlib import Path
import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()

from ...runner.pipeline import Pipeline
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.process_step_describer import ProcessStepDescriber
from ...dataclasses.databundle import DataBundle
from ...dataclasses.basedata import BaseData
from ...io.io_sources import IoSources

from ...modules.base_modules.poisson_uncertainties import PoissonUncertainties

TEST_IO_SOURCES = IoSources()
TEST_DATA = DataBundle()

@pytest.fixture
def flat_data():
    data = DataBundle()
    data["signal"] = BaseData(
        ingest_units=ureg.counts,
        internal_units=ureg.counts,
        display_units=ureg.counts,
        signal=100 * np.ones((1030, 1065)),
    )
    return data


class DummyProcessStep(ProcessStep):
    def calculate(self, data):
        return {"test": 0}


def test_processstep_pipeline():
    "tests execution of a linear processstep pipeline (not actually doing anything)"
    steps = [DummyProcessStep(TEST_IO_SOURCES, step_id=i) for i in range(3)]
    graph = {steps[i]: {steps[i + 1]} for i in range(len(steps) - 1)}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    sequence = []
    while pipeline.is_active():
        for node in pipeline.get_ready():
            sequence.append(node)
            node.execute(data=TEST_DATA)
            pipeline.done(node)
    assert pipeline._nfinished == len(steps)


def test_actual_processstep(flat_data):
    "test running the PoissonUncertainties Process step"
    graph = {PoissonUncertainties(TEST_IO_SOURCES): {}}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.execute(data=flat_data)
            pipeline.done(node)
    assert data["signal"].variances["Poisson"].mean().astype(int) == 10
