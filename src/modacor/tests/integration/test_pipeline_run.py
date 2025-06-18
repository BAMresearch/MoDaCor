import pytest
from pathlib import Path
import numpy as np

from ...runner.pipeline import Pipeline
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.process_step_describer import ProcessStepDescriber
from ...dataclasses.processing_data import ProcessingData
from ...dataclasses.databundle import DataBundle
from ...dataclasses.basedata import BaseData
from ...io.io_sources import IoSources
from modacor import ureg

from ...modules.base_modules.poisson_uncertainties import PoissonUncertainties

TEST_IO_SOURCES = IoSources()
TEST_DATA = ProcessingData()
TEST_DATA["data"] = DataBundle()


@pytest.fixture
def flat_data():
    data = ProcessingData()
    data["bundle1"] = DataBundle()
    data["bundle2"] = DataBundle()
    data["bundle1"]["signal"] = BaseData(signal=np.arange(50))
    data["bundle2"]["signal"] = BaseData(signal=np.ones((10, 10)))
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
    step.modify_config("with_processing_keys", ["bundle2"])
    graph = {step: {}}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.processing_data = flat_data
            node.execute(flat_data)
            pipeline.done(node)
    assert node.produced_outputs["bundle2"]["signal"].variances["Poisson"].mean().astype(int) == 1
