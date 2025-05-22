import pytest
from pathlib import Path

from ...runner.pipeline import Pipeline
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.process_step_describer import ProcessStepDescriber
from ...io.io_sources import IoSources
from ...dataclasses.databundle import DataBundle

TEST_IO_SOURCES = IoSources()
TEST_DATA = DataBundle()

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
            node.execute(data = TEST_DATA)
            pipeline.done(node)
    assert pipeline._nfinished == len(steps)
