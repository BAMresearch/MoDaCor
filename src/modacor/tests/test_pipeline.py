import pytest
from pathlib import Path

from ..runner.pipeline import Pipeline
from ..dataclasses.process_step import ProcessStep
from ..dataclasses.process_step_describer import ProcessStepDescriber


@pytest.fixture
def linear_pipeline():
    return {3: {2, 1}, 2: {1}}


class DummyIoSources:
    pass


class DummyProcessStepDescriber:
    pass


def test_linear_pipeline(linear_pipeline):
    "tests the sequence is expected for a linear graph"
    pipeline = Pipeline(graph=linear_pipeline)
    pipeline.prepare()
    sequence = []
    while pipeline.is_active():
        for node in pipeline.get_ready():
            sequence.append(node)
            pipeline.done(node)
    assert sequence == [1, 2, 3]


def test_node_addition(linear_pipeline):
    pipeline = Pipeline.from_dict(linear_pipeline)
    pipeline.add(ProcessStep(io_sources=DummyIoSources(), documentation=DummyProcessStepDescriber))
    pipeline.prepare()
    sequence = []
    while pipeline.is_active():
        for node in pipeline.get_ready():
            sequence.append(node)
            pipeline.done(node)
    assert sequence == [1, 2, 3]
