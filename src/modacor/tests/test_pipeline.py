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


class DummyProcessStep:
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
    ps = DummyProcessStep()
    pipeline.add(ps, *[1, 2, 3])
    pipeline.prepare()
    sequence = []
    while pipeline.is_active():
        for node in pipeline.get_ready():
            sequence.append(node)
            pipeline.done(node)
    assert sequence == [1, 2, 3, ps]


def test_branch_addition(linear_pipeline, pipeline_to_add={5: {6}}, at_node=2):
    """
    add a pipeline as a branch on an existing pipeline, using the inherited add method

    """
    pipeline_1 = Pipeline(graph=linear_pipeline)
    pipeline_2 = Pipeline(graph=pipeline_to_add)
    pipeline_1.add(at_node, *pipeline_2.static_order())
    assert [*pipeline_1.static_order()] == [1, 6, 5, 2, 3]


def test_branch_addition_method(linear_pipeline, branch_graph={5: {6}}, branching_node=2):
    pipeline = Pipeline(graph=linear_pipeline)
    branch = Pipeline(graph=branch_graph)
    pipeline.add_incoming_branch(branch, branching_node=2)
    assert [*pipeline.static_order()] == [1, 6, 5, 2, 3]
    assert pipeline.graph == {3: {2, 1}, 2: {1, 5}, 5: {6}}


def test_diverging_branch_addition(
    linear_pipeline, branch_graph={5: {6}, 6: set()}, branching_node=2
):
    pipeline = Pipeline(graph=linear_pipeline)
    branch = Pipeline(graph=branch_graph)
    pipeline.add_outgoing_branch(branch, branching_node)
    assert [*pipeline.static_order()] == [1, 2, 3, 6, 5]
    assert pipeline.graph == {3: {2, 1}, 2: {1}, 5: {6}, 6: {2}}
