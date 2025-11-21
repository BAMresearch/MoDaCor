# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Anja Hörmann", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports


import pytest
import yaml

from modacor.dataclasses.process_step import ProcessStep
from modacor.runner.pipeline import Pipeline
from modacor.runner.process_step_registry import ProcessStepRegistry


@pytest.fixture
def linear_pipeline():
    # Simple linear graph: 1 -> 2 -> 3
    return {3: {2, 1}, 2: {1}}


class DummyIoSources:
    pass


class DummyProcessStepDescriber:
    pass


class DummyProcessStep:
    pass


@pytest.fixture
def yaml_one_step():
    # Single-step pipeline, keyed by step_id "div"
    return """
    name: one_step
    steps:
      di:
        module: Divide
        requires_steps: []
        configuration:
          divisor_source: 3
    """


@pytest.fixture
def yaml_linear_pipeline():
    # Simple 3-step linear pipeline with string step_ids
    return """
    name: simple_pipeline
    steps:
      1:
        module: PoissonUncertainties
        requires_steps: []
      p2:
        module: PoissonUncertainties
        requires_steps: [1]
      mul:
        module: PoissonUncertainties
        requires_steps: [p2]
        configuration:
          multiplier: 3
          signal: sample::signal
        io_sources:
          - sample
    """


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


def test_diverging_branch_addition(linear_pipeline, branch_graph={5: {6}, 6: set()}, branching_node=2):
    pipeline = Pipeline(graph=linear_pipeline)
    branch = Pipeline(graph=branch_graph)
    pipeline.add_outgoing_branch(branch, branching_node)
    assert [*pipeline.static_order()] == [1, 2, 3, 6, 5]
    assert pipeline.graph == {3: {2, 1}, 2: {1}, 5: {6}, 6: {2}}


def test_yaml_format(yaml_linear_pipeline):
    yaml_obj = yaml.safe_load(yaml_linear_pipeline)
    assert "steps" in yaml_obj
    # now keyed by step_id, not by human-readable name
    assert 1 in yaml_obj["steps"]
    assert "p2" in yaml_obj["steps"]
    assert "mul" in yaml_obj["steps"]
    assert isinstance(yaml_obj["steps"]["mul"]["configuration"], dict)


def test_pipeline_from_yaml(yaml_one_step):
    pipeline = Pipeline.from_yaml(yaml_one_step)
    assert pipeline.name == "one_step"
    assert isinstance(pipeline, Pipeline)

    # One node with no prerequisites
    assert len(pipeline.graph) == 1
    ((node, deps),) = pipeline.graph.items()
    assert isinstance(node, ProcessStep)
    assert deps == set()


def test_pipeline_from_yaml_with_custom_registry():
    """
    Ensure Pipeline.from_yaml uses the given ProcessStepRegistry
    and instantiates the correct ProcessStep subclass.
    """

    class DummyStep(ProcessStep):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.executed = False

        def execute(self, **kwargs):
            self.executed = True

    ps_registry = ProcessStepRegistry()
    ps_registry.register(DummyStep)

    yaml_str = """
    name: dummy_pipeline
    steps:
      s1:
        module: DummyStep
        requires_steps: []
        configuration:
    """

    pipeline = Pipeline.from_yaml(yaml_str, registry=ps_registry)
    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "dummy_pipeline"

    # There should be exactly one node in the graph
    assert len(pipeline.graph) == 1
    ((node, deps),) = pipeline.graph.items()

    assert isinstance(node, DummyStep)
    assert deps == set()

    # Run the pipeline and check that our DummyStep.execute was called
    pipeline.run()
    assert node.executed is True
