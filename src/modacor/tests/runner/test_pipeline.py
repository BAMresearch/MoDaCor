# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

__coding__ = "utf-8"
__authors__ = ["Anja Hörmann", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
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


def test_pipeline_from_yaml_unknown_module_raises():
    """
    If the YAML references a module name that the registry cannot resolve,
    Pipeline.from_yaml should raise a KeyError (propagated from the registry).
    """

    # Use a registry with no base_package and no registrations,
    # so any lookup will fail with KeyError.
    registry = ProcessStepRegistry(base_package=None)

    yaml_str = """
    name: bad_pipeline
    steps:
      s1:
        module: NonExistentStep
        requires_steps: []
        configuration:
          some_param: 42
    """

    with pytest.raises(KeyError):
        Pipeline.from_yaml(yaml_str, registry=registry)


def test_to_spec_basic():
    """to_spec should expose nodes and edges consistent with the internal graph."""

    class DummyDoc:
        calling_name = "Dummy step"
        calling_module_path = Path("dummy/path.py")
        calling_version = "0.1.0"

    class DummyNode:
        def __init__(self, step_id, config):
            self.step_id = step_id
            self.configuration = config
            self.documentation = DummyDoc

    # Build a tiny graph: n1 -> n2
    n1 = DummyNode(step_id="1", config={"foo": "bar"})
    n2 = DummyNode(step_id="2", config={"baz": 123})

    graph = {
        n2: {n1},  # n2 depends on n1
        n1: set(),  # n1 has no prerequisites
    }

    pipeline = Pipeline(graph=graph, name="test_pipe")

    spec = pipeline.to_spec()

    # Basic structure
    assert spec["name"] == "test_pipe"
    assert "nodes" in spec and "edges" in spec

    # Nodes: two nodes with ids "1" and "2"
    node_ids = {n["id"] for n in spec["nodes"]}
    assert node_ids == {"1", "2"}

    # Each node should carry module name and config
    modules = {n["module"] for n in spec["nodes"]}
    assert modules == {"DummyNode"}

    # Find n1 and n2 entries
    node_map = {n["id"]: n for n in spec["nodes"]}
    assert node_map["1"]["config"] == {"foo": "bar"}
    assert node_map["2"]["config"] == {"baz": 123}

    # Edges: exactly one edge 1 -> 2
    edges = {(e["from"], e["to"]) for e in spec["edges"]}
    assert edges == {("1", "2")}


def test_to_dot_matches_spec():
    """to_dot should reflect the nodes and edges from to_spec in DOT format."""

    class DummyDoc:
        calling_name = "Dummy step"
        calling_module_path = Path("dummy/path.py")
        calling_version = "0.1.0"

    class DummyNode:
        def __init__(self, step_id):
            self.step_id = step_id
            self.configuration = {}
            self.documentation = DummyDoc

    n1 = DummyNode(step_id="1")
    n2 = DummyNode(step_id="2")
    graph = {n2: {n1}, n1: set()}

    pipeline = Pipeline(graph=graph, name="dot_test")

    dot_src = pipeline.to_dot()

    # Basic header
    assert 'digraph "dot_test"' in dot_src

    # Node labels should include "<id>: <calling_name>"
    assert '"1" [label="1: Dummy step"];' in dot_src
    assert '"2" [label="2: Dummy step"];' in dot_src

    # Edge representation
    assert '"1" -> "2";' in dot_src


def test_to_mermaid_flowchart():
    """to_mermaid should emit a valid-looking Mermaid flowchart syntax."""

    class DummyDoc:
        calling_name = "Dummy step"
        calling_module_path = Path("dummy/path.py")
        calling_version = "0.1.0"

    class DummyNode:
        def __init__(self, step_id):
            self.step_id = step_id
            self.configuration = {}
            self.documentation = DummyDoc

    n1 = DummyNode(step_id="1")
    n2 = DummyNode(step_id="2")
    graph = {n2: {n1}, n1: set()}

    pipeline = Pipeline(graph=graph, name="mermaid_test")

    mermaid_src = pipeline.to_mermaid(direction="TB")

    # Header
    assert mermaid_src.splitlines()[0] == "flowchart TB"

    # Nodes: 1 and 2 with labels "1: Dummy step" etc.
    assert '1["1: Dummy step"]' in mermaid_src
    assert '2["2: Dummy step"]' in mermaid_src

    # Edge: 1 --> 2
    assert "1 --> 2" in mermaid_src


def test_yaml_spec_roundtrip_with_edit():
    """
    End-to-end style test:

    1. Load a pipeline from YAML (using a dummy ProcessStep via a custom registry).
    2. Export to a spec for a web-based graphing tool.
    3. Modify the spec (add a node, tweak a config).
    4. Rebuild the pipeline from the modified spec.
    5. Export back to YAML and assert the structure reflects the edits.
    """

    class DummyStep(ProcessStep):
        # Extend CONFIG_KEYS so that "a" is a valid configuration key
        CONFIG_KEYS = {
            **ProcessStep.CONFIG_KEYS,
            "a": {
                "type": int,
                "allow_iterable": False,
                "allow_none": False,
                "default": 0,
            },
        }

        def calculate(self):
            # For this test we don't care about actual computation
            return {}

    # Registry that only knows about DummyStep (no lazy imports)
    registry = ProcessStepRegistry(base_package=None)
    registry.register(DummyStep)

    # 1. Original YAML: simple 2-step linear pipeline
    original_yaml = """
    name: roundtrip_pipeline
    steps:
      1:
        module: DummyStep
        requires_steps: []
        configuration:
          a: 1
      2:
        module: DummyStep
        requires_steps: [1]
        configuration:
          a: 2
    """

    pipeline = Pipeline.from_yaml(original_yaml, registry=registry)

    # Sanity: initial topology is 1 -> 2
    initial_order = [node.step_id for node in pipeline.static_order()]
    assert initial_order == ["1", "2"]

    # 2. Export to spec (what the web editor would see)
    spec = pipeline.to_spec()

    # 3. Modify the spec as if the user edited the graph in a UI:
    #    - Add a new node "3" depending on "2"
    #    - Change the config of node "2"
    spec["nodes"].append(
        {
            "id": "3",
            "label": "Third step",
            "module": "DummyStep",
            "config": {"a": 3},
        }
    )
    spec["edges"].append({"from": "2", "to": "3"})

    for node in spec["nodes"]:
        if node["id"] == "2":
            node["config"]["a"] = 42

    # 4. Rebuild a pipeline from the modified spec
    modified_pipeline = Pipeline.from_spec(spec, registry=registry)
    modified_order = [node.step_id for node in modified_pipeline.static_order()]
    assert modified_order == ["1", "2", "3"]

    # 5. Export back to YAML
    modified_yaml = modified_pipeline.to_yaml()
    yaml_obj = yaml.safe_load(modified_yaml)

    # Basic structure
    assert yaml_obj["name"] == "roundtrip_pipeline"
    assert "steps" in yaml_obj

    steps = yaml_obj["steps"]
    # Keys are step_ids as strings
    assert set(steps.keys()) == {"1", "2", "3"}

    # All modules should be DummyStep
    assert steps["1"]["module"] == "DummyStep"
    assert steps["2"]["module"] == "DummyStep"
    assert steps["3"]["module"] == "DummyStep"

    # Requires_steps reflect the edited graph: 1 -> 2 -> 3
    assert "requires_steps" not in steps["1"] or steps["1"]["requires_steps"] == []
    assert steps["2"]["requires_steps"] == ["1"]
    assert steps["3"]["requires_steps"] == ["2"]

    # Config values reflect original + edits
    assert steps["1"]["configuration"]["a"] == 1
    assert steps["2"]["configuration"]["a"] == 42  # edited
    assert steps["3"]["configuration"]["a"] == 3
