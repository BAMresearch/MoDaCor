from __future__ import annotations

import pytest

from modacor.runner.pipeline_schema import PipelineSchemaError, expand_pipeline_yaml


def test_expand_pipeline_yaml_expands_multistep_blocks_with_typed_parameters() -> None:
    source = """
    name: mapped
    step_blocks:
      prepare:
        for_each:
          sample:
            key: sample
            enabled: true
            prerequisites: [.first]
          background:
            key: background
            enabled: false
            prerequisites: [.first]
        steps:
          first:
            module: Dummy
            configuration:
              with_processing_keys: ["${key}"]
              enabled: "${enabled}"
          second:
            module: Dummy
            requires_steps: "${prerequisites}"
            configuration:
              target: "${key}"
    steps:
      finish:
        module: Dummy
        requires_steps: [prepare.sample.second, prepare.background.second]
    """

    result = expand_pipeline_yaml(source)

    assert list(result.expanded_spec["steps"]) == [
        "finish",
        "prepare.sample.first",
        "prepare.sample.second",
        "prepare.background.first",
        "prepare.background.second",
    ]
    sample_first = result.expanded_spec["steps"]["prepare.sample.first"]
    assert sample_first["configuration"] == {"with_processing_keys": ["sample"], "enabled": True}
    assert result.expanded_spec["steps"]["prepare.sample.second"]["requires_steps"] == ["prepare.sample.first"]
    assert result.expanded_spec["steps"]["finish"]["requires_steps"] == [
        "prepare.sample.second",
        "prepare.background.second",
    ]
    assert "step_blocks" not in result.expanded_spec
    assert "step_blocks" in result.authored_spec
    assert result.origins["prepare.background.second"].to_dict() == {
        "block": "prepare",
        "item": "background",
        "local_step": "second",
        "block_index": 0,
        "item_index": 1,
        "local_step_index": 1,
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            step_blocks:
              prepare:
                for_each: {sample: {key: sample}}
                steps:
                  load: {module: Dummy, configuration: {key: "prefix-${key}"}}
            """,
            "partial parameter interpolation",
        ),
        (
            """
            step_blocks:
              prepare:
                for_each: {sample: {}}
                steps:
                  load: {module: Dummy, configuration: {key: "${missing}"}}
            """,
            "missing parameter",
        ),
        (
            """
            step_blocks:
              prepare:
                for_each: {sample: {}}
                steps:
                  load: {module: Dummy, requires_steps: [.unknown]}
            """,
            "unknown local step",
        ),
        (
            """
            step_blocks:
              bad.block:
                for_each: {sample: {}}
                steps: {load: {module: Dummy}}
            """,
            "only letters, digits",
        ),
    ],
)
def test_expand_pipeline_yaml_rejects_ambiguous_templates(source: str, message: str) -> None:
    with pytest.raises(PipelineSchemaError, match=message):
        expand_pipeline_yaml(source)


def test_expand_pipeline_yaml_rejects_collisions_and_limits() -> None:
    source = """
    step_blocks:
      prepare:
        for_each: {sample: {}}
        steps: {load: {module: Dummy}}
    steps:
      prepare.sample.load: {module: Dummy}
    """

    with pytest.raises(PipelineSchemaError, match="collides"):
        expand_pipeline_yaml(source)

    with pytest.raises(PipelineSchemaError, match="exceeding the limit"):
        expand_pipeline_yaml(
            """
            step_blocks:
              prepare:
                for_each: {sample: {}, background: {}}
                steps: {load: {module: Dummy}}
            """,
            max_expanded_steps=1,
        )
