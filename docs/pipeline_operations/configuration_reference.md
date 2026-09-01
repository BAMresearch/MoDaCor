# Pipeline configuration reference

This page summarises the YAML keys understood by MoDaCor pipeline definitions.

## Step fields

Each entry under `steps` is keyed by a `step_id` and supports the following fields:

- `module` (required): ProcessStep class name to instantiate.
- `requires_steps` (optional): list of step_ids that must run before this step.
- `configuration` (optional): dictionary of ProcessStep configuration values.
- `short_title` (optional): a brief, human-friendly purpose label used in graphs (Mermaid/DOT). This is appended as a
  second line in node labels, e.g. `AU: MultiplyDatabundles` + `scaling to absolute units`.

## Step configuration validation

Each `configuration` block is checked when `Pipeline.from_yaml(...)` loads the
pipeline. MoDaCor builds the accepted key list and top-level type policy from
the selected module class:

- shared `ProcessStep` keys from `CONFIG_KEYS`
- module-specific keys from `ProcessStepDescriber.arguments`

For example, `Divide.divisor_source` is declared as a string source reference,
so this is valid:

```yaml
steps:
  normalize:
    module: Divide
    configuration:
      with_processing_keys:
        - sample
      divisor_source: sample::entry/frame_exposure_time
```

and this fails during pipeline loading because `divisor_source` is not a string:

```yaml
steps:
  normalize:
    module: Divide
    configuration:
      divisor_source: 3
```

The central validator catches unknown keys and top-level type mismatches. Module
code still performs semantic checks for values that need runtime context, such
as missing sources, non-empty required strings, mutually exclusive options, or
nested dictionary contents.
