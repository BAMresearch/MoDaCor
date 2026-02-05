# Pipeline configuration reference

This page summarises the YAML keys understood by MoDaCor pipeline definitions.

## Step fields

Each entry under `steps` is keyed by a `step_id` and supports the following fields:

- `module` (required): ProcessStep class name to instantiate.
- `requires_steps` (optional): list of step_ids that must run before this step.
- `configuration` (optional): dictionary of ProcessStep configuration values.
- `short_title` (optional): a brief, human-friendly purpose label used in graphs (Mermaid/DOT). This is appended as a
  second line in node labels, e.g. `AU: MultiplyDatabundles` + `scaling to absolute units`.
