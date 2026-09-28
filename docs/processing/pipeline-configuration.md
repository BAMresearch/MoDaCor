# Pipeline configuration

A pipeline YAML document contains a name and a mapping of step identifiers:

```yaml
name: example
steps:
  poisson:
    module: PoissonUncertainties
    requires_steps: []
    short_title: counting statistics
    configuration:
      with_processing_keys: [sample]
```

Each step has:

- `module` (required): a registered `ProcessStep` name;
- `requires_steps`: prerequisite step identifiers;
- `short_title`: optional graph annotation; and
- `configuration`: shared and module-specific values.

Configuration is validated when the pipeline loads. Shared keys come from
`ProcessStep.CONFIG_KEYS`; module-specific keys, expected top-level types, and
defaults come from `ProcessStepDescriber.arguments`. Modules perform additional
semantic validation when values depend on runtime sources or relationships
between fields.

Keep scientific constants and their provenance close to the YAML. Keep changing
file locations in runtime source/sink registration so the pipeline remains
reusable. Use stable step and processing keys because traces and partial reruns
refer to them.

See the exact [pipeline schema reference](../reference/pipeline-schema.md) and
per-step [generated configuration tables](../reference/modules/index.md).
