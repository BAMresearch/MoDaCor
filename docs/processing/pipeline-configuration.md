# Pipeline configuration

A pipeline YAML document contains a name and a mapping of step identifiers:

Pipelines written for the retired `IndexPixels`, former `IndexedAverager`, or
former geometry `wavelength_*` interfaces require the
[breaking interface migration](interface-migrations.md) before they will load.

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

Repeated lanes can be authored as a `step_blocks` entry with a `for_each`
parameter table. MoDaCor expands each item into ordinary steps before module
validation and DAG construction:

```yaml
step_blocks:
  normalize:
    for_each:
      sample: {processing_key: sample}
      background: {processing_key: background}
    steps:
      mask:
        module: ThresholdMask
        configuration:
          with_processing_keys: ["${processing_key}"]
          lower_bound: 0
      apply:
        module: ApplyMask
        requires_steps: [.mask]
        configuration:
          with_processing_keys: ["${processing_key}"]
```

This produces `normalize.sample.mask`, `normalize.sample.apply`,
`normalize.background.mask`, and `normalize.background.apply`. Each is an
independent execution node with its own validation, trace, failure report, and
partial-rerun behavior.

The block name itself is not an execution node, so a downstream step cannot use
`requires_steps: [normalize]` to depend on the whole block. It must name the
generated terminal instances it consumes, for example:

```yaml
steps:
  compare:
    module: ComparePreparedData
    requires_steps:
      - normalize.sample.apply
      - normalize.background.apply
```

Requiring each terminal instance is sufficient because its earlier local
dependencies are included transitively. Version 1 has no block-wide dependency
or wildcard shorthand.

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
