# Pipeline graphs

A pipeline is a directed acyclic graph whose nodes are `ProcessStep` instances
and whose edges express execution prerequisites. YAML uses `requires_steps` to
declare those edges:

```yaml
name: minimal_demo
steps:
  uncertainty:
    module: PoissonUncertainties
    configuration:
      with_processing_keys: [sample]
  normalize:
    module: DivideDatabundles
    requires_steps: [uncertainty]
    configuration:
      with_processing_keys: [sample, exposure]
```

`Pipeline.from_yaml(...)` validates the structure and resolves module names
through a `ProcessStepRegistry`. Step identifiers are normalized to strings and
must be unique. Every declared dependency must identify another step.

Normal execution uses `run_pipeline_job(...)`, which creates a fresh
topological scheduler for each run. Independent ready nodes may be returned
together, but the standard runner executes them deterministically in its
scheduler loop. For interactive debugging, `pipeline.create_scheduler()`
exposes the underlying `prepare`, `get_ready`, `done`, and `is_active` protocol.

The graph captures ordering, while each step's dependency contract captures
which sources and processing paths it reads and writes. Both are needed for a
correct partial rerun.

Pipelines can render Mermaid or DOT representations. Use a short `short_title`
to add scientific purpose to a node without changing its identifier or module.
