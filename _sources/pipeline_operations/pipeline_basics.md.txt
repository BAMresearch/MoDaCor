# Pipeline Basics

This page describes the core runtime model behind MoDaCor pipelines: what a
pipeline graph is, how data moves through `ProcessingData`, where sources and
sinks fit, and what "partial rerun" means in operational terms.

## Core building blocks

### `Pipeline`

A MoDaCor pipeline is a directed acyclic graph of `ProcessStep` instances. In
YAML, each step declares its module and upstream dependencies:

```yaml
name: minimal_demo
steps:
  load_sample:
    module: AppendSource
    requires_steps: []
    configuration:
      source_identifier:
        - sample
      output_key:
        - sample

  poisson:
    module: PoissonUncertainties
    requires_steps:
      - load_sample
    configuration:
      with_processing_keys:
        - sample
```

`Pipeline.from_yaml(...)` turns that specification into a graph. During
execution, the scheduler visits steps in topological order and only runs a step
once all of its declared prerequisites are complete.

### `ProcessingData`

`ProcessingData` is the in-memory workspace shared by the pipeline. Every step
reads from it, writes to it, or both. It is intentionally structured so the
pipeline can preserve intermediate results for debugging, traceability, and
partial reruns.

The hierarchy is:

- `ProcessingData`: mapping of bundle keys such as `sample`, `background`, or `result`
- `DataBundle`: mapping of named data products within one bundle
- `BaseData`: the actual array-like data plus metadata

In practice, a path like `/sample/signal/signal` means:

- bundle key: `sample`
- `BaseData` entry: `signal`
- leaf attribute inside that `BaseData`: `signal`

### `BaseData`

`BaseData` stores the numerical payload and the metadata required for traceable
correction work:

- data arrays
- units
- uncertainties
- axes / rank metadata
- optional weights and masks

This is why MoDaCor can run a correction step and still explain what happened to
units and propagated uncertainties at each stage.

## Sources and sinks

MoDaCor separates external I/O from the pipeline graph itself.

### Sources

`IoSources` is a runtime registry of named inputs such as `sample`,
`background`, or `defaults`. A source registration binds:

- a reference name
- a concrete source type such as HDF, YAML, or CSV
- a resource location
- optional reader-specific kwargs

Sources can enter the system in two ways:

- directly through the CLI or runtime API, which is how the session-based
  runtime service works
- through pipeline steps such as `AppendSource`, which pull data from the
  `IoSources` registry into `ProcessingData`

This separation is useful operationally: the same pipeline can be reused while
the actual file paths change from run to run.

### Sinks

`IoSinks` works the same way for outputs. A pipeline may emit results through a
registered sink such as `CSVSink`, or the runtime/CLI layer may export the
final `ProcessingData` snapshot directly, for example through the shared HDF
export path used by `write_hdf`.

## Execution modes

### Full run

A full run starts from a clean execution context and processes the whole graph.
This is the default safest mode when:

- there is no previous `ProcessingData` snapshot
- the pipeline configuration changed
- the scope of the input change is unclear

### Partial run

A partial run is only possible when a previous `ProcessingData` snapshot exists.
The runtime service uses `changed_sources` and `changed_keys` to find the dirty
part of the graph.

The current policy is conservative:

1. Find steps that reference the changed source or processing key.
2. Expand that set to all downstream descendants.
3. Reuse the prior `ProcessingData` snapshot.
4. Execute only the selected dirty subgraph in topological order.

This keeps unaffected upstream work intact while ensuring downstream results are
recomputed consistently.

### Auto mode

`auto` mode tries a partial rerun first and falls back to a full rerun if the
partial execution fails. This is mainly an operational convenience for long-lived
sessions where new data arrives repeatedly but robustness matters more than
forcing a partial path.

## Runtime-service session model

The runtime API wraps pipeline execution in a `PipelineSession`. A session keeps:

- the pipeline YAML
- registered sources
- tracing configuration
- the latest `ProcessingData`
- run history and error state

That session state is what enables:

- incremental source updates via the API
- dry-run invalidation previews
- partial reruns against prior results
- readiness metrics such as active runs and error-state sessions

## Partial rerun behavior in practice

When an operator updates a source like `sample`, the typical sequence is:

1. Update the source registration in the session.
2. Call `/process` with `mode="partial"` or `mode="auto"`.
3. Let the runtime service compute the dirty step set.
4. Reuse the prior `ProcessingData` for unaffected parts of the graph.
5. Record run metadata including dirty steps, skipped steps, durations, and any fallback reason.

That means MoDaCor treats partial reruns as a graph-selection problem, not as a
special alternative pipeline definition.

## Operational implications

- Prefer stable bundle keys such as `sample` and `background`; partial rerun
  detection depends on them.
- Use `changed_keys` when only a specific product changed and the dirty set can
  be narrowed further.
- Keep source registration external when file locations change frequently.
- Use dry-run before automating partial reruns against new instruments or new
  pipeline structures.

The companion pages [runtime_service_api](runtime_service_api.md) and
[backlog](backlog.md) describe how these concepts surface in the API and how the
runtime service is evolving.
