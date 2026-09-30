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

For pipelines authored with `step_blocks`, both renderers group generated
nodes automatically. The block is shown as an outer cluster or subgraph, each
`for_each` item is shown as a processing lane, and the nodes and edges remain
the real expanded execution graph. DOT also aligns matching local steps across
lanes as stages where Graphviz permits it.

```python
dot_source = pipeline.to_dot()
mermaid_source = pipeline.to_mermaid()
```

`TD` is accepted as a top-down direction alias and emitted as Mermaid's
canonical `TB` spelling so the direction also remains valid inside nested item
subgraphs.

Grouping is a visual interpretation of each node's `origin` metadata. It does
not create a block node or change scheduling and dependency semantics. To
inspect the traditional flat graph, disable it explicitly:

```python
flat_dot = pipeline.to_dot(group_step_blocks=False)
flat_mermaid = pipeline.to_mermaid(group_step_blocks=False)
```

Mermaid uses private renderer-local identifiers such as `node_0`; the visible
labels still contain the complete expanded step ids. This prevents distinct
pipeline ids containing punctuation from colliding after Mermaid identifier
sanitization.
