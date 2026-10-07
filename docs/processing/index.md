# Processing framework

MoDaCor executes a directed acyclic graph of configured `ProcessStep` objects.
Steps exchange in-memory `ProcessingData`, read registered sources, publish to
registered sinks, and emit trace information through a shared runner.

```{toctree}
:maxdepth: 1

process-steps
pipeline-graphs
pipeline-configuration
interface-migrations
sources
sinks
local-execution
dependencies-and-partial-runs
tracing-and-provenance
chunked-processing
```
