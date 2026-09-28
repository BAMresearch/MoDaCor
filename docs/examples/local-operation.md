# Local operation

The [Quickstart](../getting-started/quickstart.md) is a complete local example:
it creates synthetic `ProcessingData`, loads pipeline YAML from a string, and
calls `run_pipeline_job`.

For a reusable project, place the same YAML in `pipeline.yaml` and move data
construction or source registration into a small driver:

```python
from pathlib import Path

from modacor.runner import run_pipeline_job

result = run_pipeline_job(
    Path("pipeline.yaml"),
    processing_data=data,
    trace=True,
    trace_watch={"sample": ["signal"]},
)
```

File-backed pipelines normally register `IoSources` in the driver and use
`AppendProcessingData` steps to construct bundles. This keeps changing paths
outside the scientific pipeline definition.

The CLI uses the same runner path:

```bash
modacor run --pipeline pipeline.yaml --trace --trace-watch sample:signal
```

See [Local execution](../processing/local-execution.md) for source/sink and HDF
output options. Complete instrument drivers belong in
[MoDaCor-examples](https://github.com/BAMResearch/MoDaCor-examples).
