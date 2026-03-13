# CLI And Runner API

Use MoDaCor either from the command line (`modacor run`) or from Python via a shared execution helper
(`run_pipeline_job`). Both paths use the same pipeline scheduler backend.

## Prerequisites

- Install MoDaCor in an environment with Python 3.11+.
- If using a source checkout, install in editable mode:

```bash
pip install -e .
```

## Command-line interface

The package now installs a `modacor` command with a `run` subcommand.

### Minimal run

```bash
modacor run --pipeline processing_pipelines/MOUSE_solids.yaml
```

This is useful for pipelines that already load data sources/sinks internally (for example using `AppendSource` and
`AppendSink` steps).

### Run with external source registration

For pipelines that expect sources to be provided externally:

```bash
modacor run \
  --pipeline processing_pipelines/MOUSE_solids.yaml \
  --hdf-source sample=/data/MOUSE_sample.nxs \
  --hdf-source background=/data/MOUSE_background.nxs \
  --yaml-source defaults=/data/defaults.yaml
```

Supported source/sink registration flags:

- `--hdf-source REF=PATH` (repeatable)
- `--yaml-source REF=PATH` (repeatable)
- `--csv-sink REF=PATH` (repeatable)

### Tracing and step control

```bash
modacor run \
  --pipeline processing_pipelines/MOUSE_solids.yaml \
  --trace \
  --trace-watch sample:signal,Q \
  --trace-watch background:signal \
  --trace-report-lines 50 \
  --stop-after GX
```

- `--trace` enables trace capture and event attachment.
- `--trace-watch` is repeatable and uses `bundle:key[,key...]`.
- `--stop-after STEP_ID` stops the run after that step is executed.

### Export selected results to HDF5

Use repeatable `--write-path` selectors:

```bash
modacor run \
  --pipeline processing_pipelines/MOUSE_solids.yaml \
  --write-hdf output/results.h5 \
  --run-name run1 \
  --write-path /sample/signal/signal \
  --write-path /sample/Q/signal \
  --write-path /background/signal/signal
```

If you want to store the entire current `ProcessingData` (all `BaseData` entries) without listing paths:

```bash
modacor run \
  --pipeline processing_pipelines/MOUSE_solids.yaml \
  --write-hdf output/results_full.h5 \
  --write-all-processing-data
```

Semantics:

- `--write-hdf` sets the output file.
- each `--write-path` adds one `ProcessingData` path to `data_paths`.
- `--write-all-processing-data` auto-selects all `BaseData` entries from `ProcessingData`.
- `--run-name` maps to the HDF sink run subpath (default: `default`).
- the HDF output stores reproducibility metadata under `processing/pipeline/<run-name>/`:
  `spec` (JSON) and `yaml` (pipeline YAML).
- trace output is stored under `processing/tracer/<run-name>/` as raw `events` JSON and indexed `steps/` + `index/`.

## Shared Python runner API

Use this when driving MoDaCor in notebooks or scripts while keeping behavior consistent with the CLI:

```python
from pathlib import Path

from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sources import IoSources
from modacor.runner import run_pipeline_job

sources = IoSources()
sources.register_source(
    HDFSource(source_reference="sample", resource_location=Path("/data/sample.nxs"))
)

result = run_pipeline_job(
    Path("processing_pipelines/MOUSE_solids.yaml"),
    sources=sources,
    trace=True,
    trace_watch={"sample": ["signal"]},
)

print(result.executed_steps)
print(result.processing_data.keys())
```

`run_pipeline_job(...)` returns a `RunResult` container with:

- `processing_data`
- `pipeline`
- `tracer` (or `None`)
- `step_durations`
- `executed_steps`
- `stopped_after_step`
