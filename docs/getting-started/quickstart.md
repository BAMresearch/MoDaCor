# Quickstart

This example performs a real two-step MoDaCor correction without downloading
instrument data. It constructs a synthetic detector image, adds a named Poisson
uncertainty, and divides by exposure time.

## Create the environment

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install modacor
```

Save this as `quickstart.py`:

```python
import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.runner import run_pipeline_job
from modacor.runner.pipeline import Pipeline

data = ProcessingData()
data["sample"] = DataBundle(
    signal=BaseData(
        signal=np.array([[4.0, 9.0], [16.0, 25.0]]),
        units=ureg.count,
        rank_of_data=2,
    )
)
data["exposure"] = DataBundle(
    signal=BaseData(signal=2.0, units=ureg.second)
)

pipeline = Pipeline.from_yaml(
    """
name: synthetic_quickstart
steps:
  uncertainties:
    module: PoissonUncertainties
    configuration:
      with_processing_keys: [sample]
  normalize:
    module: DivideDatabundles
    requires_steps: [uncertainties]
    configuration:
      with_processing_keys: [sample, exposure]
"""
)

result = run_pipeline_job(
    pipeline,
    processing_data=data,
    trace=True,
    trace_watch={"sample": ["signal"]},
)
corrected = result.processing_data["sample"]["signal"]

print(result.executed_steps)
print(corrected.signal)
print(corrected.units)
print(corrected.uncertainties["Poisson"])
```

Run it:

```bash
python quickstart.py
```

The corrected signal is `[[2, 4.5], [8, 12.5]] count/s`. Its named Poisson
standard uncertainty is `[[1, 1.5], [2, 2.5]] count/s`.

## What happened

- Each physical quantity is a [`BaseData`](../data-model/basedata.md) object.
- Related quantities are grouped in [`DataBundle`](../data-model/databundle.md)
  objects under pipeline-wide [`ProcessingData`](../data-model/processingdata.md).
- The YAML defines a two-node graph. `normalize` cannot run until
  `uncertainties` completes.
- `DivideDatabundles` uses `BaseData` arithmetic, so units and uncertainty
  components propagate together.
- `run_pipeline_job` returns the corrected data, executed step identifiers,
  durations, pipeline, and tracer.

Continue with [Where to go next](where-to-go-next.md).
