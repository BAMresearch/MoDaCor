# MoDaCor

Reference-quality, traceable data corrections for monochromatic X-ray and
neutron scattering, diffraction, and imaging.

[![PyPI](https://img.shields.io/pypi/v/modacor.svg)](https://pypi.org/project/modacor)
[![Python](https://img.shields.io/pypi/pyversions/modacor.svg)](https://pypi.org/project/modacor)
[![License](https://img.shields.io/pypi/l/modacor.svg)](LICENSE)
[![CI](https://github.com/BAMresearch/MoDaCor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/BAMresearch/MoDaCor/actions/workflows/ci-cd.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://BAMresearch.github.io/MoDaCor/coverage-report/cov.json)](https://BAMresearch.github.io/MoDaCor/coverage-report/)

MoDaCor applies corrections as inspectable processing steps while preserving
physical units, multiple named uncertainty contributions, and provenance. It
prioritizes correction quality, reproducibility, and scientific review over
maximum throughput. Use it as a primary correction system or as a reference
against which faster instrument-specific implementations can be checked.

## QuickStart

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then let
it install Python and create an isolated environment:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install modacor
```

This self-contained pipeline adds Poisson uncertainties to a synthetic detector
image and normalizes the counts by exposure time:

<!-- quickstart-python-start -->
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
<!-- quickstart-python-end -->

The signal is now `[[2, 4.5], [8, 12.5]] count/s`; the named Poisson standard
uncertainty was propagated to `[[1, 1.5], [2, 2.5]] count/s`.

## Learn more

- [Full documentation](https://BAMresearch.github.io/MoDaCor)
- [Instrument notebooks, pipelines, and datasets](https://github.com/BAMResearch/MoDaCor-examples)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [BSD-3-Clause license](LICENSE)

MoDaCor implements the modular correction concepts described in
[Pauw et al. (2017)](https://doi.org/10.1107/S1600576717015096).
