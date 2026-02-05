
# Quickstart

Run a three-step MoDaCor pipeline against the bundled MOUSE example dataset, see how data sources plug into a
configuration file, and inspect the pipeline trace that records what changed at every step.

## Prerequisites

- Python 3.11 or newer
- `pip`, `curl` (or `wget`) and a POSIX-like shell
- Approximately 1.3 GB of free disk space for the sample NeXus file

If you are working from the cloned MoDaCor repository, activate the project virtual environment instead of creating a
new one and use `pip install -e .` to install the package in editable mode.

## Step 1 – Prepare a working folder

Create a clean folder, bootstrap a virtual environment, and install MoDaCor:

```bash
mkdir modacor-quickstart
cd modacor-quickstart
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install modacor
```

## Step 2 – Download example data and metadata

Grab the MOUSE sample dataset and create a small metadata file describing the detector dark current:

```bash
curl -LO https://github.com/BAMresearch/modacor/raw/main/tests/testdata/MOUSE_20250324_1_160_stacked.nxs

cat <<'YAML' > mouse_metadata.yaml
---
detector:
  darkcurrent:
    value: 1.0e-5
    units: counts/second
    uncertainty: 1.0e-6
YAML
```

The NeXus file exposes counts in `entry1/instrument/detector00/data` and the exposure time in
`entry1/instrument/detector00/frame_exposure_time`. The metadata file supplies a scalar dark-current estimate so the
last pipeline step can remove it.

## Step 3 – Create the pipeline configuration

Save the following pipeline definition as `mouse_quickstart.yaml`:

```yaml
name: mouse_quickstart
steps:
  1:
    name: add_poisson_uncertainties
    module: PoissonUncertainties
    requires_steps: []
    configuration:
      with_processing_keys:
        - sample
  2:
    name: normalize_by_exposure
    module: Divide
    requires_steps: [1]
    configuration:
      with_processing_keys:
        - sample
      divisor_source: sample::entry1/instrument/detector00/frame_exposure_time
      divisor_units_source: sample::entry1/instrument/detector00/frame_exposure_time@units
  3:
    name: subtract_darkcurrent
    module: Subtract
    requires_steps: [2]
    configuration:
      with_processing_keys:
        - sample
      subtrahend_source: metadata::detector/darkcurrent/value
      subtrahend_units_source: metadata::detector/darkcurrent/units
      subtrahend_uncertainties_sources:
        propagate_to_all: metadata::detector/darkcurrent/uncertainty
```

## Step 4 – Create a runner script

Place the script below in `run_mouse_pipeline.py`. It registers the data sources, prepares a `ProcessingData` object,
runs the pipeline, and prints both numeric results and a compact pipeline trace.

```python
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.debug.pipeline_tracer import PipelineTracer, PlainUnicodeRenderer
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sources import IoSources
from modacor.io.yaml.yaml_source import YAMLSource
from modacor.runner.pipeline import Pipeline


def _decode_unit(unit_value) -> str:
    if isinstance(unit_value, bytes):
        return unit_value.decode()
    return str(unit_value)


def build_processing_data(sources: IoSources) -> ProcessingData:
    processing = ProcessingData()
    processing["sample"] = DataBundle()

    signal = sources.get_data("sample::entry1/instrument/detector00/data")
    signal_unit = _decode_unit(
        sources.get_data_attributes("sample::entry1/instrument/detector00/data").get("units", "counts")
    )

    processing["sample"]["signal"] = BaseData(
        signal=signal,
        units=ureg.Unit(signal_unit),
        rank_of_data=2,  # last two dimensions carry detector pixels
    )
    return processing


def main() -> None:
    pipeline = Pipeline.from_yaml_file(Path("mouse_quickstart.yaml"))

    sources = IoSources()
    sources.register_source(
        YAMLSource(source_reference="metadata", resource_location=Path("mouse_metadata.yaml"))
    )
    sources.register_source(
        HDFSource(source_reference="sample", resource_location=Path("MOUSE_20250324_1_160_stacked.nxs"))
    )

    processing_data = build_processing_data(sources)
    tracer = PipelineTracer(watch={"sample": ["signal"]})

    pipeline.prepare()
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.processing_data = processing_data
            node.io_sources = sources

            start = perf_counter()
            node.execute(processing_data)
            tracer.after_step(node, processing_data, duration_s=perf_counter() - start)

            pipeline.done(node)

    sample_signal = processing_data["sample"]["signal"]
    mean_intensity = float(sample_signal.signal.mean())
    print(f"Mean intensity after corrections: {mean_intensity:.6g} {sample_signal.units}")

    print("\nPipeline trace (last few events):\n")
    print(tracer.last_report(renderer=PlainUnicodeRenderer()))

    print("\nMermaid flowchart definition:\n")
    print(pipeline.to_mermaid())


if __name__ == "__main__":
    main()
```

## Step 5 – Run the pipeline

Execute the script:

```bash
python run_mouse_pipeline.py
```

You should see the corrected mean intensity, a compact trace summarising what changed in each step (unit conversions,
shape, NaN counts, etc.), and a Mermaid flowchart definition that can be pasted into <https://mermaid.live> for a visual
graph.

## Step 6 – Where to go next

- Swap out `mouse_metadata.yaml` for the metadata produced by your instrument and adjust `with_processing_keys` for
  additional `DataBundle` entries (for example `background` or `calibration`).
- Add `pipeline.attach_tracer_event(node, tracer, include_rendered_trace=True)` inside the execution loop if you want to
  export the trace alongside the configuration.
- Explore the **Pipeline operations** and **Extending MoDaCor** sections for branching workflows, module development,
  and integration best practices.
