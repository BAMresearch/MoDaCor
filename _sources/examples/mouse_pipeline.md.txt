# MOUSE pipeline patterns

MoDaCor does not currently ship a full production MOUSE pipeline definition in
the repository, but it does already provide a stable MOUSE-oriented workflow
through the quickstart, bundled test data, and the runtime-service `mouse`
source profile.

## Current repo assets

- `docs/getting_started/quickstart.md` runs a small MOUSE correction example
  against `tests/testdata/MOUSE_20250324_1_160_stacked.nxs`.
- `src/modacor/server/source_profiles.py` defines the `mouse` source profile
  used by the runtime service and session CLI.
- `docs/pipeline_operations/runtime_service_api.md` shows a full API lifecycle
  using a MOUSE-style session.

## Typical source set

The runtime-service `mouse` profile currently expects:

- required: `sample`, `background`
- optional: `defaults`, `intensity_calibration`,
  `intensity_calibration_background`

For smaller local experiments, the quickstart uses only:

- `sample` as an HDF/NeXus source
- `metadata` as a YAML source containing scalar defaults such as detector dark
  current

## Minimal correction pattern

The quickstart demonstrates a compact three-step MOUSE-style flow:

1. `PoissonUncertainties`
2. `Divide` by frame exposure time from the sample file
3. `Subtract` a scalar dark-current estimate from YAML metadata

That pattern is useful when validating a new installation because it exercises:

- HDF-backed `IoSource` access
- YAML-backed metadata lookup
- unit propagation
- pipeline tracing and graph rendering

## Example snippet

```yaml
steps:
  1:
    module: PoissonUncertainties
    configuration:
      with_processing_keys: [sample]
  2:
    module: Divide
    requires_steps: [1]
    configuration:
      with_processing_keys: [sample]
      divisor_source: sample::entry1/instrument/detector00/frame_exposure_time
      divisor_units_source: sample::entry1/instrument/detector00/frame_exposure_time@units
  3:
    module: Subtract
    requires_steps: [2]
    configuration:
      with_processing_keys: [sample]
      subtrahend_source: metadata::detector/darkcurrent/value
      subtrahend_units_source: metadata::detector/darkcurrent/units
```

## Where to go next

- Use the quickstart if you need a runnable local example.
- Use the runtime-service docs if you need session-based orchestration with a
  `mouse` profile.
- Use `docs/pipeline_operations/pipeline_basics.md` if you need to explain how
  sources, `ProcessingData`, and partial reruns fit together.
