# SAXSess pipeline patterns

The repository currently contains a semirealistic SAXSess-style workflow in the
integration tests and an explicit runtime-service `saxsess` source profile. Use
this page as the current reference for source conventions and correction order.

## Current repo assets

- `tests/integration/test_pipeline_run.py` contains a full SAXSess-flavored
  pipeline assembled from temporary HDF and YAML inputs.
- `src/modacor/server/source_profiles.py` defines the `saxsess` runtime source
  profile.

## Typical source set

The `saxsess` profile currently expects:

- required: `sample`, `sample_background`, `defaults`
- optional: `intensity_calibration`,
  `intensity_calibration_background`, `gc_calibration`

This matches the common freestanding-solids flow where instrument defaults come
from YAML, measurements come from HDF, and an optional glassy-carbon reference
curve is supplied separately.

## Representative correction pattern

The integration test currently uses this step sequence:

1. `PoissonUncertainties` for sample and background
2. `Divide` by exposure time stored in each HDF file
3. `Subtract` dark current from YAML metadata
4. `SubtractDatabundles` to remove the measured background

## Example snippet

```yaml
steps:
  1:
    module: PoissonUncertainties
    configuration:
      with_processing_keys: [sample, sample_background]
  2:
    module: Divide
    requires_steps: [1]
    configuration:
      divisor_source: sample::detector/frame_exposure_time
      divisor_units_source: sample::detector/frame_exposure_time@units
      with_processing_keys: [sample, sample_background]
  3:
    module: Subtract
    requires_steps: [2]
    configuration:
      subtrahend_source: yaml::detector/darkcurrent/value
      subtrahend_units_source: yaml::detector/darkcurrent/units
      with_processing_keys: [sample, sample_background]
  bg:
    module: SubtractDatabundles
    requires_steps: [3]
    configuration:
      with_processing_keys: [sample, sample_background]
```

## Notes on naming

The integration test uses `yaml` as the metadata source reference, while the
runtime-service profile uses `defaults`. Both patterns are valid; choose one and
use it consistently within a given pipeline definition.

## Recommended use

Use this workflow as a regression and onboarding example when you need:

- a branched multi-source pipeline
- both scalar metadata and array data access
- a realistic background-subtraction sequence

For lower-friction local setup, start with the MOUSE quickstart first and then
adapt the source naming and pipeline graph for SAXSess-specific work.
