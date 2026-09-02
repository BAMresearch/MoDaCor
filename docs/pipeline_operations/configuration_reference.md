# Pipeline configuration reference

This page summarises the YAML keys understood by MoDaCor pipeline definitions.

## Step fields

Each entry under `steps` is keyed by a `step_id` and supports the following fields:

- `module` (required): ProcessStep class name to instantiate.
- `requires_steps` (optional): list of step_ids that must run before this step.
- `configuration` (optional): dictionary of ProcessStep configuration values.
- `short_title` (optional): a brief, human-friendly purpose label used in graphs (Mermaid/DOT). This is appended as a
  second line in node labels, e.g. `AU: MultiplyDatabundles` + `scaling to absolute units`.

## Step configuration validation

Each `configuration` block is checked when `Pipeline.from_yaml(...)` loads the
pipeline. MoDaCor builds the accepted key list and top-level type policy from
the selected module class:

- shared `ProcessStep` keys from `CONFIG_KEYS`
- module-specific keys from `ProcessStepDescriber.arguments`

For example, `Divide.divisor_source` is declared as a string source reference,
so this is valid:

```yaml
steps:
  normalize:
    module: Divide
    configuration:
      with_processing_keys:
        - sample
      divisor_source: sample::entry/frame_exposure_time
```

and this fails during pipeline loading because `divisor_source` is not a string:

```yaml
steps:
  normalize:
    module: Divide
    configuration:
      divisor_source: 3
```

The central validator catches unknown keys and top-level type mismatches. Module
code still performs semantic checks for values that need runtime context, such
as missing sources, non-empty required strings, mutually exclusive options, or
nested dictionary contents.

## Threshold masks

`ThresholdMask` creates a uint32 mask from any `BaseData` entry in a selected
`DataBundle`, not only from `signal`. This is useful when a correction map
stored with the measurement should define invalid detector pixels. For example,
to mask pixels whose flatfield correction matrix falls outside an acceptable
range:

```yaml
steps:
  load_flatfield:
    module: AppendProcessingData
    configuration:
      processing_key: sample
      signal_location: sample::entry/instrument/detector/flatfield
      rank_of_data: 2
      databundle_output_key: flatfield
      units_override: dimensionless
  flatfield_mask:
    module: ThresholdMask
    requires_steps: [load_flatfield]
    configuration:
      with_processing_keys: [sample]
      source_basedata_key: flatfield
      target_mask_key: flatfield_mask
      lower_bound: 0.8
      upper_bound: 1.2
      mask_mode: outside
```

Use `mask_mode: outside` to mask values below `lower_bound` or above
`upper_bound`. Use `mask_mode: inside` to mask values within the inclusive
range instead. The older `threshold` option is still accepted as an upper-bound
alias when `upper_bound` is not configured. The created mask keeps the same
array shape as the `source_basedata_key`; leading image or frame axes are not
collapsed before thresholding.

The same step can create geometry masks from `Q` or `Psi` BaseData entries. For
example, this masks pixels outside a radial Q range and outside an azimuthal Psi
range, combines those masks, and applies the combined mask to the sample
signal:

```yaml
steps:
  q_range_mask:
    module: ThresholdMask
    configuration:
      with_processing_keys: [sample]
      source_basedata_key: Q
      target_mask_key: q_mask
      lower_bound: 0.05
      upper_bound: 3.0
      mask_mode: outside
  psi_range_mask:
    module: ThresholdMask
    configuration:
      with_processing_keys: [sample]
      source_basedata_key: Psi
      target_mask_key: psi_mask
      lower_bound: -0.7853981633974483
      upper_bound: 0.7853981633974483
      mask_mode: outside
  combine_geometry_masks:
    module: BitwiseOrMasks
    requires_steps: [q_range_mask, psi_range_mask]
    configuration:
      with_processing_keys: [sample]
      target_mask_key: mask
      source_mask_keys: [q_mask, psi_mask]
  apply_geometry_mask:
    module: ApplyMask
    requires_steps: [combine_geometry_masks]
    configuration:
      with_processing_keys: [sample]
      mask_key: mask
      basedata_to_mask: [signal]
```

Use `mask_mode: inside` for the inverse region of interest, for example to mask
only a beamstop band or a known bad azimuthal sector while leaving the rest of
the detector unmasked.

When a pipeline is loaded through a runtime service using the restricted
runtime policy, the `module` name must resolve through the service's curated or
explicit `ProcessStepRegistry`. Filesystem discovery of unregistered module
files is disabled in that mode. If loading fails with `filesystem discovery is
disabled by runtime policy`, export/register the step in the service registry or
run a trusted local service.
