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

## DataBundle arithmetic

The source-based `Divide`, `Subtract`, and `Multiply` modules load their second
operand from `IoSources`. Their interfaces remain source-oriented.

Use `DivideDatabundles` when both operands have already been prepared as
`BaseData` entries in `ProcessingData`. `with_processing_keys` contains exactly
two keys: the dividend first and the divisor second. The default entry name is
`signal` in both bundles; `dividend_data_key` and `divisor_data_key` can select
different entries.

```yaml
steps:
  normalize_to_count_time:
    module: DivideDatabundles
    configuration:
      with_processing_keys: [sample, sample_count_time]
      dividend_data_key: signal
      divisor_data_key: signal
```

The dividend is updated in place and returned. `BaseData` arithmetic supplies
array broadcasting, unit calculation, and uncertainty propagation. This is the
division counterpart of the existing `SubtractDatabundles` and
`MultiplyDatabundles` steps.

## Sampled-data integration

`Integrate1D` integrates one or more sampled curves with trapezoidal or Simpson
quadrature. The curves share a coordinate array, which may be nonuniform but
must be strictly monotonic. Coordinate units are multiplied into the result
units and each uncertainty component is propagated independently with the
quadrature coefficients.

```yaml
steps:
  integrate_sample_and_blank:
    module: Integrate1D
    configuration:
      with_processing_keys: [sample_curve, blank_curve]
      signal_key: signal
      axis_key: q
      method: trapezoid
      output_processing_keys: [sample_integral, blank_integral]
```

Invalid, masked, or zero-weight samples in any input are omitted from every
integral so ratios use the same physical domain. Restricting this module to 1D
keeps the coordinate and uncertainty rules unambiguous; an n-dimensional
sampled-data integrator can be introduced later if a concrete pipeline
requires one.

## NeXus detector frames

MoDaCor has a generic NeXus transformation-chain resolver in the base modules.
It follows scalar `depends_on` links, resolves relative NeXus paths, and returns
a 4x4 affine transform matrix in SI length units. The resolver is deliberately
not a scattering-only feature: the same NeXus transformation-chain convention
can describe detectors, sample stages, and other instrument components.

Scattering detector coordinate modules can use this resolver through a
`detector_frame` configuration block. This keeps pipeline YAML lean because
`PixelCoordinates3D` can read the detector origin, fast/slow pixel directions,
pixel pitches, and detector normal from the configured `NXdetector` directly:

```yaml
steps:
  pixel_coordinates:
    module: PixelCoordinates3D
    configuration:
      with_processing_keys: [static]
      detector_frame:
        type: nexus
        source: calibration
        detector_path: /entry1/instrument/detector
        module_origin: first_pixel_center
```

`XSGeometryFromPixelCoordinates` accepts the same `detector_frame` block to
reuse the NeXus pixel pitches and detector normal for solid-angle calculation:

```yaml
steps:
  scattering_geometry:
    module: XSGeometryFromPixelCoordinates
    requires_steps: [pixel_coordinates]
    configuration:
      with_processing_keys: [static]
      detector_frame:
        type: nexus
        source: calibration
        detector_path: /entry1/instrument/detector
        module_origin: first_pixel_center
      sample_z_override:
        value: 0.0
        units: mm
      wavelength_source: calibration::/entry1/calibration_sample/beam/incident_wavelength
      wavelength_units_source: calibration::/entry1/calibration_sample/beam/incident_wavelength@units
```

For measurement files with a NeXus sample-stage transformation chain,
`sample_z_override` can also resolve the sample position from that chain:

```yaml
      sample_z_override:
        type: nexus
        source: sample
        transform_path: /entry1/sample/transformations/sample_z
        component: z
```

`component` defaults to `z` and is extracted from the resolved lab-frame
translation. The current scattering geometry calculation still models the
sample position as `(0, 0, sample_z)`; this override makes chained NeXus
translations usable for the z coordinate without adding a separate
preprocessing copy step.

The `source` value is an `IoSources` reference. `detector_path` points to the
`NXdetector`; by default MoDaCor reads its `detector_module` child. Use
`detector_module_name` when the module has a different child name.

`module_origin` controls how an `NXdetector_module/module_offset` is interpreted:

- `corner`: use the resolved module offset directly.
- `first_pixel_center`: treat the resolved module offset as the first pixel
  centre. MoDaCor converts it back to a corner origin because
  `PixelCoordinates3D` adds the half-pixel centre shift internally.

The older explicit configuration form remains supported. Use direct
`det_coord_*_source`, `pixel_pitch_*_source`, and `basis_*` values when data are
not NeXus encoded or when a pipeline intentionally overrides the file geometry.

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

For dynamic detector data, keep the dimensionality semantics explicit:

- Use `ThresholdMask` before normalization or frame averaging to create a
  frame-resolved raw-count mask.
- Use `ReduceMask` to collapse non-detector axes while preserving NeXus bitfield
  reason bits. `reduction: any` keeps a pixel masked if it was masked in any
  reduced frame; `reduction: all` keeps only bits present in every reduced
  frame.
- Use `BitwiseOrMasks` to combine a reduced dynamic mask with a static
  instrument mask before `IndexedAverager`.

`ReduceDimensionality` is intentionally signal-oriented and performs numeric
mean/sum reductions. It is not a replacement for `ReduceMask`, because numeric
averaging does not preserve integer mask bitfields.

Both modules accept `axes: non_data` to derive the leading reduction axes from
`BaseData.rank_of_data`. MoDaCor treats the final `rank_of_data` dimensions as
data dimensions, so a signal shaped `(frames, singleton, y, x)` with
`rank_of_data: 2` is reduced over axes `(0, 1)`. Data already at its declared
rank is left unchanged.

`ReduceDimensionality` can consume an integer mask directly without modifying
the source signal. `mask_key` names the `BaseData` mask in the same
`DataBundle`. By default every nonzero reason bit is excluded; `mask_bits` can
restrict exclusion to one uint32 bitfield value or a list of values:

```yaml
steps:
  average_frames:
    module: ReduceDimensionality
    configuration:
      with_processing_keys: [sample]
      axes: non_data
      reduction: mean
      mask_key: mask
      mask_bits: [1, 4]
      nan_policy: propagate
```

The mask must have an integer dtype and be broadcast-compatible with the
signal. Selected elements receive zero effective weight in the signal
reduction, propagated uncertainties, and scatter-derived estimators. Masked
elements are omitted independently of `nan_policy`, so a masked `NaN` does not
propagate but an unmasked `NaN` does. The source signal and mask remain
unchanged.

When every contributor to an output position is masked, a mean is `nan`, a sum
is zero, and scatter-derived estimators are `nan`.

Direct mask selection does not reduce the stored mask. Use `ReduceMask`
separately if the output bundle needs a mask with the reduced dimensionality.

The optional `uncertainty_estimation` mapping adds scatter-derived uncertainty
components under user-selected keys. Available methods are
`standard_deviation`, `standard_error_mean` (mean reductions), and
`standard_error_sum` (sum reductions). `ddof` defaults to `1`.

```yaml
steps:
  average_frames:
    module: ReduceDimensionality
    configuration:
      with_processing_keys: [sample]
      axes: non_data
      reduction: mean
      use_weights: true
      nan_policy: omit
      uncertainty_estimation:
        collision_policy: error
        estimators:
          frame_STD:
            method: standard_deviation
            ddof: 1
          frame_SEM:
            method: standard_error_mean
            ddof: 1
```

Estimator mapping keys such as `frame_STD` and `frame_SEM` are the exact output
keys in `BaseData.uncertainties`. If a key already exists after normal
uncertainty propagation, `collision_policy` may be `error` (the default),
`overwrite_existing`, `keep_existing`, or `propagate`. Here, *existing* means
the uncertainty already present at the reduced output shape. `propagate`
combines the existing and estimated components in quadrature and therefore
assumes that they are independent. An individual estimator may override the
enclosing collision policy.

For a sum reduction, `standard_error_sum` estimates uncertainty of the sum
from contributor scatter. `standard_deviation` remains available but describes
the scatter of the contributing values rather than uncertainty of the sum:

```yaml
uncertainty_estimation:
  collision_policy: error
  estimators:
    summed_repeatability:
      method: standard_error_sum
      ddof: 1
```

These estimates are distinct from propagation of uncertainty components
already attached to the input. Known per-value uncertainties, including
Poisson uncertainties, are normally best attached before reduction and allowed
to propagate through the existing mean or sum formulas. The complete design
and statistical contract is recorded in
[ReduceDimensionality uncertainty estimators](../design/reduce-dimensionality-uncertainty-estimators.md).

```yaml
steps:
  average_frames:
    module: ReduceDimensionality
    configuration:
      with_processing_keys: [sample]
      axes: non_data
      reduction: mean
  reduce_frame_mask:
    module: ReduceMask
    configuration:
      with_processing_keys: [sample]
      source_mask_key: threshold_mask
      target_mask_key: threshold_mask
      axes: non_data
      reduction: any
```

`ApplyMask` remains available when replacing masked signal values is preferred
over direct, non-mutating selection. Its `masked_value` defaults to `nan`;
explicit sentinel values remain available:

```yaml
steps:
  apply_mask:
    module: ApplyMask
    configuration:
      with_processing_keys: [sample]
      mask_key: mask
      basedata_to_mask: [signal]
      masked_value: nan
```

When a pipeline is loaded through a runtime service using the restricted
runtime policy, the `module` name must resolve through the service's curated or
explicit `ProcessStepRegistry`. Filesystem discovery of unregistered module
files is disabled in that mode. If loading fails with `filesystem discovery is
disabled by runtime policy`, export/register the step in the service registry or
run a trusted local service.
