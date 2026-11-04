# Breaking pipeline interface changes

Until the pipeline interface is declared stable at v2.0, incompatible changes
to step configuration or pipeline YAML are released as **minor** versions.
Check this page when upgrading to a new minor release; each section lists
exactly what to change in existing pipelines.

The USAXS-driven pipeline update intentionally breaks several developmental
pipeline interfaces. Existing YAML using the retired scattering-specific
indexer, the former indexed averager configuration, or the former
`XSGeometryFromPixelCoordinates` wavelength fields must be migrated. MoDaCor
does not retain compatibility aliases for these interfaces.

The synthetic [Quickstart](../getting-started/quickstart.md) is unaffected.
Ordinary explicit `steps` documents also remain valid; the `step_blocks` and
`for_each` schema is additive.

## R20: uncertainty-source units are converted

Uncertainty datasets with a `units` attribute are now converted to the signal's
units as differences. Missing or blank attributes retain the previous assumption
that the values already use signal units. Incompatible, unparseable, and offset
units raise `ValueError` naming the uncertainty, source, and both units. Correct
the source metadata or supply a compatible uncertainty dataset. A signal's
`units_override` applies to its uncertainties too: it skips unit metadata and
retains numerical values with the override label.

## R16: negative detector-incidence cosines raise ValueError

`XSGeometryFromPixelCoordinates` now rejects a detector normal pointing toward
the sample or one that does not match the pixel plane. Previously negative
`CosAlpha` values were clipped to zero, producing zero solid angles and infinite
corrected intensities. Check the sign of `detector_normal` (or the NeXus frame)
against the pixel plane. The error names the step ID, affected pixel count,
minimum cosine, and likely cause. Zero (grazing incidence) remains allowed.

## R14: required step arguments are enforced

Pipeline YAML/spec loading and step execution now raise `ValueError` when a
required argument is missing, `None`, blank (including whitespace), or an empty
container. The message names the step class, step ID, and every missing input.
Supply the argument, or remove a step that did nothing, such as
`UnitsLabelUpdate` without `update_pairs`. Zero and `False` remain valid values.

Arguments with usable defaults remain optional and keep their defaults.
Geometry steps accept either `detector_frame` or their individual detector
sources; `XSGeometryFromPixelCoordinates` also accepts `sample_z_override`
instead of `sample_z_source`. Missing alternatives now produce the same clear
validation error at load and execution time.

Module authors must use placeholder defaults for `required: True` arguments;
the describer rejects a usable default when class metadata is defined. Express
alternative input requirements by extending `missing_required_arguments()`.

## Stage 0: mixed `propagate_to_all` mappings raise ValueError

An uncertainty mapping containing both `propagate_to_all` and another key now
raises `ValueError`. Keep `propagate_to_all` as the only key, or rename it to an
ordinary named key. This applies to configured uncertainty sources as well as
mappings modified directly in Python. Validation occurs before mutation at
arithmetic, copying, indexing, unit-conversion and uncertainty-producer entry
points. Combination modules reject `propagate_to_all` as a source key.

A producer introducing named estimates into a global-only mapping folds the
global term into each estimate produced in that call and atomically removes the
reserved key. Do not pre-build a mixed mapping to request this transition.
See [Uncertainty propagation](../data-model/uncertainty-propagation.md).

## R07: `output_processing_key` only on ConcatenateDatabundles and IndexedAverager

Remove `output_processing_key` from every other step's YAML configuration.
Those steps previously accepted and ignored the setting; they now raise the
existing unknown-key `KeyError`, including a suggested key when available.
Their output continues to be written in place or to the step's documented
output setting, such as `Integrate1D.output_processing_keys`.

`ConcatenateDatabundles` requires a string output key, defaulting to
`concatenated`. It no longer accepts `None` (`null` in YAML), which previously
created a bundle literally named `"None"`. Remove a null setting to use the
default, or replace it with the intended output bundle name.

`IndexedAverager` continues to accept a string or `None`: omit the setting or
use `null` for in-place averaging; a distinct output bundle requires exactly
one input bundle.

## R01b: Use one input per setting and explicit mask keys

Removed configuration keys raise `KeyError`; update existing YAML as follows:

| Step | Removed key | Replacement |
|---|---|---|
| `PolarizationCorrection` | `polarisation_factor` | `polarization_factor` |
| `PolarizationCorrection` | `polarisation_angular_offset` | `polarization_angular_offset` |
| `PolarizationCorrection` | `polarisation_angular_offset_units` | `polarization_angular_offset_units` |
| `AttenuatorPlateCorrection`, `DetectorEfficiencyCorrection` | `material`, `chemical_composition`, `composition` | `chemical_formula` |
| `AttenuatorPlateCorrection`, `DetectorEfficiencyCorrection` | `material_source`, `chemical_composition_source` | `chemical_formula_source` |
| `ThresholdMask` | `threshold` | `upper_bound` |

`chemical_formula` is a chemical formula in xraylib syntax, e.g. `SiO2`.
Exact xraylib NIST compound names (e.g. `Kapton Polyimide Film`) are also
accepted. Bare element symbols such as `Si` are valid; informal names such
as `Silicon` or `Kapton` are not supported. Invalid formulas raise `ValueError`
naming the formula and preserve xraylib's original error as the cause.
The helper result field `MaterialAttenuation.material` is also renamed to
`MaterialAttenuation.chemical_formula`.

For attenuation and capillary parameters, configure either a literal value or
its corresponding `_source`, never both with non-`None` values. Supplying both
now raises `ValueError` naming the conflicting keys.

Both capillary corrections read exactly the configured mask key. They no
longer search for `Mask` when configured for `mask`, or vice versa. Lowercase
`mask` is the canonical convention; keys remain case-sensitive.

`CapillarySelfAbsorptionCorrection.mask_key` and
`CapillarySampleContainerCorrection.mask_key` now default to `None`, meaning
no masking even if a mask is present. Set `mask_key: mask` to enable masking.
Any configured mask key that is absent raises `KeyError` naming the key and
its input bundle. `CapillarySampleContainerCorrection.empty_mask_key` still
defaults to `None`, meaning use `mask_key` for the empty bundle as well.
With both keys `None`, neither bundle is masked. To mask only the empty
bundle, set `mask_key: null` and `empty_mask_key` to its mask's key.

`IndexedAverager.mask_key` now defaults to `None`, meaning no masking even if
a mask is present. Set `mask_key: mask` to use the canonical mask, or explicitly
name another key. A configured but absent mask raises `KeyError` naming both
the key and its input bundle. Omit `mask_key` or use `mask_key: null` to disable
masking. `PolarizationCorrection.polarization_factor` remains optional:
unset or `None` means 0.5 (unpolarized).

## Replace `IndexPixels` with `IndexByCoordinate`

`IndexPixels` combined scattering-specific coordinate selection with bin
assignment. It has been retired. `IndexByCoordinate` instead bins any one
configured `BaseData` coordinate and writes only the integer index map and
physical bin edges. Masks remain a separate concern and are consumed by the
downstream reducer.

For example, replace:

```yaml
index:
  module: IndexPixels
  configuration:
    averaging_direction: azimuthal
    q_min: 0.01
    q_max: 60.0
    q_limits_unit: 1/nm
    bin_type: log
    n_bins: 200
    with_processing_keys: [sample]
```

with:

```yaml
index:
  module: IndexByCoordinate
  configuration:
    coordinate_key: Q
    index_key: bin_index
    bin_min: 0.01
    bin_max: 60.0
    bin_units: 1/nm
    spacing: log
    n_bins: 200
    with_processing_keys: [sample]
```

The direct configuration mapping is:

| Retired `IndexPixels` field | `IndexByCoordinate` field |
|---|---|
| `q_min` or `psi_min` | `bin_min` |
| `q_max` or `psi_max` | `bin_max` |
| `q_limits_unit` or `psi_limits_unit` | `bin_units` |
| `bin_type` | `spacing` |
| selected Q/Psi direction | `coordinate_key` |
| implicit `pixel_index` output | configurable `index_key` |

Use a separate index-and-reduce sequence for each coordinate when a workflow
needs more than one one-dimensional reduction. Two-dimensional binning is not
part of this replacement.

## Configure the generic `IndexedAverager` explicitly

`IndexedAverager` is now a generic base module. It consumes a precomputed
integer index map and does not read bin edges or choose an averaging direction.
The value, index, optional measured axis, and optional mask are explicit:

```yaml
average:
  module: IndexedAverager
  requires_steps: [index]
  configuration:
    value_key: signal
    index_key: bin_index
    axis_key: Q
    mask_key: mask
    use_value_weights: true
    use_value_uncertainty_weights: false
    uncertainty_weight_key: null
    with_processing_keys: [sample]
```

Rename `use_signal_weights` to `use_value_weights` and
`use_signal_uncertainty_weights` to `use_value_uncertainty_weights`.
`averaging_direction` is removed; use `value_key`, `index_key`, and `axis_key`
to state the operation directly. The reported axis contains the mean of the
actual accepted coordinate values in each populated bin, not nominal bin
centres.

## Use one photon-metadata interface for geometry

`XSGeometryFromPixelCoordinates` replaces:

- `wavelength_source` with `photon_source`;
- `wavelength_units_source` with `photon_units_source`; and
- `wavelength_uncertainties_sources` with
  `photon_uncertainties_sources`.

The same interface is used by `AngleToQ`. The source may contain photon energy
or wavelength; MoDaCor infers which from the supplied units and converts it
with uncertainty-aware `BaseData` arithmetic. These values are acquisition or
calibration metadata and therefore remain `IoSource` references rather than
being copied into a `DataBundle`.

YAML written against intermediate development versions of `AngleToQ` must also
replace `incident_key`, `incident_source`, or separate energy/wavelength key
descriptors with the final `photon_source`, `photon_units_source`, and optional
`photon_uncertainties_sources` fields.

```yaml
geometry:
  module: XSGeometryFromPixelCoordinates
  configuration:
    photon_source: calibration::/entry/beam/incident_energy
    photon_units_source: calibration::/entry/beam/incident_energy@units
    photon_uncertainties_sources:
      energy_calibration: calibration::/entry/beam/incident_energy_error
```

The wavelength-specific fields on other modules, including
`DetectorEfficiencyCorrection` and `AttenuatorPlateCorrection`, are unchanged.

## Update HDF pipeline-provenance readers

HDF processing outputs now preserve both the compact authored description and
the expanded executable DAG below `/processing/pipeline/<run-id>/`:

```text
authored/yaml
authored/spec
expanded/yaml
expanded/spec
```

The former direct `yaml` and `spec` datasets are not retained as aliases.
Readers of MoDaCor HDF provenance must select the authored representation for
the submitted document or the expanded representation for the graph that was
executed.

## R18: AngleToQ output_units removed; use ConvertUnits

`AngleToQ` now always returns momentum transfer Q in its natural units, `1/m`.
The `output_units` configuration option is removed. Remove it from existing
AngleToQ steps and add a ConvertUnits step after angle conversion (or before
writing output). ConvertUnits scales both values and all absolute uncertainty
components, while preserving axis references and relative weights.

```yaml
name: converted_q
steps:
  angle_to_q:
    module: AngleToQ
    configuration:
      with_processing_keys: [sample]
      photon_source: measurement::/beam/wavelength
      photon_units_source: measurement::/beam/wavelength@units
      center_key: null
  output_units:
    module: ConvertUnits
    requires_steps: [angle_to_q]
    configuration:
      with_processing_keys: [sample]
      units: {Q: 1/nm}
```

Use `with_processing_keys: "*"` to convert every bundle containing Q; missing
keys are logged and skipped. Explicit bundle lists require every configured key.
This pipeline-configuration change is released as a minor version under the
pre-v2 policy. MoDaCor-examples `DLS/I22/pipelines/I22_USAXS.yaml` needs the same
migration after the release.
