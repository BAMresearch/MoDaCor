# Breaking pipeline interface changes

The USAXS-driven pipeline update intentionally breaks several developmental
pipeline interfaces. Existing YAML using the retired scattering-specific
indexer, the former indexed averager configuration, or the former
`XSGeometryFromPixelCoordinates` wavelength fields must be migrated. MoDaCor
does not retain compatibility aliases for these interfaces.

The synthetic [Quickstart](../getting-started/quickstart.md) is unaffected.
Ordinary explicit `steps` documents also remain valid; the `step_blocks` and
`for_each` schema is additive.

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
