# Capillary sample self-absorption correction

`CapillarySelfAbsorptionCorrection` corrects detector-resolved scattering from
a sample inside a straight, concentric cylindrical capillary. It supports a
two-dimensional detector, a measured or analytic two-dimensional beam profile,
separate absorption by the sample and capillary wall, and a horizontal or
tilted capillary.

The correction traces rays from illuminated scattering points to the actual
detector pixels. It does not reduce the detector to a one-dimensional
$2\theta$ curve, so pixels with the same $2\theta$ may receive different
corrections when their azimuth or the capillary orientation differs.

MoDaCor provides two process steps over the same numerical model:

- `CapillarySelfAbsorptionCorrection` applies the **sample-origin** correction
  when wall scattering is negligible or has already been removed consistently;
- `CapillarySampleContainerCorrection` jointly treats matched filled and empty
  measurements, including the different wall-origin attenuation in each.

The wall attenuation of sample scattering is included in both. The second step
is required when the wall's own scattering cannot be neglected.

## What the module calculates

The initial geometry consists of two concentric radial phases:

1. a central sample of radius `sample_radius` and linear attenuation
   coefficient `sample_mu`;
2. an optional capillary wall of thickness `wall_thickness` and coefficient
   `wall_mu`.

The radius and coefficient are deliberately separate inputs. Both phases may
therefore use independently measured or calculated values, and the code forms
products such as $\mu R$ only internally. All configured lengths and
coefficients are converted to metres and reciprocal metres before the
calculation.

For detector pixel $d$, the absolute sample-origin attenuation is

$$
A_{s,sc}(d) =
\frac{\sum_v w_v
\exp[-\mu_s L_{s,v,d} - \mu_w L_{w,v,d}]}
{\sum_v w_v}.
$$

Here $v$ indexes beam-weighted quadrature points in the illuminated sample
volume. $L_s$ and $L_w$ are the total incoming-plus-outgoing path lengths in
the sample and wall for that point and detector pixel. The calculation uses
analytic ray intersections with each cylindrical boundary and
Gauss--Legendre quadrature along only the occupied sample chords. It does not
construct a three-dimensional voxel box.

The module also predicts the direct-beam transmission

$$
T_{calc} =
\frac{\sum_b w_b
\exp[-\mu_s L^{through}_{s,b} - \mu_w L^{through}_{w,b}]}
{\sum_b w_b},
$$

where $b$ runs over the complete beam profile. Rays that miss the capillary
have zero material path and are included with transmission one.

`T_calc` is a diagnostic. It is not substituted for an experimentally measured
transmission. It is also not generally equal to the forward-scattering limit
of $A_{s,sc}$: direct transmission is beam-area weighted, whereas scattering
attenuation is illuminated-volume weighted.

## Geometry and coordinate convention

The default MoDaCor laboratory frame is:

- incident beam direction: $(0,0,1)$;
- horizontal capillary axis: $(1,0,0)$;
- vertical direction: $(0,1,0)$.

The default configuration is therefore:

```yaml
capillary_axis: [1.0, 0.0, 0.0]
incident_direction: [0.0, 0.0, 1.0]
capillary_centre: [0.0, 0.0, 0.0]
capillary_centre_units: m
```

`capillary_centre` is a point on the cylinder centreline. The beam-profile
plane passes through that point, which expresses the current assumption that
the capillary is centred in the beam.

The module reads `coord_x`, `coord_y`, and `coord_z` from each selected
`DataBundle`. These arrays must be length-valued detector-element centres and
must broadcast to a two-dimensional detector grid. They are normally produced
by `PixelCoordinates3D` before this correction runs. Tilted or rotated detector
panels work because the calculation uses their three-dimensional pixel
positions rather than an assumed flat angular grid.

### Tilted capillaries

Any finite, non-zero `capillary_axis` vector is normalized internally, so a
tilted capillary can be configured directly:

```yaml
capillary_axis: [1.0, 0.15, 0.05]
```

The implemented cylinder is straight and effectively infinite. A beam or exit
ray that is parallel, or numerically too close to parallel, to the capillary
axis is rejected. In that geometry the capillary end faces and finite length
determine the path, and the infinite-cylinder model is insufficient.

Even for a horizontal capillary, the correction need not be constant along
the detector direction parallel to its axis. At finite detector distance,
pixels displaced along that direction receive oblique rays with longer
physical paths. Mirror symmetry is expected for a centred symmetric setup;
strict one-dimensional invariance is not.

## Required upstream data

Each processing key selected through `with_processing_keys` requires:

| Key | Meaning |
|---|---|
| `signal` | Detector signal to correct. |
| `coord_x` | Lab-frame detector-pixel x coordinate. |
| `coord_y` | Lab-frame detector-pixel y coordinate. |
| `coord_z` | Lab-frame detector-pixel z coordinate. |
| `mask` | Optional boolean mask; `True` means inactive. |

The coordinate key names and mask key are configurable. The default mask name
is `mask`, with `Mask` accepted as a compatibility alias.

For a scan-shaped mask, a detector pixel is skipped only when it is masked in
every leading scan or frame position. A pixel active in at least one frame
receives the static geometric correction so that the resulting two-dimensional
factor map can broadcast across the scan.

Masked pixels retain an identity factor. They are not sent through the
point-ray integrator. Non-finite detector coordinates are accepted only at
masked pixels.

## Complete pipeline example

This example uses values supplied directly in the pipeline. Every scalar
geometry or attenuation value can instead be loaded through its corresponding
`*_source` and `*_units_source` setting.

```yaml
CAPILLARY_ABSORPTION:
  module: CapillarySelfAbsorptionCorrection
  short_title: Correct capillary sample absorption
  requires_steps: [PIXEL_COORDINATES, MERGE_MASKS, TRANSMISSION_NORMALIZATION]
  configuration:
    with_processing_keys: [sample]

    coord_x_key: coord_x
    coord_y_key: coord_y
    coord_z_key: coord_z
    mask_key: mask

    capillary_axis: [1.0, 0.0, 0.0]
    capillary_centre: [0.0, 0.0, 0.0]
    capillary_centre_units: m

    sample_radius: 0.50
    sample_radius_units: mm
    sample_mu: 8.2
    sample_mu_units: 1/mm

    wall_thickness: 0.010
    wall_thickness_units: mm
    wall_mu: 1.7
    wall_mu_units: 1/mm

    beam_profile:
      type: image
      signal_source: measurement::/entry1/instrument/beam/profile
      pixel_pitch: [0.01, 0.01]  # slow, fast
      pixel_pitch_units: mm
      image_centre: [127.5, 127.5]
      downsample: [4, 4]
      relative_weight_cutoff: 1.0e-6

    input_state: transmission_normalized
    transmission_source: measurement::/entry1/sample/transmission
    transmission_units_source: measurement::/entry1/sample/transmission@units
    transmission_uncertainties_sources:
      transmission_SEM: measurement::/entry1/sample/transmission_sem

    evaluation_mode: adaptive
    relative_tolerance: 1.0e-3
    absolute_tolerance: 1.0e-12
    max_depth: 10
    chord_order: 12
    detector_chunk_size: 256
```

The `requires_steps` identifiers are illustrative; they must match the step IDs
in the actual pipeline.

## Supplying sample and wall parameters

Direct parameters and source-based parameters have the following mapping:

| Direct value | Data source | Units source |
|---|---|---|
| `sample_radius` | `sample_radius_source` | `sample_radius_units_source` |
| `wall_thickness` | `wall_thickness_source` | `wall_thickness_units_source` |
| `sample_mu` | `sample_mu_source` | `sample_mu_units_source` |
| `wall_mu` | `wall_mu_source` | `wall_mu_units_source` |

When a source and direct value are both configured, the source takes
precedence. `sample_radius` must be positive. Wall thickness and both
coefficients must be non-negative. A zero wall thickness reduces the geometry
to a homogeneous sample cylinder; a zero wall coefficient is allowed for
limit checks.

The wall currently requires a linear attenuation coefficient. The sample
accepts either a coefficient or the effective-coefficient path described
below. The module does not fit a transmission z scan.
Composition/density/energy calculations may be performed upstream, including
with MoDaCor's existing `xraylib`-based material utilities, and the result
supplied as `sample_mu` or `wall_mu`.

### Deriving an effective sample coefficient from measurement metadata

When the sample coefficient is not otherwise known, `sample_mu` may instead be
derived from either a sample-phase transmission or a sample-phase absorbed
fraction and a transmission path thickness:

$$
\mu_{eff} = \frac{-\ln T_s}{t_s}, \qquad A_s=1-T_s.
$$

The accepted alternatives are:

| Quantity | Direct value | Data source | Units source |
|---|---|---|---|
| Sample-only transmission $T_s$ | `sample_phase_transmission` | `sample_phase_transmission_source` | dimensionless |
| Sample-only absorbed fraction $A_s=1-T_s$ | `sample_phase_absorption` | `sample_phase_absorption_source` | dimensionless |
| Transmission path thickness $t_s$ | `sample_phase_thickness` | `sample_phase_thickness_source` | `sample_phase_thickness_units_source` |

Configure exactly one of `sample_mu`, `sample_phase_transmission`, or
`sample_phase_absorption`. The two sample-phase factors are mutually exclusive
and each requires `sample_phase_thickness`. Source references take precedence
over direct values of the same quantity. Transmission must be in $(0,1]$,
absorption in $[0,1)$, and thickness must be positive.

For example, values already separated from the empty-container absorption can
be read directly from a measurement HDF5 file:

```yaml
sample_mu: null
sample_phase_absorption_source: sample::/entry1/sample/absorption_by_sample
sample_phase_thickness_source: sample::/entry1/sample/thickness_averaged/mean
sample_phase_thickness_units_source: sample::/entry1/sample/thickness_averaged/mean@units
```

Equivalently, use `sample_phase_transmission_source` for a sample-only
transmission factor. Here “sample phase” is important: the value must exclude
the capillary wall or background contribution. It is used only to derive
$\mu_{eff}$; it does not divide or otherwise normalize the scattering signal,
so it is compatible with the composite module's prohibition on a separate
measurement-transmission normalization step.

For a capillary illuminated over a finite beam height, the measured
transmission averages different chord lengths. Consequently
$-\ln(T_s)/t_s$ is generally an **effective** coefficient unless $t_s$ is a
matching effective transmission path. Do not silently substitute the
capillary diameter or centreline chord for a whole-beam effective measurement.
The calculated whole-beam transmission output provides a useful consistency
check.

## Uncertainty support

The current implementation distinguishes measurement uncertainty from model
parameter uncertainty:

- Uncertainties already attached to the input `signal` `BaseData` propagate
  through division by the sample factor. In the composite module, filled- and
  empty-signal uncertainties propagate through attenuation-aware subtraction
  and the final division. MoDaCor uses first-order, uncorrelated propagation;
  named uncertainty sources are retained.
- Covariance is not represented. In particular, a monitor or normalization
  uncertainty shared by the filled and empty measurements is still treated as
  uncorrelated by ordinary `BaseData` arithmetic; the module does not model its
  cancellation or correlation.
- For the sample-only module with `input_state: transmission_normalized`,
  measured-transmission uncertainties supplied through
  `transmission_uncertainties_sources` propagate through the residual divisor
  and corrected signal.
- The calculated sample, filled-wall, and empty-wall attenuation maps and the
  calculated direct transmissions are currently deterministic: they do not
  carry uncertainty from $\mu$, radius, wall thickness, capillary orientation,
  detector geometry, or beam-profile parameters.
- Uncertainty datasets associated with `sample_mu`, `wall_mu`, or the derived
  sample-phase absorption/transmission/thickness inputs are not currently read.

This omission is intentional for now: reliable $\mu$ uncertainty metadata is
rare, and an unused interface would imply more confidence than the input data
support. If needed later, it should be implemented at coefficient resolution
and factor evaluation, using shared nominal/perturbed coefficient sets. For the
composite correction, derivatives must be taken through the complete corrected
expression because one coefficient affects several correlated attenuation
maps; propagating those maps as independent errors would be incorrect.

## Beam profiles

Beam coordinates use `(slow, fast)` ordering in the plane normal to the
incident direction. All profile weights are non-negative and normalized
internally. Profile intensity normalization does not replace the pipeline's
separate incident-flux normalization.

### Measured image

```yaml
beam_profile:
  type: image
  signal_source: measurement::/entry1/instrument/beam/profile
  pixel_pitch: [0.01, 0.01]
  pixel_pitch_units: mm
  image_centre: [127.5, 127.5]
  rotation: 0.0
  rotation_units: degree
  downsample: [4, 4]
  relative_weight_cutoff: 1.0e-6
```

Instead of `signal_source`, a small profile may be supplied inline as `signal`.
`image_centre` is the fractional `(slow, fast)` image index aligned with
`capillary_centre`; it defaults to the geometric array centre.

Downsampling groups image blocks without losing their integrated intensity.
Each retained block is represented at its intensity-weighted centroid, which
preserves the first spatial moment. The relative cutoff removes pixels below a
fraction of the image maximum. The retained fraction of the original incident
weight is stored as a diagnostic.

Downsampling is a numerical approximation: preserving total weight and the
centroid does not guarantee that attenuation is unchanged when a block spans a
capillary boundary or a strongly varying chord. Refine it until the resulting
correction is stable.

### Two-dimensional Gaussian

```yaml
beam_profile:
  type: gaussian_2d
  standard_deviations: [0.08, 0.15]  # slow, fast principal widths
  width_units: mm
  rotation: 12.0
  rotation_units: degree
  quadrature_order: [12, 16]
  truncation_sigma: [4.0, 4.0]
```

The standard deviations are the two principal widths. Rotation permits a
correlated Gaussian in the laboratory slow/fast basis. The finite truncation
keeps the calculation bounded; the analytically retained Gaussian fraction is
recorded.

### Two-dimensional trapezoid

```yaml
beam_profile:
  type: trapezoid_2d
  plateau_width: [0.10, 0.25]
  ramp_widths:
    - [0.03, 0.04]  # slow negative, slow positive
    - [0.05, 0.06]  # fast negative, fast positive
  width_units: mm
  rotation: 0.0
  rotation_units: degree
  quadrature_order_per_region: [6, 6]
```

This is a separable product of slow- and fast-axis trapezoids. The four ramp
widths are independent, allowing an asymmetric beam. Quadrature nodes are
placed separately in each plateau and linear-ramp region, which is more
efficient and accurate than sampling the entire bounding rectangle uniformly.

## Input-state algebra

The module always calculates the absolute factor $A_{s,sc}$. What it divides
the supplied signal by depends on `input_state`:

| `input_state` | Meaning of supplied signal | Applied divisor $C(d)$ |
|---|---|---|
| `raw` | Uncorrected detector intensity. | $A_{s,sc}(d)$ |
| `flux_normalized` | Incident-flux normalized, but not transmission normalized. | $A_{s,sc}(d)$ |
| `transmission_normalized` | Already divided by measured transmission $T_{meas}$. | $A_{s,sc}(d)/T_{meas}$ |

The operation is

$$
I_{corrected}(d) = \frac{I_{input}(d)}{C(d)}.
$$

For transmission-normalized input this is equivalent to

$$
I_{corrected}(d) =
\frac{T_{meas} I_{input}(d)}{A_{s,sc}(d)}.
$$

`transmission_normalized` is the default and requires `transmission_source`.
The transmission must be an effective measurement over the same complete beam
used for the data; it must not be a centreline Beer--Lambert value. The module
does not guess the input state from numeric values.

Measured-transmission uncertainties configured through
`transmission_uncertainties_sources` are propagated through the residual
divisor and into the corrected signal. Uncertainties in $\mu$, radius, wall
thickness, detector geometry, and beam-profile calibration are not yet
propagated by this process step.

## Exact and adaptive detector evaluation

`evaluation_mode: exact` traces every active detector pixel from every
scattering quadrature point. It is the reference mode and should be used for
validation, small regions of interest, and convergence studies.

`evaluation_mode: adaptive` recursively divides the regular detector-index
grid. For each fully active cell it calculates the corners, edge midpoints,
centre, and four interior quarter points. Bilinear interpolation is accepted
when every non-corner checkpoint meets the configured mixed absolute/relative
tolerance. A failed cell subdivides. A cell that reaches `max_depth` is
calculated exactly.

Cells crossing a mask boundary subdivide independently. Fully masked cells are
discarded, and unresolved boundary cells evaluate only their active pixels.
The output diagnostic distinguishes exactly evaluated pixels from interpolated
ones.

The adaptive tolerance is an a posteriori checkpoint criterion, not a formal
error bound everywhere between checkpoints. Compare adaptive output with exact
pixels or a tighter tolerance for new geometry, very high attenuation, close
detectors, or structured beam profiles.

### Current performance indication

Development measurements on an arm64 machine used a 512 by 512 detector,
3,072 illuminated-volume points, and $\mu R=3$:

| Geometry | Exact nodes | Fraction of detector | Adaptive time | Maximum error at 1,024 withheld pixels |
|---|---:|---:|---:|---:|
| Horizontal cylinder | 4,225 | 1.61% | 1.77 s | $4.5\times10^{-4}$ |
| Axis proportional to $(1,0.3,0.1)$ | 3,903 | 1.49% | 1.60 s | $8.7\times10^{-4}$ |

The requested relative tolerance was $10^{-3}$. The same case was estimated at
about 110 s when every detector pixel was evaluated exactly. These are
development measurements, not guarantees for arbitrary instruments. Peak
temporary allocation and runtime scale strongly with `detector_chunk_size`
and the number of illuminated-volume points.

## Output data

The default output keys are:

| Key | Contents |
|---|---|
| `capillary_sample_attenuation` | Absolute sample-origin attenuation $A_{s,sc}(d)$. |
| `capillary_self_absorption` | Residual divisor $C(d)$ actually applied to `signal`. |
| `capillary_calculated_transmission` | Scalar model prediction $T_{calc}$. |
| `capillary_attenuation_evaluated` | Boolean map: `True` where an exact detector ray was calculated. |
| `capillary_beam_profile_retained_fraction` | Scalar incident-weight fraction retained after profile truncation or thresholding. |

Each name can be changed with the corresponding `*_key` configuration. Masked
pixels have attenuation and divisor one, and are false in the evaluated map.

The distinction between the first two outputs matters for
transmission-normalized data: the absolute factor remains $A_{s,sc}$ while the
applied divisor is $A_{s,sc}/T_{meas}$.

## Container scattering is a separate problem

Absorption of sample scattering by the wall is already included in
$A_{s,sc}$. Scattering generated by the wall itself has a different spatial
origin and a different attenuation factor. Ordinary empty-capillary
subtraction followed by this sample correction is therefore not generally
exact.

For raw or flux-normalized measurements, the intended composite relation is

$$
I_s = \frac{I_{sc}^{raw} -
(A_{c,sc}/A_{c,c})I_c^{raw}}{A_{s,sc}},
$$

where $A_{c,sc}$ is the wall-origin factor in the filled capillary and
$A_{c,c}$ the wall-origin factor in the empty capillary. If both measurements
were divided by their own measured transmissions, the equivalent algebra would
be

$$
I_s = \frac{T_{sc}N_{sc} -
(A_{c,sc}/A_{c,c})T_cN_c}{A_{s,sc}}.
$$

`CapillarySampleContainerCorrection` intentionally implements only the first
equation. It does not attempt to undo an earlier transmission normalization,
because the result can depend on where that normalization occurred relative to
other pipeline operations. Do not emulate this module by applying two
independent sample corrections and then subtracting: those operations are not
algebraically interchangeable.

### Composite pipeline configuration

The filled and empty signals must have the same shape, compatible units,
matching detector geometry, and comparable exposure and incident-flux
normalization. If they were measured on different grids, regrid them before
this step.

```{important}
If you use `CapillarySampleContainerCorrection`, do not include a separate
normalization by the measured filled- or empty-capillary transmission in the
same correction pipeline. Exposure-time, incident-flux, detector-response, and
other common-scale normalizations are allowed, provided the filled and empty
signals remain directly comparable.
```

```yaml
CAPILLARY_SAMPLE_AND_CONTAINER:
  module: CapillarySampleContainerCorrection
  short_title: Correct sample and capillary wall attenuation
  requires_steps: [PIXEL_COORDINATES, MERGE_MASKS, NORMALIZE_EXPOSURE_AND_FLUX]
  configuration:
    filled_processing_key: sample
    empty_processing_key: background
    mask_key: mask
    empty_mask_key: mask

    capillary_axis: [1.0, 0.0, 0.0]
    sample_radius: 0.50
    sample_radius_units: mm
    sample_mu: 8.2
    sample_mu_units: 1/mm
    wall_thickness: 0.010
    wall_thickness_units: mm
    wall_mu: 1.7
    wall_mu_units: 1/mm

    # Usually zero for an empty capillary; it may represent a gas or other
    # known material in the central region of the background measurement.
    empty_centre_mu: 0.0
    empty_centre_mu_units: 1/mm

    beam_profile:
      type: gaussian_2d
      standard_deviations: [0.08, 0.15]
      width_units: mm
      quadrature_order: [12, 16]
      truncation_sigma: [4.0, 4.0]

    evaluation_mode: adaptive
    relative_tolerance: 1.0e-3
    chord_order: 12
    # Wall-origin quadrature may need more nodes at high sample absorption.
    wall_chord_order: 32
```

The corrected sample signal replaces `signal` in the filled `DataBundle`. The
empty bundle remains unchanged. The filled and empty masks are combined with a
logical OR and stored on the filled output. At scan dimensions, a detector
pixel is ray-traced only if there is at least one paired frame in which both
measurements are active. Values at combined-mask positions are preserved from
the original filled signal and ignored downstream through the union mask.

The composite step requires a positive wall thickness because wall-origin
quadrature is part of the calculation. `empty_centre_mu` defaults to zero but
may be configured as a value or with `empty_centre_mu_source` and
`empty_centre_mu_units_source`. `wall_chord_order` controls wall-origin
integration independently and defaults to `chord_order` when omitted.

The filled- and empty-capillary wall factors share the same geometric path
calculation and adaptive detector mesh. Only their attenuation coefficients
differ. This both reduces runtime and avoids introducing two independently
chosen interpolation meshes into their ratio.

### Composite outputs

The filled output bundle receives:

| Key | Contents |
|---|---|
| `capillary_sample_attenuation` | $A_{s,sc}$, sample-origin attenuation in the filled capillary. |
| `capillary_wall_attenuation_filled` | $A_{c,sc}$, wall-origin attenuation with the sample present. |
| `capillary_wall_attenuation_empty` | $A_{c,c}$, wall-origin attenuation in the empty capillary. |
| `capillary_wall_subtraction_scale` | Pixelwise ratio $A_{c,sc}/A_{c,c}$. |
| `capillary_filled_calculated_transmission` | Calculated effective transmission of the filled capillary. |
| `capillary_empty_calculated_transmission` | Calculated effective transmission of the empty capillary. |
| `capillary_beam_profile_retained_fraction` | Incident weight retained by profile truncation or thresholding. |
| `capillary_sample_attenuation_evaluated` | Exact-evaluation mask for $A_{s,sc}$. |
| `capillary_wall_filled_attenuation_evaluated` | Exact-evaluation mask for $A_{c,sc}$. |
| `capillary_wall_empty_attenuation_evaluated` | Exact-evaluation mask for $A_{c,c}$. |

Filled- and empty-signal uncertainties propagate through the subtraction and
sample division. Model parameter uncertainties are not yet included. The two
calculated transmissions are diagnostics only and are not used to rescale the
signals. Sample-origin refinement is independent, while the two wall maps use
one shared refinement mesh whose acceptance test covers both factors. Validate
the composite output and wall-scale ratio against exact mode when establishing
settings for a new instrument.

## Accuracy controls and convergence

For a new setup, establish stability in this order:

1. Run a small detector region or selected frames with `evaluation_mode:
   exact`.
2. Increase `chord_order` and `wall_chord_order` independently until the exact
   correction changes acceptably little.
3. For analytic profiles, increase their quadrature order; for image profiles,
   reduce downsampling and cutoff.
4. Compare the intended adaptive settings against exact values, including
   high-angle detector regions and mask boundaries.
5. Compare `capillary_calculated_transmission` with the measured effective
   transmission. A discrepancy is a diagnostic for geometry, beam-profile,
   coefficient, or normalization mismatch; it is not automatically a scale
   factor to apply.

The factor must remain positive and finite. The module rejects active-pixel
values below `minimum_attenuation_factor` rather than dividing by an unstable
number.

Wall-origin quadrature can converge more slowly than sample-origin quadrature
when sample absorption is strong, because an outgoing wall ray can become
tangent to the inner cylinder within an integration segment. In one deliberately
difficult development case with radius ratio 1.1 and sample $\mu R=3$, the
maximum relative change against 96 wall nodes was about $6.0\times10^{-3}$ at
12 nodes, $2.9\times10^{-3}$ at 20, $1.2\times10^{-3}$ at 32, and
$8.8\times10^{-4}$ at 48. At sample $\mu R=0.5$, 12 nodes were already within
about $1.6\times10^{-4}$. These figures are not universal tolerances; they show
why the two chord orders are separate controls.

## Model limitations

The implemented model assumes:

- a straight, circular, concentric and effectively infinite cylinder;
- a capillary centred in the incident beam;
- parallel incident rays;
- single scattering and Beer--Lambert attenuation;
- one central sample phase and an optional concentric wall;
- a static detector geometry represented by pixel-centre coordinates.

It does not currently model finite capillary ends, eccentric or non-circular
walls, wall texture or crystallographic diffraction, refraction, multiple
scattering, or uncertainty in geometry/profile parameters. Detector pixels are
treated as point centres; detector-pixel area quadrature is not currently
performed.

## Further reading

The implementation and validation plan is documented in
[Capillary self-absorption and container correction](../design/capillary-self-absorption.md).
The principal literature references are:

- Y. Chen *et al.*, *Crystal Growth & Design* **26** (2026), 1036--1047,
  <https://doi.org/10.1021/acs.cgd.5c00551>.
- S. N. Sulyanov, A. A. Gogin & H. Boysen, *J. Appl. Cryst.* **45** (2012),
  93--97, <https://doi.org/10.1107/S0021889811048217>.
- M. E. Bowden & M. Ryan, *J. Appl. Cryst.* **43** (2010), 693--698,
  <https://doi.org/10.1107/S0021889810021114>.

The one-dimensional homogeneous-cylinder implementation in
`diffpy.labpdfproc` is used as an independent regression reference, not as a
runtime dependency. It does not cover the two-phase, arbitrary-axis,
beam-profile, or detector-azimuth requirements described here.
