# Capillary self-absorption and container correction

Status: staged implementation; Stage 1 geometry kernel complete and Stage 2
numerical prototype substantially implemented.

For configuration and operational guidance, see
[Capillary sample self-absorption correction](../corrections/capillary_self_absorption.md).

Implementation progress:

- Stage 1 completed on 2026-09-06 in
  `modacor.geometry.cylinders.ConcentricCylinderGeometry`.
- Stage 2 is in progress: the detector-driven point-ray attenuation integrator
  and uniform cross-section reference quadrature, beam-ray/chord quadrature,
  separate direct-transmission integral, and adaptive detector-grid evaluator
  are implemented in `modacor.models.attenuation.concentric_cylinder`.
- Stage 3 is in progress: the general phase-origin API now calculates sample-
  and wall-origin factors with attenuation through every concentric phase.
- Stage 4 is in progress: common normalized quadrature adapters for measured
  images, rotated 2D Gaussian profiles, and four-ramp 2D trapezoidal profiles
  are implemented in `modacor.models.attenuation.beam_profiles`.
- Stage 5 is in progress: both the sample-origin
  `CapillarySelfAbsorptionCorrection` and attenuation-aware filled/empty
  `CapillarySampleContainerCorrection` process steps are implemented with
  their respective input contracts.
- Stage 6 remains planned.

## Package boundaries

The reusable numerical implementation is deliberately separated from
pipeline modules. Dependencies point in one direction:

```text
pipeline ProcessSteps
    -> attenuation models
        -> pure geometry

NeXus geometry adapter
    -> pure geometry
```

The intended source layout is:

```text
src/modacor/
├── geometry/
│   ├── vectors.py
│   ├── transforms.py
│   ├── cylinders.py
│   └── frames.py
├── models/
│   └── attenuation/
│       ├── flat_plate.py
│       ├── concentric_cylinder.py
│       └── beam_profiles.py
├── io/
│   └── nexus/
│       └── geometry.py
└── modules/
    └── technique_modules/
        └── scattering/
            └── capillary_sample_container_correction.py
```

`modacor.geometry` contains only format-independent numerical primitives. It
must not import `ProcessStep`, `BaseData`, `IoSources`, YAML configuration, or
NeXus path handling. The concentric-cylinder ray kernel and generic vector,
ray, transform, and frame operations belong here.

`modacor.models.attenuation` combines path lengths with material attenuation
and numerical quadrature. Beer--Lambert integrals and their derivatives belong
to this layer, not to pure geometry. The flat-plate numerical formula should
eventually move here while its `ProcessStep` remains in the modules package.

The NeXus hierarchical geometry resolver is split by responsibility. Pure
affine transforms and format-independent frames belong in `modacor.geometry`;
traversal of `depends_on`, NeXus attributes, source references, and units
belongs in `modacor.io.nexus.geometry`. Compatibility imports should preserve
current callers while this later refactor is performed.

Pipeline modules remain responsible for configuration validation, source and
`BaseData` loading, dependency contracts, uncertainty attachment, provenance,
and modifying processing bundles. Neither geometry nor attenuation models may
depend back on that pipeline layer.

## Scope

MoDaCor needs an absorption correction for cylindrical capillaries measured
with a two-dimensional detector. It must not assume scattering in one
Debye--Scherrer plane or calculate a dense angular table in directions where
there is no detector. Correction factors will be evaluated by tracing rays
back from active detector pixels or explicitly configured detector segments.

The initial specimen is a straight, circular, concentric two-phase cylinder:

1. a central sample or void of radius `sample_radius` and attenuation
   coefficient `sample_mu`;
2. a capillary-wall annulus from `sample_radius` to
   `sample_radius + wall_thickness`, with coefficient `wall_mu`.

The coefficients and dimensions are separate, unit-aware inputs. Products
such as $\mu R$ are formed only inside the calculation. Internally, the
geometry kernel should accept ordered radial boundaries so that more than two
phases can be added later without replacing the intersection algorithm.

The default capillary axis is horizontal in the MoDaCor lab frame:

```yaml
capillary_axis: [1.0, 0.0, 0.0]
```

The incident beam is along $(0,0,1)$ and intersects the capillary centreline.
Tilted axes are supported under the straight, infinite-cylinder approximation.
An incident beam nearly parallel to the axis must be rejected because the
finite capillary length and end faces then determine the path lengths.

The first implementation covers single scattering and attenuation. Multiple
scattering, refraction, non-circular or eccentric walls, wall texture, and
finite capillary ends are out of scope.

## Literature basis

The design combines several treatments rather than copying one instrument
geometry:

- Chen *et al.* formulate the factor as a volume average of Beer--Lambert
  attenuation and show the homogeneous-cylinder dependence on $\mu R$. Their
  angle interpolation is a validation reference, not our detector algorithm.
- Sulyanov, Gogin & Boysen use direction cosines for an infinite cylinder,
  two-dimensional area detector, and inclined geometry.
- Bowden & Ryan treat cylindrical/annular specimens and their containers and
  motivate separate sample-origin and container-origin factors.
- The 2026 IUCr SWAXS work includes constrained illumination, measured beam
  profiles, and wall attenuation. Its line-collimated beam and specialized
  detector arrangement are useful special cases, not MoDaCor defaults.

References:

- Y. Chen *et al.*, *Crystal Growth & Design* **26** (2026), 1036--1047,
  <https://doi.org/10.1021/acs.cgd.5c00551>.
- S. N. Sulyanov, A. A. Gogin & H. Boysen, *J. Appl. Cryst.* **45** (2012),
  93--97, <https://doi.org/10.1107/S0021889811048217>.
- M. E. Bowden & M. Ryan, *J. Appl. Cryst.* **43** (2010), 693--698,
  <https://doi.org/10.1107/S0021889810021114>.
- *Angle-dependent X-ray absorption correction in small- and wide-angle X-ray
  scattering: accounting for constrained scattering volume, beam profiles and
  capillary wall attenuation*, *J. Appl. Cryst.* (2026),
  <https://journals.iucr.org/j/issues/2026/05/00/jl5129/>.

## Relationship to diffpy.labpdfproc

`diffpy.labpdfproc` is retained as an independent validation reference rather
than added as a required MoDaCor dependency. Its public correction API computes
a one-dimensional homogeneous-cylinder curve as a function of $2\theta$ and
$\mu D$. The fast backend interpolates bundled coefficients over approximately
$0.5 \leq \mu D \leq 7$ and falls back to a Python gridded-circle calculation
outside that interval.

That model does not represent the requirements of this design: it has no
separate sample and wall phases, arbitrary cylinder axis, 2D beam profile,
pixel-azimuth dependence, detector backtracing, or attenuation-aware container
subtraction. Applying its curve to a flattened 2D `TwoTheta` array would
incorrectly give identical corrections to pixels with equal $2\theta$ but
different azimuth for a horizontal or tilted capillary.

Importing the package for this restricted operation would also add
`diffpy.utils`, pandas, Gooey and its GUI dependencies to the runtime stack.
Its application-oriented `DiffractionObject` API is not a low-level ray or
layer kernel that MoDaCor can extend. Consequently:

- MoDaCor keeps its own general geometry and attenuation implementation;
- selected `diffpy.labpdfproc` outputs and its published examples are recorded
  as regression references;
- the dependency is not installed or imported in production code; and
- a specialized optional 1D backend or attributed vendoring of its BSD-3-Clause
  interpolation data is deferred unless a concrete use case justifies a
  second, deliberately restricted physical model.

`diffpy.labpdfproc` currently contains no other physical PDF corrections. Its
remaining features are file/CLI workflow, wavelength and axis conversion,
calculation of $\mu$ from composition/density/energy through `diffpy.utils`,
and estimation of $\mu D$ from a transmission z-scan. MoDaCor already provides
composition/density/energy attenuation lookup through `xraylib`. Z-scan
acquisition and fitting are explicitly out of scope here because they are
handled by separate Bluesky plans.

## Concentric-cylinder geometry kernel

Implement the geometry independently of the processing module. Inputs are a
normalized cylinder axis and centreline, ordered radii, incident and exit
rays, and scattering positions.

For each ray and cylindrical boundary, project the ray and position into the
plane normal to the cylinder axis and solve the quadratic intersection.
Sorted intersection intervals are classified by radial phase. Their lengths
give the incident and exit distances in the sample and wall. Tangent and
non-intersecting rays need robust handling, and reversing the axis must be a
numerical identity.

## Detector-driven backtracing

Evaluate only active detector pixels or configured segments:

1. read detector positions from `coord_x`, `coord_y`, and `coord_z`;
2. remove masked or inactive elements before geometric evaluation;
3. trace each contributing detector ray through the outer and inner
   cylindrical boundaries; and
4. integrate attenuation over eligible illuminated scattering positions.

This directly supports detector gaps, masks, tilted panels, and a full 2D
correction. A segment may use representative or quadrature points, but any
within-segment interpolation must be tested against pixelwise evaluation.

Two accuracy modes should be prototyped:

- **point-ray:** from every sampled scattering position to the actual detector
  pixel centre; this is the geometrically complete reference;
- **central-ray:** from the capillary centre to the pixel centre; this is a
  possible far-field optimization.

Central-ray mode should become a default only if an error study shows that
sample-size/detector-distance parallax is negligible. Work must be chunked
over detector elements and/or scattering points. Incident paths and phase
memberships should be cached per calculation. Direction grouping may be used
only with an explicit angular tolerance covered by accuracy tests.

## Beam profiles and illuminated volume

Represent the beam in the 2D lab-frame plane perpendicular to propagation.
The profile supplies physical coordinates and non-negative relative weights;
normalization is internal, while flux normalization remains a separate step.

Planned sources are:

1. **image:** an experimentally measured 2D image with physical pixel pitch,
   beam centre, and optional in-plane rotation;
2. **2D trapezoid:** a rectangular plateau with independently configurable
   ramps on four sides, including a product of horizontal and vertical
   trapezoids;
3. **2D Gaussian:** independent or correlated widths, centre, rotation, and a
   configurable finite truncation extent.

All adapters produce common beam-plane quadrature points and weights. Image
profiles support cropping insignificant tails and weight-preserving
downsampling, with the introduced error measured during validation.

For every beam-plane point, intersect its incident ray with the concentric
cylinder. Generate sample-origin integration points only along the central
sample chord and wall-origin points only in the annulus. This creates a finite,
beam-weighted illuminated volume for measured, Gaussian, trapezoidal, or
line-like profiles without assuming line collimation.

Do not construct a uniform three-dimensional voxel box. Use nested quadrature
matched to the physical problem:

1. retain only non-negligible beam-plane samples;
2. find their analytic intersections with each cylindrical phase;
3. apply one-dimensional Gauss--Legendre quadrature along each occupied chord
   (and each disjoint wall interval); and
4. include the chord-coordinate Jacobian in the volume weights.

For analytic trapezoidal and Gaussian profiles, the beam-plane quadrature
should respect ramp/plateau breakpoints or Gaussian scale. For an image, an
image pixel is initially one beam ray with its integrated intensity as weight;
optional sub-pixel quadrature is an accuracy control. This avoids evaluating
empty volume and concentrates work where both illumination and scattering
material exist.

## Factors and effective measured transmission

For scattering originating in region $j$ and reaching detector element $d$,

$$
A_j(d) =
\frac{\sum_v w_v
\exp[-\mu_s L_{s,v,d} - \mu_w L_{w,v,d}]}
{\sum_v w_v},
$$

where $v$ indexes beam-weighted volume quadrature points in region $j$.

The measured transmission is an effective value over the complete beam. It
must not be converted to $\mu$ with a centreline Beer--Lambert expression.
Calculate its model prediction as a separate beam-area integral,

$$
T_{calc} =
\frac{\sum_b w_b
\exp[-\mu_s L_{s,b}^{through} - \mu_w L_{w,b}^{through}]}
{\sum_b w_b}.
$$

This sum includes illuminated rays that miss the capillary, for which the
material path is zero. It is not generally equal to the forward limit of
$A_j(d)$: $T_{calc}$ is beam-area-weighted, whereas $A_j$ is
scattering-volume-weighted and therefore gives longer chords proportionally
more weight. `T_calc`, measured transmission, and their ratio are recorded as
a model-consistency diagnostic, but one must not be substituted for the other.

For a single isolated contribution whose upstream data were divided by
measured transmission $T_j$, the residual divisor is $A_j(d)/T_j$. This
retains the experimental scalar normalization while applying the calculated
angular and azimuthal variation. Composite sample/container subtraction must
instead use the explicit algebra below. Whether a recorded transmission
includes wall attenuation and out-of-capillary beam must be stated in source
metadata or configuration.

## Attenuation-aware container subtraction

Ordinary empty-container subtraction followed by one sample correction is not
generally exact. At minimum calculate:

- $A_{s,sc}$: sample-origin scattering in the filled capillary;
- $A_{c,sc}$: wall-origin scattering in the filled capillary;
- $A_{c,c}$: wall-origin scattering in the empty capillary.

Before scalar transmission normalization, the idealized relation is

$$
I_s = \frac{I_{sc}^{raw} - (A_{c,sc}/A_{c,c})I_c^{raw}}
{A_{s,sc}}.
$$

If the filled and empty signals have already been divided by their respective
measured transmissions, the algebra would instead require

$$
I_s = \frac{T_{sc}N_{sc} -
(A_{c,sc}/A_{c,c})T_cN_c}{A_{s,sc}}.
$$

This is why applying separate residual divisors and then performing ordinary
subtraction is not interchangeable with the composite correction. The
production composite module deliberately does not try to reverse an earlier
transmission normalization: its correctness would depend on which operations
were performed before and after that normalization. It accepts only filled
and empty signals on a comparable pre-transmission-normalization scale.

The likely public components are:

- `ConcentricCylinderAttenuation`, the geometry and integration kernel;
- `CapillarySampleContainerCorrection`, combining filled and empty bundles;
- optionally `CapillarySelfAbsorptionCorrection`, for data where container
  scattering is already removed consistently or negligible.

## Implemented sample-origin configuration

The first public process step covers the last case above. A representative
configuration is:

```yaml
module: CapillarySelfAbsorptionCorrection
configuration:
  with_processing_keys: [sample]
  capillary_axis: [1.0, 0.0, 0.0]  # horizontal default; tilted vectors work

  sample_radius: 0.5
  sample_radius_units: mm
  sample_mu: 8.2
  sample_mu_units: 1/mm
  wall_thickness: 0.01
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
  evaluation_mode: adaptive
  relative_tolerance: 1.0e-3
```

Direct values may be replaced by `sample_radius_source`,
`wall_thickness_source`, `sample_mu_source`, and `wall_mu_source`, with the
corresponding `*_units_source`. The coefficient and radius remain separate
inputs. Alternatively, an effective sample coefficient can be derived as
$-\ln(T_s)/t_s$ from mutually exclusive `sample_phase_transmission` or
`sample_phase_absorption` ($A_s=1-T_s$) and `sample_phase_thickness`, including
their IoSource forms. This is deliberately identified as an effective
coefficient: whole-beam capillary transmission is not a centreline measurement,
and its thickness must describe the corresponding effective path.

For `gaussian_2d`, the profile mapping uses `standard_deviations`,
`width_units`, optional `rotation`/`rotation_units`, `quadrature_order`, and
`truncation_sigma`. For `trapezoid_2d`, it uses `plateau_width`, `width_units`,
four `ramp_widths` ordered as `((slow_negative, slow_positive),
(fast_negative, fast_positive))`, and `quadrature_order_per_region`.

`input_state: raw` and `flux_normalized` divide the signal by the absolute
sample-origin attenuation $A_{s,sc}$. `transmission_normalized` requires the
measured effective whole-beam transmission and divides by
$A_{s,sc}/T_{meas}$. This distinction must be declared; it is not inferred
from the data values.

## Candidate configuration

Names may change during the first prototype, but the intended information is:

```yaml
module: CapillarySampleContainerCorrection
configuration:
  filled_processing_key: sample
  empty_processing_key: background
  capillary_axis: [1.0, 0.0, 0.0]

  sample_mu_source: sample::/entry1/sample/linear_attenuation_coefficient
  sample_mu_units_source: sample::/entry1/sample/linear_attenuation_coefficient@units
  sample_radius_source: sample::/entry1/sample/container/inner_radius
  sample_radius_units_source: sample::/entry1/sample/container/inner_radius@units
  wall_mu_source: sample::/entry1/sample/container/linear_attenuation_coefficient
  wall_mu_units_source: sample::/entry1/sample/container/linear_attenuation_coefficient@units
  wall_thickness_source: sample::/entry1/sample/container/wall_thickness
  wall_thickness_units_source: sample::/entry1/sample/container/wall_thickness@units

  beam_profile:
    type: image
    signal_source: sample::/entry1/instrument/beam/profile
    pixel_pitch: [0.01, 0.01]
    pixel_pitch_units: mm
    centre: [0.0, 0.0]
    centre_units: mm
    rotation: 0.0
    rotation_units: degree

  detector_coordinate_keys: [coord_x, coord_y, coord_z]
  geometry_mode: point_ray
  detector_chunk_size: 1024
  profile_weight_cutoff: 1.0e-6
  chord_order: 12
  wall_chord_order: 32
```

The filled and empty signals may be normalized by exposure and incident flux,
provided they remain mutually comparable. A pipeline using this composite
module must not contain a separate normalization by the measured filled- or
empty-capillary transmission in the same signal path.

Analytic `trapezoid_2d` and `gaussian_2d` mappings replace the image fields
with their widths, ramps, covariance/rotation, and quadrature controls.

## Uncertainties and diagnostics

The implemented modules propagate uncertainties already attached to input
signals. The composite step carries filled and empty signal terms through its
subtraction and division. The sample-only step additionally propagates a
configured measured-transmission uncertainty when correcting data that were
already transmission-normalized.

Uncertainties supplied for derived sample-phase transmission/absorption and
thickness are propagated analytically into $\mu_{eff}$. Nominal and perturbed
sample-coefficient vectors then share the existing path kernel and adaptive
mesh to obtain factor sensitivities. The composite derivative is evaluated
through the complete corrected expression because sample $\mu$ affects both
$A_{s,sc}$ and $A_{c,sc}$. The resolved coefficient, affected factor maps,
calculated filled/sample transmission, and corrected signal retain the named
uncertainty components.

Direct sample/wall-$\mu$ uncertainty and geometry/profile parameter uncertainty
remain intentionally out of scope. Reliable direct coefficient uncertainty
metadata appears uncommon. Geometry and profile derivatives could later use
checked finite differences or repeated quadrature once their convergence and
runtime are characterized.

Expose focused diagnostics:

- active detector-element and integration-point counts;
- requested and achieved quadrature resolution;
- predicted/measured forward transmissions and their ratio;
- factor extrema;
- geometry mode and any grouping tolerance;
- axis angle to the beam;
- convergence or profile-clipping warnings.

## Iterative delivery and validation gates

This feature is expected to take several iterations. Each stage must pass its
gate before performance approximations are promoted.

### Stage 1: geometry reference kernel

- Implement ordered concentric-cylinder ray intersections.
- Test analytic axial/perpendicular cases, tangent rays, axis reversal, rigid
  rotations, tilted axes, and near-parallel rejection.

Gate: paths agree with independent analytic and high-precision references.

Implementation note: the initial kernel and focused tests cover central and
off-centre analytic paths, rays starting outside the cylinder, wall-origin
rays crossing the central phase, tangency, broadcasting, axis-sign invariance,
rigid rotations (including tilted axes), input validation, and near-parallel
rejection. Published-map and numerical-quadrature comparisons belong to Stage
2, where attenuation factors are first formed from these paths.

### Stage 2: detector-driven homogeneous prototype

- Evaluate only active `coord_x/y/z` elements.
- Implement point-ray mode with provisional uniform beam-plane quadrature and
  Gauss--Legendre integration along analytically intersected sample chords.
- Compare perpendicular geometry with Chen/diffpy.labpdfproc and inclined 2D
  maps with Sulyanov.

Gate: converged factors reproduce reference cases within a stated tolerance.

Implementation note: the first Stage 2 increment accepts prefiltered detector
positions plus arbitrary scattering points and volume weights. It calculates
the incident paths once, then evaluates exact scattering-point-to-pixel exit
rays in bounded detector chunks. A Gauss--Legendre/polar uniform
cross-section quadrature provides the homogeneous-cylinder reference case.
Beam-ray/chord quadrature now generates only material-occupied points, and a
separate beam-area integral calculates direct transmission. The homogeneous
result has been cross-checked against Chen/diffpy.labpdfproc. Active-mask
adaptation is implemented through the sample-origin `ProcessStep`; a
quantitative inclined-map comparison with Sulyanov remains before this stage's
gate is complete.

The adaptive evaluator recursively subdivides the regular detector-index grid.
It evaluates exact point rays at each cell's corners, edge midpoints, centre,
and four interior quarter points, and accepts bilinear interpolation only when
every non-corner checkpoint meets a mixed absolute/relative tolerance.
Reaching the configured maximum depth triggers exact evaluation of every pixel
in that cell. The checkpoint
criterion is deliberately described as an a posteriori estimator rather than
a guaranteed error bound; tests additionally compare pixels withheld from the
adaptive decision. The exact evaluator remains available as the reference
mode. Active masks are resolved recursively: cells crossing a mask boundary
subdivide, fully inactive cells are skipped, and any unresolved boundary cell
falls back to exact evaluation of its active pixels. Masked pixels are never
passed to the ray integrator and retain identity factors.

### Stage 3: two-phase sample and wall

- Add separate coefficients, inner radius, and wall thickness.
- Calculate sample-origin and wall-origin factors.
- Test zero wall thickness, zero wall attenuation, equal coefficients, and
  empty-centre limits.

Gate: reproduce Bowden--Ryan cases and independent brute-force calculations.

Implementation note: ordered phase geometry and the phase-origin API already
support a central sample plus wall. Tests cover equal-coefficient equivalence,
the zero-wall-attenuation limit, and the difference between wall-origin
attenuation in filled and empty capillaries. Quantitative Bowden--Ryan and
independent brute-force comparisons remain before this gate is complete.

### Stage 4: realistic 2D profiles

- Add image, trapezoid, and Gaussian adapters.
- Add physical calibration, rotation, cropping, and weight-preserving image
  downsampling.
- Calculate effective whole-beam forward transmission.

Gate: uniform limits match Stage 3, analytic profile moments are reproduced,
and image downsampling errors are bounded on representative profiles.

Implementation note: the first adapters normalize relative ray weights and
share one lab-frame mapping. Image blocks retain integrated weight and are
placed at their intensity-weighted centroid; optional thresholding reports the
retained incident-weight fraction. Gaussian quadrature supports unequal
principal widths, rotation, and finite truncation. Trapezoidal quadrature
places Gauss--Legendre nodes separately in the plateau and in each of four
independently sized linear ramps. Initial tests cover calibration, rotation,
normalization, first-moment preservation, Gaussian moments, and trapezoid
symmetry. Profile-to-attenuation convergence studies remain.

### Stage 5: pipeline container correction

- Require comparable raw or exposure/incident-flux-normalized inputs.
- Implement attenuation-aware filled/empty-capillary subtraction.
- Propagate uncertainties and store factor maps and diagnostics.
- Update the MOUSE example with the correct ordering.

Gate: synthetic mixtures recover known sample signals and measured-data
results are stable under quadrature refinement.

Implementation note: `CapillarySelfAbsorptionCorrection` provides the
sample-origin subset of this stage. It resolves 2D detector coordinates and
static or scan-shaped masks, constructs a centred sample/wall geometry and one
of the three beam-profile types, calculates both absolute sample-origin
attenuation and effective direct transmission, and applies either $A_{s,sc}$
for raw/flux-normalized input or $A_{s,sc}/T_{meas}$ for
transmission-normalized input.

`CapillarySampleContainerCorrection` calculates $A_{s,sc}$ from sample-origin
quadrature, plus $A_{c,sc}$ and $A_{c,c}$ from wall-origin quadrature under the
filled and empty attenuation coefficients. It evaluates
$[F-(A_{c,sc}/A_{c,c})E]/A_{s,sc}$ directly on comparable signals before any
measured-transmission normalization. Filled and empty masks are combined before
ray tracing, including scan-shaped masks, and the filled output receives the
three factors, wall scale, calculated transmissions, and exact-evaluation
masks. BaseData arithmetic propagates filled- and empty-signal uncertainties.
Synthetic raw and common-flux-normalized mixtures recover the known sample
signal. Independent literature/brute-force validation, model-parameter
uncertainties, and a measured-data pipeline example remain before the stage
gate is complete.

The two wall factors use identical scattering points and detector paths, so
the kernel evaluates their two attenuation-coefficient vectors together. The
geometric path lengths are calculated once and both maps use one adaptive
refinement mesh. On a 32 by 32 development grid with 6,144 wall-origin points,
this reduced exact two-factor evaluation from about 1.42 s to 0.70 s (2.0x).
On a 128 by 128 adaptive grid it reduced the corresponding run from about
7.04 s and 10,180 total exact-node visits to 5.64 s and 7,079 shared nodes.
These small benchmarks characterize the implementation direction rather than
promise instrument-independent throughput.

### Stage 6: performance hardening

- Benchmark pixel/scattering-point chunking.
- Evaluate central rays, detector segmentation, direction grouping, profile
  reduction, and caching against point-ray results.
- Add memory ceilings and actionable validation errors.

Gate: choose defaults from measured accuracy/performance, not assumed detector
symmetry.

The performance benchmark must report time and peak temporary-array memory as
a function of active detector count, nonzero beam-ray count, and chord nodes.
Chunk sizes should be derived from a configurable memory budget; a raw element
count is retained only as an expert override. Inactive output pixels keep an
identity factor and their existing mask, so they do not enter the expensive
geometry or attenuation evaluation.

## Preliminary performance characterization

These measurements are development snapshots, not stable performance claims.
They were obtained on the local arm64 development machine with Python 3.14.5
and the pure NumPy point-ray implementation.

- With 576 cross-section points, two radial phases, and detector chunks of
  256, 100,000 exact detector pixels took approximately 8.6--9.0 s (about
  11,600 pixels/s) and peak resident memory was approximately 124 MiB. Runtime
  was linear from 10,000 through 100,000 pixels. A one-megapixel exact map
  therefore extrapolates to roughly 90 s for this quadrature.
- A representative compressed 16 by 16 beam profile with 12 chord nodes
  produced 3,072 volume points. Exact evaluation then took approximately
  4.2 s per 10,000 pixels (about 2,360 pixels/s), implying roughly 7 min for a
  megapixel if every point-to-pixel ray is evaluated.
- For that 3,072-point case, evaluating regular detector meshes took about
  0.11 s at 16 by 16, 0.44 s at 32 by 32, and 1.7 s at 64 by 64. On a planar
  test detector spanning approximately 53 degrees, bilinear interpolation from
  the 64 by 64 mesh had maximum relative errors of about $1.5\times10^{-5}$ at
  $\mu R=0.6$ and $1.9\times10^{-4}$ at $\mu R=3$. The 32 by 32 mesh remained
  below approximately $8\times10^{-4}$ in the tested cases.
- The implemented adaptive grid was then tested on a 512 by 512 planar
  detector, the same 3,072 volume points, and $\mu R=3$, using a requested
  relative checkpoint tolerance of $10^{-3}$. A horizontal cylinder required
  4,225 exact nodes (1.61% of pixels) and 1.77 s; a cylinder with axis
  proportional to $(1,0.3,0.1)$ required 3,903 exact nodes (1.49%) and 1.60 s.
  Neither case reached exact fallback. Against 1,024 independently selected
  detector pixels, the maximum relative errors were about $4.5\times10^{-4}$
  and $8.7\times10^{-4}$, respectively. Peak Python-traced temporary
  allocation was approximately 170 MiB at the current 256-pixel chunk size.
  Relative to the measured exact throughput, this reduces an estimated
  roughly 110 s exact 512 by 512 calculation to under 2 s in these smooth test
  geometries. It is not yet a claim for arbitrary detector layouts or beam
  profiles.

The implication is that exact point-ray evaluation is suitable as the
reference and for smaller active regions, while production megapixel use will
probably require adaptive detector segmentation/interpolation. That path is
realistic, but the segment mesh must be refined according to correction-map
curvature and checked at withheld pixels. Fixed 32 by 32 or 64 by 64 defaults
must not be inferred from these preliminary cases. Sharp geometry near a ray
parallel to the cylinder axis, stronger attenuation, close detector distance,
or structured beam profiles can require more nodes.

Quadrature convergence was also measured against an 18,432-point internal
reference over 180 scattering directions. For $\mu R=3$, 576 points gave a
maximum relative difference of about $8.1\times10^{-4}$ and 1,024 points about
$2.7\times10^{-4}$. At $\mu R=0.5$, the corresponding values were about
$4.6\times10^{-6}$ and $1.5\times10^{-6}$. Comparison over 1,791 directions
with the Chen/diffpy.labpdfproc interpolation agreed to about $1.1\times10^{-4}$
or better through $\mu R=1.5$; at $\mu R=3$ the maximum difference was about
0.25%, dominated by a stable difference from that package's polynomial
interpolation near forward scattering rather than by our quadrature order.

Wall-origin integration has a separate convergence concern. When absorption
inside the sample is strong, the outgoing wall path can cross a tangency to the
inner cylinder inside a fixed incident-chord segment. In a development case
with radius ratio 1.1 and sample $\mu R=3$, wall chord orders 12, 20, 32, 48,
and 64 differed from order 96 by maximum relative amounts of approximately
0.60%, 0.29%, 0.12%, 0.088%, and 0.055%, respectively. At sample $\mu R=0.5$,
order 12 differed by only about 0.015%. The composite module therefore exposes
`wall_chord_order` separately from sample `chord_order`; production settings
still require convergence testing. A later detector-dependent split or
adaptive chord rule may reduce this cost without sacrificing accuracy.

## Required user documentation

The implemented module documentation must state that the default axis is
horizontal; tilted, centred capillaries are supported; only actual detector
elements/segments are evaluated; beam profiles are genuinely 2D; line
collimation is only a special case; measured whole-beam transmission is not a
centreline estimate of $\mu$; sample and wall absorption are separate; and
correct empty-capillary subtraction can require wall-origin factors rather
than a post-subtraction sample correction.
