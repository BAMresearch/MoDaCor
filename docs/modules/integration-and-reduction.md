# Integration and reduction

[`ReduceDimensionality`](../reference/modules/ReduceDimensionality.md) computes
weighted or unweighted means and sums over selected axes. `axes: non_data`
reduces leading acquisition dimensions while retaining the trailing scientific
rank. Existing uncertainties propagate per name, and optional estimators can add
scatter-derived components.

[`IndexByCoordinate`](../reference/modules/IndexByCoordinate.md) assigns one
arbitrary coordinate to one-dimensional bins. It records the physical edges
for diagnostics but deliberately does not inspect signals or masks.
It replaces the retired scattering-specific `IndexPixels` step. Existing
pipelines must migrate their bin limits and selected coordinate explicitly; see
the [breaking interface migration guide](../processing/interface-migrations.md).

[`IndexedAverager`](../reference/modules/IndexedAverager.md) groups a configured
value by that index map. An optional measured axis, such as Q, is averaged over
the same accepted points and weights; it is not used to assign bins. The step
reports actual mean coordinates, their spread, and population, weight-sum, and
effective-sample-size diagnostics rather than substituting nominal bin centres.
Bins without positive total weight are omitted and their original IDs are
retained in the output.
This generic interface replaces the former scattering-specific configuration;
in particular, `averaging_direction` and the `use_signal_*` fields are no
longer accepted.

[`ConcatenateDatabundles`](../reference/modules/ConcatenateDatabundles.md)
pools compatible one-dimensional bundles before indexed reduction. Its
`sort_by` option is deliberately optional because `IndexByCoordinate` does not
require monotonic input.

[`FindCenterOfMass1D`](../reference/modules/FindCenterOfMass1D.md) estimates a
one-dimensional peak centre within an iteratively refined contiguous window.
It is appropriate when a structured or asymmetric peak should not be forced
into a Gaussian model.

[`Integrate1D`](../reference/modules/Integrate1D.md) performs trapezoidal or
Simpson integration of curves on a shared monotonic coordinate axis. It can
stably sort a scan and, when explicitly requested, consolidate exact duplicate
coordinates before quadrature.
Coordinate units multiply into the result unit. Invalid, masked, or zero-weight
points define one common domain across all integrated inputs.

Before reducing:

- apply pixel-level corrections that require detector geometry;
- decide whether a mask should mutate the signal or be consumed by the
  reduction;
- distinguish propagated input uncertainty from scatter estimation; and
- confirm whether weights express statistical importance, exposure, or another
  reviewed model.
