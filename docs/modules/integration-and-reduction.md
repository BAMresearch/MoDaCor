# Integration and reduction

[`ReduceDimensionality`](../reference/modules/ReduceDimensionality.md) computes
weighted or unweighted means and sums over selected axes. `axes: non_data`
reduces leading acquisition dimensions while retaining the trailing scientific
rank. Existing uncertainties propagate per name, and optional estimators can add
scatter-derived components.

[`IndexedAverager`](../reference/modules/IndexedAverager.md) groups data by an
index map, a common pattern for radial or azimuthal binning. It reports the
weighted mean of the coordinates that actually entered each populated bin,
their spread, and optional population, weight-sum, and effective-sample-size
diagnostics rather than substituting nominal bin centres.

[`ConcatenateDatabundles`](../reference/modules/ConcatenateDatabundles.md)
pools compatible one-dimensional bundles before indexed reduction. Its
`sort_by` option is deliberately optional because `IndexPixels` does not
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
