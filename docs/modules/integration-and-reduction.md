# Integration and reduction

[`ReduceDimensionality`](../reference/modules/ReduceDimensionality.md) computes
weighted or unweighted means and sums over selected axes. `axes: non_data`
reduces leading acquisition dimensions while retaining the trailing scientific
rank. Existing uncertainties propagate per name, and optional estimators can add
scatter-derived components.

[`IndexedAverager`](../reference/modules/IndexedAverager.md) groups data by an
index map, a common pattern for radial or azimuthal binning.

[`Integrate1D`](../reference/modules/Integrate1D.md) performs trapezoidal or
Simpson integration of curves on a shared, strictly monotonic coordinate axis.
Coordinate units multiply into the result unit. Invalid, masked, or zero-weight
points define one common domain across all integrated inputs.

Before reducing:

- apply pixel-level corrections that require detector geometry;
- decide whether a mask should mutate the signal or be consumed by the
  reduction;
- distinguish propagated input uncertainty from scatter estimation; and
- confirm whether weights express statistical importance, exposure, or another
  reviewed model.
