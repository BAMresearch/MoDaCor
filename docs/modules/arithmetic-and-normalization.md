# Arithmetic and normalization

MoDaCor provides two arithmetic patterns.

Source-based [`Divide`](../reference/modules/Divide.md),
[`Multiply`](../reference/modules/Multiply.md), and
[`Subtract`](../reference/modules/Subtract.md) obtain their second operand from
`IoSources`. They suit scalar metadata or calibration arrays that should remain
external to the processing workspace.

[`DivideDatabundles`](../reference/modules/DivideDatabundles.md),
[`MultiplyDatabundles`](../reference/modules/MultiplyDatabundles.md), and
[`SubtractDatabundles`](../reference/modules/SubtractDatabundles.md) operate on
two prepared `BaseData` entries in `ProcessingData`. Their
`with_processing_keys` order is significant and they update the first bundle.

All use `BaseData` unit algebra, uncertainty propagation, broadcasting, and
metadata checks. Do not strip values to arrays to reimplement these semantics in
a pipeline step.

[`FindScaleFactor1D`](../reference/modules/FindScaleFactor1D.md) estimates a
one-dimensional overlap scale. Its lognormal estimator uses a caller-selected
named uncertainty component rather than silently inventing a combined
uncertainty. Exact duplicate fit coordinates are inverse-variance averaged
before interpolation.

[`SubtractInterpolated1D`](../reference/modules/SubtractInterpolated1D.md)
subtracts a one-dimensional background on a different coordinate grid. Its
`nearest` mode is an explicit nearest-neighbour remap without extrapolation;
`linear` propagates background uncertainty through the interpolation
coefficients. [`Negate`](../reference/modules/Negate.md) changes the sign of a
selected `BaseData` entry while retaining its uncertainty magnitudes.

[`UnitsLabelUpdate`](../reference/modules/UnitsLabelUpdate.md)
changes a unit label only under its documented contract; it is not a substitute
for numerical conversion.
