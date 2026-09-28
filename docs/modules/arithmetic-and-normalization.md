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
one-dimensional overlap scale. [`UnitsLabelUpdate`](../reference/modules/UnitsLabelUpdate.md)
changes a unit label only under its documented contract; it is not a substitute
for numerical conversion.
