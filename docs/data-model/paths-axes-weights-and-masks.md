# Paths, axes, weights, and masks

## Paths

External source paths use `ref::path`: the part before `::` selects a registered
source and the remainder selects data inside that source. Attribute lookup may
append `@attribute`, for example `sample::/entry/data@units`.

Processing paths select an in-memory bundle, `BaseData` entry, and optionally a
field below it, for example `/sample/signal/signal`. Exact accepted paths depend
on the consuming sink or API operation.

## Axes

`BaseData.axes` can associate coordinate `BaseData` objects with signal
dimensions. Arithmetic performs inexpensive structural checks, not deep array
equality. Modules that require identical coordinates must enforce that rule.
Reductions retain only axes that were not reduced when complete axis metadata is
available.

## Weights

Weights are scalar or signal-broadcastable arrays used by reduction modules.
They are not uncertainties. A weight of zero can exclude a point from a
weighted calculation; negative weights are rejected where an estimator requires
non-negative effective weights.

## Masks

Masks are `uint32` bitfields. A nonzero value identifies an invalid or excluded
point, while individual bits can preserve multiple reasons. Mask creation and
combination should keep those reasons until a consuming operation explicitly
selects bits.

`ApplyMask` changes selected signals to `NaN` or another configured scalar.
Reduction and integration modules can instead consume a mask without mutating
the input. See the [masking guide](../modules/masking.md) for workflow choices.
