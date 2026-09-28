# Masking

MoDaCor masks are `uint32` bitfields. Zero means valid; each nonzero bit can
retain a distinct exclusion reason. Keeping reason bits separate makes the mask
auditable and lets later reductions select only relevant conditions.

## Create masks

[`ThresholdMask`](../reference/modules/ThresholdMask.md) can inspect any
`BaseData` entry, not only `signal`. Typical uses include signal saturation,
flat-field bounds, and accepted `Q` or azimuth ranges.

[`DilateMask`](../reference/modules/DilateMask.md) expands invalid regions when
neighboring pixels are affected. [`BitwiseOrMasks`](../reference/modules/BitwiseOrMasks.md)
combines reason masks without collapsing them to booleans.

## Consume masks

[`ApplyMask`](../reference/modules/ApplyMask.md) replaces selected signal values,
normally with `NaN`. It reads the mask without rewriting it. Use this when
downstream code recognizes invalid numerical values.

`ReduceDimensionality` and `Integrate1D` can instead consume masks directly,
leaving the input signal unchanged. This is preferable when the same data must
be reduced under different mask policies.

[`ReduceMask`](../reference/modules/ReduceMask.md) adapts masks when dimensional
reduction requires a corresponding lower-rank mask.

## Practical rules

- Store masks as separate `BaseData` entries with dimensionless units.
- Assign stable powers-of-two reason bits and document them near pipeline YAML.
- Combine masks with bitwise operations, not addition.
- Apply geometry and detector masks before reductions that assume a valid common
  domain.
- Do not treat masks, zero weights, and uncertainties as interchangeable; each
  has different semantics.
