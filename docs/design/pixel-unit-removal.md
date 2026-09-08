# Detector Pixel Units

MoDaCor treats detector pixels as dimensionless detector-coordinate units. Pint's
built-in `pixel`, `pixels`, `px`, and related display/CSS pixel definitions are
removed from the application unit registry during startup, then redefined as
aliases of a named dimensionless detector pixel unit.

## Rationale

Detector element coordinates are array indices. Treating them as Pint units made
metadata hard to reason about because values such as detector pitch became
`mm/pixel` even though the stored number is simply a physical length of one
detector element. The old model also leaked into signal units such as
`count/px`, which mixed detector sampling with physical dimensional analysis.

The current contract keeps unit handling physical:

- detector element indices and beam-center coordinates are dimensionless,
- unit strings such as `pixel`, `pixels`, `px`, `css_pixel`, `dot`, `pel`, and
  `picture_element` are accepted as detector-coordinate aliases,
- detector element sizes are lengths, for example `m`, `mm`, or `um`,
- solid angle outputs are `sr`,
- masks, index maps, and names such as `pixel_index` remain detector-element
  concepts and may use either `dimensionless` or a pixel alias.

This deliberately replaces Pint's display/CSS interpretation of pixel-like unit
names. In MoDaCor, `px` does not mean a CSS length.

## Migration

The preferred metadata style is still to store physical quantities in physical
units and detector indices as dimensionless values:

| Old unit string | New unit string |
| --- | --- |
| `mm/pixel` | `mm` |
| `m/pixel` | `m` |
| `count/px` | `count` |
| `counts/pixel/second` | `counts/second` |

The numeric values do not change for detector pitch or detector element size.
For example, a stored pitch value of `0.172` with old units `mm/pixel` becomes
the same value `0.172` with units `mm`.

For compatibility with existing metadata, pixel denominators are also accepted
and cancel as dimensionless factors:

- `0.172 mm/pixel` converts to `0.172 mm`,
- `5 count/px` converts to `5 count`,
- `10 counts/pixel/second` converts to `10 counts/second`.

`BaseData.is_dimensionless` checks Pint compatibility rather than exact unit
equality, so a `BaseData` stored with `pixel` units is considered dimensionless.
Calling `BaseData.to_dimensionless()` normalizes compatible named units such as
`pixel` to Pint's plain `dimensionless` unit without changing the numerical
values or uncertainties.

External pipeline repositories should prefer updating SAXSess and MOUSE-style
metadata sources to plain length units for detector pitch, but stale
`mm/pixel`-style metadata remains parseable.
