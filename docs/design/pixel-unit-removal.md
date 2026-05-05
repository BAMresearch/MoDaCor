# Pixel Unit Removal

MoDaCor no longer treats detector pixels as physical units. Pint's built-in
`pixel`, `pixels`, and `px` definitions are removed from the application unit
registry during startup, and MoDaCor does not replace them with custom detector
element units.

## Rationale

Detector element coordinates are array indices. Treating them as Pint units made
metadata hard to reason about because values such as detector pitch became
`mm/pixel` even though the stored number is simply a physical length of one
detector element. The old model also leaked into signal units such as
`count/px`, which mixed detector sampling with physical dimensional analysis.

The current contract keeps unit handling physical:

- detector element indices and beam-center coordinates are dimensionless,
- detector element sizes are lengths, for example `m`, `mm`, or `um`,
- solid angle outputs are `sr`,
- masks, index maps, and names such as `pixel_index` remain detector-element
  concepts but do not carry pixel units.

## Migration

Update metadata and pipeline inputs by removing pixel denominators from unit
strings:

| Old unit string | New unit string |
| --- | --- |
| `mm/pixel` | `mm` |
| `m/pixel` | `m` |
| `count/px` | `count` |
| `counts/pixel/second` | `counts/second` |

The numeric values do not change for detector pitch or detector element size.
For example, a stored pitch value of `0.172` with old units `mm/pixel` becomes
the same value `0.172` with units `mm`.

## Strict Failure Mode

This is an intentional breaking change. Unit strings containing `pixel`,
`pixels`, or `px` now fail during Pint parsing. MoDaCor does not provide a
compatibility warning or automatic conversion path because silent normalization
would hide stale metadata.

External pipeline repositories must update their metadata sources before running
against this version. In particular, SAXSess and MOUSE-style pipelines that read
detector pitch from HDF attributes or static YAML must store those pitch units as
plain length units.
