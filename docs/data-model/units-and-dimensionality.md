# Units and dimensionality

MoDaCor uses one application-wide Pint registry, exported as `modacor.ureg`.
Addition and subtraction require compatible units. Multiplication and division
derive result units normally, so counts divided by seconds becomes counts per
second and intensity divided by solid angle gains inverse-steradian units.

Unit conversion applies the same multiplicative factor to the signal and every
absolute uncertainty. Offset-unit conversion is deliberately rejected because
uncertainties would require an explicit delta-unit policy.

`rank_of_data` describes how many trailing dimensions carry the scientific data
grid:

- `0`: scalar metadata or a correction factor;
- `1`: a curve;
- `2`: an image; and
- `3`: a volume.

Leading dimensions can represent repeated frames or scan positions. The rank
must be between zero and three and cannot exceed `signal.ndim`.

Detector element coordinates are indices. MoDaCor therefore treats `pixel`,
`px`, `dot`, and related aliases as dimensionless detector-coordinate units.
Detector pitch remains a physical length such as `m`, `mm`, or `um`.
