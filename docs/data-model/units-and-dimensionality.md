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

## Conversion and numerical boundaries

`data.to_units("m")` converts in place, rebinding the signal and every named
uncertainty array. `converted = data.in_units("m")` returns an independent
converted object. Both promote integer signals to floating point, validate the
uncertainty mapping before conversion, and preserve axes. Weights are
**dimensionless relative weights**; unit conversions leave their values alone.

At numerical-kernel boundaries, `signal, uncertainties = data.magnitudes("m")`
returns both in the requested units without mutating the input. Identity
conversions may return borrowed arrays; use `in_units` for independent storage.
`value, components = data.scalar_components("m")` returns Python floats,
accepting scalars or exactly constant arrays for the signal and every component.
A global-only mapping retains its `propagate_to_all` key.

## Scalar metadata and configuration

`BaseData.from_scalar` is the common constructor for scalar metadata and
configuration. It folds pure-number units such as percent, permille and ppm into
the magnitude, converting absolute uncertainties with the same factor. It also
cancels matching dimensional factors: `m/mm` becomes dimensionless with a factor
of 1000. It preserves `count`, angle units defined via radian (including degree),
`steradian` and `pixel`, including their occurrences in compound units.
Thus `count/%` becomes count with a factor of 100, while `count/sr` stays intact.

This reduction applies only at scalar ingestion. General constructors and
arithmetic results retain their Pint units; there is no automatic reduction
of processing results.
