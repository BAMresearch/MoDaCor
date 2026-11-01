# Uncertainty propagation

MoDaCor stores uncertainties as absolute one-standard-deviation values in
`BaseData.uncertainties`. Each key names a contribution, for example `Poisson`,
`readout`, `transmission`, or `calibration`. Arrays may be scalar or broadcast
to the signal shape.

The current arithmetic model uses first-order propagation and assumes the two
operands' contributions are uncorrelated. It does not represent covariance.

## Binary arithmetic

For values `A` and `B`, absolute standard uncertainties `sigma_A` and
`sigma_B`, and result `R`:

| Operation | Result uncertainty |
| --- | --- |
| `R = A + B` or `A - B` | `sigma_R² = sigma_A² + sigma_B²` |
| `R = A × B` | `sigma_R² = (B sigma_A)² + (A sigma_B)²` |
| `R = A / B` | `sigma_R² = (sigma_A / B)² + (A sigma_B / B²)²` |

Pint converts addition/subtraction uncertainties to the result unit before the
quadrature sum. Multiplication and division use the derived result unit.
Division by a zero-valued divisor produces an undefined (`NaN`) propagated
uncertainty at that position.

## Named-component rules

Propagation happens independently for each name:

1. Matching names on both operands are combined with the operation's formula.
2. Non-matching names form a union; each contribution propagates from the
   operand that contains it, with zero contribution from the other operand.
3. An absent uncertainty mapping contributes no components.
4. `propagate_to_all` is a global fallback only when it is the sole key on an
   operand. It contributes to every explicit name on the other operand.
5. If both operands contain only `propagate_to_all`, the result retains that
   key. When it appears alongside other keys it is not a fallback and is not
   emitted as a result key.

This scheme preserves provenance, but a shared name is not proof of statistical
independence. Choose names and combinations to match the physical uncertainty
model.

## Unary functions

Supported unary functions use `sigma_y ≈ abs(f'(x)) sigma_x` independently for
each named component. This covers powers, square root, logarithm, exponential,
and trigonometric helpers. Outside a function's valid domain, both the signal
and propagated uncertainties are set to `NaN`.

Negation copies absolute uncertainties unchanged. Exact scalar operations carry
zero additional uncertainty while preserving existing names.

## Reductions and integration

For an unweighted mean of independent samples,
`sigma_mean = sqrt(sum(sigma_i²)) / N`. For a weighted mean,
`sigma_mean² = sum(w_i² sigma_i²) / sum(w_i)²`. A weighted sum omits the
denominator. `ReduceDimensionality` applies these rules to every existing name
and can separately add scatter-derived estimators.

`Integrate1D` derives quadrature coefficients `c_i` from the shared coordinate
axis and calculates `sigma_I = sqrt(sum((c_i sigma_i)²))` per component. The
output unit is the signal unit multiplied by the coordinate unit. Invalid,
masked, or zero-weight samples are omitted from the common integration domain.

The `nan_policy` of a reduction determines whether invalid samples are omitted
or make the reduced result `NaN`. Optional masks exclude selected samples
without changing the input arrays.

## Creating and combining components

- [`PoissonUncertainties`](../reference/modules/PoissonUncertainties.md) creates
  a `Poisson` estimate from count data.
- [`CombineUncertainties`](../reference/modules/CombineUncertainties.md)
  combines selected independent components in quadrature under a new name.
- [`CombineUncertaintiesMax`](../reference/modules/CombineUncertaintiesMax.md)
  takes the element-wise maximum when a conservative envelope, rather than an
  independent quadrature model, is intended.

Propagating an existing measurement component and estimating a new component
from scatter are different operations. Name them distinctly and choose an
explicit collision policy when a reduction estimator would reuse an existing
name.

## Limitations and scientific responsibility

The container does not currently carry covariance matrices or correlations
between pixels, samples, parameters, or named components. First-order rules can
be inadequate for strongly nonlinear transformations or large uncertainties.
Systematic contributions must not be combined in quadrature merely for
convenience. When correlation or a domain-specific model matters, implement and
document that model explicitly and retain diagnostic outputs.
