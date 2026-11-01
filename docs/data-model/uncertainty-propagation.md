# Uncertainty propagation

MoDaCor stores absolute, nonnegative one-standard-deviation uncertainties in
`BaseData.uncertainties`, in the signal's units. Each key names an uncertainty
estimate; its array must broadcast to the signal shape. NaN values follow the
operation's existing domain and arithmetic policy.

**Changed (Stage 0):** mixed `propagate_to_all` mappings now raise `ValueError`;
global-only uncertainties survive arithmetic with an empty mapping. Producers
fold a global-only term into newly introduced named estimates. Constructors
copy array inputs, in-place arithmetic is atomic and alias-safe, unary functions
preserve their inputs, and scalar arithmetic avoids full-size zero arrays.

An empty mapping means **no quantified contribution supplied (unquantified)**,
not that a measurement is exact. Stage 0 arithmetic treats absent terms as zero;
this computational convention does not establish exactness. Python real and
scalar Pint quantity operands are coerced to exact zero-dimensional operands
with no uncertainty keys.

## Binary arithmetic

For values `A` and `B`, absolute standard uncertainties `sigma_A` and
`sigma_B`, and result `R`, first-order propagation assumes independent operands:

| Operation | Result uncertainty |
| --- | --- |
| `R = A + B` or `A - B` | `sigma_R² = sigma_A² + sigma_B²` |
| `R = A × B` | `sigma_R² = (B sigma_A)² + (A sigma_B)²` |
| `R = A / B` | `sigma_R² = (sigma_A / B)² + (A sigma_B / B²)²` |

Pint supplies the unit algebra and conversion factors. Addition and subtraction
convert both uncertainties to the result's units. Division by a zero-valued
divisor produces a `NaN` propagated uncertainty at that position. Missing sigma
terms are skipped rather than materialized as zero arrays.

## Uncertainty key table

`E` means empty; `N` means one or more named keys without `propagate_to_all`;
`G` means exactly `{propagate_to_all}`; `M` means a mixed mapping containing
`propagate_to_all` and another key.

The same table applies to `+`, `-`, `*`, `/` and their in-place forms. Both
operand orders are explicit below; the numerical formula still follows the
operation and operand order.

| Left | Right | Result keys and behaviour |
| --- | --- | --- |
| E | E | E |
| N | E | N, propagated from the left operand |
| E | N | N, propagated from the right operand |
| G | E | G, propagated from the left operand |
| E | G | G, propagated from the right operand |
| G | N | N; the left global term contributes to every right named key |
| N | G | N; the right global term contributes to every left named key |
| G | G | G, combining both operands' terms |
| N | N, identical key sets | Pairwise propagation by name |
| N | N, differing key sets | Union; a missing term contributes zero; logged at DEBUG |
| M | Any | `ValueError` before mutation |
| Any | M | `ValueError` before mutation |

The reserved key must be the sole key or absent. Validation runs at
construction, every binary, in-place and unary operation, `indexed`, `copy`,
`to_units`, uncertainty combination, and uncertainty-producing modules. The
mapping remains mutable, so constructing a valid object does not exempt later
operations from validation. To repair a mixed input, keep `propagate_to_all`
alone or rename it to an ordinary key.

The union rule is a Stage 0 limitation. For example,
`{Poisson: 10, SEM: 15} - {Poisson: 7}` produces
`{Poisson: 12.2066, SEM: 15}`. The missing right `SEM` term is treated as zero;
the right `Poisson` term is not substituted for it. Key names do not establish
physical correlation, and estimates are not implicitly added across names.

## Producers and the legacy fold

A producer introducing a named estimate `X` into a global-only mapping computes
`sigma_X = hypot(sigma_X,new, sigma_global)` and replaces the mapping with
`{X: sigma_X}` in one assignment. It logs the transition at INFO as a **legacy
fold**. The global term can remain on the working signal until such a producer
introduces a name.

When one call introduces several estimates, each receives the **same original
global term**. All folded arrays are calculated first; the complete named
mapping is installed atomically, with `propagate_to_all` removed. One INFO
message lists all folded names. No mixed mapping is exposed.

For example, `PoissonUncertainties` on a signal of 100 counts with
`{propagate_to_all: 5}` produces `{Poisson: sqrt(10² + 5²)}`, approximately
`{Poisson: 11.1803}` counts. Dividing by 2 gives `{Poisson: 5.5902}` counts.
`IndexedAverager` and `ReduceDimensionality` use the same batch rule when they
introduce scatter estimates alongside propagated global-only uncertainties.
Existing named-key collision policies still apply.

A **later producer call** adding another name to the already-named mapping
performs no fold: the global term is no longer present separately. That new
estimate lacks the earlier global contribution. This is a Stage 0 limitation.

`CombineUncertainties` combines selected named components in quadrature;
`CombineUncertaintiesMax` takes an element-wise maximum. Neither may use
`propagate_to_all` as a source key or create a mixed result mapping.

## Ownership, in-place and unary operations

Constructors and `copy()` own independent signal and uncertainty arrays. Unit
conversion rebinds converted arrays. In-place arithmetic keeps the container's
identity and computes a complete checked result before installing it, so a
failed operation leaves the target's signal, units, uncertainties and metadata
unchanged. Array identities need not be preserved.

Aliases read their original operand values, including for `a /= a` and
`a *= a`. **Computational alias safety is not physical correlation.** For
`a - a`, independent-operand propagation still gives `sqrt(2) * sigma`, rather
than cancelling the uncertainty. Shared corrections and repeated measurements
require scientific assessment beyond these formulas.

Unary functions use `sigma_y = abs(f'(x)) sigma_x` for each key. A shared helper
converts the signal and every uncertainty together with an inline factor, so
logarithms, exponentials, powers and trigonometric functions leave their input
object and units unchanged. Percent inputs are converted to dimensionless
values; angular inputs are converted to radians. Invalid domain positions
produce `NaN` signal and uncertainties. Negation preserves absolute sigmas.

## Reductions and integration

For an unweighted mean of independent samples,
`sigma_mean = sqrt(sum(sigma_i²)) / N`. For a weighted mean,
`sigma_mean² = sum(w_i² sigma_i²) / sum(w_i)²`. A weighted sum omits the
denominator. `ReduceDimensionality` propagates each existing key and can add
scatter-derived estimates using the producer rule above.

`Integrate1D` uses quadrature coefficients `c_i` and computes
`sigma_I = sqrt(sum((c_i sigma_i)²))` for each key. The output unit is the signal
unit multiplied by the coordinate unit. Invalid, masked or zero-weight samples
are omitted from the common integration domain. A reduction's `nan_policy`
controls whether invalid samples are omitted or produce a `NaN` result.

## Scientific limitations

The container does not represent covariance or physical correlations between
pixels, measurements, parameters or names. First-order propagation may be
inadequate for strongly nonlinear functions or large uncertainties. In
particular, broadcasting and repeated use of a correction do not establish
independence. Choose uncertainty models and explicit combinations to match the
measurement, and retain diagnostic outputs where a stronger model is needed.
