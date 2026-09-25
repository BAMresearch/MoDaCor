# ReduceDimensionality Uncertainty-Estimator Upgrade Plan

Status: implemented and verified locally on 2026-09-24.

## Goal

Extend `ReduceDimensionality` so a dimensionality reduction can optionally
derive new uncertainty estimates from the values being reduced. The existing
propagation of uncertainty arrays attached to `BaseData` remains unchanged.

Users must be able to:

- request zero, one, or several estimators;
- choose the output key for every estimate;
- select estimators appropriate to a mean or sum reduction;
- control what happens when an output key already exists; and
- obtain weighted estimates when `ReduceDimensionality` uses data weights.

This is an opt-in extension. Pipelines that omit the new configuration must
retain their current numerical results and uncertainty keys.

## Current behavior

For every input uncertainty component `sigma[k]`, the module currently assumes
uncorrelated errors and propagates it through the configured reduction. For a
weighted mean,

```text
mean = sum(w_i * x_i) / sum(w_i)
sigma_mean[k] = sqrt(sum(w_i**2 * sigma_i[k]**2)) / sum(w_i)
```

For a weighted sum,

```text
sum = sum(w_i * x_i)
sigma_sum[k] = sqrt(sum(w_i**2 * sigma_i[k]**2))
```

These are propagated measurement-uncertainty components. The module does not
currently estimate uncertainty or scatter from the distribution of `x_i`.

## Proposed configuration contract

Add one optional top-level argument, `uncertainty_estimation`, to the
`ReduceDimensionality` configuration:

```yaml
steps:
  reduce_frames:
    module: ReduceDimensionality
    configuration:
      with_processing_keys: [sample]
      axes: non_data
      reduction: mean
      use_weights: true
      nan_policy: omit

      uncertainty_estimation:
        collision_policy: error
        estimators:
          frame_STD:
            method: standard_deviation
            ddof: 1
          frame_SEM:
            method: standard_error_mean
            ddof: 1
```

The keys below `estimators` are the exact destination keys in the output
`BaseData.uncertainties` mapping. The example therefore creates
`uncertainties["frame_STD"]` and `uncertainties["frame_SEM"]`.

The following values all disable estimation and preserve current behavior:

```yaml
uncertainty_estimation: null
```

```yaml
uncertainty_estimation: {}
```

```yaml
uncertainty_estimation:
  estimators: {}
```

The initial implementation should accept only documented estimator methods and
parameters. It must not import or evaluate arbitrary callables named in YAML.

### Collision policy

`collision_policy` controls the case where a configured destination key is
already present after the input uncertainties have been propagated to the
reduced output shape.

Supported values are:

- `error`: raise a clear error without replacing the existing component. This
  is the default.
- `overwrite_existing`: store the new estimator result under the configured
  key, replacing the existing propagated component.
- `keep_existing`: retain the existing propagated component and discard the
  newly calculated estimate for that key.
- `propagate`: combine the existing propagated component and the new estimate
  in quadrature:

  ```text
  sigma_result = sqrt(sigma_existing**2 + sigma_estimated**2)
  ```

In this contract, **existing** means the uncertainty component already present
in the reduced output uncertainty mapping. When it originates in the input
`BaseData`, it has already been propagated over the selected axes; it does not
mean the unreduced input array.

If the destination key does not exist, the estimate is added and the collision
policy has no effect. The module should log collisions resolved by
`overwrite_existing`, `keep_existing`, or `propagate`.

The `propagate` policy assumes that the two components are independent. This
may be scientifically inappropriate when a scatter-derived estimate already
contains the measurement noise represented by the existing component. The
module will perform the requested quadrature combination but must document
that it can double-count uncertainty.

An optional per-estimator `collision_policy` override may be supported without
changing the global contract:

```yaml
uncertainty_estimation:
  collision_policy: error
  estimators:
    frame_STD:
      method: standard_deviation
      ddof: 1
      collision_policy: overwrite_existing
```

The estimator-level value takes precedence over the enclosing value. Supporting
this override in the first implementation is recommended because it adds
little complexity and avoids forcing one policy on unrelated output keys.

## Estimator contract

The public method names describe the statistical quantity rather than exposing
an arbitrary Python import path. Implementations should use vectorized NumPy or
SciPy operations and may dispatch to built-in estimators where their semantics
match the request.

Initial methods:

| Method | Applicable reduction | Output meaning |
| --- | --- | --- |
| `standard_deviation` | `mean`, `sum` | Estimated scatter of the contributing values |
| `standard_error_mean` | `mean` | Estimated uncertainty of the reduced mean from contributor scatter |
| `standard_error_sum` | `sum` | Estimated uncertainty of the reduced sum from contributor scatter |

`standard_deviation` is meaningful with either reduction, but for a sum it is
only a description of the contributing-value scatter. It is not an uncertainty
of the output sum. `standard_error_mean` must be rejected for a sum reduction,
and `standard_error_sum` must be rejected for a mean reduction, so that a
plausible-looking but dimensionally mis-scaled uncertainty cannot be produced
silently.

The initial release should support `ddof` as a non-negative integer and use a
default of `1`. Invalid configurations must fail during step preparation,
before any `DataBundle` is modified.

### Unweighted implementation

For unweighted data, `standard_deviation` should use `numpy.std` or
`numpy.nanstd`, and `standard_error_mean` may use `scipy.stats.sem`. An adapter
is still needed to apply MoDaCor's axes and NaN contracts consistently,
especially for multiple reduction axes.

Let `N` be the number of participating values and let

```text
variance = sum((x_i - mean)**2) / (N - ddof)
```

Then:

```text
standard_deviation = sqrt(variance)
standard_error_mean = sqrt(variance / N)
standard_error_sum = sqrt(N * variance)
```

The last equality is also `N * standard_error_mean`.

### Weighted implementation

NumPy and SciPy do not provide all required arbitrary-weight variants through
the corresponding `std` and `sem` APIs. Weighted estimators should therefore
use vectorized NumPy moment calculations rather than Python loops.

Using the same effective weights as the signal reduction:

```text
sum_w = sum(w_i)
sum_w2 = sum(w_i**2)
N_eff = sum_w**2 / sum_w2
mean_w = sum(w_i * x_i) / sum_w
variance_0 = sum(w_i * (x_i - mean_w)**2) / sum_w
variance = variance_0 * N_eff / (N_eff - ddof)
```

The requested estimates are then:

```text
standard_deviation = sqrt(variance)
standard_error_mean = sqrt(variance / N_eff)
standard_error_sum = abs(sum_w) * standard_error_mean
```

For unit weights these reduce to the unweighted formulas. A common non-unit
weight additionally scales the weighted sum and its uncertainty by that
weight. The sum estimator can also be written as
`sqrt(sum_w2 * variance)`.

Estimator calculations require non-negative weights. Zero-weight observations
do not participate, and an output position with no positive effective weight
receives `NaN` estimates. If an estimator is requested and a participating
weight is negative, the step must raise a clear error. This validation applies
to uncertainty estimation; changing the legacy signal-reduction behavior for
negative weights is outside this upgrade.

An estimate is undefined and should be `NaN` wherever the participating count
or `N_eff` is less than or equal to `ddof`.

## Mean and sum semantics

### Mean reductions

`standard_error_mean` estimates uncertainty in the mean from observed scatter.
It is distinct from the propagated input uncertainties already calculated by
the module. Both can coexist under different uncertainty keys or be resolved
under one key using `collision_policy`.

### Sum reductions

The module can derive a scatter-based uncertainty of a sum. For an unweighted
sum of independent contributions with a common estimated variance,

```text
sigma_sum = sqrt(N) * standard_deviation
```

For a weighted sum,

```text
sigma_sum = sqrt(sum(w_i**2) * variance)
```

This is the `standard_error_sum` estimator. It assumes independent
contributions whose dispersion can be estimated from the values being summed.
It is not suitable for correlated frames without an explicit correlation
model.

Where per-contribution uncertainties are already known, the existing
uncertainty propagation is usually preferable. For example, Poisson
uncertainties can be attached before the sum and will then be propagated using
the existing `sqrt(sum(w_i**2 * sigma_i**2))` rule.

## Axes, NaNs, and no-op behavior

Estimators operate over exactly the axes selected for the signal reduction,
including `axes: non_data`, negative axes, axis tuples, and `axes: null`.

- With `nan_policy: omit`, a signal value excluded from the mean or sum is also
  excluded from all requested estimators. A `NaN` array weight excludes the
  same value, matching the existing reduction mask.
- With `nan_policy: propagate`, a non-finite participating signal or weight
  propagates to the signal result and requested estimates as it does in the
  current reduction.
- NaNs in an attached input uncertainty continue to follow the existing
  uncertainty-propagation behavior and do not independently change which
  signal values participate in a scatter estimator.
- When `axes: non_data` resolves to no axes, the existing true no-op behavior
  remains: the original `BaseData` object is retained and no estimator keys are
  added.

## Direct mask support

`ReduceDimensionality` can exclude an integer NeXus-style mask without first
replacing signal values with `NaN`:

```yaml
mask_key: mask
mask_bits: null
```

`mask_key: null` is the backward-compatible default and disables direct mask
handling. When `mask_key` is configured, its `BaseData.signal` must have an
integer dtype and be broadcast-compatible with the signal being reduced. The
source signal and mask are not modified.

With `mask_bits: null`, every nonzero mask value is excluded. `mask_bits` may
instead be one positive uint32 bitfield value or a list/tuple of values; values
in an iterable are combined with bitwise OR. An element is excluded when:

```text
(mask_value & selected_mask_bits) != 0
```

Explicitly masked elements receive zero effective weight for the signal
reduction, existing-uncertainty propagation, and every scatter-derived
estimator. They are excluded independently of `nan_policy`. Consequently, a
masked `NaN` does not propagate when `nan_policy: propagate`, while an unmasked
`NaN` still does.

If all contributors to an output position are masked, a mean is `NaN` and a
sum follows NumPy's empty-sum identity of zero. Scatter-derived estimators are
`NaN` because there is no effective sample.

The configured mask is an input selection and is not automatically reduced or
replaced. Use `ReduceMask` separately when the output `DataBundle` also needs a
mask at the reduced shape. `ApplyMask` followed by `nan_policy: omit` remains a
supported alternative when replacing the source values is desired.

## Processing order

For each selected `DataBundle`, the enhanced step should:

1. resolve and validate the reduction axes and estimator configuration;
2. construct the effective signal mask and weights;
3. calculate the mean or sum;
4. propagate every existing input uncertainty component using the current
   formulas;
5. calculate the requested scatter-derived estimates from the unreduced signal
   and effective weights;
6. merge estimator results into the propagated uncertainty mapping according
   to the applicable collision policy; and
7. construct the output `BaseData` with the existing units, weights, axes, and
   `rank_of_data` rules.

Collision errors and invalid estimator configurations should be detected
before replacing `databundle["signal"]`, preventing partial mutation of that
bundle.

## Implementation outline

1. Add `uncertainty_estimation` as a `dict | None` argument in the
   `ProcessStepDescriber` for `ReduceDimensionality`, defaulting to `None`.
2. Add a normalizer that validates the nested schema, destination keys,
   estimator names, `ddof`, and collision policies. The generic process-step
   schema validates only the outer mapping, so this validation belongs in the
   module or a reusable statistics helper.
3. Introduce an internal estimator registry. Registry entries are explicit
   functions; do not use `eval`, arbitrary imports, or unrestricted `getattr`.
4. Build a shared vectorized moment helper supporting scalar and broadcast
   weights, arbitrary reduction axes, `keepdims` for deviations, and the two
   NaN policies.
5. Reuse calculated moments when multiple estimators are requested so that
   requesting both standard deviation and standard error does not repeat full
   array passes unnecessarily.
6. Apply collision policies after the current input-uncertainty propagation
   and before constructing the result `BaseData`.
7. Keep the scalar-weight fast path where it remains beneficial, but route its
   estimator calculations through the same helper to avoid divergent
   semantics.
8. Increment the module version and update its descriptor text and reference
   documentation.
9. Add optional, non-mutating `mask_key` and `mask_bits` selection and apply
   the resulting exclusion consistently to reductions, propagated
   uncertainties, and estimators.

## Shared implementation with IndexedAverager

`ReduceDimensionality` and `IndexedAverager` use different grouping
mechanisms: regular NumPy axes versus arbitrary pixel bins accumulated with
`numpy.bincount`. That selection and accumulation code remains module-local.

Their common moment-finalization code belongs in
`modacor.modules.helpers.statistics`. Both modules should call this helper for:

- effective sample size;
- degree-of-freedom correction of weighted variance;
- standard deviation;
- standard error of the mean; and
- standard error of a sum.

The helper accepts already accumulated `sum_w`, `sum_w2`, and weighted squared
deviations. It does not know about axes, bin indices, masks, Q, or circular Psi.
This boundary lets `IndexedAverager` retain its existing population-variance
and minimum-bin-size behavior while removing its duplicate finalization
formula. Equivalence is protected by its existing focused tests.

There is another related scalar-only implementation in
`modules.helpers.scattering.detector_data`. Migrating that helper is outside
this enhancement and should happen only with equivalence tests, because its
current defaults are part of its own contract.

## Test plan

Focused tests in `tests/modules/base_modules/test_reduce_dimensionality.py`
should cover:

- omitted, `null`, and empty configuration preserving existing behavior;
- one and several estimator destination keys;
- unweighted mean standard deviation and standard error;
- weighted mean standard deviation and standard error;
- unweighted and weighted `standard_error_sum`;
- `ddof=0`, `ddof=1`, and insufficient effective sample size;
- integer, negative, tuple, `null`, and `non_data` axes;
- `omit` and `propagate` NaN policies;
- scalar, broadcast, zero, non-finite, and negative weights;
- retention of independently named propagated uncertainties;
- all four collision policies, including quadrature values for `propagate`;
- global collision policy and an estimator-level override;
- rejection of mean-only estimators for sums and sum-only estimators for
  means;
- rejection of unknown estimators, parameters, and collision policies;
- true no-op behavior when `axes: non_data` resolves to no axes; and
- output shape, units, axes, weights, and `rank_of_data` remaining unchanged
  from the current reduction contract.

The focused test file should pass before broader base-module and pipeline tests
are run.

## Documentation changes

Implementation should update:

- the `ReduceDimensionality` `ProcessStepDescriber` argument and step text;
- `docs/pipeline_operations/configuration_reference.md` with mean and sum
  examples;
- uncertainty terminology explaining propagated versus scatter-estimated
  components;
- the changelog; and
- any generated configuration reference that consumes the process-step
  descriptor.

## Compatibility and acceptance criteria

The upgrade is complete when:

- existing pipelines without `uncertainty_estimation` produce unchanged
  signals, uncertainty mappings, weights, and metadata;
- each configured estimator writes to exactly its user-selected key;
- collision policies use the terms `existing`, `overwrite_existing`, and
  `keep_existing` consistently in code, errors, tests, and documentation;
- `propagate` combines components in quadrature and documents the independence
  assumption;
- mean and sum estimators are validated against the configured reduction;
- omitted `mask_key` configuration preserves legacy behavior, while a
  configured mask is applied consistently without mutating its inputs;
- weighted, multi-axis, and NaN behavior is covered by focused tests; and
- no arbitrary function import or execution is possible through estimator
  configuration.

## Implementation record

- Added the opt-in `uncertainty_estimation` configuration and estimator-level
  collision-policy overrides to `ReduceDimensionality`.
- Added `standard_deviation`, `standard_error_mean`, and
  `standard_error_sum`, including weighted estimates and configurable `ddof`.
- Added direct, non-mutating bitfield-mask selection through `mask_key` and
  `mask_bits`.
- Added the shared `modacor.modules.helpers.statistics` moment finalizer and
  migrated `IndexedAverager` to it without changing its established results.
- Added focused estimator, collision, configuration-validation, helper, and
  compatibility tests.
- Verified the complete test suite: 847 tests passed and one optional memory
  integration test was skipped.
