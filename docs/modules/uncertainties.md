# Uncertainty workflows

Uncertainty modules create or combine named contributions; ordinary `BaseData`
arithmetic propagates them. Read the formal
[propagation rules](../data-model/uncertainty-propagation.md) before designing a
workflow.

[`PoissonUncertainties`](../reference/modules/PoissonUncertainties.md) estimates
counting uncertainty from the signal. Apply it while values still represent
counts; after normalization to count rate, propagate the existing component
rather than estimating it again.

Keep contributions such as `Poisson`, `readout`, `transmission`, and
`calibration` separate while diagnosing a pipeline. Use
[`CombineUncertainties`](../reference/modules/CombineUncertainties.md) only for
components that the physical model permits combining in quadrature. Use
[`CombineUncertaintiesMax`](../reference/modules/CombineUncertaintiesMax.md)
when the intended result is a conservative element-wise envelope.

Reduction modules can estimate uncertainty from observed scatter. That estimate
is distinct from propagating known input uncertainty. Configure a collision
policy deliberately if both would use the same name.

MoDaCor does not currently represent covariance. Correlated calibration or
systematic contributions need an explicit domain model rather than an
undocumented quadrature sum.
