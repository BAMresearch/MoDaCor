# Scientific principles

## Quantities remain explicit

Signals and correction factors carry Pint units. Compatible units are converted
when necessary, while multiplication and division use ordinary unit algebra.
Dimensionless detector indices remain distinguishable from physical pixel
pitch, which has length units.

## Uncertainty sources remain traceable

A signal may carry several named one-standard-deviation contributions, such as
counting statistics, readout noise, calibration, or transmission. MoDaCor
propagates the components independently instead of collapsing them prematurely
to one number. See [Uncertainty propagation](../data-model/uncertainty-propagation.md)
for the exact rules and limitations.

## Steps are inspectable

A pipeline is a directed acyclic graph of named processing steps. Each step has
configuration metadata, declared data dependencies, and a traceable place in
the graph. Intermediate state can be inspected, traced, or reused during a
partial rerun.

## Reproducibility includes operation

Pipeline specifications, source and sink registrations, selected outputs,
trace events, and execution metadata together describe how a result was
produced. Reproducibility therefore includes both numerical operations and the
runtime context in which they were applied.

## Quality takes precedence over throughput

MoDaCor supports chunked and server-driven operation, but scientific semantics
come first. Optimizations must preserve units, uncertainties, axes, masks,
weights, dependencies, and traceability at the `BaseData` boundary.
