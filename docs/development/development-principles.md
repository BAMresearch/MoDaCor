# Development principles

- Preserve scientific meaning: units, named uncertainties, axes, rank, weights,
  masks, and provenance are part of a result.
- Treat runtime code and focused tests as the executable contract. Reconcile
  stale prose rather than adding a third behavior.
- Prefer small, composable steps with explicit dependencies and configuration.
- Keep numerical kernels reusable and pipeline adaptation thin.
- Let provenance determine the process-step interface: acquisition or
  calibration metadata that is not computed by an upstream MoDaCor step must
  be loadable from an `IoSource`; dynamically derived or modified values must
  flow between steps as `BaseData` entries in `DataBundle` objects. Do not add
  staging steps whose only purpose is to copy static metadata into processing
  data.
- Validate at boundaries and fail explicitly for unsupported capabilities.
- Make operational optimizations preserve the same `BaseData` semantics as
  ordinary execution.
- Keep instrument-specific pipelines, notebooks, and datasets in
  [MoDaCor-examples](https://github.com/BAMResearch/MoDaCor-examples).
- Add tests and documentation with a public behavior, not as a later cleanup.
