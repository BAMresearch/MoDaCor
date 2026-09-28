# Process steps

A `ProcessStep` is one configured operation in a pipeline graph. A public step
provides `ProcessStepDescriber` metadata for its identifier, purpose, required
data, modified data, configuration fields, defaults, and dependency roles.

At runtime a step receives:

- the shared `ProcessingData` workspace;
- registered `IoSources` and `IoSinks`;
- a stable `step_id`; and
- validated step configuration.

`calculate()` makes authoritative changes directly in `self.processing_data`.
Its optional dictionary return is stored as `produced_outputs` for bookkeeping;
the runner does not merge it into the pipeline workspace. This mutation
boundary is important when writing custom steps or interpreting traces.

Dependencies describe external source references and exact `ProcessingData`
reads and writes where possible. They support invalidation and partial reruns;
they are separate from human-facing `modifies` documentation.

Browse the [module guides](../modules/index.md) to select steps, the
[generated reference](../reference/modules/index.md) for exact configuration,
or the [module author guide](../development/module-author-guide.md) to create
one.
