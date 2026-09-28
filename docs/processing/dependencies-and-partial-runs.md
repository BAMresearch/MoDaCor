# Dependencies and partial reruns

Each step declares a `ProcessStepDependencies` contract containing:

- external `source_refs`;
- `processing_reads`; and
- `processing_writes`.

Many contracts are derived from `ProcessStepDescriber.arguments` dependency
roles. Steps override the contract only for relationships the schema cannot
express. Wildcards are conservative fallbacks, not substitutes for dependencies
that can be known exactly.

## Full execution

A full run executes every graph node in topological order. Use it for the first
run, after pipeline configuration changes, or whenever the changed input scope
is uncertain.

## Selected execution

The local runner can execute selected step identifiers against caller-supplied
`ProcessingData`. The caller is responsible for providing valid prerequisites.
This is mainly useful for testing and controlled interactive work.

## Partial execution

The runtime service can reuse a previous result:

1. Identify steps that read a changed source or processing path.
2. Expand the dirty set to all downstream descendants.
3. Copy the previous `ProcessingData` snapshot.
4. Execute only the dirty subgraph.

`auto` mode attempts this path and falls back to a full run when partial
execution fails. A dry run previews the selected steps without mutating session
state.

Exact contracts matter: an undeclared read can incorrectly reuse stale data,
while an overly broad wildcard discards valid reuse. Extension authors should
assert all three dependency sets in focused tests.
