# Architecture Upgrade Plan

This document tracks the design issues identified during the September 2026
architecture review. Keep the status and notes current as each item is fixed.

Status values:

- `Open`: not started.
- `In progress`: actively being changed.
- `Done`: implemented and verified.
- `Deferred`: intentionally postponed with a reason.

## Items

1. `Open` Runtime API trust boundary

   The runtime API currently accepts local filesystem paths for pipeline YAML and
   custom source/sink class import paths from client payloads. This is acceptable
   only for a trusted local helper. A network-facing service needs path
   allowlists, disabled or registry-backed custom imports, and an explicit auth
   model.

2. `Done` Partial rerun invalidation contract

   Dirty-step selection should be based on an explicit per-step dependency
   contract instead of server-owned config heuristics. Steps should declare which
   source refs they read and which `ProcessingData` keys they read or write.

   Outcome: `ProcessStep.dependency_contract()` now provides the partial-rerun
   dependency contract. Boundary steps declare exact contracts for source
   loading, source registration, sink registration, and sink exports. Runtime
   dirty-step detection now consumes the contract and expands matching seeds to
   downstream descendants.

3. `Done` Pipeline scheduler ownership

   `Pipeline` currently inherits from `graphlib.TopologicalSorter`, which mixes
   graph data with one-shot scheduler state. Prefer keeping `Pipeline` as the
   graph/spec holder and creating a fresh scheduler per execution.

   Outcome: `Pipeline` no longer subclasses `TopologicalSorter`. It owns the
   graph and exposes `create_scheduler()` for fresh per-use schedulers. Existing
   manual scheduling helpers remain as compatibility wrappers, and the shared
   runner now uses a local scheduler per job.

4. `Done` Process-step configuration schema

   Step configuration is split between `CONFIG_KEYS` and
   `ProcessStepDescriber.arguments`. Constructor-provided configuration is
   overwritten during initialization, and documented argument types are not
   centrally validated.

   Outcome: `ProcessStep` now builds a unified effective configuration schema
   from shared `CONFIG_KEYS` and module-specific
   `ProcessStepDescriber.arguments`. Constructor-provided configuration,
   pipeline YAML/spec configuration, and manual config updates all preserve user
   values and validate unknown keys plus top-level value types through the same
   path. YAML sequences are normalized to Python tuples for configuration keys
   that explicitly declare `tuple` as their runtime type. `CONFIG_KEYS` remains
   the compatibility location for shared/base options.

5. `Done` Explicit in-place step execution

   `calculate()` is documented as returning a mapping that `execute()` merges,
   but several modules mutate shared `ProcessingData` before that merge. Partial
   rerun rollback and trace semantics would be clearer with an explicit commit
   boundary.

   Outcome: MoDaCor now treats in-place `ProcessingData` mutation as the
   authoritative process-step contract. `execute()` no longer merges returned
   mappings back into `ProcessingData`; it stores them only as optional
   `produced_outputs` bookkeeping. Built-in steps that previously relied on the
   merge path now update `self.processing_data` directly. Runtime rollback stays
   at session scope through partial snapshots and full-rerun fallback, avoiding
   per-step full data copies in the normal execution path.

6. `Open` Core data metadata contracts

   `DataBundle` is currently a permissive `dict`, and `BaseData` arithmetic
   inherits axes/rank/weights from the left operand. Mixed-axis operations need
   an explicit metadata policy.

## Update Log

- 2026-09-01: Created tracker and started item 2.
- 2026-09-01: Completed item 2. Verified with targeted `.venv-dev` pytest runs.
- 2026-09-01: Started item 3.
- 2026-09-01: Completed item 3. Verified with full `.venv-dev` pytest run.
- 2026-09-01: Started item 4.
- 2026-09-01: Completed item 4. Verified with full `.venv-dev` pytest run.
- 2026-09-01: Fixed item 4 YAML tuple compatibility regression.
- 2026-09-01: Completed item 5. Verified with full `.venv-dev` pytest run.
